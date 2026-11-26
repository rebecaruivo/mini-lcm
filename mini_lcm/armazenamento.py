"""Onde ficam itens, vetores e custos.

O resto do sistema não sabe onde os dados moram: fala com um `Armazenamento`,
que oferece sempre as mesmas operações. Há duas implementações:

- `ArmazenamentoLocal`: tudo em memória, salvo em arquivos na pasta dados/.
  Busca por força bruta (compara a consulta com todos os vetores). Com alguns
  milhares de itens isso leva milissegundos e não exige instalar nada.
- `ArmazenamentoPostgres`: Postgres com pgvector (Supabase). É o "de verdade":
  filtros em SQL, índice HNSW e dados acessíveis de qualquer lugar.

É o mesmo padrão de adaptadores do Karta (IndexedDB no navegador, memória nos testes).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from .custo import RegistroCusto
from .modelos import Avaliacao, Canal, Filtros, Item

log = logging.getLogger(__name__)


class Armazenamento(Protocol):
    def salvar_itens(self, itens: list[Item]) -> int: ...
    def obter_itens(self, ids: list[str]) -> dict[str, Item]: ...
    def itens_sem_embedding(self, modelo: str, canal: Canal, categoria: str | None = None) -> list[Item]: ...
    def salvar_embeddings(self, modelo: str, canal: Canal, vetores: list[tuple[str, np.ndarray]]) -> None: ...
    def vizinhos(self, modelo: str, canal: Canal, vetor: np.ndarray, filtros: Filtros, n: int) -> list[tuple[str, float]]:
        """Os n itens mais parecidos com o vetor, entre os que passam nos filtros: [(id, similaridade)]."""
        ...
    def salvar_avaliacoes(self, avaliacoes: list[Avaliacao]) -> int:
        """Guarda avaliações sem repetir (mesmo usuário, item e momento). Devolve quantas eram novas."""
        ...
    def avaliacoes_de(self, categoria: str | None = None, usuario: str | None = None) -> list[Avaliacao]:
        """Avaliações (de uma categoria e/ou de um usuário), em ordem cronológica."""
        ...
    def ids_categoria(self, categoria: str) -> set[str]: ...
    def remover_itens(self, ids: list[str]) -> None:
        """Tira produtos do catálogo, junto com os vetores e as avaliações deles."""
        ...
    def remover_avaliacoes_dos_itens(self, ids: list[str]) -> int:
        """Apaga as avaliações destes produtos (os produtos e os vetores ficam). Devolve quantas saíram."""
        ...
    def matriz(self, modelo: str, canal: Canal, categoria: str | None = None) -> tuple[list[str], np.ndarray]:
        """Todos os vetores de um (modelo, canal), opcionalmente de uma categoria: (ids, matriz n x d)."""
        ...
    def vetores(self, modelo: str, canal: Canal, ids: list[str]) -> dict[str, np.ndarray]:
        """Os vetores de alguns produtos (os que existirem)."""
        ...
    def registrar_custo(self, registro: RegistroCusto) -> None: ...
    def custos(self) -> list[RegistroCusto]: ...
    def estatisticas(self) -> dict: ...
    def fechar(self) -> None: ...


# ─────────────────────────── Local (arquivos) ───────────────────────────


class ArmazenamentoLocal:
    """Itens em itens.jsonl, vetores em embeddings/<modelo>__<canal>.npz,
    avaliações em avaliacoes.jsonl, custos em custos.jsonl."""

    def __init__(self, pasta: Path | None = None):
        self.pasta = pasta
        self.itens: dict[str, Item] = {}
        # (modelo, canal) -> {item_id: vetor}
        self.vetores_: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._custos: list[RegistroCusto] = []
        self.avaliacoes: dict[tuple[str, str, str], Avaliacao] = {}
        if pasta and pasta.exists():
            self._carregar()

    # persistência
    def _carregar(self) -> None:
        arq = self.pasta / "itens.jsonl"
        if arq.exists():
            for linha in arq.read_text(encoding="utf-8").splitlines():
                if linha.strip():
                    item = Item(**json.loads(linha))
                    self.itens[item.id] = item
        for npz in (self.pasta / "embeddings").glob("*.npz") if (self.pasta / "embeddings").exists() else []:
            modelo, canal = npz.stem.split("__")
            dados = np.load(npz, allow_pickle=False)
            self.vetores_[(modelo, canal)] = dict(zip(dados["ids"].tolist(), dados["vetores"]))
        arq = self.pasta / "avaliacoes.jsonl"
        if arq.exists():
            for linha in arq.read_text(encoding="utf-8").splitlines():
                if linha.strip():
                    a = Avaliacao(**json.loads(linha))
                    self.avaliacoes[(a.usuario, a.item_id, a.momento)] = a
        arq = self.pasta / "custos.jsonl"
        if arq.exists():
            self._custos = [RegistroCusto(**json.loads(l)) for l in arq.read_text().splitlines() if l.strip()]

    def _salvar(self, avaliacoes: bool = False) -> None:
        if not self.pasta:
            return
        (self.pasta / "embeddings").mkdir(parents=True, exist_ok=True)
        if avaliacoes:  # o arquivo de avaliações é o maior: só é reescrito quando elas mudam
            (self.pasta / "avaliacoes.jsonl").write_text(
                "\n".join(json.dumps(asdict(a), ensure_ascii=False) for a in self.avaliacoes.values()), encoding="utf-8"
            )
        (self.pasta / "itens.jsonl").write_text(
            "\n".join(json.dumps(asdict(i), ensure_ascii=False) for i in self.itens.values()), encoding="utf-8"
        )
        for (modelo, canal), vetores in self.vetores_.items():
            ids = list(vetores)
            np.savez(self.pasta / "embeddings" / f"{modelo}__{canal}.npz",
                     ids=np.array(ids, dtype=str), vetores=np.array([vetores[i] for i in ids], dtype=np.float32))
        (self.pasta / "custos.jsonl").write_text("\n".join(json.dumps(c.como_dict()) for c in self._custos))

    # operações
    def salvar_itens(self, itens: list[Item]) -> int:
        novos = 0
        for item in itens:
            antigo = self.itens.get(item.id)
            if antigo is None:
                novos += 1
            elif self._mudou_conteudo(antigo, item):
                # título, atributos ou foto mudaram: os vetores antigos não representam mais o anúncio
                for vetores in self.vetores_.values():
                    vetores.pop(item.id, None)
            self.itens[item.id] = item
        self._salvar()
        return novos

    @staticmethod
    def _mudou_conteudo(a: Item, b: Item) -> bool:
        return (a.titulo, a.atributos, a.foto_url) != (b.titulo, b.atributos, b.foto_url)

    def obter_itens(self, ids: list[str]) -> dict[str, Item]:
        return {i: self.itens[i] for i in ids if i in self.itens}

    def itens_sem_embedding(self, modelo: str, canal: Canal, categoria: str | None = None) -> list[Item]:
        feitos = self.vetores_.get((modelo, canal), {})
        return [i for i in self.itens.values()
                if i.id not in feitos and (categoria is None or i.categoria == categoria)
                and (canal == "texto" or i.foto_url)]

    def salvar_embeddings(self, modelo: str, canal: Canal, vetores: list[tuple[str, np.ndarray]]) -> None:
        destino = self.vetores_.setdefault((modelo, canal), {})
        for item_id, v in vetores:
            destino[item_id] = np.asarray(v, dtype=np.float32)
        self._salvar()

    def vizinhos(self, modelo: str, canal: Canal, vetor: np.ndarray, filtros: Filtros, n: int) -> list[tuple[str, float]]:
        vetores = self.vetores_.get((modelo, canal), {})
        ids = [i for i in vetores if i in self.itens and filtros.aceita(self.itens[i])]
        if not ids:
            return []
        matriz = np.stack([vetores[i] for i in ids])
        sims = matriz @ np.asarray(vetor, dtype=np.float32)  # vetores normalizados: produto escalar = cosseno
        ordem = np.argsort(-sims)[:n]
        return [(ids[k], float(sims[k])) for k in ordem]

    def salvar_avaliacoes(self, avaliacoes: list[Avaliacao]) -> int:
        antes = len(self.avaliacoes)
        for a in avaliacoes:
            self.avaliacoes[(a.usuario, a.item_id, a.momento)] = a
        self._salvar(avaliacoes=True)
        return len(self.avaliacoes) - antes

    def avaliacoes_de(self, categoria: str | None = None, usuario: str | None = None) -> list[Avaliacao]:
        saida = [a for a in self.avaliacoes.values()
                 if (usuario is None or a.usuario == usuario)
                 and (categoria is None or (a.item_id in self.itens and self.itens[a.item_id].categoria == categoria))]
        return sorted(saida, key=lambda a: (a.momento, a.usuario, a.item_id))

    def ids_categoria(self, categoria: str) -> set[str]:
        return {i for i, item in self.itens.items() if item.categoria == categoria}

    def remover_itens(self, ids: list[str]) -> None:
        fora = set(ids)
        if not fora:
            return
        for i in fora:
            self.itens.pop(i, None)
        for vetores in self.vetores_.values():
            for i in fora:
                vetores.pop(i, None)
        self.avaliacoes = {k: a for k, a in self.avaliacoes.items() if a.item_id not in fora}
        self._salvar(avaliacoes=True)

    def remover_avaliacoes_dos_itens(self, ids: list[str]) -> int:
        alvo = set(ids)
        antes = len(self.avaliacoes)
        self.avaliacoes = {k: a for k, a in self.avaliacoes.items() if a.item_id not in alvo}
        self._salvar(avaliacoes=True)
        return antes - len(self.avaliacoes)

    def matriz(self, modelo: str, canal: Canal, categoria: str | None = None) -> tuple[list[str], np.ndarray]:
        vetores = self.vetores_.get((modelo, canal), {})
        ids = sorted(i for i in vetores if i in self.itens and (categoria is None or self.itens[i].categoria == categoria))
        if not ids:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.stack([vetores[i] for i in ids])

    def vetores(self, modelo: str, canal: Canal, ids: list[str]) -> dict[str, np.ndarray]:
        todos = self.vetores_de(modelo, canal)
        return {i: todos[i] for i in ids if i in todos}

    def vetores_de(self, modelo: str, canal: Canal) -> dict[str, np.ndarray]:
        return self.vetores_.get((modelo, canal), {})

    def registrar_custo(self, registro: RegistroCusto) -> None:
        self._custos.append(registro)
        self._salvar()

    def custos(self) -> list[RegistroCusto]:
        return list(self._custos)

    def estatisticas(self) -> dict:
        por_categoria: dict[str, int] = {}
        for i in self.itens.values():
            por_categoria[i.categoria] = por_categoria.get(i.categoria, 0) + 1
        return {
            "itens": len(self.itens),
            "por_categoria": por_categoria,
            "vetores": {f"{m}/{c}": len(v) for (m, c), v in self.vetores_.items()},
            "avaliacoes": len(self.avaliacoes),
            "usuarios": len({a.usuario for a in self.avaliacoes.values()}),
        }

    def fechar(self) -> None:
        pass


# ─────────────────────────── Postgres + pgvector ───────────────────────────


def _vetor_sql(v: np.ndarray) -> str:
    """pgvector aceita o vetor como texto '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7f}" for x in np.asarray(v, dtype=np.float32)) + "]"


class ArmazenamentoPostgres:
    def __init__(self, url: str, categorias: list[tuple[str, str]] | None = None):
        try:
            import psycopg
        except ImportError as e:
            raise RuntimeError('Instale o driver do Postgres com: pip install -e ".[postgres]"') from e
        self.conn = psycopg.connect(url, autocommit=True)
        self._busca_iterativa = self._pgvector_versao() >= (0, 8, 0)
        if categorias:
            self._garantir_categorias(categorias)

    def _pgvector_versao(self) -> tuple[int, ...]:
        linha = self.conn.execute("select extversion from pg_extension where extname = 'vector'").fetchone()
        if not linha:
            raise RuntimeError("A extensão pgvector não está instalada. Rode os arquivos de sql/ primeiro (001_schema.sql, 002_lancamento.sql).")
        return tuple(int(p) for p in linha[0].split(".")[:3])

    def _garantir_categorias(self, categorias: list[tuple[str, str]]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                "insert into categorias (slug, nome) values (%s, %s) on conflict (slug) do update set nome = excluded.nome",
                categorias,
            )

    def garantir_indice(self, modelo: str, canal: Canal, dimensoes: int) -> None:
        """Índice HNSW parcial para um (modelo, canal), com o tamanho fixo que o índice exige.

        Comandos como CREATE INDEX não aceitam parâmetros (%s): os valores entram como
        literais, escapados pelo próprio psycopg (sql.Literal), nunca colados como texto.
        """
        from psycopg import sql

        nome = "emb_hnsw_" + "".join(c if c.isalnum() else "_" for c in f"{modelo}_{canal}")[:50]
        self.conn.execute(sql.SQL(
            "create index if not exists {nome} on embeddings_itens "
            "using hnsw ((vetor::vector({d})) vector_cosine_ops) "
            "where modelo = {modelo} and canal = {canal}"
        ).format(nome=sql.Identifier(nome), d=sql.Literal(int(dimensoes)),
                 modelo=sql.Literal(modelo), canal=sql.Literal(canal)))

    def salvar_itens(self, itens: list[Item]) -> int:
        sql = """
            insert into itens (id, fonte, categoria, titulo, preco, moeda, condicao, vendedor_id, link,
                               foto_url, atributos, chave_duplicata, coletado_em, lancamento)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            on conflict (id) do update set
              categoria = excluded.categoria, titulo = excluded.titulo, preco = excluded.preco,
              moeda = excluded.moeda, condicao = excluded.condicao, vendedor_id = excluded.vendedor_id,
              link = excluded.link, foto_url = excluded.foto_url, atributos = excluded.atributos,
              chave_duplicata = excluded.chave_duplicata, lancamento = excluded.lancamento,
              atualizado_em = now()
            returning (xmax = 0) as novo
        """
        # (xmax = 0) é um truque do Postgres: vale true quando a linha acabou de ser inserida
        # e false quando o "on conflict" a atualizou. Serve só para contar quantos itens são novos.
        # Se título, atributos ou foto mudaram, os vetores antigos não representam mais o anúncio
        apagar_vetores_se_mudou = """
            delete from embeddings_itens e using itens i
            where e.item_id = i.id and i.id = %s
              and (i.titulo, i.atributos, coalesce(i.foto_url, '')) is distinct from (%s, %s::jsonb, coalesce(%s, ''))
        """
        novos = 0
        with self.conn.transaction(), self.conn.cursor() as cur:
            for i in itens:
                atributos = json.dumps(i.atributos, ensure_ascii=False)
                cur.execute(apagar_vetores_se_mudou, (i.id, i.titulo, atributos, i.foto_url))
                cur.execute(sql, (i.id, i.fonte, i.categoria, i.titulo, i.preco, i.moeda, i.condicao, i.vendedor_id,
                                  i.link, i.foto_url, atributos, i.chave_duplicata, i.coletado_em, i.lancamento))
                novos += 1 if cur.fetchone()[0] else 0
        return novos

    _COLUNAS = ("id, categoria, titulo, preco, moeda, condicao, vendedor_id, link, foto_url, atributos, fonte, "
                "coletado_em, lancamento")

    @staticmethod
    def _item(linha) -> Item:
        (id_, categoria, titulo, preco, moeda, condicao, vendedor, link, foto, atributos, fonte, coletado,
         lancamento) = linha
        return Item(id=id_, categoria=categoria, titulo=titulo, preco=float(preco) if preco is not None else None,
                    moeda=moeda, condicao=condicao, vendedor_id=vendedor, link=link, foto_url=foto,
                    atributos=atributos or {}, fonte=fonte, coletado_em=coletado.isoformat(),
                    lancamento=lancamento.isoformat() if lancamento else None)

    def obter_itens(self, ids: list[str]) -> dict[str, Item]:
        if not ids:
            return {}
        linhas = self.conn.execute(f"select {self._COLUNAS} from itens where id = any(%s)", (ids,)).fetchall()
        return {l[0]: self._item(l) for l in linhas}

    def itens_sem_embedding(self, modelo: str, canal: Canal, categoria: str | None = None) -> list[Item]:
        linhas = self.conn.execute(
            f"""select {self._COLUNAS} from itens i
                where not exists (select 1 from embeddings_itens e
                                  where e.item_id = i.id and e.modelo = %s and e.canal = %s)
                  and (%s::text is null or i.categoria = %s)
                  and (%s = 'texto' or i.foto_url is not null)
                order by i.id""",
            (modelo, canal, categoria, categoria, canal),
        ).fetchall()
        return [self._item(l) for l in linhas]

    def salvar_embeddings(self, modelo: str, canal: Canal, vetores: list[tuple[str, np.ndarray]]) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(
                """insert into embeddings_itens (item_id, modelo, canal, dimensoes, vetor)
                   values (%s, %s, %s, %s, %s::vector)
                   on conflict (item_id, modelo, canal) do update set vetor = excluded.vetor,
                     dimensoes = excluded.dimensoes, criado_em = now()""",
                [(item_id, modelo, canal, len(v), _vetor_sql(v)) for item_id, v in vetores],
            )
        if vetores:  # cada modelo tem seu tamanho de vetor (CLIP: 512, e5-small: 384)
            self.garantir_indice(modelo, canal, len(vetores[0][1]))

    def vizinhos(self, modelo: str, canal: Canal, vetor: np.ndarray, filtros: Filtros, n: int) -> list[tuple[str, float]]:
        d = len(vetor)  # o tamanho vem do modelo da consulta
        # `<=>` é a distância de cosseno do pgvector (0 = idênticos). Similaridade = 1 - distância.
        # A expressão `vetor::vector(d)` e o WHERE em modelo/canal precisam ser iguais aos do
        # índice parcial para o Postgres conseguir usá-lo.
        sql = f"""
            select e.item_id, 1 - (e.vetor::vector({d}) <=> %(q)s::vector({d})) as similaridade
            from embeddings_itens e
            join itens i on i.id = e.item_id
            where e.modelo = %(modelo)s and e.canal = %(canal)s
              and (%(categoria)s::text is null or i.categoria = %(categoria)s)
              and (%(condicao)s::text is null or i.condicao = %(condicao)s)
              and (%(preco_min)s::numeric is null or i.preco >= %(preco_min)s)
              and (%(preco_max)s::numeric is null or i.preco <= %(preco_max)s)
            order by e.vetor::vector({d}) <=> %(q)s::vector({d})
            limit %(n)s
        """
        params = {"q": _vetor_sql(vetor), "modelo": modelo, "canal": canal, "categoria": filtros.categoria,
                  "condicao": filtros.condicao, "preco_min": filtros.preco_min, "preco_max": filtros.preco_max, "n": n}
        with self.conn.transaction():
            if self._busca_iterativa:
                # Com filtros, o índice HNSW pode achar os vizinhos mais próximos e depois descartar
                # quase todos no WHERE, devolvendo menos de n. A partir do pgvector 0.8, a busca
                # iterativa continua procurando até completar n resultados que passam nos filtros.
                self.conn.execute("set local hnsw.iterative_scan = relaxed_order")
            linhas = self.conn.execute(sql, params).fetchall()
        # relaxed_order pode devolver a ordem levemente fora; reordenamos aqui
        return sorted(((l[0], float(l[1])) for l in linhas), key=lambda x: -x[1])

    def salvar_avaliacoes(self, avaliacoes: list[Avaliacao]) -> int:
        novas = 0
        with self.conn.transaction(), self.conn.cursor() as cur:
            for a in avaliacoes:
                cur.execute(
                    """insert into avaliacoes (fonte, usuario_externo, item_id, nota, momento, compra_verificada)
                       values (%s, %s, %s, %s, %s, %s) on conflict do nothing""",
                    (a.fonte, a.usuario, a.item_id, a.nota, a.momento, a.compra_verificada),
                )
                novas += cur.rowcount
        return novas

    def avaliacoes_de(self, categoria: str | None = None, usuario: str | None = None) -> list[Avaliacao]:
        linhas = self.conn.execute(
            """select a.fonte, a.usuario_externo, a.item_id, a.nota, a.momento, a.compra_verificada
               from avaliacoes a join itens i on i.id = a.item_id
               where (%(c)s::text is null or i.categoria = %(c)s) and (%(u)s::text is null or a.usuario_externo = %(u)s)
               order by a.momento, a.usuario_externo, a.item_id""",
            {"c": categoria, "u": usuario},
        ).fetchall()
        return [Avaliacao(usuario=l[1], item_id=l[2], nota=float(l[3]), momento=l[4].isoformat(),
                          compra_verificada=l[5], fonte=l[0]) for l in linhas]

    def ids_categoria(self, categoria: str) -> set[str]:
        return {l[0] for l in self.conn.execute("select id from itens where categoria = %s", (categoria,)).fetchall()}

    def remover_itens(self, ids: list[str]) -> None:
        if ids:  # embeddings_itens e avaliacoes têm "on delete cascade": somem junto
            self.conn.execute("delete from itens where id = any(%s)", (list(ids),))

    def remover_avaliacoes_dos_itens(self, ids: list[str]) -> int:
        if not ids:
            return 0
        return self.conn.execute("delete from avaliacoes where item_id = any(%s)", (list(ids),)).rowcount

    def matriz(self, modelo: str, canal: Canal, categoria: str | None = None) -> tuple[list[str], np.ndarray]:
        linhas = self.conn.execute(
            """select e.item_id, e.vetor::text from embeddings_itens e join itens i on i.id = e.item_id
               where e.modelo = %s and e.canal = %s and (%s::text is null or i.categoria = %s) order by e.item_id""",
            (modelo, canal, categoria, categoria),
        ).fetchall()
        if not linhas:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [l[0] for l in linhas], np.array([json.loads(l[1]) for l in linhas], dtype=np.float32)

    def vetores(self, modelo: str, canal: Canal, ids: list[str]) -> dict[str, np.ndarray]:
        if not ids:
            return {}
        linhas = self.conn.execute(
            "select item_id, vetor::text from embeddings_itens where modelo = %s and canal = %s and item_id = any(%s)",
            (modelo, canal, list(ids)),
        ).fetchall()
        return {l[0]: np.array(json.loads(l[1]), dtype=np.float32) for l in linhas}

    def registrar_custo(self, registro: RegistroCusto) -> None:
        r = registro
        self.conn.execute(
            """insert into custos (etapa, modelo, itens, chamadas, segundos, tokens, custo_usd, falhas, criado_em)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (r.etapa, r.modelo, r.itens, r.chamadas, r.segundos, r.tokens, r.custo_usd, r.falhas, r.criado_em),
        )

    def custos(self) -> list[RegistroCusto]:
        linhas = self.conn.execute(
            "select etapa, modelo, itens, chamadas, segundos, tokens, custo_usd, falhas, criado_em from custos order by id"
        ).fetchall()
        return [RegistroCusto(etapa=l[0], modelo=l[1], itens=l[2], chamadas=l[3], segundos=l[4], tokens=l[5],
                              custo_usd=float(l[6]), falhas=l[7], criado_em=l[8].isoformat()) for l in linhas]

    def estatisticas(self) -> dict:
        por_cat = dict(self.conn.execute("select categoria, count(*) from itens group by 1 order by 1").fetchall())
        vetores = {f"{m}/{c}": n for m, c, n in
                   self.conn.execute("select modelo, canal, count(*) from embeddings_itens group by 1, 2").fetchall()}
        avaliacoes, usuarios = self.conn.execute(
            "select count(*), count(distinct usuario_externo) from avaliacoes").fetchone()
        return {"itens": sum(por_cat.values()), "por_categoria": por_cat, "vetores": vetores,
                "avaliacoes": avaliacoes, "usuarios": usuarios}

    def fechar(self) -> None:
        self.conn.close()
