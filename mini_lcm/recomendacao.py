"""Recomendação: dado o histórico de um usuário numa categoria, ordenar o catálogo.

Oito métodos, do mais ingênuo ao mais completo. Os quatro primeiros são linhas de base:
um método só vale a pena se ganhar deles.

- aleatorio: ordem sorteada. O piso: qualquer coisa deveria ganhar disto.
- popularidade: os mais avaliados primeiro, iguais para todos. Costuma ser difícil
  de bater, porque a demanda se concentra em poucos produtos.
- em_alta: os mais avaliados numa janela recente (por exemplo, os últimos 90 dias antes
  de "agora"), iguais para todos. Captura lançamentos e ciclos de produto que a
  popularidade de todos os tempos não vê.
- ultimo: os mais parecidos (em conteúdo) com o ÚLTIMO produto que a pessoa avaliou.
  Captura "interesse do momento" e é a recomendação mais comum em lojas ("parecidos com
  o que você viu").
- conteudo: os mais parecidos com o PERFIL (média ponderada de todo o histórico; ver
  perfil.py). Aprende o gosto a partir do conteúdo (texto e foto) do que a pessoa avaliou.
- coocorrencia: filtragem colaborativa item-item. "Quem avaliou o que você avaliou
  também avaliou...". Não olha conteúdo nenhum, só interações entre pessoas: é a
  ideia do LCM (aprender com comportamento) na sua forma mais simples.
- hibrido: conteúdo + coocorrência fundidos por RRF, como na busca.
- hibrido_alta: conteúdo + coocorrência + em_alta, por RRF, com peso próprio para a
  lista em alta. Junta o gosto da pessoa com o momento do mercado. Opcionalmente,
  reserva as últimas `vagas_novos` posições do top 10 para produtos NOVOS (sem nenhuma
  interação ainda), escolhidos pelo perfil de conteúdo. Sem a reserva, produtos novos
  quase nunca chegam ao top 10: a lista em alta e a coocorrência só conhecem produtos
  com interações, e a fusão empurra as sugestões de conteúdo para baixo.

Detalhes que valem para todos:
- Produtos que a pessoa já avaliou nunca são recomendados de novo.
- Empate (ex.: na coocorrência, muitos produtos com pontuação zero) é desempatado pela
  popularidade, e não pela ordem do arquivo, que seria arbitrária.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import sparse

from .armazenamento import Armazenamento
from .busca import K_RRF
from .config import ParametrosRecomendacao
from .embeddings import Embedder
from .modelos import CANAIS, Avaliacao, Canal
from .perfil import construir_perfil, pesos_do_historico

METODOS = ("aleatorio", "popularidade", "em_alta", "ultimo", "conteudo", "coocorrencia", "hibrido", "hibrido_alta")
DESEMPATE = 1e-6  # peso da popularidade no desempate: pequeno demais para mudar uma ordem que não seja empate


def _instante(momento: str) -> datetime:
    """ISO 8601 -> datetime com fuso (datas sem fuso são lidas como UTC)."""
    d = datetime.fromisoformat(momento.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def ranks(pontuacoes: np.ndarray) -> np.ndarray:
    """Posição (1 = melhor) de cada item, a partir das pontuações (maior = melhor)."""
    ordem = np.argsort(-pontuacoes, kind="stable")
    r = np.empty(len(pontuacoes), dtype=np.int64)
    r[ordem] = np.arange(1, len(pontuacoes) + 1)
    return r


def rrf_completo(listas: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """RRF sobre o catálogo inteiro: [(pontuações, peso)] -> pontuação fundida."""
    return sum(peso / (K_RRF + ranks(p)) for p, peso in listas)


@dataclass
class Catalogo:
    """Os vetores de uma categoria, alinhados: a linha i de cada matriz é o produto ids[i]."""

    ids: list[str]
    indice: dict[str, int]
    matrizes: dict[Canal, np.ndarray]  # linhas zeradas onde o produto não tem vetor naquele canal
    tem_vetor: dict[Canal, np.ndarray]  # máscara booleana
    lancamentos: list[str | None] | None = None  # data de lançamento de cada produto (AAAA-MM-DD), se houver

    @classmethod
    def carregar(cls, armazenamento: Armazenamento, embedder: Embedder, categoria: str) -> "Catalogo":
        ids = sorted(armazenamento.ids_categoria(categoria))
        indice = {i: n for n, i in enumerate(ids)}
        matrizes, mascaras = {}, {}
        for canal in CANAIS:
            ids_canal, m = armazenamento.matriz(embedder.chave(canal), canal, categoria)
            if not ids_canal:
                continue
            completa = np.zeros((len(ids), m.shape[1]), dtype=np.float32)
            mascara = np.zeros(len(ids), dtype=bool)
            for linha, item_id in zip(m, ids_canal):
                if item_id in indice:
                    completa[indice[item_id]] = linha
                    mascara[indice[item_id]] = True
            matrizes[canal], mascaras[canal] = completa, mascara
        if not matrizes:
            raise ValueError(f"Nenhum vetor para a categoria '{categoria}'. Rode `mini-lcm indexar`.")
        itens = armazenamento.obter_itens(ids)
        lancamentos = [itens[i].lancamento if i in itens else None for i in ids]
        return cls(ids, indice, matrizes, mascaras, lancamentos)

    def vetores_de(self, ids: list[str]) -> dict[Canal, dict[str, np.ndarray]]:
        return {canal: {i: m[self.indice[i]] for i in ids if i in self.indice and self.tem_vetor[canal][self.indice[i]]}
                for canal, m in self.matrizes.items()}


# Meia-vida padrão (em avaliações), escolhida na validação das bonecas: ver docs/resultados-recomendacao.md.
MEIA_VIDA_PADRAO = ParametrosRecomendacao().meia_vida


class Recomendador:
    def __init__(self, catalogo: Catalogo, meia_vida: float | None = None,
                 janela_dias: int = 365, peso_em_alta: float = 1.0, vagas_novos: int = 0):
        """meia_vida: decaimento pela recência no perfil e na coocorrência (None = desligado)."""
        self.catalogo = catalogo
        self.meia_vida = meia_vida
        self.janela_dias = janela_dias
        self.peso_em_alta = peso_em_alta
        self.vagas_novos = vagas_novos
        n = len(catalogo.ids)
        self.disponiveis = np.ones(n, dtype=bool)
        self.popularidade = np.zeros(n)
        self.em_alta = np.zeros(n)
        self.desempate = np.zeros(n)
        self.similaridade: sparse.csr_matrix = sparse.csr_matrix((n, n))

    def ajustar(self, interacoes: list[Avaliacao], agora: str | None = None,
                disponiveis: np.ndarray | None = None) -> "Recomendador":
        """Aprende popularidade, em alta e coocorrência a partir de interações (na avaliação: só as de treino).

        agora: o instante da recomendação (ISO 8601). A janela do em_alta termina nele. Na
        avaliação por data, é a data de corte; sem valor, vale a interação mais recente.
        Interações a partir de `agora` são ignoradas: o futuro não entra em nada.
        Sem `agora`, todas as interações valem e a janela termina na mais recente.
        disponiveis: máscara dos produtos que existem agora (None = todos). As vagas de
        produtos novos só são preenchidas com produtos disponíveis.
        """
        if disponiveis is not None:
            self.disponiveis = np.asarray(disponiveis, dtype=bool)
        idx = self.catalogo.indice
        n = len(self.catalogo.ids)
        validas = [a for a in interacoes if a.item_id in idx]
        if agora is not None:
            validas = [a for a in validas if _instante(a.momento) < _instante(agora)]
            fim = _instante(agora)
        else:
            fim = max((_instante(a.momento) for a in validas), default=datetime.now(timezone.utc))
        inicio = fim - timedelta(days=self.janela_dias)
        usuarios: dict[str, int] = {}
        linhas, colunas = [], []
        recentes: set[tuple[str, str]] = set()
        for a in validas:
            linhas.append(usuarios.setdefault(a.usuario, len(usuarios)))
            colunas.append(idx[a.item_id])
            if _instante(a.momento) >= inicio:
                recentes.add((a.usuario, a.item_id))
        self.em_alta = np.zeros(n)
        for _, item_id in recentes:  # pessoas distintas por produto, dentro da janela
            self.em_alta[idx[item_id]] += 1
        # Matriz usuário x produto com 1 onde houve interação (repetições contam uma vez)
        x = sparse.csr_matrix((np.ones(len(linhas)), (linhas, colunas)), shape=(len(usuarios), n))
        x.sum_duplicates()
        x.data[:] = 1.0
        self.popularidade = np.asarray(x.sum(axis=0)).ravel()
        self.desempate = DESEMPATE * self.popularidade / (self.popularidade.max() or 1.0)
        # Coocorrência: c[i, j] = quantos usuários interagiram com i E com j.
        # Similaridade de cosseno = c[i, j] / sqrt(n_i * n_j): sem a divisão, os produtos
        # populares seriam "parecidos" com tudo só por aparecerem muito.
        c = (x.T @ x).tocsr()
        c.setdiag(0)
        c.eliminate_zeros()
        graus = np.sqrt(np.maximum(self.popularidade, 1))
        d = sparse.diags(1.0 / graus)
        self.similaridade = (d @ c @ d).tocsr()
        return self

    # ───── pontuações (maior = melhor), uma por produto do catálogo ─────

    def _conteudo(self, historico: list[Avaliacao]) -> np.ndarray:
        perfil = construir_perfil(historico, self.catalogo.vetores_de([a.item_id for a in historico]), self.meia_vida)
        listas = []
        for canal, vetor in perfil.items():
            sims = self.catalogo.matrizes[canal] @ vetor
            sims[~self.catalogo.tem_vetor[canal]] = -1.0  # sem vetor neste canal: vai para o fim desta lista
            listas.append((sims, 1.0))
        if not listas:
            return np.zeros(len(self.catalogo.ids))
        return rrf_completo(listas) if len(listas) > 1 else listas[0][0]

    def _coocorrencia(self, historico: list[Avaliacao]) -> np.ndarray:
        pesos = np.zeros(len(self.catalogo.ids))
        for item_id, peso in pesos_do_historico(historico, self.meia_vida).items():
            if item_id in self.catalogo.indice:
                pesos[self.catalogo.indice[item_id]] = peso
        # o desempate entra ANTES de qualquer fusão: na coocorrência, muitos produtos empatam em zero
        return self.similaridade @ pesos + self.desempate

    def pontuar(self, metodo: str, historico: list[Avaliacao], semente: str = "") -> np.ndarray:
        historico = sorted(historico, key=lambda a: a.momento)
        if metodo == "aleatorio":
            s = int.from_bytes(hashlib.sha256(semente.encode()).digest()[:8], "little")
            return np.random.default_rng(s).random(len(self.catalogo.ids))
        if metodo == "popularidade":
            return self.popularidade.astype(float)
        if metodo == "ultimo":
            base = self._conteudo(historico[-1:]) if historico else np.zeros(len(self.catalogo.ids))
        elif metodo == "conteudo":
            base = self._conteudo(historico)
        elif metodo == "coocorrencia":
            base = self._coocorrencia(historico)
        elif metodo == "hibrido":
            base = rrf_completo([(self._conteudo(historico), 1.0), (self._coocorrencia(historico), 1.0)])
        elif metodo == "em_alta":
            base = self.em_alta.astype(float)
        elif metodo == "hibrido_alta":
            conteudo = self._conteudo(historico)
            base = rrf_completo([(conteudo, 1.0), (self._coocorrencia(historico), 1.0),
                                 (self.em_alta + self.desempate, self.peso_em_alta)]) + self.desempate
            if self.vagas_novos > 0:
                return self._reservar_novos(base, conteudo, {a.item_id for a in historico})
            return base
        else:
            raise ValueError(f"Método desconhecido: {metodo}. Use um de {METODOS}")
        return base + self.desempate

    def _reservar_novos(self, base: np.ndarray, conteudo: np.ndarray, ja_tem: set[str], k: int = 10) -> np.ndarray:
        """Reordena para que as últimas `vagas_novos` posições do top k sejam produtos novos.

        Novo = disponível e sem nenhuma interação até agora (popularidade zero). As k - vagas
        primeiras posições seguem a ordem do híbrido; as vagas recebem os novos de maior
        pontuação de conteúdo que ainda não estão no topo; o resto do catálogo vem depois,
        na ordem do híbrido. Devolve pontuações que produzem exatamente essa ordem.
        """
        idx = self.catalogo.indice
        candidatos = self.disponiveis.copy()
        for item_id in ja_tem:
            if item_id in idx:
                candidatos[idx[item_id]] = False
        ordem = [i for i in np.argsort(-base, kind="stable") if candidatos[i]]
        vagas = min(self.vagas_novos, k)
        topo = ordem[: k - vagas]
        no_topo = set(topo)
        novos = np.flatnonzero(candidatos & (self.popularidade == 0))
        novos = [i for i in novos[np.argsort(-conteudo[novos], kind="stable")] if i not in no_topo][:vagas]
        escolhidos = topo + novos
        if len(escolhidos) < k:  # faltaram novos: completa com o híbrido
            usados = set(escolhidos)
            escolhidos += [i for i in ordem if i not in usados][: k - len(escolhidos)]
        pontos = base - base.max() - 1.0  # todos abaixo de zero, na ordem do híbrido
        for posicao, i in enumerate(escolhidos):
            pontos[i] = float(k - posicao)  # o topo escolhido, acima de tudo e na ordem certa
        return pontos

    def recomendar(self, metodo: str, historico: list[Avaliacao], k: int = 10, semente: str = "") -> list[tuple[str, float]]:
        pontos = self.pontuar(metodo, historico, semente).astype(float)
        for a in historico:
            if a.item_id in self.catalogo.indice:
                pontos[self.catalogo.indice[a.item_id]] = -np.inf  # nunca recomendar o que já foi avaliado
        ordem = np.argsort(-pontos, kind="stable")[:k]
        return [(self.catalogo.ids[i], float(pontos[i])) for i in ordem if np.isfinite(pontos[i])]


def explicar(catalogo: Catalogo, recomendado: str, historico: list[Avaliacao], canal: Canal = "texto") -> str | None:
    """O produto do histórico mais parecido com o recomendado: o "porque você avaliou X"."""
    if canal not in catalogo.matrizes or recomendado not in catalogo.indice:
        return None
    alvo = catalogo.matrizes[canal][catalogo.indice[recomendado]]
    melhores = [(float(catalogo.matrizes[canal][catalogo.indice[a.item_id]] @ alvo), a.item_id)
                for a in historico if a.item_id in catalogo.indice and catalogo.tem_vetor[canal][catalogo.indice[a.item_id]]]
    return max(melhores)[1] if melhores else None
