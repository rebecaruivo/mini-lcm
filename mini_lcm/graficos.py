"""Gráficos das grades de peso da imagem (saída de `mini-lcm avaliar --pesos`).

Lê o texto que o `avaliar` imprime (colado num arquivo, ou vindo de um pipe) e desenha, para
cada categoria e conjunto de consultas, o NDCG@10 de cada modelo em função do peso da imagem.

Dois formatos:
- no terminal, com barras de texto (sem dependências);
- em PNG, com matplotlib (extra `graficos`), para guardar ou pôr na documentação.

Um bloco começa numa linha "### EMBEDDER=<modelo> · <consultas>" (o loop com `tee` do
passo-a-passo escreve essa linha) ou na linha "Grade do peso da imagem na fusão · notas: <fonte>
· consultas de <arquivo>". Bloco sem nenhuma das duas recebe o modelo "desconhecido N".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

NOMES = {"gemma2": "Gemma sozinho", "clip+gemma2": "CLIP + Gemma", "clip+e5": "CLIP + e5", "clip": "CLIP",
         "fake": "falso"}
METRICAS = {"ndcg": 3, "recall": 4, "mrr": 5}  # coluna da tabela
TITULO_METRICA = {"ndcg": "NDCG@10", "recall": "recall@10", "mrr": "MRR"}

PERGUNTAS = """Perguntas para ler os gráficos (responda antes de olhar os números de perto):
  1. As linhas começam no mesmo ponto no peso 0? O que é igual entre os sistemas para isso acontecer?
  2. Em cada gráfico, a linha sobe, cai ou fica parada quando a foto ganha peso? Onde está o máximo?
  3. O que isso diz sobre a foto de cada modelo nesta categoria? O que nos produtos explica isso?
  4. Escolha o peso de cada modelo SÓ no conjunto de escolha; depois leia a conferência nesse mesmo
     peso, sem procurar o melhor ponto. Por que essa ordem importa?
  5. Com esses pesos, quem ganha em cada categoria, e por quanto? A diferença é maior que o ruído
     de poucas consultas?
  6. O que mais entra na decisão além do NDCG (custo de indexar, GPU, manutenção, robustez)?
  7. Algum peso extremo (0 ou 1) joga fora um canal? Que tipo de busca sofreria com isso?"""


@dataclass
class Bloco:
    modelo: str
    consultas: str  # "escolha", "conferência" ou o nome do arquivo
    fonte: str  # notas: ia, humano, combinado ("?" se o bloco não disse)
    linhas: list[tuple[str, int, float, float, float, float]] = field(default_factory=list)
    # (categoria, n_consultas, peso, ndcg, recall, mrr)


def _nome_consultas(texto: str) -> str:
    base = Path(texto.strip().split("=")[-1].strip()).name  # aceita "consultas" e "consultas=consultas"
    if base.startswith(("consultas-conferencia", "consultas-validacao")):  # aceita o nome antigo
        return "conferência"
    if base.startswith("consultas"):  # consultas-escolha (ou o nome antigo, consultas.yaml)
        return "escolha"
    return base or "?"


def ler_grades(texto: str) -> list[Bloco]:
    """Separa o texto em blocos (um por tabela) e lê as linhas de cada um."""
    blocos: list[Bloco] = []
    atual: Bloco | None = None
    pendente: dict = {}  # informação de cabeçalho que chegou antes da tabela
    desconhecidos = 0

    for bruta in texto.splitlines():
        linha = bruta.strip()
        m = re.match(r"^#+\s*EMBEDDER=(\S+)\s*(?:·\s*(.+))?$", linha)
        if m:
            atual = None
            pendente = {"modelo": m.group(1)}
            if m.group(2):
                pendente["consultas"] = _nome_consultas(m.group(2))
            continue
        if linha.startswith("Grade do peso"):
            atual = None
            f = re.search(r"notas:\s*(\w+)", linha)
            c = re.search(r"consultas de\s+(\S+)", linha)
            if f:
                pendente["fonte"] = f.group(1)
            if c:
                pendente["consultas"] = _nome_consultas(c.group(1))
            continue
        if not linha.startswith("|"):
            continue
        celulas = [c.strip() for c in linha.strip("|").split("|")]
        if len(celulas) < 6 or celulas[0] in ("Categoria", "") or set(celulas[0]) <= {"-"}:
            continue
        try:
            peso = float(celulas[2].split("←")[0].strip().replace(",", "."))
            n = int(celulas[1])
            ndcg, recall, mrr = (float(x.replace(",", ".")) for x in celulas[3:6])
        except ValueError:
            continue
        if atual is None:
            modelo = pendente.get("modelo")
            if modelo is None:
                desconhecidos += 1
                modelo = f"desconhecido {desconhecidos}"
            atual = Bloco(modelo, pendente.get("consultas", "?"), pendente.get("fonte", "?"))
            blocos.append(atual)
            pendente = {}
        atual.linhas.append((celulas[0], n, peso, ndcg, recall, mrr))
    return blocos


def agrupar(blocos: list[Bloco], metrica: str = "ndcg") -> dict[tuple[str, str], dict[str, dict[float, float]]]:
    """{(categoria, consultas): {modelo: {peso: valor}}}, na ordem em que aparecem."""
    col = METRICAS[metrica]
    grupos: dict[tuple[str, str], dict[str, dict[float, float]]] = {}
    for b in blocos:
        for linha in b.linhas:
            chave = (linha[0], b.consultas)
            grupos.setdefault(chave, {}).setdefault(b.modelo, {})[linha[2]] = linha[col]
    # escolha antes da conferência, mantendo a ordem das categorias
    ordem = {"escolha": 0, "conferência": 1}
    return dict(sorted(grupos.items(), key=lambda kv: ordem.get(kv[0][1], 2)))


def avisos(blocos: list[Bloco]) -> list[str]:
    """Situações em que a comparação entre modelos não é justa."""
    saida = []
    fontes = {b.fonte for b in blocos if b.fonte != "?"}
    if len(fontes) > 1:
        saida.append(f"ATENÇÃO: fontes de notas diferentes ({', '.join(sorted(fontes))}). "
                     "Compare modelos só com a mesma fonte (--fonte ia).")
    for b in blocos:
        if b.modelo.startswith("desconhecido"):
            saida.append(f"Aviso: um bloco sem a linha '### EMBEDDER=' virou '{b.modelo}'. "
                         "Rode pelo loop com tee do passo-a-passo para os nomes saírem certos.")
    # com o mesmo canal de texto, o peso 0 tem que dar o mesmo valor: diferente sugere gabaritos diferentes
    por_grupo: dict[tuple[str, str], set[float]] = {}
    for b in blocos:
        for cat, _, peso, ndcg, _, _ in b.linhas:
            if peso == 0:
                por_grupo.setdefault((cat, b.consultas), set()).add(ndcg)
    for (cat, cons), valores in por_grupo.items():
        if len(valores) > 1:
            saida.append(f"Aviso: em {cat} · {cons}, o peso 0 (só texto) difere entre os modelos "
                         f"({', '.join(f'{v:.3f}' for v in sorted(valores))}). Se o modelo de texto é o mesmo, "
                         "os números vêm de gabaritos diferentes (rodadas de julgar diferentes).")
    return saida


def _rotulo_peso(p: float) -> str:
    return f"{p:g}".replace(".", ",")


def texto_terminal(blocos: list[Bloco], metrica: str = "ndcg", largura: int = 24) -> str:
    """Barras de texto: uma tabela por categoria e conjunto de consultas, uma coluna por modelo."""
    partes = []
    titulo_m = TITULO_METRICA[metrica]
    n_consultas = {(l[0], b.consultas): l[1] for b in blocos for l in b.linhas}
    for (cat, cons), modelos in agrupar(blocos, metrica).items():
        nomes = [NOMES.get(m, m) for m in modelos]
        pesos = sorted({p for serie in modelos.values() for p in serie})
        col = largura + 10
        partes.append(f"\n{cat} · consultas {cons} ({n_consultas[(cat, cons)]}) · {titulo_m} "
                      f"(barra cheia = 1)")
        partes.append("peso  " + "".join(n.ljust(col) for n in nomes))
        melhores = {m: max(serie, key=serie.get) for m, serie in modelos.items()}
        for p in pesos:
            celulas = []
            for m, serie in modelos.items():
                if p not in serie:
                    celulas.append("".ljust(col))
                    continue
                v = serie[p]
                cheio = round(max(0.0, min(1.0, v)) * largura)
                marca = " ←" if melhores[m] == p else ""
                celulas.append((("█" * cheio).ljust(largura) + f" {v:.3f}{marca}").ljust(col))
            partes.append(f"{_rotulo_peso(p):<5} " + "".join(celulas))
    if _saltos(blocos):
        partes.append("\nObs.: os pesos não estão igualmente espaçados (" +
                      ", ".join(_rotulo_peso(p) for p in sorted({l[2] for b in blocos for l in b.linhas})) + ").")
    partes.append("← = maior valor de cada modelo")
    return "\n".join(partes)


def _saltos(blocos: list[Bloco]) -> list[float]:
    pesos = sorted({l[2] for b in blocos for l in b.linhas})
    passos = [round(b - a, 6) for a, b in zip(pesos, pesos[1:])]
    return [p for p in passos if passos and p != min(passos)]


def salvar_png(blocos: list[Bloco], pasta: Path, metrica: str = "ndcg") -> list[Path]:
    """Um PNG por categoria e conjunto de consultas. Exige matplotlib (pip install -e ".[graficos]")."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit('Para PNG, instale o matplotlib: pip install -e ".[graficos]"') from e

    pasta.mkdir(parents=True, exist_ok=True)
    n_consultas = {(l[0], b.consultas): l[1] for b in blocos for l in b.linhas}
    arquivos = []
    for (cat, cons), modelos in agrupar(blocos, metrica).items():
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for m, serie in modelos.items():
            pesos = sorted(serie)
            ax.plot(pesos, [serie[p] for p in pesos], marker="o", label=NOMES.get(m, m))
        ax.set_ylim(0, 1)
        ax.set_xlabel("peso da imagem (0 = só texto, 1 = só foto)")
        ax.set_ylabel(TITULO_METRICA[metrica])
        ax.set_title(f"{cat} · consultas {cons} ({n_consultas[(cat, cons)]})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        nome = re.sub(r"[^a-z0-9]+", "-", f"{cat}-{cons}-{metrica}".lower().replace("ç", "c").replace("ã", "a"))
        caminho = pasta / f"{nome.strip('-')}.png"
        fig.savefig(caminho, dpi=120)
        plt.close(fig)
        arquivos.append(caminho)
    return arquivos
