"""Página de rotulagem: uma alternativa ao `rotular` no terminal para muitas consultas.

`mini-lcm rotular --lote avaliacao/consultas-escolha.yaml --html avaliacao/rotular-escolha.html` monta o
pool de cada consulta (os primeiros de cada modo, juntos e embaralhados, como no
terminal) e grava UMA página HTML autocontida. No navegador, cada produto aparece com a
foto, o título e o preço; as notas são dadas com o teclado (0 a 3) ou com o mouse. O
progresso fica no próprio navegador, e o botão "Baixar notas" gera um JSON que
`mini-lcm importar-notas` junta ao gabarito.

Por que uma página: a rubrica pede para usar a foto como evidência, e no terminal cada
foto é um link para abrir. Com centenas de produtos, isso tornaria a rotulagem lenta e
cansativa, e notas dadas com cansaço são piores.

Com `--amostra N`, a página mostra só N produtos já julgados pela IA (metade sorteada ao
acaso, para a calibração; metade estratificada pela nota da IA, para a auditoria; ver
`calibracao.amostra_calibracao`). A nota da IA não aparece, e as duas metades vêm
misturadas: quem rotula não sabe de qual grupo cada produto veio.

A página não guarda fotos: só as URLs da Amazon, que o navegador carrega.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import yaml

from .avaliacao import ConsultaRotulada, obter_ou_criar
from .modelos import Item

VERSAO_ARQUIVO = 1


def carregar_lote(caminho: Path) -> list[dict]:
    """Lê o YAML de consultas: [{texto, filtros, perfil, tipo}]."""
    dados = yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or {}
    consultas = []
    for c in dados.get("consultas", []):
        if not c.get("texto"):
            raise ValueError(f"Consulta sem texto em {caminho}: {c}")
        consultas.append({"texto": str(c["texto"]), "filtros": dict(c.get("filtros") or {}),
                          "perfil": str(c.get("perfil") or ""), "tipo": str(c.get("tipo") or "")})
    return consultas


def _produto(item: Item, auditoria: bool = False) -> dict:
    return {"id": item.id, "auditoria": auditoria, "titulo": item.titulo, "preco": item.preco_formatado, "foto": item.foto_url or "",
            "link": item.link or "", "loja": item.vendedor_id or "",
            "categoria": item.atributos.get("Categoria", item.categoria)}


def gerar_pagina(lote: list[tuple], rubrica: str) -> str:
    """lote: (consulta, tipo, itens sem nota humana[, ids da auditoria]). Devolve o HTML."""
    dados = []
    for entrada in lote:
        c, tipo, itens = entrada[:3]
        auditoria = entrada[3] if len(entrada) > 3 else set()
        dados.append({"texto": c.texto, "filtros": c.filtros, "perfil": c.perfil, "tipo": tipo,
                      "produtos": [_produto(i, i.id in auditoria) for i in itens]})
    bruto = json.dumps(dados, ensure_ascii=False, sort_keys=True)
    chave = "mini-lcm-rotulagem-" + hashlib.sha256(bruto.encode()).hexdigest()[:12]
    # "</" dentro de um <script> fecharia a tag antes da hora
    dados_js = bruto.replace("</", "<\\/")
    total = sum(len(d["produtos"]) for d in dados)
    return _MODELO.replace("__DADOS__", dados_js).replace("__CHAVE__", chave) \
        .replace("__RUBRICA__", html.escape(rubrica)).replace("__TOTAL__", str(total))


def importar_notas(gabarito: list[ConsultaRotulada], caminho: Path) -> tuple[int, int]:
    """Junta ao gabarito as notas baixadas da página. Devolve (notas, consultas)."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if dados.get("versao") != VERSAO_ARQUIVO:
        raise ValueError(f"Arquivo de notas com versão {dados.get('versao')!r}; esperada {VERSAO_ARQUIVO}.")
    notas = consultas = 0
    for c in dados.get("consultas", []):
        validas = {i: int(n) for i, n in (c.get("notas") or {}).items() if int(n) in (0, 1, 2, 3)}
        if not validas:
            continue
        consulta = obter_ou_criar(gabarito, c["texto"], c.get("filtros") or {}, c.get("perfil") or "")
        consulta.notas_humanas.update(validas)
        for item_id in c.get("auditados") or []:  # vieram da metade estratificada da amostra
            if item_id in validas and item_id not in consulta.auditados:
                consulta.auditados.append(item_id)
        notas += len(validas)
        consultas += 1
    return notas, consultas


_MODELO = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-lcm · rotulagem</title>
<style>
  :root { --fundo:#f6f5f2; --cartao:#fff; --texto:#1d1d1b; --suave:#6b6a66; --borda:#e2e0da;
          --foco:#2f5bd3; --n0:#b42318; --n1:#b54708; --n2:#2e7d32; --n3:#1b5e20; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--fundo); color:var(--texto); }
  header { position:sticky; top:0; z-index:5; background:var(--texto); color:#fff; padding:10px 20px;
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header b { font-size:16px; }
  .barra { flex:1; min-width:160px; height:8px; background:#444; border-radius:4px; overflow:hidden; }
  .barra i { display:block; height:100%; background:#7bd389; width:0; }
  button { font:inherit; cursor:pointer; border-radius:6px; border:1px solid var(--borda); background:#fff; padding:6px 12px; }
  header button { background:#fff; color:var(--texto); border:0; }
  main { max-width:1200px; margin:0 auto; padding:16px 20px 80px; }
  details.rubrica { background:var(--cartao); border:1px solid var(--borda); border-radius:8px; padding:10px 14px; margin-bottom:16px; }
  details.rubrica pre { white-space:pre-wrap; font:13px/1.5 ui-monospace, monospace; }
  .ajuda { color:var(--suave); font-size:13px; margin:0 0 16px; }
  section { margin:28px 0; }
  section h2 { font-size:19px; margin:0 0 4px; }
  .meta { color:var(--suave); font-size:13px; margin-bottom:12px; }
  .grade { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:12px; }
  .produto { background:var(--cartao); border:2px solid var(--borda); border-radius:10px; padding:10px;
             display:flex; flex-direction:column; gap:6px; outline:none; }
  .produto:focus { border-color:var(--foco); box-shadow:0 0 0 3px rgba(47,91,211,.2); }
  .produto img { width:100%; height:180px; object-fit:contain; background:#fafafa; border-radius:6px; }
  .produto .titulo { font-size:13px; max-height:5.8em; overflow:hidden; }
  .produto .preco { font-size:13px; color:var(--suave); }
  .produto a { font-size:12px; color:var(--foco); }
  .notas { display:flex; gap:6px; margin-top:auto; }
  .notas button { flex:1; padding:6px 0; font-weight:600; }
  .produto[data-nota="0"] { border-color:var(--n0); } .produto[data-nota="0"] .notas button[data-n="0"] { background:var(--n0); color:#fff; }
  .produto[data-nota="1"] { border-color:var(--n1); } .produto[data-nota="1"] .notas button[data-n="1"] { background:var(--n1); color:#fff; }
  .produto[data-nota="2"] { border-color:var(--n2); } .produto[data-nota="2"] .notas button[data-n="2"] { background:var(--n2); color:#fff; }
  .produto[data-nota="3"] { border-color:var(--n3); } .produto[data-nota="3"] .notas button[data-n="3"] { background:var(--n3); color:#fff; }
  .produto[data-nota] { opacity:.8; }
</style>
</head>
<body>
<header>
  <b>Rotulagem</b>
  <span id="contagem">0 / __TOTAL__</span>
  <div class="barra"><i id="progresso"></i></div>
  <button id="proximo" title="Ir para o próximo sem nota (N)">Próximo sem nota</button>
  <button id="baixar">Baixar notas</button>
</header>
<main>
  <details class="rubrica"><summary><b>Rubrica</b> (0 irrelevante · 1 relacionado · 2 relevante · 3 exato)</summary><pre>__RUBRICA__</pre></details>
  <p class="ajuda">Teclado: <b>0</b>–<b>3</b> dá a nota e pula para o próximo · <b>←</b> <b>→</b> navegam ·
  <b>N</b> vai ao próximo sem nota · <b>Esc</b> apaga a nota. O progresso fica salvo neste navegador;
  ao terminar (ou ao parar), clique em <b>Baixar notas</b> e rode <code>mini-lcm importar-notas</code> com o arquivo.</p>
  <div id="consultas"></div>
</main>
<script>
const DADOS = __DADOS__;
const CHAVE = "__CHAVE__";
let notas = {};
try { notas = JSON.parse(localStorage.getItem(CHAVE) || "{}"); } catch (e) { notas = {}; }
function salvar() { try { localStorage.setItem(CHAVE, JSON.stringify(notas)); } catch (e) {} }

const raiz = document.getElementById("consultas");
const cartoes = [];
DADOS.forEach((c, ci) => {
  const s = document.createElement("section");
  const filtros = Object.entries(c.filtros).map(([k, v]) => k + ": " + v).join(" · ");
  s.innerHTML = "<h2></h2><div class='meta'></div><div class='grade'></div>";
  s.querySelector("h2").textContent = (ci + 1) + ". " + c.texto;
  s.querySelector(".meta").textContent = [c.tipo && ("tipo: " + c.tipo), filtros, c.perfil && ("perfil: " + c.perfil)]
    .filter(Boolean).join("  —  ") + "  (" + c.produtos.length + " produtos)";
  const grade = s.querySelector(".grade");
  c.produtos.forEach(p => {
    const d = document.createElement("div");
    d.className = "produto"; d.tabIndex = 0; d.dataset.chave = ci + "|" + p.id;
    d.innerHTML = "<img loading='lazy' alt=''><div class='titulo'></div><div class='preco'></div>" +
      "<a target='_blank' rel='noopener'>ver na Amazon</a><div class='notas'>" +
      [0,1,2,3].map(n => "<button data-n='" + n + "'>" + n + "</button>").join("") + "</div>";
    if (p.foto) d.querySelector("img").src = p.foto; else d.querySelector("img").remove();
    d.querySelector(".titulo").textContent = p.titulo;
    d.querySelector(".preco").textContent = [p.preco, p.loja, p.categoria].filter(Boolean).join(" · ");
    const a = d.querySelector("a"); if (p.link) a.href = p.link; else a.remove();
    d.querySelectorAll(".notas button").forEach(b => b.addEventListener("click", () => dar(d, +b.dataset.n, false)));
    grade.appendChild(d); cartoes.push(d);
    if (notas[d.dataset.chave] !== undefined) d.dataset.nota = notas[d.dataset.chave];
  });
  raiz.appendChild(s);
});

function atualizar() {
  const feitos = cartoes.filter(d => d.dataset.nota !== undefined).length;
  document.getElementById("contagem").textContent = feitos + " / " + cartoes.length;
  document.getElementById("progresso").style.width = (100 * feitos / Math.max(1, cartoes.length)) + "%";
}
function proximoSemNota(de) {
  const i0 = de ? cartoes.indexOf(de) : -1;
  for (let k = 1; k <= cartoes.length; k++) {
    const d = cartoes[(i0 + k) % cartoes.length];
    if (d.dataset.nota === undefined) { d.focus(); d.scrollIntoView({block: "center", behavior: "smooth"}); return; }
  }
}
function dar(d, n, avancar) {
  if (n === null) { delete notas[d.dataset.chave]; delete d.dataset.nota; }
  else { notas[d.dataset.chave] = n; d.dataset.nota = n; }
  salvar(); atualizar();
  if (avancar && n !== null) proximoSemNota(d);
}
document.addEventListener("keydown", e => {
  const d = document.activeElement && document.activeElement.classList.contains("produto") ? document.activeElement : null;
  if (e.key === "n" || e.key === "N") { proximoSemNota(d); return; }
  if (!d) return;
  if ("0123".includes(e.key) && e.key.length === 1) { dar(d, +e.key, true); e.preventDefault(); }
  else if (e.key === "Escape") dar(d, null, false);
  else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const i = cartoes.indexOf(d) + (e.key === "ArrowRight" ? 1 : -1);
    if (cartoes[i]) { cartoes[i].focus(); e.preventDefault(); }
  }
});
document.getElementById("proximo").addEventListener("click", () => proximoSemNota(null));
document.getElementById("baixar").addEventListener("click", () => {
  const saida = {versao: 1, gerado_em: new Date().toISOString(), consultas: DADOS.map((c, ci) => {
    const n = {}, aud = [];
    c.produtos.forEach(p => { const v = notas[ci + "|" + p.id];
      if (v !== undefined) { n[p.id] = v; if (p.auditoria) aud.push(p.id); } });
    return {texto: c.texto, filtros: c.filtros, perfil: c.perfil, notas: n, auditados: aud};
  })};
  const url = URL.createObjectURL(new Blob([JSON.stringify(saida, null, 1)], {type: "application/json"}));
  const a = document.createElement("a"); a.href = url; a.download = "notas-rotulagem.json"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
atualizar();
proximoSemNota(null);
</script>
</body>
</html>
"""
