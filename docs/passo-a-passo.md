# Passo a passo: da instalação aos números do README

Tempo total estimado: uma tarde, quase toda esperando downloads e modelos.
O trabalho manual de verdade é o passo 7 (rotular), umas 2 horas.

## 0. Pré-requisitos
- Python 3.11 ou mais novo (`python3 --version`).
- Uns 20 GB livres em disco (os arquivos da Amazon são grandes).
- Uma chave do Gemini: https://aistudio.google.com/apikey

## 1. Instalar
```bash
cd mini-lcm
python3 -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -e ".[clip,dev]"
pytest                             # deve terminar com "passed" (alguns "skipped" são normais)
```

## 2. Configurar
```bash
cp .env.example .env
```
No `.env`, preencha `GEMINI_API_KEY`. Para o custo sair em dólar, copie os
preços do `gemini-3.5-flash-lite` (entrada e saída, por 1 milhão de tokens)
de https://ai.google.dev/gemini-api/docs/pricing para `JUIZ_PRECO_ENTRADA_1M`
e `JUIZ_PRECO_SAIDA_1M`. Deixe `DATABASE_URL` vazio por enquanto.

## 3. Baixar os dados
```bash
bash scripts/baixar_amazon.sh
```
Pode levar bastante tempo (Electronics é o maior). Se cair, rode de novo: ele continua de onde parou.

## 4. Conferir os filtros das categorias
```bash
mini-lcm inspecionar Toys_and_Games
mini-lcm inspecionar Electronics
```
Procure na lista os nomes que correspondem a bonecas e a placas de vídeo e
confira se batem com `categorias_contem` em `categorias.yaml`. Se o nome real
for diferente (ex.: "Fashion Dolls" ou "Graphics Cards" escritos de outro
jeito), corrija o YAML.

## 5. Importar e indexar
A importação escolhe usuários com histórico (`selecao: usuarios` no `categorias.yaml`)
e passa duas vezes pelo arquivo de avaliações: espere de 10 a 40 minutos por categoria
(Electronics é a mais demorada). Se você já tinha importado com a versão antiga, rode
de novo: o catálogo de cada categoria é substituído, e os vetores de quem continua são
mantidos.
```bash
mini-lcm -v importar     # o -v mostra o progresso (milhões de linhas lidas)
mini-lcm status          # quantos produtos, avaliações e usuários entraram
mini-lcm indexar         # na 1ª vez baixa ~1 GB de modelos; as fotos levam mais tempo
```
Se `indexar` for interrompido, rode de novo: ele só processa o que falta.

## 6. Testar a busca
```bash
mini-lcm buscar "boneca gótica de cabelo rosa" --categoria bonecas
mini-lcm buscar "boneca gótica de cabelo rosa" --categoria bonecas --modo texto
mini-lcm buscar "boneca gótica de cabelo rosa" --categoria bonecas --modo imagem
mini-lcm buscar "placa boa para jogar em 1080p" --categoria placas-de-video --preco-max 300
```
Anote as buscas em que os modos divergem: são casos úteis para analisar o comportamento de cada canal.

## 6b. Escolher a variante de embedding do texto (antes de rotular)
O primeiro teste mostrou o canal de texto fraco (ver `docs/decisoes.md`).
Compare três variantes. Só o texto é reindexado; as fotos são reaproveitadas.
```bash
# variante atual
mini-lcm sondar "monster high" barbie troll "baby doll" "american girl" --categoria bonecas --salvar docs/sondagem.md

# variante 1: só título + loja (~10 min de indexação)
TEXTO_ITEM=titulo mini-lcm indexar --canal texto
TEXTO_ITEM=titulo mini-lcm sondar "monster high" barbie troll "baby doll" "american girl" --categoria bonecas --salvar docs/sondagem.md

# variante 2: modelo de texto dedicado (baixa ~500 MB na 1ª vez)
EMBEDDER=clip+e5 mini-lcm indexar --canal texto
EMBEDDER=clip+e5 mini-lcm sondar "monster high" barbie troll "baby doll" "american girl" --categoria bonecas --salvar docs/sondagem.md

# variante 3: as duas mudanças juntas
EMBEDDER=clip+e5 TEXTO_ITEM=titulo mini-lcm indexar --canal texto
EMBEDDER=clip+e5 TEXTO_ITEM=titulo mini-lcm sondar "monster high" barbie troll "baby doll" "american girl" --categoria bonecas --salvar docs/sondagem.md
```
Faça o mesmo para placas de vídeo, com termos como `"rtx 4060" "rx 7600" "gtx 1650" asus msi`.
Escolha a variante com melhor acerto e faixa de similaridade mais larga,
grave a escolha no `.env` (`EMBEDDER=` e `TEXTO_ITEM=`) e teste as buscas descritivas do passo 6 de novo.
Resultado e escolha (e5 com título): [sondagem.md](sondagem.md).

## 6c. Fase 2: personalização
Guia completo, com os conceitos: [fase-2-personalizacao.md](fase-2-personalizacao.md).
```bash
# quem tem mais histórico (escolha 2 ou 3 usuários bem diferentes para os exemplos)
mini-lcm usuarios --categoria bonecas

# o histórico de um usuário e o que o sistema recomenda, com o motivo
mini-lcm perfil USUARIO --categoria bonecas
mini-lcm perfil USUARIO --categoria bonecas --metodo coocorrencia

# a mesma busca para duas pessoas diferentes
mini-lcm buscar "boneca colecionável" --categoria bonecas --usuario USUARIO_1
mini-lcm buscar "boneca colecionável" --categoria bonecas --usuario USUARIO_2

# escolha de parâmetros na validação (cortes mais antigos; o teste não é usado)
for j in 30 90 180 365 730; do mini-lcm avaliar-recomendacao --categoria bonecas --validacao --metodos em_alta --janela-dias $j; done
for mv in 1 2 3; do for p in 0.25 0.5 1 2 4; do
  mini-lcm avaliar-recomendacao --categoria bonecas --validacao --metodos hibrido_alta --meia-vida $mv --peso-em-alta $p
done; done
for v in 0 1 2 3 4 5; do mini-lcm avaliar-recomendacao --categoria bonecas --validacao --metodos hibrido_alta --vagas-novos $v; done
# os valores escolhidos vão para categorias.yaml (bloco recomendacao)

# teste, uma vez, com os parâmetros de categorias.yaml (cortes semestrais por data)
mini-lcm avaliar-recomendacao --categoria bonecas --curva --salvar docs/resultados-recomendacao.md
mini-lcm avaliar-recomendacao --categoria placas-de-video --curva --salvar docs/resultados-recomendacao.md

# protocolo anterior (última avaliação de cada pessoa) e linha de base sem recência
mini-lcm avaliar-recomendacao --categoria bonecas --curva --corte usuario
mini-lcm avaliar-recomendacao --categoria bonecas --curva --sem-recencia
```
Copie para o README e para a seção 10 do guia da fase 2: a tabela de métodos, a
curva e 1 ou 2 exemplos de `perfil` que mostrem a personalização funcionando (ou falhando).
Commit: `git add -A && git commit -m "Resultados da fase 2: avaliação temporal da recomendação" && git push`

## 7. A IA julga todos os produtos das consultas
As consultas ficam em `avaliacao/consultas.yaml`: 12 (6 de cada categoria), de quatro
tipos (nome, visual, funcional, intenção com filtro de preço), cada uma com o perfil de
quem busca. Use as variantes de embedding escolhidas (`.env` com `EMBEDDER=clip+e5` e
`TEXTO_ITEM=titulo`) e a `GEMINI_API_KEY` preenchida.

```bash
# cria as consultas no gabarito e julga o pool de cada uma (os 10 primeiros de cada modo)
mini-lcm julgar --lote avaliacao/consultas.yaml --profundidade 10
mini-lcm custos
```
O julgamento é salvo a cada 20 produtos; se for interrompido, rodar de novo continua de
onde parou e não paga de novo pelo que já foi julgado.

## 8. Você rotula uma amostra de 100 (sem ver a nota da IA)
```bash
mini-lcm rotular --lote avaliacao/consultas.yaml --html avaliacao/rotular.html --amostra 100
```
A página mostra 100 produtos, agrupados por consulta: 50 sorteados ao acaso (o conjunto
de **calibração**) e 50 sorteados com a mesma quantidade de cada nota da IA (a
**auditoria**). Os dois grupos aparecem misturados, e a nota da IA não aparece.

Abra `avaliacao/rotular.html` no navegador, leia a rubrica no topo e dê as notas com o
teclado (0 a 3 dá a nota e pula para o próximo; N vai ao próximo sem nota). O progresso
fica salvo no navegador. Ao terminar, clique em **Baixar notas** e importe:
```bash
mini-lcm importar-notas /mnt/c/Users/<usuario>/Downloads/notas-rotulagem.json
```

## 9. Medir a concordância
```bash
mini-lcm concordancia                                       # calibração: decide se o juiz é aprovado
mini-lcm concordancia --origem auditoria --min-pares 20     # auditoria: onde ele erra nas notas raras
```
- **APROVADO:** siga para o passo 11.
- **NÃO APROVADO:** olhe a matriz e o viés. A IA é generosa demais? Confunde
  1 com 2? Faça o passo 10 e, se o problema for de critério, ajuste
  `avaliacao/rubrica.md` (ex.: deixe mais claro o que é "relacionado") e
  rode `mini-lcm julgar --lote avaliacao/consultas.yaml --profundidade 10` de novo.
  Mudar a rubrica faz a IA rejulgar sozinha; as suas notas continuam valendo.

Depois de ajustar a rubrica, a concordância nas mesmas consultas fica otimista (a
rubrica foi escrita olhando os desacordos delas). Valide em consultas que não guiaram o
ajuste:
```bash
mini-lcm julgar --lote avaliacao/consultas-validacao.yaml --profundidade 10
mini-lcm rotular --lote avaliacao/consultas-validacao.yaml --html avaliacao/rotular-validacao.html --amostra 60
mini-lcm importar-notas /mnt/c/Users/<usuario>/Downloads/notas-rotulagem.json
mini-lcm concordancia --lote avaliacao/consultas-validacao.yaml --min-pares 25
```

Sem `--amostra`, `rotular --lote` gera uma página com o pool inteiro de cada consulta
(para rotular tudo à mão). Uma consulta avulsa também pode ser rotulada no terminal:
`mini-lcm rotular "boneca de vampira" --categoria bonecas --perfil "colecionadora adulta"`.

## 10. Revisar as discordâncias (passo 4)
```bash
mini-lcm revisar --distancia 2     # primeiro as discordâncias grandes
```
Você vê a nota e a justificativa da IA. Às vezes ela viu algo na foto que
passou batido; às vezes errou. Corrija a sua nota só se ela estava errada.
Depois rode `mini-lcm concordancia` de novo.

## 11. Escalar com a IA
Crie mais consultas e deixe só a IA julgar. Use `--limite` para controlar o custo:
```bash
mini-lcm julgar "boneca colecionável dos anos 2010" --categoria bonecas --perfil "colecionadora adulta"
mini-lcm julgar "placa de vídeo silenciosa para edição" --categoria placas-de-video --limite 60
mini-lcm custos                    # quanto custou até agora
```

## 12. Auditar uma amostra do que só a IA julgou
```bash
mini-lcm auditar --amostra 30
mini-lcm concordancia --origem auditoria --min-pares 20
```
Se a concordância da auditoria for bem pior que a da calibração, a IA vai bem
nas consultas que você rotulou e mal nas outras: volte à rubrica.

## 13. Avaliar a busca e preencher o README
```bash
mini-lcm avaliar --fonte humano --salvar docs/resultados-humano.md
mini-lcm avaliar --fonte combinado --salvar docs/resultados-combinado.md
mini-lcm custos
```
Copie para as tabelas de "Resultados" do README: recall@10, MRR e NDCG@10 por
modo; o kappa da calibração e da auditoria; e o custo (embeddings, busca e julgamento).

## 14. Publicar
```bash
git add -A && git commit -m "Resultados da fase 1"
```
Crie o repositório no GitHub (sem README, ele já existe) e siga as
instruções de "push an existing repository". O `.env` e a pasta `dados/` não
sobem (estão no `.gitignore`). O gabarito (`avaliacao/gabarito.json`) sobe, e
é bom que suba: mostra o trabalho de avaliação.

## Se algo der errado
Rode o comando com `-v` (ex.: `mini-lcm -v importar`) e me mande a saída.
