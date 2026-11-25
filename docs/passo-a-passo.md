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
```bash
mini-lcm importar        # lê os arquivos inteiros: alguns minutos
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
Anote buscas em que os modos dão resultados bem diferentes: são bons exemplos para a entrevista.

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

## 7. Montar o gabarito humano (passo 1 da calibração)
Escolha umas **10 consultas variadas** (5 de cada categoria): algumas fáceis
("monster high draculaura"), algumas descritivas ("boneca de vampira com
roupa rosa"), algumas difíceis ("presente para fã de terror de 10 anos").
Para cada uma:
```bash
mini-lcm rotular "boneca de vampira com roupa rosa" --categoria bonecas --perfil "colecionadora adulta de Monster High"
```
Leia `avaliacao/rubrica.md` antes. Para cada produto, abra o link e a foto e
dê a nota de 0 a 3. "q" para no meio e salva; rodar de novo continua de onde parou.
Meta: pelo menos 50 pares no total (10 consultas × ~20 produtos dá bem mais).

## 8. A IA julga os mesmos pares (passo 2)
```bash
mini-lcm julgar --apenas-rotulados
```

## 9. Medir a concordância (passo 3)
```bash
mini-lcm concordancia
```
- **APROVADO:** siga para o passo 11.
- **NÃO APROVADO:** olhe a matriz e o viés. A IA é generosa demais? Confunde
  1 com 2? Faça o passo 10 e, se o problema for de critério, ajuste
  `avaliacao/rubrica.md` (ex.: deixe mais claro o que é "relacionado") e
  repita os passos 8 e 9. Mudar a rubrica faz a IA rejulgar sozinha.

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
