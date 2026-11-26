# mini-lcm

Um motor de personalização em miniatura, inspirado no **Large Commerce Model
(LCM)** da Prosus: em vez de aprender com documentos, como um LLM, ele aprende o
gosto de cada pessoa a partir do que ela faz. Funciona para vários usuários e
para qualquer categoria de produto.

> **Status: fases 1 e 2 de 5.** Fase 1 (busca): embeddings de texto e de imagem,
> busca em linguagem natural com filtros, avaliação com notas de relevância
> minhas e de um juiz de IA calibrado contra elas, e medição de custo.
> Fase 2 (personalização): perfil de cada usuário aprendido com as avaliações
> reais, seis métodos de recomendação comparados entre si, busca reordenada pelo
> gosto da pessoa e avaliação temporal sem rótulos. Os números serão preenchidos
> depois da execução com os dados reais.
> Os números de avaliação e custo serão preenchidos depois da primeira
> execução com os dados reais.

## O problema

Buscar produto por palavra-chave falha de dois jeitos: quando a pessoa descreve
o que quer com outras palavras ("boneca gótica de cabelo rosa"), e quando o
título do produto é pobre. E mesmo uma busca boa devolve a mesma lista para
todos os usuários. O objetivo é chegar a uma lista que entende a consulta, respeita os
filtros e, a partir da fase 2, é ordenada pelo gosto de quem busca.

## Como funciona (fase 1)

```text
Amazon Reviews 2023 (arquivos .jsonl.gz)
   │  importar: filtro por categoria (categorias.yaml) + amostra aleatória
   ▼
produtos ───────────────┐          avaliações reais (para a fase 2)
   │ indexar            │
   ├─ título → multilingual-e5 ───┐  384 dimensões (texto contra texto)
   └─ foto   → CLIP imagem ───────┤  512 dimensões (texto contra imagem)
                                  ▼
                    pgvector (ou arquivos locais)
                                  ▲
consulta "boneca gótica até US$ 30"
   │ e5 (canal texto) e CLIP     │ vizinhos mais próximos em cada canal,
   │ multilíngue (canal imagem)  │ já filtrados por preço/condição/categoria
   └─────────────────────────────┘
                                │
                     Reciprocal Rank Fusion + remoção de duplicatas
                                ▼
                          top 10 resultados
```

- **Consulta contra a foto:** o CLIP põe texto e imagem no mesmo espaço, então a
  consulta é comparada diretamente com a foto do produto. Com o codificador
  multilíngue, consultas em português encontram produtos descritos em inglês.
- **Um modelo por canal:** no canal de texto, o multilingual-e5 (treinado para busca)
  acerta 94% dos nomes testados, contra 30–40% do texto do CLIP
  ([docs/sondagem.md](docs/sondagem.md)). Cada canal compara a consulta só com vetores
  do próprio espaço.
- **Fusão por posição (RRF):** os dois canais têm escalas de similaridade
  diferentes. Combinar pelas posições evita que um canal domine o outro.
- **Filtros antes da ordenação:** "até US$ 30" nunca devolve um produto mais caro.
- **Nada é específico de categoria:** uma categoria nova é uma entrada em `categorias.yaml`.

## Personalização (fase 2)

```text
avaliações reais de cada usuário (nota 1–5, data)
   │
   ├─ perfil = média dos vetores do que avaliou, ponderada pela nota e pela recência (conteúdo)
   ├─ "quem avaliou o que você avaliou também avaliou..."                          (colaborativa)
   └─ o que está em alta nas últimas semanas                                        (momento)
                     │
          recomendação (híbrido por RRF)  ·  busca reordenada pelo perfil
                     │
   avaliação: cortes semestrais por data; o modelo vê só o passado e tenta prever a próxima compra
```

- **Oito métodos, quatro deles linhas de base** (aleatório, popularidade, em alta, último
  produto): um método só vale se ganhar das réguas simples.
- **A busca personalizada só reordena** o que a consulta trouxe; nunca empurra algo
  sem relação com a busca só porque combina com o gosto.
- **Avaliação temporal sem rótulos:** em cada data de corte, o modelo aprende só com o que
  veio antes e tenta prever a próxima compra de cada pessoa. Parâmetros escolhidos num
  período de validação, separado do teste. Também mede a **cobertura** (diversidade), a
  **curva de histórico** (usuário novo) e o acerto em **produtos novos**.
- **Cada recomendação diz por quê** ("parecido com: ...").

Conceitos, escolhas e limitações: [docs/fase-2-personalizacao.md](docs/fase-2-personalizacao.md).

## Como a busca é avaliada

Rotular relevância à mão não escala, e confiar cegamente numa IA também não.
O projeto usa as duas coisas, nesta ordem:

1. **Eu rotulo um conjunto pequeno** (notas de 0 a 3, rubrica em
   [avaliacao/rubrica.md](avaliacao/rubrica.md)): o padrão-ouro.
2. **Um juiz de IA** (Gemini, vendo texto + foto + perfil de quem busca) julga os mesmos pares.
3. **Mede-se a concordância** com kappa ponderado e intervalo de confiança.
   A IA só é usada em escala se o limite inferior do kappa passar de 0,6.
4. **Eu arbitro as discordâncias** e audito às cegas uma amostra do que só a IA julgou.

Métricas da busca: recall@10, MRR e NDCG@10, comparando os modos texto, imagem e fusão.

Cada escolha, com alternativas e motivos, está em [docs/decisoes.md](docs/decisoes.md).
O passo a passo completo de execução está em [docs/passo-a-passo.md](docs/passo-a-passo.md).

## Como rodar

Requer Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[clip,dev]"            # + ".[postgres]" para usar o Supabase
cp .env.example .env                     # preencha GEMINI_API_KEY

# 1. Baixar os dados (arquivos grandes, vários GB)
bash scripts/baixar_amazon.sh            # Toys_and_Games e Electronics

# 2. (opcional) Ver quais categorias existem, para ajustar os filtros
mini-lcm inspecionar Toys_and_Games

# 3. Importar produtos e avaliações das categorias de categorias.yaml
mini-lcm importar

# 4. Gerar os embeddings (na primeira vez, baixa ~1 GB de modelos)
mini-lcm indexar

# 5. Buscar
mini-lcm buscar "boneca gótica de cabelo rosa" --categoria bonecas --preco-max 30
mini-lcm buscar "placa para jogar em 1080p" --categoria placas-de-video --modo imagem

# 6. Personalização (fase 2)
mini-lcm usuarios --categoria bonecas
mini-lcm perfil USUARIO --categoria bonecas
mini-lcm buscar "boneca colecionável" --categoria bonecas --usuario USUARIO
mini-lcm avaliar-recomendacao --categoria bonecas --curva
mini-lcm avaliar-recomendacao --categoria bonecas --validacao --janela-dias 90   # escolha de parâmetros

# 7. Avaliar a busca (detalhes em docs/passo-a-passo.md)
mini-lcm rotular "boneca gótica de cabelo rosa" --categoria bonecas --perfil "colecionadora adulta"
mini-lcm julgar --apenas-rotulados
mini-lcm concordancia
mini-lcm revisar
mini-lcm avaliar --salvar docs/resultados-busca.md

# Custos e tamanho dos dados
mini-lcm custos
mini-lcm status
```

**Com Postgres (Supabase):** preencha `DATABASE_URL` no `.env`, rode
`psql "$DATABASE_URL" -f sql/001_schema.sql -f sql/002_lancamento.sql` uma vez (ou cole os arquivos no SQL
Editor do Supabase) e use os mesmos comandos.

**Testes:** `pytest`. Eles rodam sem rede e sem modelos (com um embedder falso
e determinístico). Para testar o Postgres, use um banco **de teste** (o teste
apaga as tabelas): `DATABASE_URL_TESTE=postgresql://... pytest tests/test_postgres.py`.

## Resultados

*Preenchido à medida que cada etapa roda com os dados reais.*

**Sondagem do canal de texto** (5 nomes por categoria; acerto@10 = resultados com o nome
no título. Detalhes em [docs/sondagem.md](docs/sondagem.md))

| Variante de texto | bonecas: só texto | bonecas: fusão | placas: só texto | placas: fusão |
|---|---|---|---|---|
| CLIP, texto completo | 40% | 52% | 6% | 2% |
| CLIP, só título | 30% | 52% | 0% | 0% |
| multilingual-e5, texto completo | **96%** | 88% | **68%** | **38%** |
| multilingual-e5, só título (escolhida) | 94% | **96%** | 66% | **38%** |
| teto (produtos com o nome) | 100% | 100% | 76% | 76% |

Na recomendação, que compara produto com produto, as variantes empatam. Nas placas, que
parecem todas iguais na foto, a imagem atrapalha a fusão: o peso da imagem deve variar
por categoria.

**Busca** (notas combinadas: minhas onde existem, do juiz no resto)

| Métrica | texto | imagem | fusão |
|---|---|---|---|
| recall@10 | | | |
| MRR | | | |
| NDCG@10 | | | |

**Juiz de IA**

| | Pares | Acordo exato | Kappa ponderado (IC 95%) |
|---|---|---|---|
| Calibração | | | |
| Auditoria às cegas | | | |

**Recomendação** (cortes semestrais por data, k = 10, catálogo completo da categoria;
parâmetros escolhidos na validação. Bonecas: 1.876 casos de 1.101 pessoas, 2019 a 2023;
placas: 228 casos de 144 pessoas, 2018 a 2023. Detalhes em
[docs/resultados-recomendacao.md](docs/resultados-recomendacao.md))

| Método | hit@10 bonecas (IC 95%) | hit@10 placas (IC 95%) | bonecas: alvos novos / já vistos |
|---|---|---|---|
| aleatório | 0,001 (0,000–0,003) | 0,004 (0,000–0,014) | 0,000 / 0,002 |
| popularidade | 0,009 (0,005–0,013) | 0,035 (0,013–0,063) | 0,000 / 0,014 |
| em alta | 0,104 (0,090–0,119) | **0,083** (0,049–0,122) | 0,000 / 0,173 |
| último produto | 0,048 (0,038–0,058) | 0,031 (0,009–0,054) | **0,031** / 0,059 |
| conteúdo (perfil) | 0,045 (0,036–0,055) | 0,022 (0,004–0,043) | 0,025 / 0,058 |
| coocorrência | 0,051 (0,042–0,062) | 0,018 (0,004–0,036) | 0,000 / 0,085 |
| híbrido | 0,063 (0,052–0,076) | 0,031 (0,012–0,054) | 0,004 / 0,103 |
| híbrido + em alta | **0,126** (0,110–0,142) | 0,070 (0,040–0,105) | 0,000 / **0,209** |

- **O momento prevê mais que a pessoa:** a popularidade das últimas semanas ganha de
  qualquer método personalizado isolado. Nas bonecas, somar o gosto da pessoa ao que está
  em alta acrescenta 0,021 (IC 95% de +0,009 a +0,034); nas placas, nada.
- **Produtos novos** (40% dos alvos nas bonecas, 65% nas placas) só são alcançados pelos
  métodos de conteúdo, e por pouco (cerca de 3%). Reservar vagas do top 10 para eles não
  compensou. É o caso típico de classificados, e onde melhores embeddings fariam diferença.
- O primeiro protocolo (esconder a última avaliação de cada pessoa) dava 0,267 ao híbrido
  das bonecas: compras feitas juntas e vazamento entre pessoas inflavam o resultado.

**Custo**

| Etapa | Itens | Tempo | ms/item | US$ |
|---|---|---|---|---|
| embeddings de texto | | | | 0 |
| embeddings de imagem | | | | 0 |
| busca (por consulta) | | | | 0 |
| julgamento (por par) | | | | |

## Limitações

- Os produtos da Amazon são novos e bem descritos. O caso típico de
  classificados (usado, título ruim) aparece pouco nestes dados.
- O padrão-ouro é de uma pessoa só. A concordância mede o quanto a IA
  concorda comigo, não com "a verdade".
- O gabarito sai dos resultados dos próprios modos (pooling): um produto
  relevante que nenhum modo encontrou nunca é avaliado. O recall é um teto otimista.
- Os modelos locais (CLIP e multilingual-e5-small) são gratuitos, mas ficam abaixo dos
  modelos pagos de ponta. A troca seria medida, não suposta.
- Preços em dólar, no momento da coleta do dataset (até 2023).
- Avaliação escrita não é clique nem compra: o sinal de comportamento é mais esparso
  e enviesado que o de um marketplace real.
- Só 30% das bonecas têm data de lançamento nos metadados; produto novo sem data não
  concorre na avaliação por data (nas placas, 96% têm data).
- Nas placas, o teste tem só 228 casos: os intervalos de confiança são largos.

## Próximas fases

3. **Personas sintéticas:** usuários simulados com preferências ocultas
   escritas como regras, que servem de gabarito para medir se o sistema aprende.
4. **Avaliação ampliada:** efeito da personalização na busca e custo por recomendação.
   (O corte por data e o teste de vagas para produtos novos já foram feitos na fase 2.)
5. **Opcional:** treinar um modelo pequeno com as interações (two-tower).

## Dados e créditos

Amazon Reviews 2023, McAuley Lab (UC San Diego). Hou, Y., Li, J., He, Z., Yan, A.,
Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and
Recommendation*. arXiv:2403.03952. Os dados não são redistribuídos aqui: o
script baixa direto da fonte.

Modelos: CLIP (OpenAI), clip-ViT-B-32-multilingual-v1 (sentence-transformers) e
multilingual-e5-small (Wang et al., 2024, arXiv:2402.05672);
juiz: Gemini (Google). Método de calibração do juiz inspirado em Thomas, Spielman,
Craswell & Mitra (2024), *Large Language Models can Accurately Predict Searcher
Preferences*, SIGIR.
