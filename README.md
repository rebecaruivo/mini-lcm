# mini-lcm

Busca e recomendação de produtos personalizadas, em miniatura, inspiradas no
**Large Commerce Model (LCM)** da Prosus: em vez de aprender com documentos, como um LLM,
o sistema aprende o gosto de cada pessoa a partir do que ela fez. Dados reais da Amazon
(Amazon Reviews 2023) em duas categorias de comportamento oposto: **bonecas**
(colecionismo, compras frequentes) e **placas de vídeo** (compra rara, guiada pela
geração do produto). Nada no código é específico dessas categorias.

O projeto tem três partes:

1. **Busca** em linguagem natural, por texto e por foto, com filtros.
2. **Recomendação** a partir do histórico de cada pessoa.
3. **Avaliação** das duas, feita para não enganar quem a faz.

## Principais resultados

- **Um modelo por canal na busca.** O codificador de texto do CLIP, feito para comparar
  texto com imagem, acertava 30–40% dos nomes buscados nas bonecas; modelos feitos para
  busca de texto (multilingual-e5, EmbeddingGemma 2) acertam 94–96%. Escolher o peso da
  imagem por categoria leva o NDCG@10 médio de 0,62 (peso igual para todas) a 0,79.
- **Testando um modelo novo com a mesma régua.** O EmbeddingGemma 2 no canal de texto
  superou o e5 (NDCG@10 de 0,680 contra 0,652, e 0,580 contra 0,490 na conferência), mas só
  depois de cada modelo receber o seu próprio peso da imagem.
- **O peso da foto depende da categoria e do modelo de texto.** Toda placa parece igual na
  foto: com o e5 no texto, a foto atrapalha nas placas (melhor peso 0) e ajuda nas bonecas
  (0,5). Com o EmbeddingGemma 2, que entende melhor as buscas descritivas, a foto pesa
  menos: 0,4 nas bonecas e 0,2 nas placas, onde quase não muda o resultado.
- **O momento prevê mais que a pessoa.** Na recomendação, "o que está em alta nas últimas
  semanas" ganha de qualquer método personalizado isolado. Juntar o gosto da pessoa ao
  que está em alta é o melhor nas bonecas (hit@10 de 0,126) e não acrescenta nada nas placas.
- **Produtos novos são o ponto cego.** 40% das próximas compras nas bonecas e 65% nas
  placas são de produtos sem nenhuma interação anterior. Só os métodos de conteúdo
  (texto e foto) alcançam algum deles.
- **Três resultados bons demais eram vazamento, e um ruim demais era viés.** A primeira
  avaliação da recomendação dava 0,267; com a correção, 0,063. Um modelo novo parecia bem
  pior e, medido com a mesma régua, ganhou (ver "Como a avaliação evita se enganar").

## Como funciona

### Busca

```text
consulta "boneca gótica de cabelo rosa"  +  filtros (categoria, preço, condição)
   │
   ├─ EmbeddingGemma 2 ─► compara com os TÍTULOS ──► lista A
   └─ CLIP (texto) ─────► compara com as FOTOS ────► lista B
                                                      │
          Reciprocal Rank Fusion, com o peso da imagem da categoria
                         │  remoção de duplicatas
                         ▼
                     top 10 resultados
```

- **Fusão por posição (RRF):** os canais têm escalas de similaridade diferentes;
  combinar pelas posições evita que um domine o outro. O peso da imagem é escolhido por
  categoria (`categorias.yaml`).
- **Filtros antes da ordenação:** "até US$ 30" nunca devolve um produto mais caro.
- **Consultas em português** encontram produtos descritos em inglês (modelos multilíngues).
- **Vetores** em pgvector (Postgres/Supabase) ou em arquivos locais, com a mesma interface.

### Recomendação

```text
histórico de cada pessoa (avaliações com nota e data)
   │
   ├─ perfil: média dos vetores do que avaliou, com peso pela nota e pela recência  (conteúdo)
   ├─ "quem avaliou o que você avaliou também avaliou..."                          (colaborativa)
   └─ o que está em alta nas últimas semanas                                        (momento)
                         │
              fusão por RRF  ·  cada recomendação diz por quê
```

Oito métodos comparados, quatro deles linhas de base (aleatório, popularidade, em alta,
último produto). A mesma ideia de perfil reordena a busca pelo gosto de quem busca, sem
trazer nada que a consulta não pediu.

## Como a avaliação evita se enganar

A maior parte do trabalho foi descobrir quando um número bom não é verdade. Três vezes,
um resultado bom demais era vazamento ou viés de medição:

1. **"Esconder a última avaliação de cada pessoa"**, o protocolo mais comum, dava 0,267 de
   hit@10 à recomendação. Metade dos casos eram compras feitas no mesmo pedido, e o
   modelo aprendia com o futuro de outras pessoas. A troca por **cortes semestrais por
   data** (o modelo só vê o passado; o alvo é a próxima compra) levou o número a 0,063.
2. **Vagas reservadas para produtos novos** pareciam mais que dobrar o acerto nas placas. O
   catálogo só tinha produtos que alguém da amostra ia comprar. Com o **catálogo
   completo** e a data de lançamento dos metadados, as vagas deixaram de compensar.
3. **A rubrica do juiz de IA** foi ajustada olhando os desacordos de um conjunto de
   consultas; medir nelas daria um número otimista. A conferência foi feita em **consultas
   novas**, que não guiaram o ajuste.

E duas armadilhas no sentido contrário, ao comparar o EmbeddingGemma 2 com o e5: o Gemma
parecia bem pior porque os produtos do e5 tinham notas humanas e os dele, só notas da IA,
mais rigorosa; depois, parecia empatar porque usava o peso da imagem escolhido para o e5.
Com a mesma fonte de notas e o peso de cada um, o Gemma ganhou.

Outros cuidados: todo parâmetro é escolhido num **conjunto de escolha** e medido uma vez
num **conjunto de conferência** separado (nos termos usuais, validação e teste); intervalos de confiança por bootstrap (pareado, e por pessoa quando a mesma pessoa
aparece várias vezes); critérios definidos antes de ver os resultados.

**Juiz de IA.** As notas de relevância (0 a 3, [avaliacao/rubrica.md](avaliacao/rubrica.md))
são dadas pelo Gemini e calibradas contra notas humanas dadas às cegas. O critério
combinado (limite inferior do kappa ≥ 0,6) não foi atingido nas consultas novas, por um
efeito conhecido do kappa quando uma nota domina; as conclusões sobre a busca são as
mesmas com as notas da IA e com as humanas, e é para isso que o juiz é usado. Detalhes
em [docs/resultados-busca.md](docs/resultados-busca.md).

Cada decisão, com alternativas e motivos, está em [docs/decisoes.md](docs/decisoes.md).

## Resultados

**Busca** (18 consultas de quatro tipos: nome, visual, funcional e intenção com preço;
[docs/resultados-busca.md](docs/resultados-busca.md))

| Estratégia | NDCG@10, escolha (12 consultas) | NDCG@10, conferência (6 consultas) |
|---|---|---|
| só texto | 0,753 | 0,570 |
| texto + imagem, peso 0,5 para todas | 0,616 | 0,650 |
| **texto + imagem, peso por categoria (bonecas 0,5, placas 0)** | **0,790** | **0,681** |

(Canal de texto com o multilingual-e5, notas combinadas. A troca para o EmbeddingGemma 2
está na tabela do canal de texto, logo abaixo, medida só com notas da IA.)

**Canal de texto** (acerto@10 em buscas por nome; [docs/sondagem.md](docs/sondagem.md))

| Modelo de texto | bonecas | placas (máximo possível: 76%) |
|---|---|---|
| CLIP | 30–40% | 0–6% |
| multilingual-e5 | **94–96%** | **66–68%** |
| EmbeddingGemma 2 | 92–96% | 44–70% |

Na avaliação com gabarito (só notas da IA, cada modelo com o seu melhor peso da imagem):

| Canal de texto | NDCG@10, escolha | NDCG@10, conferência | ms por produto |
|---|---|---|---|
| multilingual-e5 (pesos 0,5 e 0) | 0,652 | 0,490 | 30 |
| **EmbeddingGemma 2 (pesos 0,4 e 0,2)** | **0,680** | **0,580** | 358 |

O EmbeddingGemma 2 é o padrão no texto; o canal de imagem dele leva cerca de 20 s por
foto sem GPU, e o CLIP continua nas fotos.

**Recomendação** (hit@10 na conferência, cortes por data de 2018–2019 a 2023, catálogo completo;
bonecas: 1.876 casos de 1.101 pessoas; placas: 228 casos de 144 pessoas;
[docs/resultados-recomendacao.md](docs/resultados-recomendacao.md))

| Método | bonecas (IC 95%) | placas (IC 95%) |
|---|---|---|
| aleatório | 0,001 (0,000–0,003) | 0,004 (0,000–0,014) |
| popularidade de todos os tempos | 0,009 (0,005–0,013) | 0,035 (0,013–0,063) |
| em alta (janela recente) | 0,104 (0,090–0,119) | **0,083** (0,049–0,122) |
| último produto (conteúdo) | 0,048 (0,038–0,058) | 0,031 (0,009–0,054) |
| híbrido (perfil + coocorrência) | 0,063 (0,052–0,076) | 0,031 (0,012–0,054) |
| **híbrido + em alta** | **0,126** (0,110–0,142) | 0,070 (0,040–0,105) |

**Custo** (CPU de notebook, sem GPU; modelos locais gratuitos)

| Etapa | Itens | ms por item | US$ |
|---|---|---|---|
| vetores de texto (EmbeddingGemma 2, só título) | 13.984 | 358 | 0 |
| vetores de texto (multilingual-e5, só título) | 13.984 | 30 | 0 |
| vetores de imagem (CLIP, com download da foto) | 18.339 | 191 | 0 |
| julgamento de relevância (Gemini 3.5 Flash-Lite) | 629 | 7.067 | ~1,2 milhão de tokens no total |

## Como rodar

Requer Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[clip,gemma,dev]"      # + ".[postgres]" para usar o Supabase
cp .env.example .env                     # EMBEDDER, TEXTO_ITEM e GEMINI_API_KEY

bash scripts/baixar_amazon.sh            # dados (vários GB): Toys_and_Games e Electronics
mini-lcm importar                        # produtos e avaliações das categorias de categorias.yaml
mini-lcm indexar                         # vetores de texto e imagem

mini-lcm buscar "boneca gótica de cabelo rosa" --categoria bonecas --preco-max 30
mini-lcm usuarios --categoria bonecas
mini-lcm buscar "boneca colecionável" --categoria bonecas --usuario USUARIO   # busca personalizada
mini-lcm perfil USUARIO --categoria bonecas                                   # recomendações com o porquê

mini-lcm avaliar                                          # busca, com o gabarito em avaliacao/
mini-lcm avaliar-recomendacao --categoria bonecas --curva # recomendação, cortes por data
mini-lcm custos
```

Passo a passo completo, incluindo rotulagem, calibração do juiz e escolha de
parâmetros: [docs/passo-a-passo.md](docs/passo-a-passo.md).

**Postgres (Supabase):** preencha `DATABASE_URL` no `.env`, rode
`psql "$DATABASE_URL" -f sql/001_schema.sql -f sql/002_lancamento.sql` uma vez e, se os
dados já estiverem em `dados/`, copie-os com `mini-lcm migrar-para-postgres`.

**Testes:** `pytest` (sem rede e sem modelos, com um embedder falso e determinístico).
O teste do Postgres usa um banco de teste, que ele apaga:
`DATABASE_URL_TESTE=postgresql://... pytest tests/test_postgres.py`.

## Estrutura

```text
mini_lcm/
  amazon.py                  importação (centrada em usuários, catálogo completo, datas)
  embeddings.py, indexacao.py  vetores de texto (EmbeddingGemma 2, e5 ou CLIP) e de imagem (CLIP)
  armazenamento.py           arquivos locais ou pgvector, mesma interface
  busca.py                   busca por canal, filtros e fusão RRF
  perfil.py, recomendacao.py perfis, oito métodos de recomendação, explicações
  avaliacao.py               métricas de busca (recall, MRR, NDCG) e pooling
  avaliacao_recomendacao.py  cortes por data, escolha e conferência, bootstrap
  sondagem.py                teste rápido de variantes de embedding, sem rótulos
  juiz.py, calibracao.py     juiz de IA, kappa, amostras de calibração e auditoria
  rotulagem_html.py          página de rotulagem com fotos
  graficos.py                gráficos das grades de peso (terminal ou PNG)
  custo.py, cli.py, config.py  medição de custo, linha de comando, configuração
avaliacao/                   rubrica, consultas e gabarito (notas humanas e da IA)
notebooks/                   tarefas que precisam de GPU (Google Colab)
scripts/                     download dos dados e análises avulsas
docs/                        decisões, resultados e passo a passo
categorias.yaml              categorias, filtros e parâmetros escolhidos no conjunto de escolha
```

## Limitações

- Os produtos da Amazon são novos e bem descritos. O caso típico de classificados
  (usado, título pobre, foto como principal informação) aparece pouco nestes dados.
- As notas humanas de relevância são de uma pessoa só.
- O gabarito da busca sai dos resultados dos próprios modos (pooling): um produto
  relevante que nenhum modo encontrou nunca é avaliado, e o recall é otimista.
- Avaliação escrita não é clique nem compra: o sinal de comportamento é mais esparso
  que o de um marketplace real.
- Só 30% das bonecas têm data de lançamento nos metadados; nas placas, a conferência da
  recomendação tem 228 casos, com intervalos largos.

## Próximos passos

- EmbeddingGemma 2 também nas fotos (um modelo só para texto e imagem), com os vetores das
  fotos calculados numa GPU do Colab (`notebooks/gemma_fotos_colab.ipynb`).
- Fine-tuning do EmbeddingGemma 2 com buscas sintéticas geradas a partir dos produtos,
  medido nas mesmas consultas do gabarito.
- Peso da imagem que depende da consulta (mais imagem em buscas visuais, mais texto em
  buscas por nome), com mais consultas julgadas pela IA para escolher e conferir.
- Personas sintéticas com preferências conhecidas, para medir se o perfil aprende o gosto.
- Treinar um modelo próprio com as interações (two-tower).

## Dados e créditos

Amazon Reviews 2023, McAuley Lab (UC San Diego). Hou, Y., Li, J., He, Z., Yan, A.,
Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and
Recommendation*. arXiv:2403.03952. Os dados não são redistribuídos aqui: o script baixa
direto da fonte.

Modelos: CLIP (OpenAI), clip-ViT-B-32-multilingual-v1 (sentence-transformers) e
multilingual-e5-small (Wang et al., 2024, arXiv:2402.05672), EmbeddingGemma 2 (Google
DeepMind, Apache 2.0); juiz: Gemini (Google).
Calibração do juiz inspirada em Thomas, Spielman, Craswell & Mitra (2024), *Large
Language Models can Accurately Predict Searcher Preferences*, SIGIR.
