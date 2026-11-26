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
   ├─ texto → CLIP multilíngue ─┐
   └─ foto  → CLIP imagem ──────┤  vetores de 512 dimensões, no mesmo espaço
                                ▼
                    pgvector (ou arquivos locais)
                                ▲
consulta "boneca gótica até US$ 30"
   │ CLIP multilíngue           │ vizinhos mais próximos em cada canal,
   └────────────────────────────┘ já filtrados por preço/condição/categoria
                                │
                     Reciprocal Rank Fusion + remoção de duplicatas
                                ▼
                          top 10 resultados
```

- **Texto e foto no mesmo espaço:** o CLIP permite comparar a consulta
  diretamente com a foto do produto. Com o codificador multilíngue, consultas
  em português encontram produtos descritos em inglês.
- **Fusão por posição (RRF):** os dois canais têm escalas de similaridade
  diferentes. Combinar pelas posições evita que um canal domine o outro.
- **Filtros antes da ordenação:** "até US$ 30" nunca devolve um produto mais caro.
- **Nada é específico de categoria:** uma categoria nova é uma entrada em `categorias.yaml`.

## Personalização (fase 2)

```text
avaliações reais de cada usuário (nota 1–5, data)
   │
   ├─ perfil = média dos vetores do que avaliou, ponderada pela nota   (conteúdo)
   └─ "quem avaliou o que você avaliou também avaliou..."             (colaborativa)
                     │
          recomendação (híbrido por RRF)  ·  busca reordenada pelo perfil
                     │
   avaliação: esconde a avaliação mais recente de cada pessoa e mede se o sistema a adivinha
```

- **Seis métodos, três deles linhas de base** (aleatório, popularidade, último produto):
  um método só vale se ganhar das réguas simples.
- **A busca personalizada só reordena** o que a consulta trouxe; nunca empurra algo
  sem relação com a busca só porque combina com o gosto.
- **Avaliação temporal sem rótulos:** o passado prevê o futuro, sem vazar a resposta
  (popularidade e coocorrência aprendem só com o treino). Também mede a **cobertura**
  (diversidade) e a **curva de histórico** (usuário novo).
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
`psql "$DATABASE_URL" -f sql/001_schema.sql` uma vez (ou cole o arquivo no SQL
Editor do Supabase) e use os mesmos comandos.

**Testes:** `pytest`. Eles rodam sem rede e sem modelos (com um embedder falso
e determinístico). Para testar o Postgres, use um banco **de teste** (o teste
apaga as tabelas): `DATABASE_URL_TESTE=postgresql://... pytest tests/test_postgres.py`.

## Resultados

*A preencher depois da primeira execução com os dados reais.*

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

**Recomendação** (avaliação temporal, k = 10)

| Método | hit@10 bonecas | hit@10 placas | cobertura bonecas |
|---|---|---|---|
| aleatório | | | |
| popularidade | | | |
| último produto | | | |
| conteúdo (perfil) | | | |
| coocorrência | | | |
| híbrido | | | |

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
- O CLIP local é gratuito, mas fica abaixo dos modelos pagos de ponta em
  texto contra texto. A troca será medida, não suposta.
- Preços em dólar, no momento da coleta do dataset (até 2023).
- Avaliação escrita não é clique nem compra: o sinal de comportamento é mais esparso
  e enviesado que o de um marketplace real. A avaliação temporal corta por usuário,
  não por data única, e a coocorrência aproveita um pouco do futuro de outras pessoas.

## Próximas fases

3. **Personas sintéticas:** usuários simulados com preferências ocultas
   escritas como regras, que servem de gabarito para medir se o sistema aprende.
4. **Avaliação ampliada:** corte temporal por data única (sem vazamento entre
   usuários), efeito da personalização na busca e custo por recomendação.
5. **Opcional:** treinar um modelo pequeno com as interações (two-tower).

## Dados e créditos

Amazon Reviews 2023, McAuley Lab (UC San Diego). Hou, Y., Li, J., He, Z., Yan, A.,
Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and
Recommendation*. arXiv:2403.03952. Os dados não são redistribuídos aqui: o
script baixa direto da fonte.

Modelos: CLIP (OpenAI) e clip-ViT-B-32-multilingual-v1 (sentence-transformers);
juiz: Gemini (Google). Método de calibração do juiz inspirado em Thomas, Spielman,
Craswell & Mitra (2024), *Large Language Models can Accurately Predict Searcher
Preferences*, SIGIR.
