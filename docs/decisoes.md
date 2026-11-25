# Decisões

Cada decisão com: o que foi escolhido, as alternativas e o motivo. Os conceitos
aparecem explicados na primeira vez em que são usados.

---

## 2026-11-25 · Objetivo: um mini-LCM, com foco em personalização

**Decisão:** construir em miniatura a ideia do Large Commerce Model da Prosus
(lançado em out/2025, desenvolvido primeiro com o iFood e treinado também com
outras empresas do portfólio, entre elas a OLX). LLMs aprendem com documentos;
um LCM aprende com **interações** (o que cada pessoa busca, vê, salva, compra)
para entender intenção e preferência.

O sistema deve funcionar para **vários usuários** e **qualquer categoria de
produto**. O projeto tem 5 fases; este repositório está na fase 1.

| Fase | O quê |
|---|---|
| 1 | Catálogo de 2 categorias + embeddings + busca semântica com filtros |
| 2 | Perfil de usuário a partir das interações + reordenação personalizada |
| 3 | Personas sintéticas com preferências ocultas (gabarito controlado) |
| 4 | Avaliação: NDCG@10, hit rate, curva de cold start, custo |
| 5 | Opcional: treinar um modelo pequeno com as interações (two-tower) |

---

## 2026-11-25 · Fonte de dados: Amazon Reviews 2023

**Decisão:** usar o dataset acadêmico Amazon Reviews 2023 (McAuley Lab, UC San
Diego): 33 domínios, produtos com título, preço, loja, árvore de categorias e
fotos, e **avaliações de usuários reais** (571 milhões no total).

**Alternativas consideradas:**
- *API do Mercado Livre:* descartada. É concorrente direto da OLX; não faz
  sentido apresentar à OLX um projeto construído sobre ela.
- *Raspar a própria OLX:* descartada. Não há API pública de busca, e raspar o
  site viola os termos de uso.
- *OLX Jobs Interactions (Kaggle):* dados da própria OLX, 65 milhões de eventos
  reais, ótimo para personalização. Mas só tem IDs (sem texto nem foto) e uma
  única categoria (empregos). Fica como ideia de extensão.
- *Mercari (Kaggle):* marketplace de usados, mais parecido com a OLX, mas sem
  fotos e sem usuários.

**Motivo:** é o único dataset avaliado que tem ao mesmo tempo conteúdo (texto
e foto, necessários para a busca da fase 1) e comportamento real de muitos
usuários em muitas categorias (necessário para a personalização da fase 2).

**Limitações assumidas:**
- Os produtos são novos, de loja. Anúncios de usados (com título ruim, o caso
  típico da OLX) aparecem menos. O campo `condicao` existe para quando
  houver uma fonte de usados.
- Dados em inglês e preços em dólar. A busca aceita consultas em português
  (ver o modelo de embedding abaixo).
- Uso acadêmico: citar Hou et al. (2024), arXiv:2403.03952.

---

## 2026-11-25 · Categorias como recortes configuráveis

**Decisão:** cada categoria é definida em `categorias.yaml` como um recorte de
um domínio da Amazon (ex.: "bonecas" = Toys_and_Games filtrado pela árvore de
categorias "Dolls" ou pelo título "monster high"). Nenhuma linha de código
conhece uma categoria específica.

**Motivo:** o requisito é não se limitar a um tipo de produto. Adicionar uma
categoria é uma entrada no YAML. O comando `mini-lcm inspecionar <domínio>`
lista os nomes de categoria existentes, para escrever os filtros sem adivinhar.

**Amostragem de reservatório:** quando o filtro aceita mais produtos que o
limite (`max_itens`), a importação sorteia uma amostra uniforme de todos os
aceitos em uma única passada pelo arquivo, em vez de pegar os primeiros (o
arquivo pode estar ordenado por algo que não conhecemos). Com semente fixa, a
importação é reprodutível.

**`min_avaliacoes`:** produtos com pouquíssimas avaliações ficam de fora.
Para a personalização, um produto com que ninguém interagiu ensina pouco.

---

## 2026-11-25 · Python

**Decisão:** Python.

**Alternativa:** TypeScript/Next.js, que já usei no Karta e no Investments Tracker.

**Motivo:** as fases 3 a 5 são simulação, métricas e treino de modelos, onde o
ecossistema de Python (numpy, sentence-transformers, PyTorch) é o padrão.

---

## 2026-11-25 · Embeddings: CLIP local, multilíngue, texto e imagem no mesmo espaço

**Conceito.** Um *embedding* é um vetor (aqui, 512 números) que representa o
significado de um texto ou de uma imagem. Coisas parecidas viram vetores que
apontam em direções parecidas. A *similaridade de cosseno* mede o ângulo entre
dois vetores: 1 = mesma direção, 0 = sem relação. Com vetores normalizados
(comprimento 1), o cosseno é só o produto escalar, que é barato de calcular.

**Decisão:** dois modelos do sentence-transformers rodando localmente:
- imagens: `clip-ViT-B-32` (CLIP original, da OpenAI);
- textos: `clip-ViT-B-32-multilingual-v1`, treinado para colocar textos de
  50+ línguas no mesmo espaço do CLIP.

O CLIP foi treinado com centenas de milhões de pares (imagem, legenda), e por
isso texto e imagem caem **no mesmo espaço**: a consulta "boneca de cabelo
rosa" pode ser comparada diretamente com a foto de um produto. Com o
codificador multilíngue, uma consulta em português encontra títulos em inglês.

**Alternativas:**
- *APIs pagas de embedding de texto* (OpenAI, Gemini, Cohere): melhores para
  texto contra texto, mas cobram por token e, em geral, não comparam texto com
  imagem no mesmo espaço.
- *APIs multimodais pagas:* fazem texto + imagem, mas custam por chamada.
- *Um modelo só de texto:* ignoraria as fotos.

**Motivo:** custo zero (roda no computador), funciona com consultas em
português e compara texto com foto. O preço disso é qualidade menor que a
dos modelos pagos de ponta no texto contra texto. Essa troca será medida
(recall@10) em vez de suposta.

**Trocar de modelo é barato:** os vetores ficam numa tabela separada,
identificados pelo nome do modelo. Um modelo novo gera vetores novos sem
apagar os antigos, e as duas versões podem ser comparadas na avaliação.

**Texto de cada produto:** título + loja + categoria mais específica + três
primeiras características, cortado em 40 palavras. O codificador lê textos
curtos; cortamos nós, para garantir que o título fique.

**Foto:** a principal (variante MAIN), no tamanho "large". O CLIP reduz tudo
para 224x224 pixels, então baixar a versão em alta resolução só gastaria
banda. As fotos são baixadas em paralelo (8 de cada vez), porque baixar é
esperar a rede, não usar processador.

---

## 2026-11-25 · Busca: dois canais combinados por Reciprocal Rank Fusion

**Decisão:** cada produto tem dois vetores (texto e foto). A consulta é
comparada com os dois canais separadamente, e as listas são combinadas por
**RRF**: pontuação = soma de peso / (60 + posição na lista do canal).

**Alternativa:** somar as similaridades dos dois canais.

**Motivo:** as escalas não são comparáveis. No CLIP, texto contra texto costuma
dar similaridades bem mais altas que texto contra imagem (que ficam, em geral,
entre 0,2 e 0,35). Somando,
o canal de texto venceria sempre. O RRF usa só a posição: ser o 1º de uma
lista vale o mesmo em qualquer escala. K = 60 vem do artigo original (Cormack
et al., 2009) e suaviza a diferença entre posições vizinhas.

O comando `buscar` aceita `--modo texto|imagem|fusao` e `--peso-imagem`, e a
avaliação compara os três modos.

**Filtros estruturados** (categoria, preço, condição) são aplicados no banco,
antes da ordenação, e não depois. Assim, "até US$ 30" nunca devolve um produto
de US$ 80 só porque ele era muito parecido.

**Duplicatas:** mesma loja + mesmo título normalizado = o mesmo produto
cadastrado de novo. A busca mostra só o melhor de cada grupo. Produtos iguais
de lojas diferentes não são duplicatas: são ofertas diferentes, e a
personalização precisa poder escolher entre elas.

---

## 2026-11-25 · Armazenamento: arquivos locais para começar, Postgres + pgvector para valer

**Decisão:** duas implementações com as mesmas operações (o padrão de
adaptadores do Karta):
- **local** (padrão, `DATABASE_URL` vazio): tudo em `dados/`, busca por força
  bruta (compara a consulta com todos os vetores). Com alguns milhares de
  itens, isso leva milissegundos e não exige instalar nada.
- **Postgres + pgvector** (Supabase): filtros em SQL e índice vetorial.

**Conceito: índice HNSW.** Comparar a consulta com todos os vetores custa
proporcionalmente ao número de itens. O HNSW (Hierarchical Navigable Small
World) monta um grafo em camadas, em que cada vetor se liga a vizinhos
próximos. A busca começa na camada de cima, com poucos nós e saltos longos, e
desce refinando, como procurar um endereço partindo do país, depois a cidade,
depois a rua. É *aproximado*: troca uma pequena chance de perder um vizinho
por buscas muito mais rápidas. Com 6 mil itens não faz diferença; com
milhões, faz toda a diferença.

**Detalhes do pgvector:**
- A coluna `vetor` não fixa o número de dimensões, porque modelos diferentes
  têm tamanhos diferentes. O índice HNSW exige tamanho fixo, então é criado
  por (modelo, canal), como índice parcial sobre `vetor::vector(512)`.
- Com filtros, o HNSW pode achar os vizinhos e depois descartar quase todos no
  WHERE, devolvendo menos resultados que o pedido. A partir do pgvector 0.8,
  a busca iterativa (`hnsw.iterative_scan`) continua procurando até completar
  a lista. O código liga essa opção quando a versão permite.

---

## 2026-11-25 · Avaliações reais importadas desde já

**Decisão:** importar as avaliações dos produtos escolhidos (usuário anônimo,
nota, data, compra verificada) já na fase 1, numa tabela própria (`avaliacoes`).

**Motivo:** são o sinal real de comportamento com que a fase 2 vai aprender o
gosto de cada pessoa. Ficam separadas de `eventos` (que registra o que
acontece dentro do nosso sistema e é append-only) porque são um histórico que
já existia antes dele.

---

## 2026-11-25 · Avaliação da busca: gabarito com pooling, recall@10 e MRR

**Métricas:**
- **recall@10:** dos produtos relevantes para a consulta, quantos aparecem
  entre os 10 primeiros.
- **MRR** (Mean Reciprocal Rank): 1 / posição do primeiro relevante, na média
  das consultas. Mede se o que importa aparece logo no topo.

**Gabarito:** marcado à mão com `mini-lcm rotular`. Para cada consulta, os
primeiros resultados dos três modos (texto, imagem, fusão) são juntados e
embaralhados (*pooling*). Assim, quem rotula não sabe de qual modo veio cada
item, e nenhum modo ganha a avaliação por construção.

**Limitação:** um produto relevante que nenhum modo encontrou nunca entra no
gabarito. O recall medido é, portanto, um teto otimista.

---

## 2026-11-25 · Custo medido, não estimado

**Decisão:** cada etapa (importação, embeddings de texto, de imagem, busca)
grava tempo, número de itens, chamadas externas, falhas e, quando houver
modelo pago, tokens e US$. `mini-lcm custos` mostra o resumo.

**Motivo:** com o CLIP local o custo em dinheiro é zero e o custo real é tempo.
O registro é o mesmo para comparar com uma API paga no futuro, ponto central
do post da OLX ("custos de token decidem se a solução se paga em escala").

---

## 2026-11-25 · Segurança e dados de terceiros

- Segredos (`DATABASE_URL`) só no `.env`, fora do git.
- `dados/` inteiro fica fora do git: arquivos brutos, produtos importados e vetores.
- Fotos nunca são copiadas: guardamos só a URL e o vetor.
