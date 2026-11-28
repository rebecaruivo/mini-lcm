# Fase 2: personalização

Documentação técnica da fase 2: o problema, as decisões de projeto, os métodos
implementados, a metodologia de avaliação e as limitações. As decisões resumidas
estão em [decisoes.md](decisoes.md); os comandos, em [passo-a-passo.md](passo-a-passo.md).

## Em uma frase

A fase 1 ordena produtos pelo que a pessoa **digitou**; a fase 2 ordena também pelo
que a pessoa **fez antes**. É a ideia do Large Commerce Model da Prosus (aprender
com interações, não com documentos) em escala de projeto pessoal.

## Sumário

1. [O problema dos dados e a importação centrada em usuários](#1-o-problema-dos-dados)
2. [O perfil do usuário](#2-o-perfil-do-usuário)
3. [Oito formas de recomendar](#3-oito-formas-de-recomendar)
4. [Busca personalizada](#4-busca-personalizada)
5. [Como medir sem rótulos: a avaliação temporal](#5-como-medir-sem-rótulos-a-avaliação-temporal)
6. [Como ler os resultados](#6-como-ler-os-resultados)
7. [Limitações](#7-limitações)
8. [Discussão técnica](#8-discussão-técnica)
9. [Glossário](#9-glossário)
10. [Resultados](#10-resultados)

---

## 1. O problema dos dados

Na fase 1, o catálogo foi montado **sorteando produtos**: 3.000 bonecas e 3.000
placas de vídeo. Com eles vieram 136.597 avaliações de 120.326 usuários, mas:

| Avaliações do usuário na mesma categoria | Usuários |
|---|---|
| 1 | 110.489 |
| 2 | 7.296 |
| 3 ou 4 | 2.003 |
| 5 ou mais | 660 |

Com uma avaliação só, não há gosto a aprender, nem passado para separar do futuro
na avaliação. Isso acontece porque, sorteando produtos, a chance de pegar vários
produtos da MESMA pessoa é pequena.

### A correção: escolher pessoas, depois produtos

`importar_por_usuarios` (em `mini_lcm/amazon.py`):

1. **Candidatos:** todos os produtos que passam no filtro da categoria. Os que casam
   com `titulo_contem` ("monster high") são **garantidos** no catálogo, até um terço
   dele, para a busca da fase 1 continuar com esses produtos.
2. **1ª passada pelas avaliações:** conta quantas avaliações cada usuário fez em candidatos.
3. **Elegíveis:** quem tem pelo menos `min_avaliacoes_usuario` (5) avaliações na categoria.
4. **2ª passada:** guarda o histórico completo dos elegíveis.
5. **Seleção gulosa:** em ordem aleatória (semente fixa, reprodutível), cada elegível
   entra **com todos os produtos que avaliou**, enquanto couber em `max_itens`. Quem
   estouraria o limite é pulado, nunca cortado: um histórico cortado pela metade faria
   a avaliação temporal esconder um produto que talvez nem esteja no catálogo.
6. **Avaliações importadas:** as de todos os elegíveis sobre produtos do catálogo.
   Mesmo quem não foi "selecionado" ajuda a calcular popularidade e coocorrência.

Detalhe de engenharia: o arquivo de avaliações de Electronics tem dezenas de milhões
de linhas, e interpretar JSON é a parte cara. Uma expressão regular extrai o id do
produto direto do texto, e só as linhas de produtos candidatos são interpretadas.

A reimportação **substitui** o catálogo da categoria. Os produtos que saem levam
junto vetores e avaliações; os que continuam mantêm os vetores, e o `indexar`
seguinte só processa os novos.

---

## 2. O perfil do usuário

`mini_lcm/perfil.py`

### A ideia

Cada produto já tem vetores (fase 1): um do texto, um da foto. Produtos parecidos
têm vetores parecidos. O **perfil** de uma pessoa é a média ponderada dos vetores
dos produtos que ela avaliou. Ele aponta para a "região" do espaço de que ela gosta.
Recomendar vira buscar os vizinhos do perfil, a mesma operação da busca. A
diferença é que a "consulta" agora é a pessoa.

### O peso de cada avaliação

    peso = 0,5 + (nota - 3) / 2

| Nota | Peso | Leitura |
|---|---|---|
| 5 | 1,5 | gostou muito: puxa forte |
| 4 | 1,0 | gostou |
| 3 | 0,5 | neutro, mas se interessou a ponto de comprar e avaliar |
| 2 | 0,0 | não puxa nem empurra |
| 1 | -0,5 | empurra o perfil para longe |

**Por que a base de 0,5?** Avaliar já é sinal de interesse: a pessoa comprou ou
considerou o produto. Sem a base, a nota 3 valeria zero, e quem dá só notas
medianas ficaria sem perfil. É uma escolha de projeto, testável: a avaliação
temporal diz se ela ajuda.

### Exemplo em duas dimensões

Imagine vetores com 2 números, onde o 1º mede "gótico" e o 2º mede "bebê":

- boneca vampira, nota 5 → vetor (1, 0), peso 1,5
- boneca bebê, nota 1 → vetor (0, 1), peso -0,5

Perfil = 1,5·(1, 0) + (-0,5)·(0, 1) = (1,5; -0,5), normalizado ≈ (0,95; -0,32).
Ele aponta para o gótico e para longe do bebê. Um produto (0,9; 0,1) tem cosseno
alto com o perfil; um produto (0,1; 0,9), cosseno negativo.

### Um perfil por categoria e por canal

- **Por categoria:** o gosto por bonecas não pode contaminar as recomendações de placas.
- **Por canal:** os vetores de texto (e5 ou CLIP) e os de foto (CLIP) vivem em espaços
  diferentes. Somar os dois seria somar coisas sem relação. Cada canal tem seu perfil,
  e as listas resultantes são fundidas por posição (RRF), como na busca.

### Recência: o interesse atual pesa mais

A média simples trata uma compra de três anos atrás igual à de ontem. Os resultados
(seção 10) mostraram que isso dilui o perfil: quanto mais histórico, pior a
recomendação. Por isso o peso da nota é multiplicado por um decaimento exponencial:

    peso = peso_da_nota × 0,5 ^ (idade / meia_vida)

A **idade** é quantas avaliações vieram depois desta (0 para a mais recente). A
**meia-vida** é a idade em que o peso cai pela metade. Com meia-vida 1, as quatro
avaliações mais recentes valem 1; 0,5; 0,25 e 0,125. A idade conta avaliações, e não
dias, porque o ritmo de compra varia muito entre pessoas: "duas compras atrás" descreve
melhor o interesse atual de quem compra pouco e de quem compra muito.

O mesmo peso vale para a coocorrência. O valor padrão (meia-vida 1) foi escolhido num
conjunto de escolha, sem olhar a conferência (seção 5).

Casos especiais: se a pessoa avaliou o mesmo produto duas vezes, vale a mais
recente. Se a soma dos pesos não é positiva (só notas 1 e 2), não há direção de
gosto, e o canal fica sem perfil.

---

## 3. Oito formas de recomendar

`mini_lcm/recomendacao.py`. As quatro primeiras são **linhas de base** (*baselines*):
um método só se justifica se ganhar delas.

| Método | O que faz | Usa conteúdo? | Usa outras pessoas? |
|---|---|---|---|
| `aleatorio` | ordem sorteada | não | não |
| `popularidade` | os mais avaliados de todos os tempos, iguais para todos | não | sim |
| `em_alta` | os mais avaliados numa janela recente (ex.: 30 dias), iguais para todos | não | sim |
| `ultimo` | parecidos com o último produto avaliado | sim | não |
| `conteudo` | parecidos com o perfil (todo o histórico, com recência) | sim | não |
| `coocorrencia` | "quem avaliou o que você avaliou também avaliou..." | não | sim |
| `hibrido` | conteúdo + coocorrência, fundidos por RRF | sim | sim |
| `hibrido_alta` | conteúdo + coocorrência + em alta, por RRF, com peso próprio para o em alta | sim | sim |

O `em_alta` entrou depois que a avaliação mostrou que, em categorias guiadas por
lançamentos, o que vende agora prevê melhor a próxima compra do que o gosto acumulado
de cada pessoa. A janela e o peso no `hibrido_alta` são parâmetros de cada categoria
(`categorias.yaml`), escolhidos no conjunto de escolha.

### Por que linhas de base importam

Popularidade costuma ser surpreendentemente difícil de bater em comércio
eletrônico, porque a demanda se concentra em poucos produtos. Um sistema "inteligente"
que perde para a lista dos mais vendidos não vale o custo. Mostrar a comparação é
o que torna o resultado confiável.

### Filtragem por conteúdo × filtragem colaborativa

- **Conteúdo** (`ultimo`, `conteudo`): recomenda pelo que o produto É (texto, foto).
  Funciona com produto novo, que ninguém avaliou ainda, mas só sugere "mais do mesmo"
  e não descobre relações que o conteúdo não mostra.
- **Colaborativa** (`coocorrencia`): recomenda pelo que as PESSOAS fazem. Descobre
  relações invisíveis no conteúdo (quem compra uma Draculaura também compra o
  carrinho de acessórios dela), mas não sabe nada de produto sem interações (o
  "cold start" do item).

O **híbrido** junta os dois para cobrir a fraqueza de cada um. O LCM, em escala
real, é um modelo que aprende essas duas coisas juntas a partir de bilhões de interações.

### Como a coocorrência é calculada

1. Matriz usuário × produto com 1 onde houve avaliação.
2. `C = Xᵀ · X`: `C[i][j]` é quantas pessoas avaliaram i **e** j.
3. Similaridade de cosseno: `C[i][j] / √(nᵢ · nⱼ)`, onde nᵢ é quantas pessoas
   avaliaram i. Sem a divisão, os produtos muito populares seriam "parecidos" com tudo
   só por aparecerem muito.
4. Pontuação de um produto para a pessoa: a soma das similaridades dele com os
   produtos do histórico, ponderada pelo mesmo peso da nota.

Exemplo: 10 pessoas avaliaram a Draculaura, 4 avaliaram o carrinho e 3 avaliaram os
dois. Similaridade = 3 / √(10 × 4) ≈ 0,47.

### Regras comuns

- Nunca recomendar o que a pessoa já avaliou.
- Empates (comuns na coocorrência, com muitos produtos em zero) são desempatados
  pela popularidade, e não pela ordem do arquivo, que seria arbitrária.

### Explicabilidade

`mini-lcm perfil USUARIO --categoria ...` mostra, para cada recomendação, o produto
do histórico mais parecido com ela ("parecido com: ..."). Saber dizer **por que**
algo foi recomendado é essencial para depurar o sistema e para a confiança de quem usa.

---

## 4. Busca personalizada

`mini_lcm/busca.py` (`_listas_de_perfil`)

    mini-lcm buscar "boneca" --categoria bonecas --usuario USUARIO --peso-perfil 0.5

A fase 1 combina duas listas por RRF: candidatos por texto e por foto. A busca
personalizada acrescenta mais duas: **os mesmos candidatos**, ordenados pela
afinidade com o perfil da pessoa (uma lista por canal).

Duas propriedades de projeto:

1. **A personalização só reordena o que a busca trouxe.** Ela nunca põe no topo um
   produto sem relação com a consulta só porque combina com o gosto da pessoa. Quem
   busca "placa de vídeo" não recebe bonecas, por mais que goste delas (há um teste
   para isso).
2. **`peso_perfil` controla o equilíbrio** entre a consulta e o gosto. Com 0, é
   exatamente a busca da fase 1 (também testado).

Cada candidato é comparado com o perfil **da categoria dele**: o gosto em bonecas
julga bonecas, o gosto em placas julga placas.

---

## 5. Como medir sem rótulos: a avaliação temporal

`mini_lcm/avaliacao_recomendacao.py`

A pergunta: **olhando o passado de alguém, o sistema adivinha o que a pessoa fez
depois?** A resposta está nos próprios dados, e não precisa de ninguém rotulando.

Dois protocolos foram implementados. O primeiro (corte por usuário) é o mais comum em
artigos e o mais simples de explicar; o segundo (cortes por data) é o principal, porque
o primeiro se mostrou otimista nestes dados.

### Protocolo anterior: "deixa a última de fora" (*leave-last-out*)

Para cada usuário com pelo menos 3 produtos distintos:

    histórico em ordem de data:  A  →  B  →  C  →  D
                                 └── treino ──┘    └ alvo (escondido)

O sistema recebe só A, B e C, ordena o catálogo inteiro, e verificamos em que
posição ficou o D.

**Por que dividir no tempo, e não sortear?** Sorteando, o sistema poderia usar uma
avaliação de 2022 para prever uma de 2019, usando o futuro para prever o passado,
e o número sairia otimista. Na vida real, só se conhece o passado.

### Cuidados contra vazamento (*data leakage*)

Vazamento é quando a resposta escapa para dentro do que o sistema usa para
aprender, e a avaliação fica boa demais para ser verdade.

- Popularidade e coocorrência são aprendidas **só com as interações de treino**. Se
  o D de alguém entrasse na contagem, a popularidade já "saberia" a resposta.
- O que a pessoa já tem no treino não concorre.
- Se o D também aparece no treino (a pessoa avaliou o mesmo produto duas vezes), o
  usuário fica de fora: não há o que adivinhar.

### Protocolo principal: cortes semestrais por data

O corte por usuário tem dois problemas, medidos nos dados (seção 10):

- **Vazamento entre pessoas:** o treino de todos inclui avaliações posteriores ao alvo
  de cada um. Popularidade e coocorrência aprendem com esse futuro.
- **Compras em lote:** metade dos produtos escondidos das bonecas foi avaliada no mesmo
  dia que o anterior (comprados juntos). Prever o segundo item de um pedido é muito mais
  fácil do que prever a próxima compra.

No corte por data, há uma data única para todos:

    linha do tempo:  ....... avaliações anteriores ....... | 1º/jul/2019 | ... seis meses ...
                     tudo isto é treino, de todo mundo       data de corte   alvo: a 1ª avaliação
                                                                             de cada pessoa aqui

- O modelo (popularidade, em alta, coocorrência) aprende **só** com o que veio antes da
  data. O perfil de cada pessoa usa o histórico dela até a data.
- O catálogo é completo: todos os produtos que passam no filtro da categoria
  (`catalogo_completo: true`), e não só os que a amostra avaliou. Um catálogo montado com
  o que a amostra avaliou conteria, em cada corte, só os produtos novos que alguém da
  amostra vai comprar, e o alvo estaria entre poucos candidatos.
- Concorrem só os produtos que já existiam naquele semestre: os que já tinham alguma
  avaliação antes da data, ou cuja data de lançamento (metadados da Amazon) é anterior ao
  fim da janela. Nenhuma avaliação posterior ao corte entra nessa regra.
- Uma data só deixa poucos casos (460 nas bonecas, 66 nas placas). Por isso, há um corte
  a cada seis meses, cada um com o seu modelo ("rolling origin"), e os casos de todos os
  cortes são somados.
- **Escolha e conferência no tempo:** os cortes mais antigos, até somar metade dos casos,
  são o conjunto de escolha; os mais recentes, o de conferência. Os parâmetros são
  escolhidos olhando só o conjunto de escolha.
- A mesma pessoa aparece em vários cortes. O bootstrap sorteia **pessoas**, levando todos
  os casos de cada uma, para o intervalo não ficar estreito demais.

### Produtos novos (*cold start* do item)

Em cada corte, o relatório separa os alvos **novos** (produtos sem nenhuma interação
antes da data) dos **já vistos**. Popularidade, em alta e coocorrência não conseguem
recomendar um produto novo, porque não há nada a contar; só os métodos de conteúdo
conseguem. Nos dados, 40% dos alvos das bonecas e 65% dos das placas são novos. Em
classificados, quase todo anúncio é novo.

O `hibrido_alta` aceita reservar as últimas posições do top 10 para produtos novos,
escolhidos pelo perfil de conteúdo (`vagas_novos`). O número de vagas é escolhido no
conjunto de escolha, como os outros parâmetros.

### Métricas (k = 10, um único produto certo por caso)

| Métrica | Definição | Exemplo |
|---|---|---|
| hit@10 | o produto escondido está no top 10? | 0,25 = acertou para 1 em cada 4 pessoas |
| NDCG@10 | 1 / log2(posição + 1) se estiver no top 10, senão 0 | 1º → 1,0 · 3º → 0,5 · 10º → 0,29 |
| MRR | 1 / posição, na lista inteira | 2º → 0,5 · 100º → 0,01 |
| cobertura | fração do catálogo que aparece em algum top 10 | popularidade: ~0,3% (os mesmos 10 para todos) |

A **cobertura** mede diversidade. Um sistema pode ter hit@10 razoável recomendando
só best-sellers, mas então os produtos de nicho nunca são exibidos, o que é um
problema central em catálogos de cauda longa, como os de marketplaces de usados.

### Curva de histórico: o problema do usuário novo (*cold start*)

`--curva` repete a avaliação usando só as últimas N avaliações de treino (N = 1, 2, 3, 5,
todas). A pergunta é: quanto histórico cada método precisa para valer a pena? Na
prática: a partir de quantas interações vale trocar a lista de populares por
recomendações personalizadas para um usuário novo?

### Incerteza: intervalos de confiança por bootstrap

Um hit@10 de 0,233 em 3.500 pessoas não é um número exato: com outra amostra de
pessoas, sairia um pouco diferente. Cada hit@10 vem com um intervalo de 95% por
bootstrap: os usuários testados são sorteados com reposição 2.000 vezes, o hit@10 é
recalculado em cada sorteio e o intervalo vai do percentil 2,5 ao 97,5.

Para comparar dois métodos, o bootstrap é **pareado**: cada sorteio usa as mesmas
pessoas para os dois, e o que se mede é a diferença. Algumas pessoas são fáceis de
prever para qualquer método e outras são difíceis; parear desconta essa variação e
produz um intervalo bem mais estreito do que comparar os dois intervalos separados.
Se o intervalo da diferença não contém zero, a vantagem não é explicada só pela
amostra de usuários.

### Escolha e conferência: escolher parâmetros sem olhar a conferência

Parâmetros como a meia-vida da recência precisam ser escolhidos por algum número.
Escolher pelo resultado da conferência é ajustar o modelo à prova: com vários valores
experimentados, algum vai parecer bom por acaso, e o número reportado sai otimista.

`--escolha` resolve isso sem precisar de mais dados. Nos cortes por data, avalia nos
cortes mais antigos (conjunto de escolha) em vez dos mais recentes (conferência). No corte por usuário,
retira a última avaliação de cada pessoa e repete a divisão no que sobra, com a
penúltima como alvo. Os parâmetros são escolhidos no conjunto de escolha, e a conferência é
rodada uma única vez no final, com os valores escolhidos.

---

## 6. Como ler os resultados

O que se espera, para verificar com os números reais:

- **Aleatório** dá o piso: com 3.000 produtos, hit@10 ≈ 10 / 3.000 ≈ 0,3%.
- **Popularidade** costuma ser forte na Amazon, com cobertura mínima.
- **Em alta** deve ganhar da popularidade de todos os tempos em categorias guiadas por
  lançamentos.
- **Coocorrência** tende a ser o melhor método individual quando há interações
  suficientes, e piora com histórico curto.
- **Conteúdo** depende da qualidade dos vetores. Com o canal de texto fraco da
  fase 1 (ver o diagnóstico em decisoes.md), é de esperar que ele perca. Isso
  conecta as duas fases: melhorar os embeddings melhora também a recomendação.
- **Híbrido** deveria ser pelo menos tão bom quanto o melhor dos dois que combina.
  Se não for, a fusão está atrapalhando, e isso também é um achado.

Resultados que contrariem essas expectativas são investigados e documentados aqui,
junto com a causa encontrada, em vez de omitidos.

---

## 7. Limitações

- **Data de lançamento incompleta:** só 30% das bonecas têm data nos metadados (96% das
  placas). Produto novo sem data não concorre; se for o alvo, conta como erro para todos.
- **Filtro com informação futura:** `min_avaliacoes` usa o total de avaliações do produto
  na Amazon até 2023. Não depende da amostra, mas também é posterior aos cortes.
- **Poucos casos nas placas:** 228 na conferência, com intervalos de confiança largos
  (cerca de ±0,035 no hit@10). Diferenças pequenas entre métodos não são conclusivas.
- **Avaliação (review) não é compra nem clique:** só uma fração de quem compra
  escreve avaliação. O sinal é mais esparso e mais enviesado (para experiências muito
  boas ou muito ruins) que os cliques de um marketplace real.
- **Um produto certo por pessoa:** se a pessoa compraria qualquer um entre 5 produtos
  parecidos, acertar 4 deles não conta. As métricas subestimam a utilidade real.
- **Seleção por usuários enviesa o catálogo** para produtos de quem avalia muito.
- **O perfil é uma média:** quem gosta de duas coisas bem diferentes na mesma
  categoria (vampiras E bebês) ganha um perfil "no meio", que não representa nenhuma
  das duas. Modelos com vários vetores por pessoa (ou aprendidos, fase 5) resolvem.

---

## 8. Discussão técnica

**Por que não usar só um LLM para recomendar?**
Custo e escala: ordenar 3.000 produtos para cada pessoa com chamadas de LLM custaria
caro e seria lento. Vetores e matrizes fazem isso em milissegundos e sem custo de API. O LLM
entra onde agrega (julgar relevância, explicar), não no laço principal. É a mesma
lógica do LCM, que a Prosus diz custar 60 vezes menos que modelos gerais.

**Qual a diferença entre filtragem por conteúdo e colaborativa?**
Ver a seção 3. Conteúdo: pelo que o produto é; resolve produto novo, mas fica no
"mais do mesmo". Colaborativa: pelo que as pessoas fazem; descobre relações, mas não
conhece produto sem interações.

**O que é cold start e como ele é tratado?**
Usuário novo: popularidade (por categoria ou região) até ter algumas
interações; a curva de histórico mostra a partir de quantas vale personalizar.
Produto novo: conteúdo (texto e foto), porque a colaborativa não o enxerga.

**Como garantir que a avaliação não está otimista demais?**
Divisão no tempo, popularidade e coocorrência aprendidas só com o treino, nada do
histórico concorrendo, e as limitações declaradas (o corte por usuário vaza um pouco
de futuro).

**Por que hit@10 e NDCG@10, e não acurácia?**
Recomendação é ordenação: importa se o produto certo está no topo e em que posição,
não um "certo ou errado" por produto.

**Por que medir cobertura?**
Porque um sistema que só recomenda best-sellers pode ter boas métricas e ainda assim
não servir para descobrir produtos, o que importa em catálogos de cauda longa.

**Como isso escalaria para um marketplace grande?**
Os vetores dos produtos ficam num banco vetorial (pgvector com índice HNSW, como na
fase 1); o perfil é recalculado quando há interações novas; a coocorrência vira uma
tarefa periódica. Em escala maior, troca-se a média ponderada por um modelo treinado
com as interações (two-tower, fase 5), na linha do que o LCM propõe: aprender com
interações em escala muito maior.

---

## 9. Glossário

- **Embedding / vetor:** lista de números que representa um texto, foto ou produto.
- **Similaridade de cosseno:** o quanto dois vetores apontam na mesma direção (1 = igual, 0 = sem relação).
- **Perfil:** vetor que representa o gosto de uma pessoa (média ponderada do que ela avaliou).
- **Filtragem por conteúdo:** recomendar pelo que o produto é.
- **Filtragem colaborativa:** recomendar pelo comportamento de outras pessoas.
- **Coocorrência:** quantas pessoas interagiram com dois produtos.
- **Linha de base (baseline):** método simples que serve de régua.
- **RRF (Reciprocal Rank Fusion):** combinar listas pela posição de cada item, não pelo valor.
- **Leave-last-out:** esconder a interação mais recente de cada pessoa e tentar adivinhá-la.
- **Rolling origin:** avaliar em várias datas de corte, cada uma com um modelo que só vê o passado.
- **Em alta:** popularidade contada só numa janela recente antes do momento da recomendação.
- **Conjunto de escolha / de conferência:** o primeiro serve para escolher parâmetros; o
  segundo, separado, mede o resultado uma vez. Nos termos usuais, validação e teste.
- **Vazamento (data leakage):** a resposta escapando para os dados de aprendizado.
- **Cold start:** recomendar sem histórico (usuário novo) ou sem interações (produto novo).
- **Cobertura:** quanto do catálogo aparece nas recomendações.

---

## 10. Resultados

Tabelas completas, grades do conjunto de escolha e comparações pareadas em
[resultados-recomendacao.md](resultados-recomendacao.md).

### Protocolo principal (cortes por data, catálogo completo)

| Método | bonecas (1.876 casos, 2019–2023) | placas (228 casos, 2018–2023) |
|---|---|---|
| aleatório | 0,001 | 0,004 |
| popularidade | 0,009 | 0,035 |
| em alta | 0,104 | **0,083** |
| último produto | 0,048 | 0,031 |
| conteúdo (perfil) | 0,045 | 0,022 |
| coocorrência | 0,051 | 0,018 |
| híbrido | 0,063 | 0,031 |
| híbrido + em alta | **0,126** | 0,070 |

(hit@10 na conferência; parâmetros escolhidos no conjunto de escolha: bonecas com meia-vida 2, janela de 30 dias e
peso 2; placas com meia-vida 1, janela de 90 dias e peso 2; nenhuma vaga reservada.)

Confronto com as expectativas da seção 6:

- **Popularidade forte:** só a recente. A popularidade de todos os tempos é quase inútil
  nas bonecas (0,009); a dos últimos 30 dias, a mais forte das linhas de base (0,104).
- **Coocorrência como melhor método individual:** não se confirmou no protocolo honesto.
  Nas bonecas, empata com o conteúdo; nas placas, fica perto do acaso.
- **Conteúdo perde:** perde no total, mas é o **único** método que acerta produtos novos
  (cerca de 3% dos alvos novos, contra 0% dos métodos de interações e menos de 1% do acaso).
- **Híbrido ≥ melhor componente:** confirmado nas bonecas. O híbrido + em alta é o melhor
  método, 0,021 acima do em alta sozinho (IC 95% de +0,009 a +0,034).
- **Vagas para produtos novos:** não compensaram. Com 3 vagas, o acerto em novos nas
  bonecas sobe de 0,000 para 0,035, mas o acerto nos já vistos cai de 0,209 para 0,173.

A conclusão principal da fase: nestes dados, **o momento do mercado prevê a próxima
compra melhor do que o gosto individual**; o gosto individual acrescenta algo em
colecionismo (bonecas) e nada em compra rara de tecnologia (placas). A personalização é o
único caminho para os produtos novos, mas o sinal de conteúdo ainda é fraco demais para
ganhar posições no top 10. É exatamente o caso de classificados, e o argumento para
investir em embeddings melhores.

### Protocolo anterior (corte por usuário)

| Método | bonecas, média simples | bonecas, com recência | placas, com recência |
|---|---|---|---|
| popularidade | 0,051 | 0,051 | 0,146 |
| último produto | 0,169 | 0,169 | 0,085 |
| conteúdo (perfil) | 0,113 | 0,160 | 0,070 |
| coocorrência | 0,233 | 0,260 | 0,041 |
| híbrido | 0,216 | 0,267 | 0,060 |

Os números deste protocolo são otimistas. Nas bonecas, 52% dos alvos foram avaliados no
mesmo dia que o produto anterior, e nesses casos o último produto acerta 0,244 contra
0,088 nos demais. A popularidade das placas caía de 0,146 para 0,092 quando contada só
com o passado. Foi este protocolo, porém, que revelou o efeito da recência (o perfil-média
piorava com mais histórico), confirmado depois no protocolo principal.
