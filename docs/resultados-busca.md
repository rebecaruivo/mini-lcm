# Resultados da busca

Avaliação com gabarito: 18 consultas (`avaliacao/consultas-escolha.yaml` e
`avaliacao/consultas-conferencia.yaml`), 9 de cada categoria, de quatro tipos (nome,
visual, funcional, intenção com filtro de preço). Notas de 0 a 3 pela rubrica
(`avaliacao/rubrica.md`): relevante = nota 2 ou 3.

As 12 consultas do **conjunto de escolha** servem para escolher configurações (peso da
imagem, modelo, rubrica do juiz); as 6 do **conjunto de conferência** conferem a escolha uma
vez, sem influenciá-la. Nos termos usuais de aprendizado de máquina, validação e teste.

As seções até "Peso da imagem por categoria" usam o multilingual-e5 no canal de texto
(só título) e o CLIP na imagem. A última seção compara o e5 com o EmbeddingGemma 2, que
passou a ser o padrão no texto, com pesos da imagem de 0,4 (bonecas) e 0,2 (placas).

```bash
mini-lcm avaliar --fonte ia
mini-lcm avaliar --fonte combinado
```

## O juiz de IA

O pool de cada consulta (os 10 primeiros de cada modo) foi julgado pelo Gemini
(`gemini-3.5-flash-lite`), e uma amostra foi rotulada por uma pessoa, sem ver a nota da IA.
Histórico completo na decisão "Rubrica v2 do juiz" em `decisoes.md`.

| Conjunto | Pares | Acordo exato | Diferença ≤ 1 | Kappa ponderado (IC 95%) | Viés (IA − humano) |
|---|---|---|---|---|---|
| Rubrica v1, consultas de escolha | 100 | 50% | — | 0,67 (0,56 a 0,77) | −0,48 |
| Rubrica v2, consultas de escolha (otimista) | 100 | 66% | 94% | 0,76 (0,64 a 0,85) | −0,29 |
| Rubrica v2, auditoria estratificada | 60 | 68% | — | 0,83 (0,73 a 0,91) | −0,22 |
| Rubrica v2, consultas novas (conferência) | 100 | 74% | 95% | 0,64 (0,46 a 0,78) | −0,01 |

**O critério definido antes (limite inferior do kappa ≥ 0,6) não foi atingido na
conferência.** Nas consultas novas, o acordo é o maior de todos (74% de notas iguais,
95% com diferença de até 1 ponto, viés nulo), mas 68% dos pares são nota 1. Com quase
tudo numa nota só, o acordo esperado por acaso é alto, e os poucos desacordos nas notas
raras derrubam o kappa (o "paradoxo do kappa"). A pessoa que rotulou também relatou
dificuldade em vários casos de fronteira, o que indica um teto de concordância abaixo
de 1 mesmo entre pessoas.

**Teste de decisão:** o juiz serve para comparar modos de busca. A pergunta que importa
é se a comparação muda quando as notas humanas substituem as da IA onde existem
(cerca de 40% dos pares). Não muda: a ordem entre os modos é a mesma nas três métricas
(tabela abaixo). O juiz é usado com duas ressalvas: o kappa da conferência ficou abaixo do
critério, e a IA quase nunca dá nota 2 (2% das notas na conferência, contra 14% das
humanas), empurrando os casos médios para 1 ou 3.

## Resultado por modo

| Modo | recall@10 (IA) | recall@10 (combinado) | MRR (IA) | MRR (combinado) | NDCG@10 (IA) | NDCG@10 (combinado) |
|---|---|---|---|---|---|---|
| só texto | **0,604** | **0,653** | **0,708** | **0,703** | **0,662** | **0,692** |
| só imagem | 0,249 | 0,210 | 0,262 | 0,331 | 0,367 | 0,390 |
| fusão (peso 0,5) | 0,481 | 0,510 | 0,641 | 0,650 | 0,595 | 0,627 |

Só com as notas da IA, 4 das 18 consultas não têm nenhum produto relevante ("placa de
vídeo branca", "barbie sereia", "boneca que fala português", "placa de vídeo com rgb") e
ficam fora da média (14 consultas); com as notas combinadas, todas entram.

## Por consulta (NDCG@10, notas combinadas)

| Consulta | Tipo | Relevantes | texto | imagem | fusão |
|---|---|---|---|---|---|
| monster high draculaura | nome | 15 | **0,92** | 0,23 | 0,79 |
| boneca gótica de cabelo rosa | visual | 10 | 0,31 | **0,61** | 0,45 |
| boneca com vestido de noiva | visual | 18 | 0,59 | **0,95** | 0,94 |
| boneca bebê que chora e faz xixi | funcional | 7 | 0,60 | 0,53 | **0,68** |
| boneca negra de cabelo cacheado | visual | 11 | **0,50** | 0,42 | **0,50** |
| presente para menina de 6 anos | intenção | 26 | 0,93 | **1,00** | 0,94 |
| boneca vampira com roupa preta | visual | 13 | 0,33 | 0,53 | **0,76** |
| barbie sereia | nome | 7 | 0,57 | 0,52 | **0,77** |
| boneca que fala português | funcional | 1 | 0,40 | **0,69** | 0,44 |
| rtx 3060 12gb | nome | 13 | **1,00** | 0,13 | 0,61 |
| placa de vídeo branca | visual | 1 | **1,00** | 0,00 | 0,60 |
| placa de vídeo low profile para pc pequeno | funcional | 13 | **1,00** | 0,10 | 0,64 |
| placa silenciosa sem ventoinha | funcional | 7 | **0,89** | 0,00 | 0,52 |
| radeon com 16gb de memória | nome | 3 | **0,66** | 0,17 | 0,38 |
| placa barata para jogar em full hd | intenção | 2 | **0,63** | 0,29 | 0,33 |
| gtx 1660 super | nome | 10 | **0,94** | 0,23 | 0,72 |
| placa de vídeo com rgb | visual | 2 | **0,92** | 0,30 | 0,61 |
| placa para edição de vídeo até 400 dólares | intenção | 9 | 0,28 | 0,31 | **0,61** |

## Leitura

1. **O texto sozinho é o melhor modo na média** (NDCG@10 0,69), e a fusão com peso 0,5
   fica abaixo dele (0,63). A troca do texto do CLIP pelo e5 deixou o canal de texto forte
   o bastante para que a imagem, com peso fixo, mais atrapalhe do que ajude na média.
2. **A imagem depende da categoria.** Nas placas, o texto ganha em 8 de 9 consultas, e a
   imagem fica perto de zero: toda placa parece igual na foto. Nas bonecas, a imagem ou a
   fusão ganham em 7 de 9, em especial nas buscas visuais (noiva, gótica, vampira).
3. **Consequência:** o peso da imagem na fusão deve ser escolhido por categoria (seção
   seguinte).

As notas por consulta vêm de poucos produtos relevantes em vários casos (1 a 3): as
diferenças por consulta são indicativas, não conclusivas.

## Peso da imagem por categoria

Regra fixada antes de ver os números: o peso de maior NDCG@10 nas 12 consultas de escolha
(`avaliacao/consultas-escolha.yaml`); as 6 de conferência só conferem.

```bash
mini-lcm avaliar --lote avaliacao/consultas-escolha.yaml --pesos 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
mini-lcm avaliar --lote avaliacao/consultas-conferencia.yaml --pesos 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
```

NDCG@10 da fusão (notas combinadas; 0 = só texto, 1 = só imagem):

| Peso | bonecas, escolha (6) | bonecas, conferência (3) | placas, escolha (6) | placas, conferência (3) |
|---|---|---|---|---|
| 0 | 0,643 | 0,430 | **0,863** | 0,709 |
| 0,1 | 0,656 | 0,430 | 0,837 | 0,716 |
| 0,2 | 0,659 | 0,431 | 0,814 | 0,702 |
| 0,3 | 0,667 | 0,474 | 0,792 | 0,741 |
| 0,4 | 0,680 | 0,538 | 0,744 | **0,752** |
| 0,5 | **0,716** | 0,653 | 0,515 | 0,646 |
| 0,6 | 0,666 | **0,739** | 0,221 | 0,375 |
| 0,7 | 0,680 | 0,633 | 0,168 | 0,354 |
| 0,8 | 0,653 | 0,583 | 0,120 | 0,354 |
| 0,9 | 0,630 | 0,583 | 0,116 | 0,325 |
| 1 | 0,622 | 0,583 | 0,116 | 0,279 |

**Escolhidos para o e5: 0,5 nas bonecas e 0 nas placas** (depois substituídos pelos do
EmbeddingGemma 2, 0,4 e 0,2; última seção). Na conferência, os
picos ficam vizinhos da escolha (0,6 nas bonecas, 0,4 nas placas), mas com 3 consultas
por categoria essas diferenças cabem no ruído.

Comparação das três estratégias (média das duas categorias):

| Estratégia | NDCG@10, escolha | NDCG@10, conferência |
|---|---|---|
| só texto | 0,753 | 0,570 |
| peso 0,5 para todas (antes) | 0,616 | 0,650 |
| **peso por categoria (0,5 e 0)** | **0,790** | **0,681** |

O peso por categoria é o melhor nos dois conjuntos, inclusive nas consultas que não
entraram na escolha.

**Leitura:**
- Nas placas, com o e5, a imagem só atrapalha. Com peso acima de 0,5, o NDCG despenca (0,12 com só
  imagem): a foto não distingue uma RTX 3060 de uma RX 6600.
- Nas bonecas, o melhor é metade texto e metade imagem. Só imagem também piora (0,62): a
  foto mostra o estilo, mas não lê o nome da personagem nem a função ("chora e faz xixi").
- Um peso fixo por categoria ainda é grosso. Dentro das bonecas, as buscas por nome
  ("monster high draculaura": texto 0,92, imagem 0,23) e as visuais ("boneca com vestido
  de noiva": texto 0,59, imagem 0,95) pedem pesos opostos. O próximo nível seria um peso
  que depende da consulta (ver `decisoes.md`, 27/11).

## EmbeddingGemma 2 contra o e5 no canal de texto

O EmbeddingGemma 2 (Google DeepMind) foi testado no lugar do e5, só no canal de texto,
com o CLIP nas fotos (o canal de imagem do Gemma leva cerca de 20 s por foto sem GPU; ver
`decisoes.md`). Os produtos que ele trouxe ao topo e que ainda não tinham nota foram
julgados pela IA (223 julgamentos).

**A primeira comparação foi injusta.** Com as notas combinadas, o Gemma parecia perder
(média de 0,66 contra 0,79 nas consultas de escolha). Mas os produtos que o e5 traz ao
topo têm muitas notas humanas, mais generosas nos casos médios, e os novos que o Gemma
trouxe só têm notas da IA, que quase nunca dá nota 2. Comparar os dois com notas de
fontes diferentes favorecia o e5. A comparação justa usa só as notas da IA, para os dois,
sobre o mesmo conjunto de produtos julgados.

NDCG@10, só notas da IA:

| | e5 | EmbeddingGemma 2 |
|---|---|---|
| bonecas, escolha, peso 0,5 | 0,628 | 0,673 |
| placas, escolha, peso 0 | 0,675 | 0,641 |
| **média, escolha (pesos escolhidos)** | **0,652** | **0,657** |
| média, conferência (pesos escolhidos) | 0,490 | 0,524 |
| bonecas, escolha, só texto | 0,544 | 0,664 |
| bonecas, conferência, só texto | 0,289 | 0,504 |
| placas, conferência, só texto | 0,571 | 0,651 |

Na conferência, só 2 consultas por categoria têm algum produto relevante pelas notas da IA:
as diferenças ali não são conclusivas.

**Ainda injusta: os pesos eram do e5.** O peso da imagem (0,5 e 0) tinha sido escolhido
com a grade inteira para o e5; o Gemma recebeu os mesmos valores. A comparação final dá a
cada modelo o melhor peso dele nas consultas de escolha (regra fixada antes: se o Gemma
ganhar na escolha e não perder na conferência, vira o padrão).

NDCG@10, só notas da IA, grade de 0 a 0,7:

| Peso | e5 bonecas | Gemma bonecas | e5 placas | Gemma placas |
|---|---|---|---|---|
| 0 | 0,544 | 0,664 | **0,675** | 0,641 |
| 0,1 | 0,561 | 0,667 | 0,669 | 0,639 |
| 0,2 | 0,568 | 0,684 | 0,650 | **0,644** |
| 0,3 | 0,592 | 0,699 | 0,642 | 0,632 |
| 0,4 | 0,602 | **0,715** | 0,604 | 0,582 |
| 0,5 | **0,628** | 0,673 | 0,419 | 0,405 |
| 0,6 | 0,589 | 0,608 | 0,181 | 0,201 |
| 0,7 | 0,564 | 0,558 | 0,136 | 0,142 |

(consultas de escolha)

| Com o melhor peso de cada um | e5 (0,5 e 0) | Gemma (0,4 e 0,2) |
|---|---|---|
| bonecas, escolha | 0,628 | **0,715** |
| placas, escolha | **0,675** | 0,644 |
| **média, escolha** | 0,652 | **0,680** |
| bonecas, conferência | 0,408 | **0,518** |
| placas, conferência | 0,571 | **0,641** |
| **média, conferência** | 0,490 | **0,580** |

**Leitura:**
- Com o peso próprio, o Gemma ganha na escolha (+0,028) e na conferência (+0,090). Pela
  regra fixada, passa a ser o padrão (`EMBEDDER=clip+gemma2`, pesos 0,4 e 0,2).
- A vantagem está nas bonecas, nos dois conjuntos (+0,09 e +0,11). Nas placas, os dois
  ficam próximos (o e5 à frente na escolha, o Gemma na conferência).
- O Gemma pede menos foto nas bonecas (0,4 contra 0,5): o texto dele já entende parte do
  que só a imagem trazia. Nas placas, um pouco de foto (0,2) praticamente não muda o Gemma (0,644 contra 0,641
  sem foto; na conferência, o melhor foi 0), enquanto atrapalhava o e5.
- São 6 consultas por categoria na escolha e 2 na conferência: a direção é a mesma nos
  dois conjuntos, mas não há consultas suficientes para um intervalo de confiança firme.
- Custo: o Gemma leva 358 ms por produto na indexação (uma vez, 1,4 h para o catálogo) e
  cerca de 0,3 s a mais por consulta. Aceitável aqui; num catálogo de milhões de anúncios
  novos por dia, o custo entraria na decisão.
