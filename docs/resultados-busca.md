# Resultados da busca

Avaliação com gabarito: 18 consultas (`avaliacao/consultas.yaml` e
`avaliacao/consultas-validacao.yaml`), 9 de cada categoria, de quatro tipos (nome,
visual, funcional, intenção com filtro de preço). Notas de 0 a 3 pela rubrica
(`avaliacao/rubrica.md`): relevante = nota 2 ou 3. Embeddings: multilingual-e5 com só o
título no canal de texto, CLIP na imagem (ver `sondagem.md`). Fusão com peso 0,5 na imagem.

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
| Rubrica v1, consultas originais | 100 | 50% | — | 0,67 (0,56 a 0,77) | −0,48 |
| Rubrica v2, consultas originais (otimista) | 100 | 66% | 94% | 0,76 (0,64 a 0,85) | −0,29 |
| Rubrica v2, auditoria estratificada | 60 | 68% | — | 0,83 (0,73 a 0,91) | −0,22 |
| Rubrica v2, consultas novas (validação) | 100 | 74% | 95% | 0,64 (0,46 a 0,78) | −0,01 |

**O critério definido antes (limite inferior do kappa ≥ 0,6) não foi atingido na
validação.** Nas consultas novas, o acordo é o maior de todos (74% de notas iguais,
95% com diferença de até 1 ponto, viés nulo), mas 68% dos pares são nota 1. Com quase
tudo numa nota só, o acordo esperado por acaso é alto, e os poucos desacordos nas notas
raras derrubam o kappa (o "paradoxo do kappa"). A pessoa que rotulou também relatou
dificuldade em vários casos de fronteira, o que indica um teto de concordância abaixo
de 1 mesmo entre pessoas.

**Teste de decisão:** o juiz serve para comparar modos de busca. A pergunta que importa
é se a comparação muda quando as notas humanas substituem as da IA onde existem
(cerca de 40% dos pares). Não muda: a ordem entre os modos é a mesma nas três métricas
(tabela abaixo). O juiz é usado com duas ressalvas: o kappa da validação ficou abaixo do
critério, e a IA quase nunca dá nota 2 (2% das notas na validação, contra 14% das
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

Regra fixada antes de ver os números: o peso de maior NDCG@10 nas 12 consultas originais
(`avaliacao/consultas.yaml`); as 6 de validação só conferem.

```bash
mini-lcm avaliar --lote avaliacao/consultas.yaml --pesos 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
mini-lcm avaliar --lote avaliacao/consultas-validacao.yaml --pesos 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
```

NDCG@10 da fusão (notas combinadas; 0 = só texto, 1 = só imagem):

| Peso | bonecas, originais (6) | bonecas, validação (3) | placas, originais (6) | placas, validação (3) |
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

**Escolhidos (em `categorias.yaml`): 0,5 nas bonecas e 0 nas placas.** Na validação, os
picos ficam vizinhos da escolha (0,6 nas bonecas, 0,4 nas placas), mas com 3 consultas
por categoria essas diferenças cabem no ruído.

Comparação das três estratégias (média das duas categorias):

| Estratégia | NDCG@10, originais | NDCG@10, validação |
|---|---|---|
| só texto | 0,753 | 0,570 |
| peso 0,5 para todas (antes) | 0,616 | 0,650 |
| **peso por categoria (0,5 e 0)** | **0,790** | **0,681** |

O peso por categoria é o melhor nos dois conjuntos, inclusive nas consultas que não
entraram na escolha.

**Leitura:**
- Nas placas, a imagem só atrapalha. Com peso acima de 0,5, o NDCG despenca (0,12 com só
  imagem): a foto não distingue uma RTX 3060 de uma RX 6600.
- Nas bonecas, o melhor é metade texto e metade imagem. Só imagem também piora (0,62): a
  foto mostra o estilo, mas não lê o nome da personagem nem a função ("chora e faz xixi").
- Um peso fixo por categoria ainda é grosso. Dentro das bonecas, as buscas por nome
  ("monster high draculaura": texto 0,92, imagem 0,23) e as visuais ("boneca com vestido
  de noiva": texto 0,59, imagem 0,95) pedem pesos opostos. O próximo nível seria um peso
  que depende da consulta (ver `decisoes.md`, 27/11).
