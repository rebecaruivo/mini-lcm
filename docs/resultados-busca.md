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
3. **Consequência:** o peso da imagem na fusão deve ser escolhido por categoria. A
   escolha é feita nas 12 consultas originais e conferida nas 6 de validação.

As notas por consulta vêm de poucos produtos relevantes em vários casos (1 a 3): as
diferenças por consulta são indicativas, não conclusivas.
