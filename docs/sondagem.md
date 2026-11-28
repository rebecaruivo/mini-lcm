# Sondagem do canal de texto

Comparação de variantes do embedding de texto dos produtos, sem rótulos. Contexto e
hipóteses na decisão "canal de texto fraco" em [decisoes.md](decisoes.md).

## Método

`mini-lcm sondar` busca nomes que existem no catálogo e mede, para cada modo de busca
(só texto, só imagem, fusão), quantos dos 10 primeiros resultados têm o termo no título
(**acerto@10**), e o **teto**: o maior acerto possível, min(10, produtos com o termo no
título) / 10. Mede também a **faixa de similaridade** do canal de texto: a diferença
entre o 1º e o 100º resultado. Uma faixa estreita indica similaridades espremidas, em
que o modelo quase não separa o relevante do resto.

É uma medida grosseira: cobre só buscas pelo nome, e um produto relevante sem o termo no
título conta como erro. Serve para achar defeitos grandes, não para escolher entre
variantes parecidas.

Variantes (as fotos são sempre do CLIP; só o texto muda):

| Variante | Modelo de texto | Texto do produto |
|---|---|---|
| `EMBEDDER=clip`, `TEXTO_ITEM=completo` | CLIP ViT-B/32 multilíngue | título + loja + categoria + destaques |
| `EMBEDDER=clip`, `TEXTO_ITEM=titulo` | CLIP ViT-B/32 multilíngue | título + loja |
| `EMBEDDER=clip+e5`, `TEXTO_ITEM=completo` | multilingual-e5-small | título + loja + categoria + destaques |
| `EMBEDDER=clip+e5`, `TEXTO_ITEM=titulo` | multilingual-e5-small | título + loja |

## Bonecas (catálogo de 8.407 produtos)

Termos: "monster high", "barbie", "troll", "baby doll", "american girl" (todos com pelo
menos 51 produtos no catálogo: teto de 100%).

| Variante | acerto@10 texto | imagem | fusão |
|---|---|---|---|
| CLIP, completo | 40% | 74% | 52% |
| CLIP, título | 30% | 74% | 52% |
| e5, completo | **96%** | 74% | 88% |
| e5, título | 94% | 74% | **96%** |

Por termo:

| Termo | CLIP completo | CLIP título | e5 completo | e5 título |
|---|---|---|---|---|
| monster high | 20% | 10% | 100% | 100% |
| barbie | 90% | 100% | 100% | 100% |
| troll | 0% | 0% | 100% | 100% |
| baby doll | 10% | 10% | 80% | 70% |
| american girl | 80% | 30% | 100% | 100% |

(acerto@10 do canal de texto. A imagem não muda entre variantes: 100%, 100%, 70%, 70%, 30%.)

A faixa de similaridade não separou as variantes (entre 0,017 e 0,055 em todas): o e5
também produz similaridades próximas entre si, mas ordena os produtos bem melhor.

## Efeito na recomendação

A recomendação usa os vetores de outro jeito: compara produtos com produtos (o perfil é
uma média de vetores de produtos), e não uma consulta curta com produtos. A mesma
avaliação por cortes de data da fase 2 foi rodada com cada variante de texto (parâmetros
de `categorias.yaml`; hit@10, com o acerto nos alvos novos entre parênteses):

| Variante | bonecas: último produto | bonecas: híbrido + em alta | placas: último produto | placas: híbrido + em alta |
|---|---|---|---|---|
| escolha, CLIP completo | 0,046 (0,021) | 0,075 | 0,060 (0,063) | 0,121 |
| escolha, CLIP título | 0,044 (0,021) | 0,073 | 0,056 (0,063) | 0,121 |
| escolha, e5 completo | 0,050 (0,023) | 0,071 | 0,047 (0,051) | 0,121 |
| escolha, e5 título | 0,052 (0,023) | 0,074 | 0,052 (0,057) | 0,116 |
| conferência, CLIP completo | 0,048 (0,031) | 0,126 | 0,031 (0,034) | 0,070 |
| conferência, e5 título | 0,051 (0,029) | 0,118 | 0,031 (0,034) | 0,079 |

As diferenças ficam todas dentro da incerteza da medida (cerca de ±0,015 nas bonecas e
±0,03 nas placas), inclusive nos produtos novos.

## Placas de vídeo (catálogo de 5.577 produtos)

Termos: "rtx 4060", "rx 7600", "gtx 1650", "asus", "msi".

| Variante | acerto@10 texto | imagem | fusão |
|---|---|---|---|
| CLIP, completo | 6% | 2% | 2% |
| CLIP, título | 0% | 2% | 0% |
| e5, completo | **68%** | 2% | **38%** |
| e5, título | 66% | 2% | **38%** |
| teto | 76% | 76% | 76% |

Por termo (canal de texto):

| Termo | produtos com o termo | teto | CLIP completo | CLIP título | e5 completo | e5 título |
|---|---|---|---|---|---|---|
| rtx 4060 | 4 | 40% | 0% | 0% | 0% | 0% |
| rx 7600 | 4 | 40% | 0% | 0% | **40%** | 30% |
| gtx 1650 | 72 | 100% | 0% | 0% | **100%** | **100%** |
| asus | 564 | 100% | 10% | 0% | **100%** | **100%** |
| msi | 566 | 100% | 20% | 0% | **100%** | **100%** |

O dataset vai até 2023: a RTX 4060 e a RX 7600 tinham acabado de sair e aparecem em só 4
títulos cada, o que limita o acerto a 40%.

## Leitura

1. **Para busca, o e5 é muito melhor.** O codificador de texto do CLIP foi treinado para
   comparar texto com imagem; com uma consulta curta contra títulos, erra nomes óbvios
   ("troll" nas bonecas, "gtx 1650" nas placas: 0% contra 100%). O e5, treinado para
   busca, chega perto do teto nas duas categorias (bonecas 94–96% de 100%; placas 66–68%
   de 76%). O único erro grande do e5 é "rtx 4060" (0 de 4 possíveis), provavelmente
   confundida com as várias "RTX 4060 Ti" e "RTX 3060".
2. **Para recomendação, tanto faz.** Comparar produto com produto é mais fácil: títulos
   da mesma linha compartilham palavras, e a foto carrega boa parte da semelhança. A
   fraqueza do CLIP estava em entender a consulta, não em descrever o produto.
3. **Com o e5, a foto ajuda nas bonecas e atrapalha nas placas.** Nas bonecas, a imagem acerta 74%
   sozinha e a fusão chega a 96%. Nas placas, todas parecem iguais na foto (um retângulo
   preto com ventoinhas), a imagem acerta 2%, e a fusão puxa o e5 de 66% para 38%. O peso
   da imagem na fusão deveria depender da categoria.
4. **Título ou texto completo:** empate no canal de texto nas duas categorias; na fusão,
   o título ganha nas bonecas (96% contra 88%) e empata nas placas.

**EmbeddingGemma 2 no canal de texto** (`EMBEDDER=clip+gemma2`, só título), testado depois:

| | bonecas: só texto | bonecas: fusão | placas: só texto | placas: fusão |
|---|---|---|---|---|
| e5, título | 94% | 96% | 66% | 38% |
| EmbeddingGemma 2, título | 96% | 92% | 70% | 44% |

Empate, com leve vantagem do Gemma nas placas: foi o único a achar a "rtx 4060" (3 dos 4
produtos possíveis). Comparação com gabarito em `resultados-busca.md`.

**Escolha inicial:** `EMBEDDER=clip+e5` e `TEXTO_ITEM=titulo`. Depois, na avaliação com
gabarito e com o peso da imagem ajustado para cada modelo, o EmbeddingGemma 2 superou o e5
e passou a ser o padrão (`EMBEDDER=clip+gemma2`; ver `resultados-busca.md`).
Os resultados de recomendação publicados em `resultados-recomendacao.md` foram medidos
com o texto do CLIP; reproduzi-los exige `EMBEDDER=clip TEXTO_ITEM=completo`.

**Ajuste seguinte (feito, ver `resultados-busca.md`):** peso da imagem na fusão por categoria. A sondagem só mede buscas
pelo nome, em que a foto naturalmente ajuda pouco; o peso deve ser escolhido com a
avaliação com gabarito (passo 7), que inclui buscas descritivas ("placa branca", "boneca
de cabelo rosa").
