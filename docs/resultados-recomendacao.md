# Resultados da recomendação

Avaliação temporal *leave-last-out* descrita em
[fase-2-personalizacao.md](fase-2-personalizacao.md#5-como-medir-sem-rótulos-a-avaliação-temporal).
Para cada pessoa, a última avaliação da categoria é escondida e o sistema tenta
colocá-la entre os 10 primeiros a partir das anteriores.

Comandos:

```bash
# linha de base: perfil e coocorrência com a média simples do histórico
mini-lcm avaliar-recomendacao --categoria bonecas --curva --sem-recencia
# escolha da meia-vida (validação, sem tocar no teste)
mini-lcm avaliar-recomendacao --categoria bonecas --validacao --meia-vida 2
# sistema final (meia-vida padrão = 1)
mini-lcm avaliar-recomendacao --categoria bonecas --curva
```

Resumo (hit@10 nas bonecas, 3.504 usuários testados):

| Método | média simples | com recência (meia-vida 1) |
|---|---|---|
| conteúdo (perfil) | 0,113 | 0,160 |
| coocorrência | 0,233 | 0,260 |
| híbrido | 0,216 | **0,267** |

## Bonecas

### Dados

| | Valor |
|---|---|
| Produtos no catálogo | 3.000 |
| Avaliações (após remover repetições) | 28.843 |
| Usuários na categoria | 3.699 (3.103 com 5 ou mais avaliações) |
| Usuários testados | 3.504 |
| Interações de treino | 25.339 |
| Histórico médio de treino por usuário | 7,1 |
| Vetores | CLIP ViT-B/32 multilíngue (imagem e texto, 512 dimensões) |

A importação lê 29.496 avaliações; 653 repetem usuário, produto e momento de outra
e ficam de fora pela chave da tabela.
Usuários sem pelo menos 3 produtos distintos, ou cujo último produto já aparece no
histórico, não viram caso de teste, mas as interações deles entram no treino.

## Bonecas: linha de base (média simples)

### Histórico completo

| Método | hit@10 | IC 95% | NDCG@10 | MRR | cobertura |
|---|---|---|---|---|---|
| aleatório | 0,003 | 0,001–0,004 | 0,001 | 0,003 | 100,0% |
| popularidade | 0,051 | 0,044–0,059 | 0,023 | 0,022 | 0,7% |
| último produto | 0,169 | 0,156–0,181 | 0,097 | 0,085 | 78,8% |
| conteúdo (perfil) | 0,113 | 0,102–0,124 | 0,064 | 0,059 | 57,0% |
| coocorrência | **0,233** | 0,219–0,247 | **0,140** | **0,123** | **82,9%** |
| híbrido | 0,216 | 0,203–0,230 | 0,125 | 0,110 | 67,7% |

### Curva de histórico (hit@10 usando só as últimas N avaliações)

| N | aleatório | popularidade | último | conteúdo | coocorrência | híbrido |
|---|---|---|---|---|---|---|
| 1 | 0,003 | 0,051 | 0,169 | 0,169 | 0,232 | 0,249 |
| 2 | 0,003 | 0,051 | 0,169 | 0,150 | 0,254 | **0,261** |
| 3 | 0,003 | 0,051 | 0,169 | 0,140 | 0,254 | 0,248 |
| 5 | 0,003 | 0,051 | 0,169 | 0,124 | 0,241 | 0,228 |
| todas | 0,003 | 0,051 | 0,169 | 0,113 | 0,233 | 0,216 |

Aleatório, popularidade e último produto não dependem de N (os dois primeiros ignoram
o histórico; o terceiro só usa o produto mais recente).

### Comparações pareadas

Diferença de hit@10 entre duas configurações nos mesmos 3.504 usuários, com intervalo
de 95% por bootstrap pareado (2.000 reamostras).

| Configuração A | Configuração B | A − B | IC 95% |
|---|---|---|---|
| híbrido (N=2) | coocorrência (N=2) | +0,007 | −0,004 a +0,018 |
| híbrido (N=2) | último produto | +0,092 | +0,079 a +0,105 |
| híbrido (N=2) | popularidade | +0,209 | +0,194 a +0,225 |
| híbrido (N=2) | híbrido (todas) | +0,045 | +0,033 a +0,057 |
| coocorrência (todas) | híbrido (todas) | +0,017 | +0,005 a +0,028 |
| último produto | conteúdo (todas) | +0,056 | +0,044 a +0,069 |

As quatro primeiras linhas saem do próprio relatório do comando. As duas últimas foram
calculadas com a mesma função (`intervalo_bootstrap` aplicada à diferença por usuário).

### Leitura

1. **Personalização supera as linhas de base com folga.** O melhor resultado (híbrido
   com as 2 últimas avaliações, 0,261) acerta para cerca de 1 em cada 4 pessoas, contra
   1 em 20 da popularidade e 1 em 300 do acaso. A cobertura também sobe de 0,7% para
   68–83% do catálogo: a lista deixa de ser a mesma para todos.

2. **Verificação de sanidade: conteúdo com N=1 é igual a último produto (0,169).** Com
   uma avaliação só, o perfil é o vetor desse produto, e os dois métodos ordenam o
   catálogo da mesma forma. A coincidência exata confirma que o perfil e a exclusão dos
   produtos já avaliados estão implementados de forma consistente.

3. **O perfil-média piora com mais histórico.** O conteúdo cai de 0,169 (N=1) para 0,113
   (todas). Somar vetores de produtos diferentes produz um ponto "no meio" que se parece
   pouco com qualquer um deles, e as compras antigas pesam tanto quanto as recentes. O
   último produto sozinho é melhor que o perfil completo (+0,056, IC sem zero).

4. **A coocorrência também prefere o histórico recente.** Ela sobe até N=2–3 (0,254) e
   cai com o histórico completo (0,233). Produtos avaliados há muito tempo trazem
   vizinhos menos relacionados ao interesse atual. Junto com o item 3, isso indica que
   recência é um sinal forte nesta categoria.

5. **A expectativa "híbrido ≥ melhor componente" falhou com histórico completo.** Com
   todas as avaliações, a coocorrência sozinha supera o híbrido (+0,017, IC sem zero):
   a fusão RRF com pesos iguais deixa a lista de conteúdo, que se degradou, puxar a
   lista melhor para baixo. Com histórico curto (N=1 e N=2), em que o conteúdo ainda é
   bom, o híbrido fica à frente. Em N=2 a vantagem sobre a coocorrência (+0,007) não se
   distingue de zero.

6. **Popularidade é fraca neste catálogo.** Com 0,051, fica longe do que costuma
   acontecer em lojas generalistas. A seleção centrada em usuários escolhe colecionadores
   com históricos longos e de nicho, para quem os 10 produtos mais avaliados raramente
   são o próximo.

### Efeito da limpeza dos dados

Uma execução anterior, com avaliações de uma importação antiga ainda misturadas
(70.885 em vez de 28.843), dava coocorrência 0,211, híbrido 0,217 e popularidade 0,024.
Com os dados limpos, a ordem entre coocorrência e híbrido se inverteu. A correção está no
commit "Reimportação substitui também as avaliações da categoria".

## Bonecas: peso pela recência

### Hipótese

Os itens 3 e 4 da leitura acima indicam que as avaliações recentes descrevem melhor o
próximo interesse do que as antigas. Em vez de cortar o histórico (usar só as últimas
N), cada avaliação passa a valer mais quanto mais recente for:

    peso = peso_da_nota × 0,5 ^ (idade / meia_vida)

A idade conta avaliações, não dias: 0 para a mais recente, 1 para a anterior e assim por
diante. Com meia-vida 1, as quatro mais recentes valem 1; 0,5; 0,25 e 0,125. O mesmo peso
vale para o perfil (conteúdo) e para a soma da coocorrência. Contar avaliações em vez de
dias evita depender do ritmo de compra, que varia muito entre pessoas.

### Escolha da meia-vida sem olhar o teste

Escolher o parâmetro pelo resultado do teste seria ajustar o modelo à prova, e o
número reportado sairia otimista. A escolha foi feita num conjunto de **validação**: a
última avaliação de cada pessoa (o teste) é retirada e a mesma divisão é repetida no
que sobra, com a penúltima avaliação como alvo (3.336 usuários). O teste só foi rodado
depois, uma vez, com o valor escolhido.

| Meia-vida | conteúdo | coocorrência | híbrido |
|---|---|---|---|
| sem decaimento | 0,122 | 0,233 | 0,216 |
| 0,5 | 0,163 | 0,236 | **0,259** |
| 1 | 0,154 | 0,253 | **0,258** |
| 2 | 0,143 | **0,257** | 0,251 |
| 3 | 0,136 | 0,253 | 0,243 |
| 5 | 0,133 | 0,247 | 0,232 |
| 8 | 0,131 | 0,245 | 0,227 |

(hit@10 na validação, histórico completo.)

Escolha: **meia-vida 1**. No híbrido, 0,5 e 1 diferem em 0,001, bem abaixo da incerteza
da medida (intervalos de cerca de ±0,014). Entre os dois, 1 mantém a coocorrência forte
(0,253 contra 0,236), o que deixa o resultado menos dependente da fusão. O valor fica em
`MEIA_VIDA_PADRAO`, em `mini_lcm/recomendacao.py`.

### Resultado no teste (meia-vida 1)

| Método | hit@10 | IC 95% | NDCG@10 | MRR | cobertura |
|---|---|---|---|---|---|
| aleatório | 0,003 | 0,001–0,004 | 0,001 | 0,003 | 100,0% |
| popularidade | 0,051 | 0,044–0,059 | 0,023 | 0,022 | 0,7% |
| último produto | 0,169 | 0,156–0,181 | 0,097 | 0,085 | 78,8% |
| conteúdo (perfil) | 0,160 | 0,147–0,172 | 0,093 | 0,082 | 66,9% |
| coocorrência | 0,260 | 0,246–0,275 | **0,166** | **0,148** | **86,6%** |
| híbrido | **0,267** | 0,253–0,282 | 0,165 | 0,144 | 73,3% |

Curva de histórico (hit@10):

| N | conteúdo | coocorrência | híbrido |
|---|---|---|---|
| 1 | 0,169 | 0,232 | 0,249 |
| 2 | 0,164 | 0,254 | 0,264 |
| 3 | 0,162 | 0,256 | **0,269** |
| 5 | 0,160 | 0,260 | 0,268 |
| todas | 0,160 | 0,260 | 0,267 |

Comparações pareadas no teste (bootstrap pareado, IC 95%):

| A | B | A − B | IC 95% |
|---|---|---|---|
| híbrido com recência | híbrido sem recência | +0,051 | +0,040 a +0,062 |
| híbrido com recência | coocorrência sem recência (melhor da linha de base) | +0,034 | +0,021 a +0,047 |
| coocorrência com recência | coocorrência sem recência | +0,027 | +0,016 a +0,037 |
| conteúdo com recência | conteúdo sem recência | +0,047 | +0,037 a +0,057 |
| conteúdo com recência | último produto | −0,009 | −0,018 a +0,000 |
| híbrido com recência | coocorrência com recência | +0,007 | −0,004 a +0,018 |

### Leitura

1. **A hipótese se confirmou.** O híbrido sobe de 0,216 para 0,267 (+0,051, IC sem
   zero), e a recência melhora os três métodos que usam o histórico.
2. **A curva deixou de cair.** Sem recência, todos os métodos pioravam a partir de
   N = 2 ou 3. Com recência, o hit@10 se estabiliza a partir de N = 3: o histórico
   antigo deixa de atrapalhar, mesmo sem ajudar muito. Na prática, isso dispensa
   escolher um corte N por usuário.
3. **A expectativa "híbrido ≥ melhor componente" volta a valer**, agora com histórico
   completo (+0,007 sobre a coocorrência, diferença ainda indistinguível de zero). A
   fusão não atrapalha mais porque a lista de conteúdo deixou de se degradar.
4. **O conteúdo sozinho ainda não supera o último produto** (−0,009, intervalo
   encostando em zero). Com o canal de texto fraco da fase 1, olhar além do produto
   mais recente ainda não acrescenta informação útil ao perfil. A sondagem das
   variantes de texto (passo 6b) é o próximo teste dessa explicação.
5. **A cobertura subiu junto** (coocorrência 82,9% → 86,6%; híbrido 67,7% → 73,3%): as
   recomendações seguem o interesse recente de cada pessoa e ficam mais variadas
   entre pessoas.

### Próximos experimentos

- Repetir linha de base, validação e teste nas placas de vídeo, que têm outro padrão
  de compra (poucas compras por pessoa, mais acessórios ao redor). A meia-vida escolhida
  nas bonecas é testada lá sem reajuste, para ver se generaliza.
- Variantes de texto (passo 6b): se o canal de texto melhorar, o conteúdo deve passar
  a contribuir além do último produto.
- Pesos da fusão no híbrido, escolhidos na mesma validação.
