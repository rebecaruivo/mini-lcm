# Resultados da recomendação

Dois protocolos de avaliação temporal, descritos em
[fase-2-personalizacao.md](fase-2-personalizacao.md#5-como-medir-sem-rótulos-a-avaliação-temporal):

- **Cortes por data (principal):** datas de corte semestrais iguais para todos. O modelo
  aprende só com o que veio antes de cada data, e o alvo de cada pessoa é a primeira
  avaliação nos seis meses seguintes. Sem vazamento entre pessoas.
- **Corte por usuário (anterior):** a última avaliação de cada pessoa é escondida. Foi o
  primeiro protocolo do projeto; os resultados ficam registrados mais abaixo, junto com
  os dois problemas que levaram à troca.

Comandos:

```bash
# protocolo principal; parâmetros de categorias.yaml
mini-lcm avaliar-recomendacao --categoria bonecas --curva
# escolha de parâmetros (validação: os cortes mais antigos; o teste não é usado)
mini-lcm avaliar-recomendacao --categoria bonecas --validacao --janela-dias 90
# protocolo anterior
mini-lcm avaliar-recomendacao --categoria bonecas --curva --corte usuario
# análise do vazamento da popularidade no protocolo anterior
python scripts/analise_popularidade_temporal.py --categoria bonecas
```

## Resumo

hit@10 no teste, com os parâmetros escolhidos na validação:

| Método | bonecas, por data | placas, por data | bonecas, por usuário (anterior) | placas, por usuário (anterior) |
|---|---|---|---|---|
| aleatório | 0,001 | 0,004 | 0,003 | 0,013 |
| popularidade | 0,009 | 0,035 | 0,051 | 0,146 |
| em alta | 0,104 | **0,083** | — | — |
| último produto | 0,048 | 0,031 | 0,169 | 0,085 |
| conteúdo (perfil) | 0,045 | 0,022 | 0,160 | 0,070 |
| coocorrência | 0,051 | 0,018 | 0,260 | 0,041 |
| híbrido | 0,063 | 0,031 | 0,267 | 0,060 |
| híbrido + em alta | **0,126** | 0,070 | — | — |

1. **O protocolo anterior inflava a personalização.** No protocolo por data, o híbrido das
   bonecas cai de 0,267 para 0,063. A diferença vem de compras em lote e do vazamento
   entre pessoas, medidos abaixo.
2. **"O que está em alta" é a linha de base a bater.** A popularidade numa janela recente
   (30 dias nas bonecas, 90 nas placas) supera todos os métodos personalizados isolados
   nas duas categorias.
3. **Gosto + momento:** nas bonecas, o híbrido + em alta é o melhor método, 0,021 acima do
   em alta sozinho (IC 95% de +0,009 a +0,034). Nas placas, o em alta sozinho é o melhor.
4. **Produtos novos:** 40% dos alvos das bonecas e 65% dos das placas são produtos sem
   nenhuma interação antes do corte. Só os métodos de conteúdo acertam algum (cerca de
   3%, contra menos de 1% do acaso). Reservar vagas do top 10 para eles não compensou:
   o que se ganha nos novos se perde, em dobro, nos já vistos.

## Protocolo principal: cortes semestrais por data

### Por que trocar de protocolo

Dois problemas do corte por usuário, medidos nos dados (com o catálogo da primeira
importação, de 3.000 bonecas):

**Compras em lote.** Metade dos alvos das bonecas (52%) foi avaliada no mesmo dia da
avaliação anterior da mesma pessoa, o que normalmente indica produtos comprados juntos.
São casos muito mais fáceis do que prever a próxima compra:

| hit@10 no corte por usuário (bonecas, com recência) | alvo no mesmo dia da anterior | alvo em outro dia |
|---|---|---|
| último produto | 0,244 | 0,088 |
| coocorrência | 0,392 | 0,118 |

Nas placas, 24% dos alvos estão nessa situação. No corte por data, nenhum: a próxima
compra depois da data de corte não pode estar no mesmo pedido que o histórico.

**Vazamento entre pessoas.** O treino incluía avaliações de outras pessoas posteriores ao
teste de cada um (em média, 39% do treino nas bonecas e 30% nas placas). A popularidade e
a coocorrência aproveitavam esse futuro (seção "a popularidade se aproveita do futuro",
no protocolo anterior).

As duas coisas respondem perguntas diferentes: o corte por usuário mede "o próximo item,
incluindo o mesmo pedido"; o corte por data mede "a próxima compra, numa visita futura",
que é a situação de um sistema de recomendação em produção.

### Desenho

- **Cortes:** 1º de janeiro e 1º de julho de cada ano. Em cada corte, o modelo
  (popularidade, em alta, coocorrência) aprende só com as avaliações anteriores à data.
  Caso de teste: cada pessoa com pelo menos 3 produtos antes do corte e alguma avaliação
  nos seis meses seguintes; o alvo é a primeira delas.
- **Catálogo completo:** todos os produtos que passam no filtro da categoria (8.407
  bonecas e 5.577 placas), avaliados ou não pelos usuários da amostra.
- **Catálogo do período:** num corte, concorre um produto que já tinha alguma avaliação
  antes da data ou cuja data de lançamento (metadados da Amazon) é anterior ao fim da
  janela. Nenhuma avaliação posterior ao corte entra na regra. Produto novo sem data de
  lançamento não concorre; se for o alvo, conta como erro para todos os métodos.
- **Validação e teste:** os cortes mais antigos, até somar metade dos casos, formam a
  validação; os demais, o teste. Todos os parâmetros foram escolhidos na validação, e o
  teste foi rodado uma vez.
- **Incerteza:** a mesma pessoa aparece em vários cortes, e seus casos não são
  independentes. O bootstrap sorteia pessoas, levando todos os casos de cada uma.

| | Bonecas | Placas |
|---|---|---|
| Validação | 2.166 casos, cortes de jul/2001 a jul/2018 | 232 casos, cortes de jul/2002 a jan/2018 |
| Teste | 1.876 casos de 1.101 pessoas, cortes de jan/2019 a jul/2023 | 228 casos de 144 pessoas, cortes de jul/2018 a jan/2023 |
| Produtos com data de lançamento | 2.550 de 8.407 (30%) | 5.367 de 5.577 (96%) |
| Produtos novos candidatos por corte (teste) | 922 em média | 3.759 em média |
| Alvos novos no teste | 39,8% (73% deles com data, portanto candidatos) | 64,9% (93% com data) |

### Escolha dos parâmetros na validação

**Janela do em alta** (hit@10 do em alta sozinho):

| Janela | 30 dias | 90 dias | 180 dias | 365 dias | 730 dias |
|---|---|---|---|---|---|
| Bonecas | **0,066** | 0,054 | 0,045 | 0,036 | 0,027 |
| Placas | 0,086 | **0,129** | 0,112 | 0,125 | 0,095 |

**Meia-vida e peso do em alta** (hit@10 do híbrido + em alta, janela já escolhida):

Bonecas:

| Meia-vida \ peso | 0,5 | 1 | 2 | 4 |
|---|---|---|---|---|
| sem recência | 0,058 | 0,067 | 0,068 | 0,067 |
| 1 | 0,064 | 0,073 | 0,072 | 0,068 |
| 2 | 0,065 | 0,071 | **0,075** | 0,070 |
| 3 | 0,066 | 0,070 | 0,074 | 0,072 |

Placas:

| Meia-vida \ peso | 0,5 | 1 | 2 | 4 |
|---|---|---|---|---|
| sem recência | 0,091 | 0,108 | 0,108 | 0,108 |
| 1 | 0,078 | 0,112 | **0,121** | 0,103 |
| 2 | 0,082 | 0,116 | 0,116 | 0,108 |
| 3 | 0,078 | 0,112 | 0,112 | 0,108 |

**Vagas reservadas para produtos novos** (as últimas posições do top 10 do híbrido + em
alta vão para os produtos novos de maior pontuação de conteúdo):

| Vagas | bonecas: total | novos | já vistos | placas: total | novos | já vistos |
|---|---|---|---|---|---|---|
| 0 | **0,075** | 0,001 | 0,122 | **0,121** | 0,000 | 0,378 |
| 1 | 0,074 | 0,013 | 0,113 | **0,121** | 0,006 | 0,365 |
| 2 | 0,071 | 0,020 | 0,103 | 0,116 | 0,006 | 0,351 |
| 3 | 0,066 | 0,020 | 0,095 | 0,112 | 0,006 | 0,338 |
| 5 | 0,054 | 0,030 | 0,069 | 0,108 | 0,032 | 0,270 |

Escolhidos (em `categorias.yaml`): bonecas com meia-vida 2, janela de 30 dias, peso 2 e
nenhuma vaga reservada; placas com meia-vida 1, janela de 90 dias, peso 2 e nenhuma vaga.
As diferenças vizinhas na grade são menores que a incerteza da validação (cerca de
±0,012 nas bonecas e ±0,04 nas placas): os valores exatos importam pouco. O que os dados
sustentam é a ordem de grandeza da janela (semanas nas bonecas, meses nas placas), um
peso alto para o em alta e alguma recência no perfil.

### Resultado no teste: bonecas

1.876 casos, 1.101 pessoas, 10 cortes (jan/2019 a jul/2023); 39,8% dos alvos são novos.

| Método | hit@10 | IC 95% | NDCG@10 | cobertura | hit@10 alvos novos | hit@10 alvos já vistos |
|---|---|---|---|---|---|---|
| aleatório | 0,001 | 0,000–0,003 | 0,000 | 54,9% | 0,000 | 0,002 |
| popularidade | 0,009 | 0,005–0,013 | 0,004 | 0,2% | 0,000 | 0,014 |
| em alta | 0,104 | 0,090–0,119 | 0,058 | 1,4% | 0,000 | 0,173 |
| último produto | 0,048 | 0,038–0,058 | 0,024 | 37,6% | **0,031** | 0,059 |
| conteúdo (perfil) | 0,045 | 0,036–0,055 | 0,022 | 25,6% | 0,025 | 0,058 |
| coocorrência | 0,051 | 0,042–0,062 | 0,028 | 38,4% | 0,000 | 0,085 |
| híbrido | 0,063 | 0,052–0,076 | 0,032 | 29,2% | 0,004 | 0,103 |
| híbrido + em alta | **0,126** | 0,110–0,142 | **0,071** | 7,9% | 0,000 | **0,209** |

Curva de histórico (hit@10): o híbrido + em alta vai de 0,110 (só a última avaliação) a
0,128 (as duas últimas) e fica estável daí em diante (0,126 com todas).

### Resultado no teste: placas de vídeo

228 casos, 144 pessoas, 10 cortes (jul/2018 a jan/2023); 64,9% dos alvos são novos.

| Método | hit@10 | IC 95% | NDCG@10 | cobertura | hit@10 alvos novos | hit@10 alvos já vistos |
|---|---|---|---|---|---|---|
| aleatório | 0,004 | 0,000–0,014 | 0,002 | 23,5% | 0,007 | 0,000 |
| popularidade | 0,035 | 0,013–0,063 | 0,019 | 0,3% | 0,000 | 0,100 |
| em alta | **0,083** | 0,049–0,122 | **0,057** | 1,8% | 0,000 | **0,237** |
| último produto | 0,031 | 0,009–0,054 | 0,011 | 19,9% | **0,034** | 0,025 |
| conteúdo (perfil) | 0,022 | 0,004–0,043 | 0,012 | 19,0% | 0,020 | 0,025 |
| coocorrência | 0,018 | 0,004–0,036 | 0,006 | 12,1% | 0,000 | 0,050 |
| híbrido | 0,031 | 0,012–0,054 | 0,018 | 15,1% | 0,013 | 0,062 |
| híbrido + em alta | 0,070 | 0,040–0,105 | 0,041 | 3,5% | 0,000 | 0,200 |

### Comparações pareadas no teste

Bootstrap pareado por pessoa, IC 95%.

| A | B | Bonecas | Placas |
|---|---|---|---|
| híbrido + em alta | em alta | +0,021 (+0,009 a +0,034) | −0,013 (−0,040 a +0,010) |
| híbrido + em alta | híbrido | +0,062 (+0,047 a +0,077) | +0,040 (+0,004 a +0,077) |
| em alta | popularidade | +0,096 (+0,081 a +0,112) | +0,048 (+0,009 a +0,086) |
| híbrido | coocorrência | +0,012 (+0,002 a +0,022) | +0,013 (−0,009 a +0,036) |
| conteúdo | último produto | −0,003 (−0,012 a +0,006) | −0,009 (−0,023 a +0,000) |
| último produto | aleatório | +0,047 (+0,037 a +0,057) | +0,026 (+0,004 a +0,051) |

Como referência (não escolhido na validação): com 3 vagas reservadas, o híbrido + em alta
fica em 0,118 nas bonecas (novos 0,035; já vistos 0,173) e 0,075 nas placas (novos 0,014;
já vistos 0,188).

### Leitura

1. **O momento prevê mais que a pessoa.** O em alta, igual para todos, ganha de qualquer
   método personalizado isolado nas duas categorias. Nas bonecas, a popularidade de todos
   os tempos (0,009) é quase inútil, e a dos últimos 30 dias (0,104) é dez vezes melhor:
   lançamentos e reedições concentram as compras em semanas.
2. **Nas bonecas, gosto + momento é melhor que cada um.** O híbrido + em alta soma 0,021
   ao em alta (IC sem zero) e 0,062 ao híbrido. Nas placas, compra rara e guiada pela
   geração do produto, o gosto individual não acrescenta.
3. **O ganho está todo nos produtos já vistos.** Nos alvos novos, o híbrido + em alta
   acerta zero: a lista em alta e a coocorrência só contêm produtos com interações, e a
   fusão empurra as sugestões de conteúdo para fora do top 10.
4. **Só o conteúdo alcança produtos novos, e ainda pouco.** O último produto acerta 3,1%
   (bonecas) e 3,4% (placas) dos alvos novos, entre centenas a milhares de candidatos
   novos por semestre. É sinal real (o acaso fica abaixo de 1%), mas fraco demais para
   ganhar posições no top 10: reservar vagas para novos troca cada acerto ganho neles
   por cerca de dois perdidos nos já vistos.
5. **Consequência:** a parte do problema que mais pesa em classificados, em que quase
   todo anúncio é novo, é a que depende da qualidade dos embeddings. Melhorar o canal de
   conteúdo (fase 1) é o caminho para os produtos novos, antes de qualquer ajuste de fusão.

### Versão anterior do catálogo

A primeira rodada deste protocolo usou o catálogo montado só com os produtos avaliados
pela amostra (3.000 bonecas, 1.118 placas) e a data da primeira avaliação como data de
lançamento. Com isso, os produtos novos candidatos de cada semestre eram justamente os
que alguém da amostra ia avaliar, e as vagas reservadas pareciam levar as placas de 0,13
a 0,33 na validação. Os métodos baseados em interações deram números parecidos nas duas
versões (bonecas: em alta 0,117, híbrido + em alta 0,133); o acerto do conteúdo em
produtos novos caiu de cerca de 5% para cerca de 3% com o catálogo completo. Detalhes na
decisão de 26/11 em `decisoes.md`.

## Protocolo anterior: corte por usuário

Resultados do primeiro protocolo (a última avaliação de cada pessoa é escondida). Ficam
registrados porque motivaram a recência e a troca de protocolo; os números absolutos são
otimistas pelos motivos da seção anterior.

### Bonecas: dados

#### Dados

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

### Bonecas: linha de base (média simples)

#### Histórico completo

| Método | hit@10 | IC 95% | NDCG@10 | MRR | cobertura |
|---|---|---|---|---|---|
| aleatório | 0,003 | 0,001–0,004 | 0,001 | 0,003 | 100,0% |
| popularidade | 0,051 | 0,044–0,059 | 0,023 | 0,022 | 0,7% |
| último produto | 0,169 | 0,156–0,181 | 0,097 | 0,085 | 78,8% |
| conteúdo (perfil) | 0,113 | 0,102–0,124 | 0,064 | 0,059 | 57,0% |
| coocorrência | **0,233** | 0,219–0,247 | **0,140** | **0,123** | **82,9%** |
| híbrido | 0,216 | 0,203–0,230 | 0,125 | 0,110 | 67,7% |

#### Curva de histórico (hit@10 usando só as últimas N avaliações)

| N | aleatório | popularidade | último | conteúdo | coocorrência | híbrido |
|---|---|---|---|---|---|---|
| 1 | 0,003 | 0,051 | 0,169 | 0,169 | 0,232 | 0,249 |
| 2 | 0,003 | 0,051 | 0,169 | 0,150 | 0,254 | **0,261** |
| 3 | 0,003 | 0,051 | 0,169 | 0,140 | 0,254 | 0,248 |
| 5 | 0,003 | 0,051 | 0,169 | 0,124 | 0,241 | 0,228 |
| todas | 0,003 | 0,051 | 0,169 | 0,113 | 0,233 | 0,216 |

Aleatório, popularidade e último produto não dependem de N (os dois primeiros ignoram
o histórico; o terceiro só usa o produto mais recente).

#### Comparações pareadas

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

#### Leitura

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

#### Efeito da limpeza dos dados

Uma execução anterior, com avaliações de uma importação antiga ainda misturadas
(70.885 em vez de 28.843), dava coocorrência 0,211, híbrido 0,217 e popularidade 0,024.
Com os dados limpos, a ordem entre coocorrência e híbrido se inverteu. A correção está no
commit "Reimportação substitui também as avaliações da categoria".

### Bonecas: peso pela recência

#### Hipótese

Os itens 3 e 4 da leitura acima indicam que as avaliações recentes descrevem melhor o
próximo interesse do que as antigas. Em vez de cortar o histórico (usar só as últimas
N), cada avaliação passa a valer mais quanto mais recente for:

    peso = peso_da_nota × 0,5 ^ (idade / meia_vida)

A idade conta avaliações, não dias: 0 para a mais recente, 1 para a anterior e assim por
diante. Com meia-vida 1, as quatro mais recentes valem 1; 0,5; 0,25 e 0,125. O mesmo peso
vale para o perfil (conteúdo) e para a soma da coocorrência. Contar avaliações em vez de
dias evita depender do ritmo de compra, que varia muito entre pessoas.

#### Escolha da meia-vida sem olhar o teste

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

#### Resultado no teste (meia-vida 1)

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

#### Leitura

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

### Placas de vídeo

#### Dados

| | Bonecas | Placas de vídeo |
|---|---|---|
| Produtos no catálogo | 3.000 | 1.118 |
| Usuários elegíveis (5+ avaliações na categoria) | 3.722 | 345 |
| Avaliações (após remover repetições) | 28.843 | 2.023 |
| Avaliações por produto (média) | 9,6 | 1,8 |
| Usuários testados | 3.504 | 316 |
| Histórico médio de treino | 7,1 | 5,2 |

Num universo de 1,6 milhão de produtos de Electronics, só 345 pessoas avaliaram 5 ou
mais placas de vídeo. Placa de vídeo é compra rara e cara: a maioria das pessoas compra
uma a cada poucos anos. A importação leu 2.220 avaliações; 197 eram repetições.

#### Resultado no teste

| Método | hit@10 média simples | hit@10 com recência (IC 95%) | cobertura (com recência) |
|---|---|---|---|
| aleatório | 0,013 | 0,013 (0,003–0,025) | 94,6% |
| popularidade | **0,146** | **0,146** (0,108–0,183) | 1,2% |
| último produto | 0,085 | 0,085 (0,057–0,117) | 70,4% |
| conteúdo (perfil) | 0,038 | 0,070 (0,044–0,098) | 65,1% |
| coocorrência | 0,029 | 0,041 (0,022–0,063) | 70,0% |
| híbrido | 0,060 | 0,060 (0,035–0,089) | 57,9% |

Curva de histórico, com recência (hit@10):

| N | conteúdo | coocorrência | híbrido |
|---|---|---|---|
| 1 | 0,085 | 0,092 | 0,070 |
| 2 | 0,076 | 0,057 | 0,082 |
| 3 | 0,070 | 0,048 | 0,060 |
| todas | 0,070 | 0,041 | 0,060 |

A meia-vida não foi reajustada para as placas: o valor das bonecas foi aplicado como
está, para testar se generaliza. Na validação das placas (304 usuários), as meias-vidas
de 0,5 a 8 ficam todas entre 0,049 e 0,059 no híbrido, diferenças muito menores que a
incerteza da medida.

Comparações pareadas (bootstrap pareado, IC 95%):

| A | B | A − B | IC 95% |
|---|---|---|---|
| híbrido com recência | popularidade | −0,085 | −0,127 a −0,044 |
| coocorrência com recência | popularidade | −0,104 | −0,146 a −0,070 |
| híbrido com recência | último produto | −0,025 | −0,054 a +0,003 |
| conteúdo com recência | conteúdo sem recência | +0,032 | +0,006 a +0,060 |
| híbrido com recência | híbrido sem recência | +0,000 | −0,025 a +0,029 |

#### Diagnóstico: a popularidade se aproveita do futuro

A avaliação corta o tempo por usuário (limitação já registrada): a popularidade é
contada com o treino de todas as pessoas, incluindo avaliações feitas depois do
produto escondido de alguém. Em média, 30% das interações de treino das placas são
posteriores ao teste de cada pessoa. Numa categoria movida por gerações de produto, isso
pesa: a popularidade "sabe" quais placas vão fazer sucesso.

Teste: recalcular a popularidade de cada pessoa só com as interações anteriores ao
momento do teste dela.

| Popularidade | Bonecas | Placas |
|---|---|---|
| contada com todo o treino (como no relatório) | 0,051 | 0,146 |
| contada só com o passado | 0,042 | 0,092 |
| diferença (IC 95% pareado) | +0,009 (+0,002 a +0,017) | +0,054 (+0,025 a +0,085) |

Um terço do hit@10 da popularidade nas placas vem desse vazamento. A coocorrência tem o
mesmo vazamento e mesmo assim fica abaixo, então a conclusão principal se mantém.

#### Popularidade recente

Se o futuro ajuda tanto, o passado recente deve ajudar também. Variante testada:
contar só as interações de uma janela antes do momento do teste ("o que está em alta").

| Janela | Bonecas | Placas |
|---|---|---|
| 90 dias | **0,131** | 0,085 |
| 180 dias | 0,122 | 0,108 |
| 365 dias | 0,097 | 0,117 |
| 730 dias | 0,077 | **0,127** |
| todo o passado | 0,042 | 0,092 |

(hit@10; só interações anteriores ao teste, então sem vazamento.)

A popularidade recente é uma linha de base bem mais forte que a popularidade global, sem
nenhum vazamento: nas bonecas, sobe de 0,042 para 0,131 (ainda metade do híbrido, 0,267);
nas placas, chega a 0,127 e supera todos os métodos personalizados. A janela boa depende
da categoria: bonecas seguem lançamentos e modas de meses; placas, ciclos de geração de
um a dois anos.

As duas tabelas desta seção e da anterior são reproduzidas por
`python scripts/analise_popularidade_temporal.py --categoria <categoria>`. A análise olha o
teste diretamente; para virar método do sistema, a janela precisa ser escolhida na
validação, o que foi feito no protocolo principal (acima).

#### Leitura

1. **Nas placas, a personalização não vence com estes dados.** Com 1,8 avaliação por
   produto, a coocorrência quase não tem pares para aprender (pior que o último produto),
   e o histórico de cada pessoa é curto e espalhado no tempo: placas compradas com anos de
   intervalo dizem pouco sobre a próxima.
2. **O que prevê a próxima placa é o momento, não a pessoa.** A popularidade recente
   (0,127) supera tudo: quem compra placa compra o modelo da geração atual.
3. **A recência ajuda o conteúdo também aqui** (+0,032, IC sem zero), mas não o híbrido.
4. **A comparação entre as categorias é o resultado.** O mesmo sistema, com os mesmos
   parâmetros, ganha 5 vezes da popularidade nas bonecas (colecionismo, compras
   frequentes, gosto estável) e perde nas placas (compra rara, guiada por ciclo de
   produto). Saber quando personalizar faz parte do problema. No protocolo por data,
   contra a linha de base certa (o em alta), a vantagem das bonecas ficou bem menor.
5. **A avaliação precisa de corte por data global.** O vazamento medido acima é pequeno
   nas bonecas e grande nas placas. Isso levou ao protocolo principal, acima.

## Próximos experimentos

- **Variantes de texto (passo 6b):** o conteúdo é o único caminho para produtos novos;
  melhorar o canal de texto deve aparecer diretamente no hit@10 dos alvos novos, e só
  então vale testar de novo as vagas reservadas.
- **Perfil com mais de um vetor:** colecionadores misturam linhas diferentes, e a média
  (mesmo com recência) representa mal quem tem dois gostos.
