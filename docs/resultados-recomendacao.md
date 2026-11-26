# Resultados da recomendação

Avaliação temporal *leave-last-out* descrita em
[fase-2-personalizacao.md](fase-2-personalizacao.md#5-como-medir-sem-rótulos-a-avaliação-temporal).
Para cada pessoa, a última avaliação da categoria é escondida e o sistema tenta
colocá-la entre os 10 primeiros a partir das anteriores.

Comando:

```bash
mini-lcm avaliar-recomendacao --categoria bonecas --curva
```

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

### Próximos experimentos

- **Perfil com peso pela recência:** multiplicar o peso de cada avaliação por um
  decaimento exponencial pela posição no histórico (meia-vida de poucas avaliações). Os
  itens 3 e 4 sugerem que isso deve recuperar o conteúdo com histórico longo.
- **Pesos da fusão:** dar mais peso à coocorrência no RRF, ou escolher o peso pela
  quantidade de histórico disponível.
- Repetir tudo nas placas de vídeo, que têm outro padrão de compra (poucas compras por
  pessoa, mais acessórios ao redor).
