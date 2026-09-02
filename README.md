# Simulador de rebaixamento do Brasileirão

Aplicativo Streamlit que estima, rodada a rodada, a probabilidade de rebaixamento dos clubes da Série A. O resultado de cada partida futura é gerado por um modelo Bradley–Terry–Davidson (BTD), com empates e vantagem do mandante; a temporada é concluída por Monte Carlo.

## O que está implementado

- ingestão da Série A (`BSA`) pela API gratuita [football-data.org](https://www.football-data.org/coverage);
- corte temporal por rodada, sem usar resultados posteriores no ajuste;
- forças dos clubes, parâmetro de empate, mando médio e desvios de mando por clube estimados por máxima verossimilhança penalizada;
- placares amostrados condicionalmente a vitória/empate/derrota, para acumular saldo e gols pró;
- ordenação por pontos, vitórias, saldo de gols e gols pró;
- probabilidade de rebaixamento, intervalos empíricos de pontos, histograma e boxplot em Plotly;
- modo demonstrativo offline e importação/exportação CSV.

## Executar

Requer Python 3.11 ou superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

O aplicativo abre inicialmente em **Demonstração**, com dados artificiais. Para dados reais:

1. crie uma chave no [cadastro da football-data.org](https://www.football-data.org/client/register);
2. selecione `football-data.org` na barra lateral;
3. informe a chave e a temporada.

A chave também pode ser definida sem entrar no código:

```powershell
$env:FOOTBALL_DATA_TOKEN = "sua-chave"
streamlit run app.py
```

No Streamlit Community Cloud, salve a chave em `Secrets` como `FOOTBALL_DATA_TOKEN`. Nunca versione a chave.

## Contrato do CSV

O upload aceita o mesmo contrato interno da API:

| coluna | tipo/descrição |
|---|---|
| `matchday` | rodada inteira, iniciando em 1 |
| `status` | `FINISHED`/`AWARDED` para jogo encerrado; outro status para jogo futuro |
| `home_id`, `away_id` | identificadores estáveis dos clubes |
| `home_team`, `away_team` | nomes dos clubes |
| `home_goals`, `away_goals` | inteiros para jogos encerrados; vazios para futuros |
| `utc_date` | opcional, data/hora ISO-8601 |
| `match_id` | opcional, identificador único da partida |

## Metodologia

Para o mandante `H` e visitante `A`, com forças `θ`, mando médio `h`, desvio de mando do clube `δ` e parâmetro de empate positivo `ν`:

```text
a = exp(θ_H + h + δ_H)
b = exp(θ_A)
D = a + b + ν sqrt(a b)

P(H vence) = a / D
P(empate)   = ν sqrt(a b) / D
P(A vence) = b / D
```

As somas das forças `θ` e dos desvios de mando `δ` são fixadas em zero para identificabilidade. Assim, `h` representa o efeito médio de jogar em casa no campeonato e `δ_H` representa quanto o mandante se beneficia acima ou abaixo dessa média. O multiplicador total de mando do clube é `exp(h + δ_H)`.

Há três penalizações L2 independentes: uma para as forças `θ`, uma para o mando médio `h` e outra para os desvios de mando `δ`. A terceira é normalmente mais forte, pois cada clube tem bem menos jogos em casa para sustentar um efeito individual. Um desvio estimado não deve ser interpretado como efeito causal do estádio.

### Peso temporal e seleção dos parâmetros

O ajuste pode dar mais peso aos jogos recentes. Para um jogo da rodada `r`, observado na rodada de referência `R`, o peso é:

```text
w_r = 0,5 ^ ((R - r) / meia_vida)
```

Se a meia-vida for 8 rodadas, um jogo de oito rodadas atrás recebe metade do peso bruto de um jogo da rodada atual. Os pesos são normalizados para média 1 dentro de cada ajuste; isso separa o efeito de recência da intensidade das penalizações.

Por padrão, o aplicativo escolhe a meia-vida e as três regularizações por backtest temporal expansivo. A grade inclui também **sem decaimento** como controle: a recência só é adotada quando melhora a previsão fora da amostra. Em cada fold, o modelo usa somente rodadas anteriores para prever a rodada seguinte. A configuração com menor log loss médio entre os ajustes convergidos é escolhida; o Brier score e o erro-padrão do log loss são exibidos como diagnósticos. Partidas posteriores nunca entram no treino daquele fold. Quando `utc_date` está disponível, jogos adiados disputados depois do início da rodada de validação também são excluídos do treino.

A seleção automática usa somente dados disponíveis até a rodada de corte. Para avaliar historicamente o sistema completo, inclusive a escolha dos hiperparâmetros, o backtest externo deve repetir essa seleção em cada corte — selecionar uma vez com a temporada completa e reutilizar o resultado em rodadas antigas produziria vazamento temporal.

O BTD não produz placares. Depois de sortear o resultado, o aplicativo amostra um placar compatível de uma distribuição empírica por tipo de resultado. Placares reais observados recebem peso maior; um pequeno conjunto de pseudoplacares estabiliza o começo da temporada. Essa segunda camada permite aplicar saldo e gols pró sem afirmar que o BTD seja um modelo de gols.

## Critérios e limitações

A CBF publica a tabela e o regulamento específico de cada edição; consulte o [documento oficial da Série A](https://www.cbf.com.br/futebol-brasileiro/tabelas/campeonato-brasileiro/serie-a/2026?documento=Regulamento) antes do uso operacional. O aplicativo implementa exatamente os critérios solicitados: pontos, vitórias, saldo de gols e gols pró. Se ainda houver empate, usa sorteio residual, pois confronto direto e cartões não estão disponíveis no endpoint gratuito adotado.

Riscos metodológicos relevantes:

- as forças são associativas e não causais; o decaimento captura mudanças recentes somente pelos resultados, sem observar escalações, lesões ou trocas de técnico;
- os jogos são condicionais aos parâmetros ajustados e tratados como independentes;
- a incerteza atual é Monte Carlo condicional; não inclui integralmente a incerteza dos parâmetros estimados;
- em poucas rodadas, regularização e pseudoplacares têm impacto material;
- o mando específico por clube pode refletir composição dos adversários e ruído do calendário, sobretudo com amostra pequena;
- a disponibilidade histórica depende da conta, temporada e política da API;
- o backtest interno escolhe hiperparâmetros para o corte atual, mas não substitui uma avaliação externa em várias temporadas;
- um backtest externo correto deve reajustar o modelo e refazer a seleção em cada rodada, nunca reutilizar parâmetros estimados com a temporada completa.

## Testes

```powershell
python -m pytest -q
```

Os testes cobrem soma das probabilidades, mando médio e específico por clube, regularizações independentes, decaimento temporal, seleção expansiva sem rodadas futuras, restrições de identificabilidade, corte temporal, reprodutibilidade e ordenação pelos critérios simulados.

## Estrutura

```text
app.py                         interface Streamlit
brasileirao/api.py             cliente e normalização da API
brasileirao/data.py            validação, corte e classificação
brasileirao/model.py           ajuste Bradley–Terry–Davidson
brasileirao/simulation.py      placares e Monte Carlo
tests/                         testes automatizados
```
