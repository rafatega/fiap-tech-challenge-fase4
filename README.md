# LSTM Stock Predictor, PETR4.SA

Tech Challenge Fase 4 (Pós-Tech FIAP): pipeline completo de deep learning para prever o preço de fechamento de PETR4.SA com uma LSTM, servido por uma API REST e um dashboard, ambos em produção no Render, sem Docker.

## Pipeline

1. **Dados** (`src/stock_predictor/data.py`): histórico via `yfinance` desde 2018-01-01. O modelo prevê **log-retornos** (`log(P_t/P_{t-1})`), não o preço bruto, preço bruto é não-estacionário e um LSTM treinado sobre ele tende só a "copiar" o último valor observado, o que infla artificialmente as métricas sem representar poder preditivo real. O preço é reconstruído (`P_t = P_{t-1} * exp(retorno previsto)`) só na camada de saída. As janelas de entrada usam `SEQUENCE_LENGTH = 30` dias úteis (~1 mês de pregão), dentro da faixa comum na literatura de LSTM para séries financeiras (tipicamente 20-60 dias), equilibrando contexto histórico suficiente sem diluir demais o sinal com um passado muito distante.
2. **Modelo** (`src/stock_predictor/model.py`, `train.py`): LSTM em PyTorch Lightning, com `EarlyStopping` e `ModelCheckpoint`. Tracking de experimentos via MLflow (backend local em arquivo, pasta `mlruns/`, sem servidor). **Descoberta empírica:** a configuração inicial (`hidden_size=64`, `num_layers=2`) não batia a baseline naive, o modelo, com capacidade demais para um sinal ruidoso e poucas amostras, entrava em overfitting no treino sem generalizar para o teste. Reduzir para `hidden_size=16`, `num_layers=1` (valores atuais em `config.py`) resolveu, produzindo um modelo que efetivamente supera a persistência simples nas métricas de teste. Essa configuração foi depois confirmada por uma busca sistemática de hiperparâmetros (ver seção "Busca de hiperparâmetros" abaixo).
3. **Avaliação** (`src/stock_predictor/baseline.py`): MAE, RMSE e MAPE (em R$, na escala de preço reconstruída) do LSTM comparados lado a lado com uma baseline naive (persistence, "amanhã = hoje"), para evidenciar que o modelo tem poder preditivo real além da alta autocorrelação natural de preços de ações.
4. **Exportação**: `models/lstm_model.joblib` (pesos + hiperparâmetros), `models/scaler.joblib` (normalização) e `models/metrics.json`.
5. **API** (`src/stock_predictor/api/`): FastAPI com `POST /predict`, `GET /health` e `GET /metrics` (formato Prometheus).
6. **Dashboard** (`dashboard/app.py`): Streamlit, consome a API, mostra histórico + previsão e as métricas de desempenho.

## Como rodar localmente

```bash
pip install -r requirements.txt

# Treinar o modelo (gera os artefatos em models/)
PYTHONPATH=src python -m stock_predictor.train

# Rodar a busca de hiperparâmetros (avalia candidatos na validação, registra tudo no MLflow)
PYTHONPATH=src python -m stock_predictor.sweep

# Ver os experimentos de treino no MLflow
mlflow ui  # abre em http://127.0.0.1:5000

# Subir a API
PYTHONPATH=src uvicorn stock_predictor.api.main:app --reload

# Em outro terminal, subir o dashboard
PYTHONPATH=src streamlit run dashboard/app.py

# Rodar os testes
pytest
```

No Windows PowerShell, troque `PYTHONPATH=src comando` por `$env:PYTHONPATH="src"; comando`.

## Endpoints da API

### `POST /predict`

```json
{
  "closes": [38.1, 38.4, 38.0, "..."],
  "days_ahead": 5
}
```

Retorna as previsões de preço de fechamento para os próximos `days_ahead` dias úteis. `closes` precisa ter pelo menos `sequence_length + 1` preços (ver `models/lstm_model.joblib`).

O campo `ticker` na resposta identifica o modelo servido (é um modelo especialista em PETR4.SA), não os dados de entrada enviados em `closes`, a API não valida que os preços informados são de fato de PETR4.SA.

### `GET /health`

Retorna `{"status": "ok", "model_loaded": true}`.

### `GET /metrics`

Métricas em formato Prometheus (requisições, latência, uso de CPU/RAM), prontas para serem raspadas por um Prometheus caso um dia seja configurado.

## Métricas do modelo

Números do conjunto de teste (dados nunca vistos durante o treino), gerados automaticamente pelo `train.py` e persistidos em `models/metrics.json`:

| Métrica  | LSTM   | Baseline naive (persistência) | Melhora do LSTM |
|----------|--------|--------------------------------|------------------|
| MAE (R$) | 0,412  | 0,416                          | ~1,0%            |
| RMSE (R$)| 0,579  | 0,581                          | ~0,3%            |
| MAPE (%) | 1,156% | 1,168%                         | ~1,0%            |

O LSTM supera a baseline naive nas três métricas, mas por margem estreita (~0,3–1,0%), dentro da variabilidade amostral esperada de um conjunto de teste de poucas centenas de pregões, não há teste estatístico formal (ex: Diebold-Mariano) validando significância. Isso é consistente com a hipótese de mercado eficiente: evidência de que o modelo não é pior que a persistência, não de poder preditivo explorável.

## Busca de hiperparâmetros

Dado que a vantagem do LSTM sobre a baseline naive é pequena (tabela acima), foi feita uma busca sistemática por hiperparâmetros (`src/stock_predictor/sweep.py`) para verificar se a configuração de produção já era o melhor resultado possível com esses dados, ou se dava para melhorar de forma real.

**Metodologia:** cada candidato foi avaliado em MAE/RMSE/MAPE de preço na **validação**, nunca no teste, para não contaminar o conjunto de teste como critério de escolha de modelo (o teste só é tocado uma única vez, no final, depois que a configuração já está decidida). Todas as execuções foram registradas no MLflow, num experimento separado (`lstm-hp-sweep`), para comparação lado a lado.

**Candidatos testados**, variando um eixo por vez a partir da configuração de produção (`sequence_length=30, hidden_size=16, num_layers=1, learning_rate=1e-3, dropout=0,0`):

| Variação | MAE validação (LSTM) | MAE validação (naive) | LSTM vs. naive |
| --- | --- | --- | --- |
| Produção (referência) | 0,3352 | 0,3341 | pior (-0,33%) |
| sequence_length=20 | 0,3350 | 0,3333 | pior (-0,52%) |
| sequence_length=45 | 0,3369 | 0,3359 | pior (-0,31%) |
| sequence_length=60 | 0,3339 | 0,3318 | pior (-0,60%) |
| hidden_size=8 | 0,3337 | 0,3341 | melhor (+0,10%) |
| hidden_size=32 | 0,3354 | 0,3341 | pior (-0,40%) |
| dropout=0,2 | 0,3344 | 0,3341 | pior (-0,10%) |
| num_layers=2, dropout=0,2 | 0,3353 | 0,3341 | pior (-0,37%) |
| learning_rate=5e-4 | 0,3355 | 0,3341 | pior (-0,43%) |

Na validação, praticamente todas as configurações testadas, inclusive a de produção, ficaram ligeiramente **abaixo** do naive. Só uma (`hidden_size=8`, resto igual à produção) ficou marginalmente acima, por 0,10%, uma diferença pequena diante da variação natural entre as próprias configurações.

**Confirmação no teste:** mesmo com uma vantagem tão pequena, essa configuração foi retreinada e avaliada uma única vez no conjunto de teste, para verificar se o resultado se sustentava fora da validação. Resultado: MAE de 0,4148 (0,32% melhor que o naive), **pior** do que a configuração de produção atual (MAE de 0,4120, 1,0% melhor que o naive). A vantagem observada na validação não se confirmou no teste, evidência de que era ruído amostral, não um sinal real.

**Conclusão:** a configuração de produção original permanece a melhor encontrada, e esse resultado agora está respaldado por uma busca sistemática, não apenas por uma única rodada de treino. Nenhuma das configurações testadas apresentou vantagem robusta e consistente sobre a persistência simples, achado consistente com a hipótese de mercado eficiente, e mais uma evidência de que ganhos de precisão nesse tipo de dado tendem a ser pequenos e frágeis quando a única entrada é o preço passado.

## Limitações conhecidas

- A previsão multi-dia é recursiva: cada dia usa a previsão do dia anterior como entrada, então o erro se acumula, previsões de `days_ahead` maiores são menos confiáveis que a de 1 dia.
- O split treino/validação/teste é cronológico único, não walk-forward. Walk-forward validation (retreinar/reavaliar em janelas sucessivas ao longo do tempo) é a extensão natural para tornar a avaliação mais robusta a diferentes regimes de mercado, mas está fora do escopo deste entregável.
- Como a maioria dos modelos desse tipo, o resultado deve ser interpretado à luz da hipótese de mercado eficiente: nenhuma métrica de erro baixa aqui implica uma estratégia de investimento lucrativa.

## Deploy

- API: [https://lstm-stock-api-tj0s.onrender.com/docs](https://lstm-stock-api-tj0s.onrender.com/docs) (docs interativos em `/docs`)
- Dashboard: [https://lstm-stock-dashboard-ueg1.onrender.com](https://lstm-stock-dashboard-ueg1.onrender.com)

Deploy sem Docker: `render.yaml` define dois serviços web nativos Python no Render, `lstm-stock-api` (a API FastAPI, com `healthCheckPath: /health`) e `lstm-stock-dashboard` (o Streamlit, que consome a API via a env var `API_URL`). Ambos rodam no plano free do Render, que hiberna após 15 min de inatividade, a primeira requisição depois disso pode levar de 30 a 60s para responder (cold start).

## Vídeo

Vídeo demonstrando o funcionamento da API: `[<link>](https://www.youtube.com/watch?v=qIpQSamZDSc)`.
