# Datathon - Passos Mágicos: Predição de Defasagem Escolar

## Visão Geral do Projeto

Este projeto foi desenvolvido para o Datathon da FIAP em parceria com a ONG **Passos Mágicos**, que atua na transformação da vida de crianças e jovens em situação de vulnerabilidade social por meio da educação.

### Problema

Identificar precocemente alunos com risco de **piora na defasagem escolar** (diferença entre a série esperada e a série real do aluno), permitindo intervenções preventivas por parte da equipe pedagógica.

### Solução

Um modelo de classificação binária (Random Forest) que prediz se a defasagem de um aluno irá piorar no próximo período, com base em indicadores do framework PEDE (Pedagógico, Engajamento, Desenvolvimento, Emocional):

| Feature | Descrição |
|---------|-----------|
| `ieg` | Indicador de Engajamento |
| `iaa` | Indicador de Autoavaliação |
| `ips` | Indicador Psicossocial |
| `ida` | Indicador de Desempenho Acadêmico |
| `ian` | Indicador de Adequação de Nível |
| `ipv` | Indicador de Ponto de Virada |
| `inde` | Índice de Desenvolvimento Geral |
| `stone` | Categoria de pedra (Quartzo=1, Ágata=2, Ametista=3, Topázio=4) |
| `age` | Idade do aluno |

### Stack Tecnológica

- **Linguagem**: Python 3.13+
- **Gerenciador de pacotes**: [uv](https://docs.astral.sh/uv/)
- **Banco de dados**: DuckDB
- **ML**: scikit-learn (Random Forest), pandas, numpy
- **API**: FastAPI + Uvicorn
- **Observabilidade**: Prometheus (métricas), logging estruturado, detecção de drift (PSI)
- **Containerização**: Docker + Docker Compose

## Estrutura do Projeto

```
datathon/
├── data/
│   ├── duckdb/datathon.db              # Banco DuckDB com dados brutos e refinados
│   └── queries/                        # SQL para criação e merge de tabelas
├── datathon/
│   ├── api/
│   │   ├── main.py                     # Entrypoint FastAPI (rotas + middleware)
│   │   ├── models/                     # Schemas Pydantic (request/response)
│   │   ├── routes/
│   │   │   ├── predict.py              # POST /api/predict, /api/predict/single
│   │   │   └── model.py               # GET /api/model/info, /api/model/drift
│   │   └── util/
│   │       ├── logging.py              # Logging estruturado + middleware
│   │       ├── metrics.py              # Métricas Prometheus + drift monitor (PSI)
│   │       └── model.py               # Carregamento singleton do modelo
│   ├── database/
│   │   └── client.py                   # Cliente DuckDB
│   ├── modeling/
│   │   ├── __main__.py                 # Entrypoint: python -m datathon.modeling
│   │   ├── config.py                   # FEATURE_COLUMNS
│   │   ├── types.py                    # TrainedModel, ModelMetrics, FeatureBaseline
│   │   ├── baseline.py                 # Cálculo de distribuições baseline
│   │   ├── evaluation.py              # Avaliação e importância de features
│   │   └── train.py                    # Orquestração do treinamento
│   └── preprocessing/
│       ├── mapping.py                  # Mapeamentos de renomeação por ano
│       ├── cleaning.py                 # Limpeza: renomear, dropar, padronizar
│       ├── encoding.py                 # Encoding de categóricos + listas de colunas
│       ├── imputation.py              # Imputação (mediana/moda)
│       ├── outliers.py                 # Detecção e tratamento de outliers (IQR)
│       ├── pipeline.py                 # Pipeline completo de pré-processamento
│       └── transformations.py          # Re-export shim (compatibilidade)
├── models/
│   └── lag_worsening.pkl               # Modelo treinado serializado
├── reports/
│   └── outlier_boxplots.png            # Visualização de outliers
├── scripts/
│   ├── api_examples.sh                 # Exemplos de chamadas à API
│   └── test_drift.sh                   # Teste de detecção de drift
├── tests/                              # Testes automatizados (pytest)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Instruções de Deploy

### Pré-requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Docker e Docker Compose (para deploy containerizado)

### Instalação local

```bash
# Instalar dependências
uv sync

# Executar pipeline de pré-processamento (requer dados brutos no DuckDB)
make preprocess

# Treinar o modelo
make train

# Iniciar a API
make api
```

A API estará disponível em `http://localhost:8000`.

### Deploy com Docker

```bash
# Subir o container (build + run)
make up

# Ver logs
make logs

# Parar o container
make down
```

### Todos os comandos Make

| Comando | Descrição |
|---------|-----------|
| `make api` | Inicia a API localmente (porta 8000) |
| `make preprocess` | Executa o pipeline de pré-processamento |
| `make train` | Treina o modelo de classificação |
| `make test` | Executa os testes automatizados |
| `make up` | Sobe o container Docker |
| `make down` | Para o container Docker |
| `make logs` | Exibe logs do container |
| `make examples` | Executa exemplos de chamadas à API |
| `make test-drift` | Testa a detecção de drift (requer API rodando) |
| `make db` | Abre o DuckDB em modo leitura |

## Exemplos de Chamadas à API

### Health Check

```bash
curl http://localhost:8000/
# {"status": "ok"}
```

### Predição individual

```bash
curl -X POST http://localhost:8000/api/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "ieg": 7.5,
    "iaa": 8.0,
    "ips": 6.5,
    "ida": 7.2,
    "ian": 8.5,
    "ipv": 6.0,
    "inde": 7.3,
    "stone": 3,
    "age": 14
  }'
```

Resposta:
```json
{
  "will_worsen": false,
  "probability": 0.32
}
```

### Predição em lote

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {"ieg": 7.5, "iaa": 8.0, "ips": 6.5, "ida": 7.2, "ian": 8.5, "ipv": 6.0, "inde": 7.3, "stone": 3, "age": 14},
      {"ieg": 5.0, "iaa": 4.5, "ips": 5.2, "ida": 4.8, "ian": 5.0, "ipv": 3.5, "inde": 4.7, "stone": 1, "age": 12}
    ]
  }'
```

Resposta:
```json
{
  "predictions": [
    {"will_worsen": false, "probability": 0.32},
    {"will_worsen": true, "probability": 0.78}
  ]
}
```

### Informações do modelo

```bash
curl http://localhost:8000/api/model/info
```

Retorna métricas (accuracy, precision, recall, F1, AUC-ROC, cross-validation) e importância de features.

### Detecção de drift

```bash
curl http://localhost:8000/api/model/drift
```

Retorna o status de drift (`no_data`, `no_drift`, `warning`, `alert`) e o PSI por feature.

Para testar o fluxo completo de drift (envia dados normais, verifica ausência de drift, envia dados com drift, verifica alerta):

```bash
make test-drift
```

### Métricas Prometheus

```bash
curl http://localhost:8000/metrics
```

## Etapas do Pipeline de ML

### 1. Pré-processamento (`make preprocess`)

O pipeline processa dados brutos de 2022 a 2024, executando as seguintes etapas:

1. **Limpeza** (`cleaning.py`): Renomeia colunas para nomes padronizados, padroniza gênero e instituição de ensino, remove colunas desnecessárias.

2. **Encoding** (`encoding.py`): Converte colunas numéricas para `float`, codifica variáveis categóricas:
   - Stone: ordinal (Quartzo=1 < Ágata=2 < Ametista=3 < Topázio=4)
   - Gênero: binário (Feminino=0, Masculino=1)
   - Instituição de ensino: categórico inteiro (0-9)

3. **Tratamento de outliers** (`outliers.py`): Detecção via IQR (Interquartile Range) com winsorização (capping nos limites Q1-1.5*IQR e Q3+1.5*IQR).

4. **Imputação** (`imputation.py`): Mediana para colunas numéricas, moda para colunas categóricas.

5. **Arredondamento**: Valores numéricos arredondados para 2 casas decimais.

### 2. Treinamento (`make train`)

- **Algoritmo**: Random Forest Classifier com regularização (`max_depth=5`, `min_samples_leaf=10`, `class_weight='balanced'`)
- **Target**: `lag_next > lag_current` (se a defasagem piorou)
- **Scaling**: StandardScaler nas features
- **Split**: 80/20 estratificado
- **Validação cruzada**: 5-fold stratified com métrica F1
- **Artefato**: Salvo como `models/lag_worsening.pkl` com modelo, scaler, métricas e distribuições baseline para drift detection

### 3. Monitoramento em produção

- **Métricas Prometheus**: Duração de requests, contagem de predições, distribuição de probabilidades, distribuição de features
- **Detecção de drift**: Calcula o PSI (Population Stability Index) entre a distribuição das features em produção e a baseline de treinamento. Thresholds: PSI > 0.2 (warning), PSI > 0.25 (alert)
- **Logging estruturado**: Logs de requests, predições e eventos do modelo com timestamps e contexto
