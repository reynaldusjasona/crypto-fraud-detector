# crypto-fraud-detector

Detect fraudulent Ethereum wallet addresses using graph-based feature engineering and machine learning classifiers.

Given an Ethereum address, the system analyzes its transaction history, builds a local transaction graph, extracts structural and statistical features, and returns a fraud risk score with explanations.

## Why this exists

Blockchain fraud (wash trading, phishing, pump-and-dump schemes) costs billions annually. Tools like Chainalysis and Elliptic solve this at enterprise scale. This project demonstrates the same core approach at a portfolio scale: collect on-chain data, engineer meaningful features from transaction graphs, and train classifiers that distinguish fraudulent wallets from legitimate ones.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Etherscan   │────>│  Data Collector   │────>│  Raw Data    │
│  API         │     │  (rate-limited)   │     │  (CSV/JSON)  │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                    ┌──────────────────┐              │
                    │  Feature Engine   │<────────────┘
                    │                  │
                    │  - Transaction   │
                    │  - Graph (NX)    │
                    │  - Statistical   │
                    └───────┬──────────┘
                            │
                    ┌───────v──────────┐
                    │  ML Pipeline     │
                    │                  │
                    │  - Random Forest │
                    │  - XGBoost       │
                    │  - SHAP explain  │
                    └───────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
      ┌───────v────────┐         ┌────────v───────┐
      │  FastAPI        │         │  Streamlit     │
      │  REST API       │         │  Dashboard     │
      │                │         │                │
      │  POST /score   │         │  - Risk score  │
      │  GET /health   │         │  - Graph viz   │
      └────────────────┘         │  - SHAP chart  │
                                 └────────────────┘
```

## Features

**Transaction features**: total sent/received, average value, min/max value, transaction frequency, time span of activity.

**Graph features**: in-degree, out-degree, clustering coefficient, PageRank, self-loop detection, circular path detection, community membership (Louvain).

**Statistical features**: value distribution skewness/kurtosis, burst detection (sudden activity spikes), ratio of unique counterparties to total transactions.

## Tech stack

| Layer              | Tools                                      |
|--------------------|---------------------------------------------|
| Data collection    | Etherscan API, Kaggle dataset               |
| Data processing    | pandas, NumPy                               |
| Graph analysis     | NetworkX                                    |
| ML training        | scikit-learn, XGBoost                       |
| Explainability     | SHAP                                        |
| API                | FastAPI, Pydantic                           |
| Dashboard          | Streamlit, Plotly                            |
| Deployment         | Docker, Docker Compose                      |
| Testing            | pytest                                      |

## Project structure

```
crypto-fraud-detector/
├── src/
│   ├── collector/          # Etherscan API client, dataset loaders
│   ├── features/           # Feature engineering (transaction, graph, statistical)
│   ├── models/             # Training, evaluation, inference
│   ├── api/                # FastAPI REST endpoints
│   └── dashboard/          # Streamlit interactive dashboard
├── notebooks/              # Exploration and analysis notebooks
├── tests/                  # Unit tests
├── data/                   # Raw and processed data (gitignored)
├── models/                 # Trained model artifacts (gitignored)
├── docs/                   # Architecture and design notes
├── Makefile                # Common commands (make train, make api, etc.)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- Free Etherscan API key ([etherscan.io/apis](https://etherscan.io/apis))

### Install

```bash
git clone https://github.com/reynaldusjasona/crypto-fraud-detector.git
cd crypto-fraud-detector

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ETHERSCAN_API_KEY
```

### Run

```bash
# Collect data
make collect

# Train model
make train

# Start API server
make api

# Launch dashboard
make dashboard
```

### Docker

```bash
docker compose up --build
```

## Data sources

1. **Ethereum Fraud Detection Dataset** (Kaggle): ~10K labeled addresses (fraud/legitimate) with pre-computed transaction features. Used for initial model training.
2. **Etherscan API** (free tier, 5 req/sec): live transaction history lookup for scoring new addresses.

## Model performance

*Results will be added after training is complete.*

| Model          | Precision | Recall | F1    | AUC-ROC |
|----------------|-----------|--------|-------|---------|
| Random Forest  | -         | -      | -     | -       |
| XGBoost        | -         | -      | -     | -       |

## Roadmap

- [x] Project scaffold and repo setup
- [ ] Data collection pipeline (Etherscan API client + Kaggle loader)
- [ ] Feature engineering (transaction, graph, statistical)
- [ ] Model training and evaluation
- [ ] FastAPI REST API
- [ ] Streamlit dashboard with graph visualization
- [ ] Docker deployment
- [ ] Deploy dashboard to Streamlit Cloud

## License

MIT
