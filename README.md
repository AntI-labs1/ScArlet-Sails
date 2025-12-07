# ScArlet-Sails

**Algorithmic trading system combining quantitative strategies with LLM Council for pattern-based decision making.**
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/268ee985-4994-47c5-ad04-30524610cb77" />

## Overview

ScArlet-Sails is a research and trading system that:

- Combines **multiple strategies** (rule-based, ML, hybrid/RL) into a unified framework
- Analyzes **dispersion** between strategy decisions for risk management
- Uses **LLM Council** to interpret patterns and provide human-readable recommendations
- Keeps **human operator** in the loop for final decisions

The system is built around **Council of Agents** architecture, where:
- Quant modules provide numerical signals (P_rb, P_ml, P_hyb)
- LLM agents interpret patterns and context from RAG
- Human operator makes final trading decisions

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     DATA & STATE LAYER                      │
│  Market data → Feature Engine → Canonical State S(t)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   QUANT STRATEGIES LAYER                    │
│  S(t) → P_rb (Rule-Based)                                   │
│  S(t) → P_ml (XGBoost ML)                                   │
│  S(t) → P_hyb (Hybrid + RL)                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    COUNCIL & RAG LAYER                      │
│  [Quant Signals] + [S(t)] + [RAG Context]                   │
│           ↓                                                 │
│  LLM Council: Pattern Detection → Risk Assessment           │
│           ↓                                                 │
│  Structured Recommendation                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   HUMAN DECISION LAYER                      │
│  Recommendation → Human Review → ACCEPT/MODIFY/REJECT       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   EXECUTION & RISK LAYER                    │
│  Position sizing, SL/TP, Kill-switch, Trade logging         │
└─────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Key Features

**Quantitative Foundation:**
- Rule-based strategy with opportunity scoring and risk penalties
- XGBoost ML with multi-timeframe features (4 TF × 31 features)
- Hybrid strategy combining quant signals with RL value estimation

**Risk Management:**
- GARCH volatility, CVaR tail risk, OOD detection
- Position sizing: `size = risk_per_trade / SL_distance`
- Daily/weekly loss limits with kill-switch

**LLM Council:**
- Pattern detection from screenshots (vision) + numerical data
- RAG retrieval of similar historical states
- Structured recommendations with confidence and dissent

**Research Goal:**
- Dispersion analysis between P_rb, P_ml, P_hyb
- ANOVA, Kolmogorov-Smirnov tests
- Variance decomposition across market regimes

## Project Structure
```
scarlet-sails/
├── core/                    # Data processing and state building
│   ├── feature_engine_v2.py
│   ├── data_loader.py
│   └── canonical_state.py   # Unified S(t) builder
│
├── components/              # Reusable scoring components
│   ├── opportunity_scorer.py
│   └── advanced_risk_penalty.py
│
├── strategies/              # Quant strategy implementations
│   ├── rule_based_v2.py     # P_rb(S)
│   └── xgboost_ml_v3.py     # P_ml(S)
│
├── council/                 # LLM Council agents
│   ├── base_agent.py
│   ├── pattern_detector.py
│   └── recommendation.py
│
├── rag/                     # Knowledge base
│   ├── patterns/            # Pattern library
│   ├── trades/              # Trade history
│   └── lessons/             # Lessons learned
│
├── execution/               # Order management and risk
├── analysis/                # Dispersion analysis
├── data/features/           # Parquet files (14 coins × 4 TF)
└── tests/                   # Unit and integration tests
```

## Data

The system uses pre-computed features stored in parquet format:
- **Coins:** BTC, ETH, SOL, AVAX, DOT, LINK, UNI, LTC, ALGO, HBAR, LDO, SUI, ENA, ONDO
- **Timeframes:** 15m, 1h, 4h, 1d
- **Features:** 74 technical indicators per state

## Installation
```bash
git clone https://github.com/AntI-labs1/ScArlet-Sails.git
cd ScArlet-Sails
pip install -r requirements.txt
```

## Usage
```bash
# Run tests
python -m pytest tests/

# Backtest rule-based strategy
python scripts/run_backtest.py --strategy rule_based --coin BTC --tf 4h

# Train XGBoost model
python scripts/train_xgboost_v3.py --coin BTC --tf 4h
```

## Research

The primary research goal is to prove that P_rb, P_ml, and P_hyb produce **significantly different decisions** for the same market state S(t).

This dispersion is not just academic — it's used for:
- **Risk sizing:** High agreement → larger position, high disagreement → smaller or skip
- **Regime detection:** Understanding when each strategy performs best
- **Publication:** Formal statistical analysis for academic paper

## Team

- **ANT_S** — Operator, Researcher, Final Decision Maker
- **Egor 1, Egor 2** — Pattern annotation, RAG maintenance
- **Mathematicians** — Statistical validation

## Status

- [x] Data pipeline (59 parquet files)
- [x] Feature engine v2
- [x] Rule-based strategy
- [x] XGBoost ML strategy
- [x] Risk components (GARCH, CVaR, OOD)
- [ ] Canonical state builder
- [ ] Council agents
- [ ] RAG retrieval
- [ ] Hybrid/RL strategy
- [ ] Human interface
- [ ] Dispersion analysis

## License

Private repository. All rights reserved.
