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
┌──────────────────────────────────────────────────────────────────┐
│                     DATA & STATE LAYER                           │
│  Market data → Feature Engine → Canonical State S(t)             │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   SIGNAL LAYER (P_rb, P_ml, P_hyb)               │
├──────────────────────────────────────────────────────────────────┤
│  P_rb (Rule-Based)                                               │
│  ├─ Opportunity Scorer: trend, momentum, volume                  │
│  └─ Risk Penalty: GARCH, CVaR, drawdown                          │
│                                                                  │
│  P_ml (XGBoost, threshold from config.yaml: models.xgboost.threshold) │
│  ├─ Multi-TF features: 4 timeframes × 31 indicators             │
│  └─ Binary classifier: Long/Neutral                             │
│                                                                  │
│  P_hyb (Q-Learner, α=0.6, β=0.4)                                 │
│  ├─ State discretization: (P_rb, P_ml, regime, dispersion)      │
│  ├─ Q-learning: ε-greedy exploration, TD updates                │
│  └─ Hybrid combination: 0.6×V(Q) + 0.4×P_rb                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    RISK LAYER (Week 2)                           │
├──────────────────────────────────────────────────────────────────┤
│  OOD Detector (Mahalanobis Distance)                             │
│  ├─ Detects out-of-distribution market states                   │
│  └─ Trained on historical feature covariance                    │
│                                                                  │
│  Regime Detector (ATR-based)                                     │
│  ├─ Low volatility: ATR < 33rd percentile                       │
│  ├─ Normal: 33rd-66th percentile                                │
│  └─ High volatility: ATR > 66th percentile                      │
│                                                                  │
│  Rolling Dispersion                                              │
│  └─ Measures agreement between P_rb, P_ml, P_hyb                │
│                                                                  │
│  Dynamic Position Sizer                                          │
│  ├─ Inputs: OOD score, regime, dispersion                       │
│  ├─ Range: 0.1 (minimal) to 1.5 (max leverage)                  │
│  └─ Default: 1.0 (neutral conditions)                           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   COUNCIL & RAG LAYER                            │
│  [Quant Signals] + [S(t)] + [RAG Context]                        │
│           ↓                                                      │
│  LLM Council: Pattern Detection → Risk Assessment                │
│           ↓                                                      │
│  Structured Recommendation                                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   HUMAN DECISION LAYER                           │
│  Recommendation → Human Review → ACCEPT/MODIFY/REJECT            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   EXECUTION & RISK LAYER                         │
│  Position sizing, SL/TP, Kill-switch, Trade logging              │
└──────────────────────────────────────────────────────────────────┘
```

See [docs/SYSTEM_ARCHITECTURE_DETAILED.md](docs/SYSTEM_ARCHITECTURE_DETAILED.md) for detailed documentation.

## Key Features

### Quantitative Foundation
- **Rule-based strategy** (P_rb) with opportunity scoring and risk penalties
- **XGBoost ML** (P_ml) with multi-timeframe features (4 TF × 31 features)
- **Hybrid Q-Learning strategy** (P_hyb) combining quant signals with RL value estimation

### Risk Management (Week 2 Updates)
- **OOD Detection**: Mahalanobis distance-based outlier detection for unusual market states
- **Regime Detection**: ATR-based volatility regime classification (low/normal/high)
- **Rolling Dispersion**: Real-time measurement of strategy agreement
- **Dynamic Position Sizing**: Adaptive position sizing based on OOD score, regime, and dispersion
  - Range: 0.1 (high uncertainty) to 1.5 (high confidence)
  - Prevents overexposure in risky conditions
  - Allows leverage in favorable setups

### Performance Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Tests | 173+ | Unit + integration coverage |
| Walk-forward Sharpe | **2.91** | Average across 5 folds |
| Walk-forward Win Rate | 65.7% | Consistent across time periods |
| Dynamic Sizing Sharpe | **2.11** | vs 1.51 static |
| Dynamic Sizing Calmar | **8.98** | vs 3.19 static |
| Max DD (dynamic) | -15.4% | vs -36.4% static |
| Avg Position Size | 0.84 | Conservative sizing in practice |

### LLM Council
- Pattern detection from screenshots (vision) + numerical data
- RAG retrieval of similar historical states
- Structured recommendations with confidence and dissent

### Research Goal
- Dispersion analysis between P_rb, P_ml, P_hyb
- ANOVA, Kolmogorov-Smirnov tests
- Variance decomposition across market regimes

## Project Structure
```
scarlet-sails/
├── core/                    # Data processing and state building
│   ├── feature_engine_v2.py
│   ├── data_loader.py
│   ├── canonical_state.py   # Unified S(t) builder
│   ├── ood_detector.py      # [WEEK 2] Out-of-distribution detection
│   ├── regime_detector.py   # [WEEK 2] Volatility regime classifier
│   ├── rolling_dispersion.py # [WEEK 2] Strategy agreement tracker
│   ├── dynamic_position_sizer.py # [WEEK 2] Adaptive position sizing
│   └── sanitize_features.py # [WEEK 2] Feature validation
│
├── components/              # Reusable scoring components
│   ├── opportunity_scorer.py
│   └── advanced_risk_penalty.py
│
├── strategies/              # Quant strategy implementations
│   ├── rule_based_v2.py     # P_rb(S)
│   ├── xgboost_ml_v3.py     # P_ml(S)
│   └── hybrid_q_learner.py  # [WEEK 2] P_hyb(S) with Q-learning
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
├── analysis/                # Backtesting and validation
│   ├── walk_forward_validation.py # [WEEK 2] WFV implementation
│   ├── backtest_dynamic_sizing.py # [WEEK 2] Dynamic vs static comparison
│   ├── dispersion_analyzer.py
│   └── dispersion_visualizer.py
│
├── execution/               # Order management and risk
├── data/features/           # Parquet files (14 coins × 4 TF)
├── docs/                    # Documentation
│   ├── MATHEMATICAL_FRAMEWORK.md
│   ├── SYSTEM_ARCHITECTURE_DETAILED.md
│   └── PHASE3_STATUS.md
└── tests/                   # Unit and integration tests
```

## Data

The system uses pre-computed features stored in parquet format:
- **Coins:** BTC, ETH, SOL, AVAX, DOT, LINK, UNI, LTC, ALGO, HBAR, LDO, SUI, ENA, ONDO
- **Timeframes:** 15m, 1h, 4h, 1d
- **Features:** 74 technical indicators per state
- **History:** ~4 years (2021-12 to 2025-10)

## Installation
```bash
git clone https://github.com/AntI-labs1/ScArlet-Sails.git
cd ScArlet-Sails
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # подтянет vectorbt, quantstats и пр.
```
Python требуется **>=3.10** (см. `pyproject.toml`).

## Usage

Канонический бэктест — через единый vectorbt-движок:
```bash
# Health check (config, models, data layout)
python main.py

# Single-asset backtest (рекомендуемый путь)
python run_backtest.py --strategy rsi --coin BTC --timeframe 15m

# Multi-asset comparison
python run_backtest.py --strategy combined --coins BTC ETH SOL --timeframe 15m

# Training pipeline (XGBoost v3)
python scripts/train_xgboost_v3.py --coin BTC --tf 4h

# Walk-forward validation (legacy; будет смигрирована на vbt)
python analysis/walk_forward_validation.py

# Tests
pytest tests/ -q
```

**Старые скрипты `backtesting/honest_backtest*.py`, `analysis/backtest_*.py`,
`core/backtest_engine.py` помечены DEPRECATED** — см. `backtesting/MIGRATION_NOTES.md`.

## Research

The primary research goal is to prove that P_rb, P_ml, and P_hyb produce **significantly different decisions** for the same market state S(t).

This dispersion is not just academic — it's used for:
- **Risk sizing:** High agreement → larger position, high disagreement → smaller or skip
- **Regime detection:** Understanding when each strategy performs best
- **Publication:** Formal statistical analysis for academic paper

### Week 2 Achievements
1. ✅ **OOD Detection**: Mahalanobis-based outlier detection with 95th percentile threshold
2. ✅ **Regime Detection**: ATR-based volatility classification (3 regimes)
3. ✅ **Dynamic Position Sizing**: Risk-adjusted sizing improves Calmar from 3.19→8.98
4. ✅ **Q-Learning Strategy**: Hybrid P_hyb with episodic learning and value function
5. ✅ **Walk-Forward Validation**: 5-fold validation with avg Sharpe 2.91
6. ✅ **Feature Sanitization**: Robust NaN/Inf handling for production stability
7. ✅ **173+ Tests**: Comprehensive unit and integration test coverage

## Team

- **ANT_S** — Operator, Researcher, Final Decision Maker
- **Egor 1, Egor 2** — Pattern annotation, RAG maintenance
- **Mathematicians** — Statistical validation

## Status

### Phase 1: Foundation ✅
- [x] Data pipeline (59 parquet files)
- [x] Feature engine v2
- [x] Rule-based strategy (P_rb)
- [x] XGBoost ML strategy (P_ml)
- [x] Risk components (GARCH, CVaR)

### Phase 2: Advanced Risk & RL ✅ (Week 2)
- [x] OOD detection
- [x] Regime detection
- [x] Dynamic position sizing
- [x] Hybrid Q-learner strategy (P_hyb)
- [x] Walk-forward validation
- [x] Feature sanitization

### Phase 3: Council & Human-in-Loop 🚧
- [ ] Canonical state builder
- [ ] Council agents
- [ ] RAG retrieval
- [ ] Human interface
- [ ] Full dispersion analysis

## License

Private repository. All rights reserved.
