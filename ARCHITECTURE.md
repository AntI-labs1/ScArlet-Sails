# ScArlet-Sails — System Architecture

## 1. Purpose

ScArlet-Sails is a research and trading system that:

- Combines **multiple strategies** (rule-based, ML, hybrid/RL) into decision functions
- Rigorously analyzes **dispersion** between strategy decisions
- Uses **LLM Council** to interpret patterns and provide recommendations
- Keeps **human operator** as final decision maker

The system is NOT about finding a single "perfect edge." It's about building a **council of agents** where:
- Quant strategies provide numerical opinions
- LLM agents interpret context and patterns
- Human makes informed final decisions

## 2. System Layers

### 2.1 Data & State Layer

**Purpose:** Load market data, compute features, build canonical state S(t).

**Components:**

| File | Description |
|------|-------------|
| `core/feature_engine_v2.py` | Multi-timeframe feature computation (4 TF × 31 features) |
| `core/data_loader.py` | OHLCV data loading |
| `core/canonical_state.py` | Unified state builder |
| `components/opportunity_scorer.py` | Market opportunity scoring W_opp(S) |
| `components/advanced_risk_penalty.py` | Risk penalty R(S) with GARCH, CVaR, OOD |

**Canonical State S(t):**
```python
S(t) = {
    # Price features (per timeframe)
    'rsi_14': float,
    'price_to_ema9': float,
    'price_to_ema21': float,
    'bb_width_pct': float,
    'atr_pct': float,
    'returns_5': float,
    'volume_ratio': float,
    # ... 74 total features
    
    # Derived scores
    'opportunity_score': float,  # W_opp(S)
    'risk_penalty': float,       # R(S)
    'regime': str,               # 'low_vol' | 'normal' | 'high_vol' | 'crisis'
}
```

### 2.2 Quant Strategies Layer

**Purpose:** Generate numerical decision signals from S(t).

**Three Decision Functions:**

#### Rule-Based Strategy (P_rb)
```
P_rb(S) = W_opportunity(S) · ∏ I_i(S) - C_fixed - R_penalty(S)

Where:
- W_opportunity: Market opportunity score [0,1]
- ∏ I_i(S): Binary filters (RSI, EMA, Volume, BB)
- C_fixed: Transaction costs (0.15%)
- R_penalty: Risk penalty (GARCH + CVaR + OOD + DD)
```

**File:** `strategies/rule_based_v2.py`

**Role:** Conservative anchor. "Should we even look at this?"

#### XGBoost ML Strategy (P_ml)
```
P_ml(S) = σ(f_XGB(Φ(S))) · ∏ F_k(S) - C_adaptive(S) - R_ood(S)

Where:
- f_XGB: Trained XGBoost classifier
- Φ(S): Feature transformation (31 features × 4 TF)
- F_k(S): Regime filters (crisis, drawdown)
- C_adaptive: Volatility-adjusted costs
- R_ood: Out-of-distribution penalty
```

**File:** `strategies/xgboost_ml_v3.py`

**Role:** Nonlinear pattern detector. Probabilistic view.

#### Hybrid Strategy (P_hyb)
```
P_hyb(S) = α(t)·P_rb(S) + β(t)·P_ml(S) + γ·V_RL(S)

Where:
- α(t), β(t): Adaptive weights based on rolling performance
- V_RL(S): RL value estimation (DQN)
- γ: RL weight (0.95 discount)
```

**File:** `strategies/hybrid_v1.py` (planned)

**Role:** Dynamic policy. Combines short-term signals with long-term value.

### 2.3 Council & RAG Layer

**Purpose:** Interpret quant signals + context, generate human-readable recommendation.

**Pipeline:**
```
1. Screenshot of current market state
       ↓
2. LLM (vision) describes what it sees
   "Price touching MA50, RSI low, volume increasing"
       ↓
3. Parallel: S(t) → numerical vector (74 features)
       ↓
4. RAG retrieval by S(t):
   - Find top-5 similar historical states
   - Return: pattern, outcome, PnL
       ↓
5. LLM receives:
   - Visual description (step 2)
   - Numerical data S(t)
   - RAG context (similar cases)
   - Quant signals: P_rb, P_ml, P_hyb
       ↓
6. LLM generates recommendation
```

**RAG Structure:**
```
rag/
├── patterns/
│   └── library.json      # Pattern definitions
│
├── states/
│   └── historical.parquet  # S(t) + outcome for retrieval
│
├── trades/
│   └── trade_log.json    # All trades with context
│
└── lessons/
    └── lessons.json      # Lessons learned
```

**Council Agents:**

| Agent | Role |
|-------|------|
| `pattern_detector.py` | Identifies current pattern from visuals + data |
| `risk_assessor.py` | Evaluates risk and position size |
| `contrarian.py` | Devil's advocate, finds weak points |

**Recommendation Output:**
```python
{
    'pattern': 'ma50_bounce',
    'direction': 'long',
    'confidence': 0.75,
    'position_size_pct': 0.8,
    'sl_pct': 4.0,
    'tp_pct': 8.0,
    'quant_signals': {
        'P_rb': 0.65,
        'P_ml': 0.58,
        'P_hyb': 0.61
    },
    'agreement': 0.93,  # 1 - max_spread
    'dissent': 'contrarian: volume declining last 3 bars',
    'similar_cases': [...]
}
```

### 2.4 Human Decision Layer

**Purpose:** Final decision by human operator.

**Interface shows:**
- Pattern identified (with historical examples)
- Current screenshot vs similar historical cases
- Council arguments (for/against)
- Recommended size and risk
- Historical win rate for this pattern

**Human options:**
- **ACCEPT** → Execute as recommended
- **MODIFY** → Change size, SL/TP
- **REJECT** → Log reason to RAG

### 2.5 Execution & Risk Layer

**Purpose:** Safe execution with strict risk limits.

**Risk Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `risk_per_trade_pct` | Max equity risk per trade | 0.5% |
| `max_sl_distance_pct` | Max price movement to SL | 15% |
| `max_position_size_pct` | Hard cap on position | 10% |
| `daily_loss_limit_pct` | Kill-switch daily | 3% |
| `weekly_loss_limit_pct` | Kill-switch weekly | 7% |
| `max_dd_portfolio_pct` | Max portfolio drawdown | 20% |

**Position Sizing Formula:**
```
position_size = risk_per_trade / SL_distance

Example:
- risk_per_trade = 0.75% equity
- SL_distance = 12% price movement
- position_size = 0.0075 / 0.12 = 6.25% of capital

If SL hits → lose 0.75% equity (not 12%)
```

## 3. Research Goal: Dispersion Analysis

**Hypothesis:** P_rb, P_ml, P_hyb produce significantly different decisions for the same S(t).

**Method:**

1. Compute P_rb, P_ml, P_hyb for N market states
2. Apply statistical tests:
   - ANOVA: F-statistic, p-value
   - Kolmogorov-Smirnov: Distribution differences
   - Variance decomposition: Between vs Within

**Expected Results:**
```
F-statistic > 100
p-value < 0.001
Var_between / Var_total > 0.6
```

**Practical Use:**

Dispersion is not just academic — it's a risk signal:
```
Agreement = 1 - |P_rb - P_ml|

Agreement > 0.8: All agree → larger position
Agreement < 0.5: Disagreement → smaller or skip
```

## 4. Data Flow Diagram

See README.md for visual representation.

## 5. File Structure
```
scarlet-sails/
│
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
│
├── core/
│   ├── __init__.py
│   ├── feature_engine_v2.py      # Multi-TF features
│   ├── data_loader.py            # OHLCV loading
│   ├── canonical_state.py        # S(t) builder [NEW]
│   └── position_sizer.py         # Position calculations
│
├── components/
│   ├── __init__.py
│   ├── opportunity_scorer.py     # W_opp(S)
│   └── advanced_risk_penalty.py  # R(S)
│
├── strategies/
│   ├── __init__.py
│   ├── rule_based_v2.py          # P_rb(S)
│   ├── xgboost_ml_v3.py          # P_ml(S)
│   └── hybrid_v1.py              # P_hyb(S) [PLANNED]
│
├── council/
│   ├── __init__.py
│   ├── base_agent.py             # Abstract agent [NEW]
│   ├── pattern_detector.py       # Pattern LLM [NEW]
│   ├── risk_assessor.py          # Risk LLM [NEW]
│   ├── contrarian.py             # Devil's advocate [NEW]
│   ├── discussion.py             # Orchestration [NEW]
│   └── recommendation.py         # Output structure [NEW]
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py              # KNN/vector search [NEW]
│   ├── patterns/
│   │   └── library.json
│   ├── states/
│   │   └── historical.parquet
│   ├── trades/
│   │   └── trade_log.json
│   └── lessons/
│       └── lessons.json
│
├── execution/
│   ├── __init__.py
│   ├── order_manager.py          # Order execution [NEW]
│   ├── position_monitor.py       # Position tracking [NEW]
│   └── risk_manager.py           # Global limits [NEW]
│
├── interface/
│   ├── __init__.py
│   └── cli.py                    # Human interface [NEW]
│
├── analysis/
│   ├── __init__.py
│   └── dispersion_analysis.py    # Research module
│
├── data/
│   └── features/                 # 59 parquet files
│
├── models/
│   ├── xgboost/                  # Trained XGBoost models
│   └── llm/                      # Fine-tuned LLMs [NEW]
│
├── configs/
│   ├── council.yaml              # Council settings [NEW]
│   └── risk_limits.yaml          # Risk parameters [NEW]
│
├── scripts/
│   ├── train_xgboost_v3.py
│   ├── run_backtest.py
│   └── run_council.py            # [NEW]
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── docs/
    ├── pattern_template.md
    └── annotation_guide.md
```

## 6. Configuration

### council.yaml
```yaml
council:
  agents:
    pattern_detector:
      enabled: true
      model: "local_llm"  # or "gpt-4-vision"
    risk_assessor:
      enabled: false  # Phase 2
    contrarian:
      enabled: false  # Phase 2

  discussion:
    max_rounds: 3
    consensus_threshold: 0.7

  human_override:
    enabled: true
    timeout_seconds: 300
```

### risk_limits.yaml
```yaml
risk:
  risk_per_trade_pct: 0.5
  max_sl_distance_pct: 15
  max_position_size_pct: 10
  daily_loss_limit_pct: 3
  weekly_loss_limit_pct: 7
  max_dd_portfolio_pct: 20

markets:
  coins:
    - BTC
    - ETH
    - SOL
    # ... 14 total
  
  timeframes:
    primary: "4h"
    confirmation: ["1h", "15m"]
    context: "1d"
```

## 7. Development Phases

### Phase 1: Foundation (Current)
- [x] Feature engine
- [x] Rule-based strategy
- [x] XGBoost strategy
- [x] Risk components
- [ ] Canonical state builder
- [ ] Basic RAG structure

### Phase 2: Council MVP
- [ ] Base agent class
- [ ] Pattern detector (1 agent)
- [ ] Simple retriever (KNN)
- [ ] CLI interface
- [ ] Integration test

### Phase 3: Full Council
- [ ] Risk assessor agent
- [ ] Contrarian agent
- [ ] Discussion protocol
- [ ] Hybrid/RL strategy

### Phase 4: Research & Production
- [ ] Dispersion analysis module
- [ ] Paper trading (1 month)
- [ ] Vector DB for RAG
- [ ] Research paper draft
