# 🏗️ SCARLET SAILS - ARCHITECTURE

**Алгоритмическая торговая система для криптовалютных рынков**

**Version**: 1.0.0  
**Last Updated**: 23 ноября 2025  
**Status**: ✅ Архитектура готова

---

## 📋 СОДЕРЖАНИЕ

1. [Обзор системы](#обзор-системы)
2. [Ключевые компоненты](#ключевые-компоненты)
3. [Поток данных](#поток-данных)
4. [Стратегии](#стратегии)
5. [Оркестрация](#оркестрация)
6. [Риск-менеджмент](#риск-менеджмент)
7. [Модели](#модели)
8. [Структура файлов](#структура-файлов)

---

## 🎯 ОБЗОР СИСТЕМЫ

Scarlet Sails - модульная система алгоритмической торговли, объединяющая:

- **Rule-Based**: RSI mean-reversion
- **Machine Learning**: XGBoost (AUC 0.70)
- **Reinforcement Learning**: DQN (Episode 87)
- **Hybrid**: Взвешенный ансамбль (α=0.45, β=0.45, γ=0.10)

### Целевые метрики

| Метрика | Цель | Статус |
|---------|------|--------|
| Monthly ROI | 10-25% | 🔄 Phase 4 |
| Profit Factor | > 2.0 | ✅ 2.12 |
| Max Drawdown | < 15% | ✅ 9.44% |
| Sharpe Ratio | > 1.0 | 🔄 Phase 4 |
| Win Rate | > 55% | 🔄 Phase 4 |

---

## 🧩 КЛЮЧЕВЫЕ КОМПОНЕНТЫ

### 1. Стратегии

```
strategies/
├── rule_based_v2.py      # RSI mean-reversion
├── xgboost_ml_v2.py      # ML predictions (AUC 0.70)
├── hybrid_v2.py          # Ensemble
└── __init__.py
```

### 2. Reinforcement Learning

```
rl/
├── dqn.py                   # DQN agent (12→[128,128]→3)
├── trading_environment.py   # Trading environment
└── __init__.py
```

### 3. Оркестрация

```
orchestrator.py              # Unified management
```

Функции:
- Signal aggregation (voting)
- Portfolio management
- Risk validation
- Performance tracking

### 4. Анализ

```
analysis/
├── dispersion_analyzer.py   # ANOVA, KS tests
├── dispersion_visualizer.py # Charts
└── __init__.py
```

---

## 🔄 ПОТОК ДАННЫХ

```
Market Data (OHLCV)
        ↓
 [Feature Engineering]
        ↓
    [4 Strategies]
   /    |    |    \
  RB   ML  Hybrid  DQN
   \    |    |    /
        ↓
  [Orchestrator]
  (Voting/Aggregation)
        ↓
   [Risk Check]
    /        \
 PASS       FAIL
   ↓          ↓
[Execute]  [Hold]
   ↓
[Portfolio Update]
   ↓
[Metrics Logging]
```

---

## 🎲 СТРАТЕГИИ

### 1. Rule-Based (RSI)

**Формула:**
```
P_rb = W_opportunity × ∏[filters] - C - R_adaptive
```

**Характеристики:**
- Сигналы: 195/2000 (9.75%)
- Mean: 0.060
- Sparse, high-conviction

### 2. XGBoost ML

**Формула:**
```
P_ml = f_ML(X) - C - R_ood
```

**Модель:**
- Features: 31
- AUC: 0.6978
- Training: 10k samples (SMOTE)

**Характеристики:**
- Сигналы: 2000/2000 (100%)
- Mean: 0.172
- Consistent

### 3. Hybrid

**Формула:**
```
P_hyb = α·P_rb + β·P_ml + γ·V
```

**Веса:**
- α (Rule): 0.45
- β (ML): 0.45
- γ (RL): 0.10

**Характеристики:**
- Сигналы: 1879/2000 (93.95%)
- Mean: 0.191
- Best balance

### 4. DQN RL

**Архитектура:**
```
Input: 12 features
Hidden: [128, 128] (ReLU)
Output: 3 actions (Buy/Sell/Hold)
```

**Состояние (12 features):**
- Normalized price
- RSI, MACD, Bollinger
- Volume, Volatility
- Position status, PnL

**Характеристики:**
- Сигналы: 2000/2000 (100%)
- Mean: 0.176
- Adaptive

---

## 🎛️ ОРКЕСТРАЦИЯ

### StrategyOrchestrator

```python
class StrategyOrchestrator:
    def __init__(strategies, capital, risk_config):
        # Initialize
        
    def aggregate_signals(signals):
        # Voting: majority wins
        
    def execute_signal(signal):
        # Risk check → Execute
        
    def step(data, time):
        # Process one timestep
        
    def get_performance():
        # Return metrics
```

**Signal Aggregation:**
```
Buy votes > Sell votes → BUY
Sell votes > Buy votes → SELL
Otherwise → HOLD
```

---

## 🛡️ РИСК-МЕНЕДЖМЕНТ

### RiskManager

**Параметры:**
```python
max_position_size = 0.10     # 10% equity
max_total_exposure = 0.50    # 50% equity
max_drawdown = 0.15          # 15% drawdown
```

**Проверки:**

1. **Position Size:**
   ```
   position_value / equity ≤ 10%
   ```

2. **Total Exposure:**
   ```
   Σ(positions) / equity ≤ 50%
   ```

3. **Drawdown:**
   ```
   (equity - peak) / peak ≥ -15%
   ```

---

## 📊 МОДЕЛИ

### XGBoost V2

**Файл:** `models/xgboost_trained_v2.json`  
**Размер:** 1.3 MB

**Метрики:**
- AUC: 0.6978
- Precision: 0.64
- Recall: 0.66

**Улучшения:**
- +12% AUC vs V1
- 2x больше данных
- SMOTE balancing

### DQN

**Файл:** `models/dqn_best_pnl.pth`  
**Размер:** 313 KB

**Config:**
```python
state_dim: 12
action_dim: 3
hidden: [128, 128]
gamma: 0.95
lr: 0.0001
```

**Статус:**
- Episode: 87
- Steps: 173,913
- Epsilon: 0.65

---

## 📁 СТРУКТУРА ФАЙЛОВ

```
scarlet-sails/
│
├── strategies/              # Стратегии
│   ├── rule_based_v2.py
│   ├── xgboost_ml_v2.py
│   └── hybrid_v2.py
│
├── rl/                      # RL компоненты
│   ├── dqn.py
│   └── trading_environment.py
│
├── analysis/                # Анализ
│   ├── dispersion_analyzer.py
│   └── dispersion_visualizer.py
│
├── models/                  # Обученные модели
│   ├── xgboost_trained_v2.json (1.3 MB)
│   └── dqn_best_pnl.pth (313 KB)
│
├── orchestrator.py          # Оркестратор
├── backtester.py            # Backtesting
├── test_integration.py      # Тесты
├── run_dispersion_analysis.py
│
├── README.md
├── ARCHITECTURE.md          # Этот файл
├── requirements.txt
└── .gitignore
```

---

## 📈 DISPERSION ANALYSIS

**Статистика:**
```
F-statistic: 611.86
p-value: < 0.001
eta²: 19.64% (large effect)

Вывод: Стратегии принимают
статистически разные решения ✅
```

**Корреляции:**
```
         RB     ML    Hyb    DQN
RB     1.00  -0.02   0.02   0.01
ML    -0.02   1.00  -0.10  -0.13
Hyb    0.02  -0.10   1.00   0.04
DQN    0.01  -0.13   0.04   1.00
```

**Вывод:** Независимость подтверждена ✅

---

## 🚀 DEPLOYMENT

### Локальная разработка

```bash
# 1. Clone
git clone https://github.com/USER/scarlet-sails.git
cd scarlet-sails

# 2. Install
pip install -r requirements.txt

# 3. Test
python test_integration.py
python run_dispersion_analysis.py

# 4. Backtest
python backtester.py
```

### Требования

- Python 3.8+
- 8GB+ RAM
- GPU опционально (для обучения DQN)

---

## 🔮 ROADMAP

### Phase 4: "Глубоководный горизонт - 1" (45 дней)

**Week 1-2:** Real Market Data
- Binance API integration
- Historical data pipeline
- Feature engineering

**Week 3-4:** Model Optimization
- Hyperparameter tuning
- Feature selection
- Cross-validation

**Week 5-6:** Ensemble Methods
- Model stacking
- Voting strategies
- Performance validation

**Week 7-8:** Risk Management
- Stop-loss optimization
- Position sizing
- Portfolio allocation

### После Phase 4

- Live paper trading
- Multi-asset support
- Advanced RL (PPO, A3C)
- Real-time monitoring
- Production deployment

---

**Version**: 1.0.0  
**Author**: STAR_ANT + Team  
**Date**: 23 ноября 2025  
**Status**: ✅ Готово к Phase 4