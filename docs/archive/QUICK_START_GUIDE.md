# QUICK START GUIDE - SCARLET SAILS

**For:** Resuming work after pause or onboarding new team member
**Updated:** 2025-11-10

---

## 🚀 ONE-MINUTE QUICK START

```powershell
# 1. Navigate to project
cd C:\Users\Dmitriy\scarlet-sails

# 2. Activate environment (if using venv)
# venv\Scripts\activate

# 3. Check git status
git status
git log --oneline -5

# 4. Read summary
cat DAY12_FINAL_SUMMARY.md

# 5. Run master audit (optional - takes 30min)
python scripts/master_comprehensive_audit.py
```

---

## 📁 PROJECT STRUCTURE

```
scarlet-sails/
├── models/                    # Trading models (position manager, regime detector)
├── features/                  # Feature engineering
├── scripts/                   # Analysis & backtest scripts
├── backtesting/               # Backtest engine
├── reports/                   # Generated reports
│   ├── master_audit/         # Day 12: 56 combination results
│   └── day11_forensics/      # Day 11: 10 hardcore tests
├── data/
│   ├── raw/                  # Raw OHLCV data (*.parquet) - NOT in git
│   ├── features/             # Engineered features - NOT in git
│   └── processed/            # Processed data - NOT in git
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

---

## 🎯 CURRENT STATUS (Day 12)

### ✅ Completed:
1. **Core System:** HybridPositionManager with adaptive stops
2. **Validation:** 56 combinations tested (14 assets × 4 timeframes)
3. **Honest Assessment:** Reality-checked backtest results
4. **Key Files:** All code committed to GitHub

### 🔄 Next Tasks (Week 2):
1. **Build execution layer** (CCXT integration)
2. **Paper trading** (2 weeks on testnet)
3. **Risk management** (position sizing, drawdown limits)

---

## 📊 KEY RESULTS (Backtest)

### Top 3 (with >5 years data):
1. **ALGO 15m:** 342.3% annual (7.81 years)
2. **ETH 15m:** 181.2% annual (7.81 years)
3. **HBAR 15m:** 175.5% annual (7.81 years)

### **Realistic Expectations (After Adjustments):**
- **Conservative:** 50% annual
- **Base Case:** 70-80% annual
- **Optimistic:** 110% annual

*See DAY12_FINAL_SUMMARY.md for honest reality check*

---

## 🔧 COMMON TASKS

### 1. Run Master Audit (56 combinations)
```powershell
cd C:\Users\Dmitriy\scarlet-sails
python scripts/master_comprehensive_audit.py
```
**Output:** `reports/master_audit/raw_results.json` + `comprehensive_report.txt`
**Time:** ~30 minutes

### 2. Run Day 11 Forensic Analysis
```powershell
python scripts/day11_forensic_analysis.py
```
**Output:** `reports/day11_forensics/forensic_report.md` + CSVs
**Time:** ~15 minutes

### 3. Test Week 1 Improvements
```powershell
python scripts/test_week1_improvements.py
```
**Output:** Console output with strategy comparison
**Time:** ~5 minutes

### 4. Check Data Availability
```powershell
ls data/raw/*.parquet | wc -l    # Linux/Mac
dir data\raw\*.parquet           # Windows
```
**Expected:** 56 files (14 assets × 4 timeframes)

---

## 📝 IMPORTANT FILES TO READ

### Priority 1 (Must Read):
1. **DAY12_FINAL_SUMMARY.md** - Complete 12-day summary
2. **reports/master_audit/comprehensive_report.txt** - 56 combination results
3. **reports/day11_forensics/forensic_report.md** - 10 hardcore tests

### Priority 2 (Reference):
4. **FILE_INVENTORY.md** - Complete file list
5. **models/hybrid_position_manager.py** - Core position logic
6. **scripts/master_comprehensive_audit.py** - How master audit works

---

## 🔍 UNDERSTANDING THE SYSTEM

### Entry Signal:
```python
# RSI < 30 (oversold mean reversion)
if rsi < 30:
    signal = True
```

### Position Management:
```python
HybridPositionManager:
    - Detects market regime
    - Sets adaptive stop-loss (ATR-based)
    - Uses trailing stop in profit
    - Partial exits at key levels
    - Max holding time: 7 days (168 bars on 1h)
```

### Exit Conditions:
1. **Stop-Loss:** Hit adaptive stop
2. **Take-Profit:** Hit regime-based TP
3. **Trailing Stop:** Follow price in profit
4. **Max Hold Time:** 7 days exceeded
5. **Regime Change:** Detected regime shift

---

## 🐛 TROUBLESHOOTING

### Problem: "No data found"
```powershell
# Check if data files exist
dir data\raw\BTCUSDT_*.parquet

# If missing, data not downloaded yet
# Data files are ~20GB, not in git
# You need to download them separately
```

### Problem: "ModuleNotFoundError"
```powershell
# Install requirements
pip install -r requirements.txt

# Check Python version (need 3.9+)
python --version
```

### Problem: "Git branch not found"
```powershell
# Check current branch
git branch

# Switch to correct branch
git checkout claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH

# Or create if doesn't exist
git checkout -b claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
```

### Problem: "Results differ from DAY12_FINAL_SUMMARY.md"
```
This is expected! Results vary based on:
1. Data period used
2. Random seed (if any)
3. Exact parameter values

Small variations (<5%) are normal.
Large variations (>20%) indicate a problem.
```

---

## 📈 NEXT WEEK PRIORITIES

### Week 2 Goal: Production Readiness
1. **CCXT Integration** (2-3 days)
   - Connect to Binance Testnet
   - Implement order placement
   - Handle partial fills, rejections

2. **Real Slippage Tracking** (1-2 days)
   - Log intended price vs filled price
   - Measure real slippage over 1 week
   - Update cost assumptions

3. **Paper Trading** (ongoing)
   - Run system on Testnet for 2 weeks
   - Monitor all trades
   - Track:
     - Slippage
     - Latency
     - Fill rates
     - Execution failures

4. **Risk Management** (2-3 days)
   - Position sizing based on volatility
   - Portfolio correlation limits
   - Drawdown monitoring
   - Emergency stop system

---

## 🔐 SECURITY REMINDERS

### ❌ NEVER COMMIT:
- `.env` files (API keys)
- `config.yaml` with secrets
- Private keys
- Personal data

### ✅ ALWAYS:
- Use `.gitignore` for sensitive files
- Test on Testnet first
- Start with small position sizes
- Monitor system 24/7 in early stages

---

## 📞 GETTING HELP

### Resources:
1. **This Project:**
   - Read `DAY12_FINAL_SUMMARY.md`
   - Check `reports/master_audit/`
   - Review `reports/day11_forensics/`

2. **External:**
   - CCXT docs: https://docs.ccxt.com/
   - Binance API: https://binance-docs.github.io/apidocs/
   - Python docs: https://docs.python.org/3/

---

## ✅ HEALTH CHECK COMMANDS

Run these to verify everything is working:

```powershell
# 1. Git status
git status
# Expected: "nothing to commit, working tree clean"

# 2. Python imports
python -c "import pandas, numpy, ccxt; print('OK')"
# Expected: "OK"

# 3. Check key files
dir models\hybrid_position_manager.py
dir scripts\master_comprehensive_audit.py
dir reports\master_audit\raw_results.json
# Expected: Files exist

# 4. Quick data check
python -c "import pandas as pd; df=pd.read_parquet('data/raw/BTCUSDT_1h.parquet'); print(f'BTC 1h: {len(df)} bars')"
# Expected: "BTC 1h: XXXXX bars"
```

---

## 🎯 REMEMBER

1. **Backtest ≠ Production**
   - Results WILL be worse in production
   - Expect 50-70% of backtest performance

2. **Start Small**
   - First week: $100-500
   - First month: $1000-2000
   - Scale gradually after validation

3. **Monitor Everything**
   - Slippage
   - Fill rates
   - Execution time
   - Costs

4. **Be Honest**
   - Log all trades
   - Track real costs
   - Don't cherry-pick results
   - Admit when something doesn't work

---

## 📌 QUICK REFERENCE

### Key Metrics (Backtest):
- **Best Performer:** ALGO 15m (342.3% annual, 7.81 years)
- **Most Stable:** BTC 1d (18.2% annual, Sharpe 5.89)
- **Average (Top 10):** ~200% annual (backtest)
- **Realistic Expectation:** 50-110% annual (production)

### Key Parameters:
- **Entry:** RSI < 30
- **Stop-Loss:** ATR-based (adaptive)
- **Max Hold:** 7 days
- **Cost Assumption:** 0.15% per trade (backtest)
- **Real Cost:** 0.3-0.5% per trade (expected)

---

**Good luck! Remember: Trade responsibly, start small, and always use proper risk management.**

---

*Last updated: Day 12 (2025-11-10)*
