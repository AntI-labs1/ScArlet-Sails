"""
HYBRID SYSTEM BACKTEST - 56 Combinations Test
==============================================

Test NEW Hybrid system (3-layer) vs OLD Rule-based baseline

NEW Hybrid (3-layer):
- Layer 1: RSI < 30 (Rule-based)
- Layer 2: XGBoost ML filter
- Layer 3: Crisis detection

OLD Baseline (from master audit):
- Only RSI < 30 (Rule-based)
- 7,464 trades, WR 47.3%, PF 1.17

Expected improvement:
- Hybrid: ~3,000 trades, WR 55-60%, PF 1.8-2.0
- Better quality through ML filtering
- Crisis protection

Author: Scarlet Sails Team
Date: 2025-11-10
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_entry_system import HybridEntrySystem

print("="*100)
print("HYBRID SYSTEM BACKTEST - 56 COMBINATIONS")
print("="*100)
print("\nComparing:")
print("  OLD: Rule-based (RSI < 30) - 7,464 trades")
print("  NEW: Hybrid (Rules + ML + Crisis) - Expected ~3,000 trades")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

ASSETS = [
    'BTC', 'ETH', 'SOL',           # Top 3
    'LINK', 'LDO', 'SUI', 'HBAR',  # Mid caps
    'ENA', 'ALGO', 'AVAX',         # Additional
    'DOT', 'LTC', 'ONDO', 'UNI'    # More
]

TIMEFRAMES = ['15m', '1h', '4h', '1d']

DATA_DIR = Path('data/raw')
OUTPUT_DIR = Path('reports/hybrid_backtest')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Hybrid system config
ML_THRESHOLD = 0.6  # ML probability threshold
ENABLE_ML_FILTER = False  # Disabled for now
ENABLE_CRISIS_GATE = False  # Disabled for now

# ============================================================================
# HYBRID BACKTEST ENGINE
# ============================================================================

class HybridBacktestEngine:
    """
    Backtest engine using HybridEntrySystem (3-layer)

    Differences from old master_audit:
    - Uses HybridEntrySystem.should_enter() instead of simple RSI < 30
    - ML filter reduces signal count
    - Crisis detector protects during crashes
    """

    def __init__(
        self,
        ml_threshold=0.6,
        enable_ml_filter=True,
        enable_crisis_gate=True
    ):
        self.entry_system = HybridEntrySystem(
            ml_threshold=ml_threshold,
            enable_ml_filter=enable_ml_filter,
            enable_crisis_gate=enable_crisis_gate
        )

        # Exit management (same as master audit)
        self.atr_multipliers = {
            'Bull Trend': 3.0,
            'Bear Market': 2.0,
            'Sideways/Choppy': 1.5,
            'Crisis Event': 1.0
        }
        self.trailing_activation = 0.08  # Activate at +8%
        self.partial_exits = [0.33, 0.33, 0.34]
        self.tp_levels = [1.05, 1.10, 1.15]  # +5%, +10%, +15%

    def generate_signals(self, df):
        """
        Generate entry signals using HybridEntrySystem

        KEY DIFFERENCE: Uses 3-layer system instead of simple RSI < 30
        """
        signals = []

        for i in range(200, len(df)):
            # Use Hybrid system to check entry
            should_enter, reason = self.entry_system.should_enter(df, i)

            if should_enter:
                signals.append({
                    'bar_index': i,
                    'timestamp': df.index[i],
                    'price': df['close'].iloc[i],
                    'rsi': df['rsi'].iloc[i] if 'rsi' in df.columns else df['RSI_14'].iloc[i],
                    'reason': reason,
                    'regime': self._detect_regime(df.iloc[:i+1])
                })

        return signals

    def _detect_regime(self, df):
        """Simple regime detection (for exit management)"""
        if len(df) < 200:
            return 'Sideways/Choppy'

        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma200 = df['close'].rolling(200).mean().iloc[-1]

        if ma20 > ma200 * 1.05:
            return 'Bull Trend'
        elif ma20 < ma200 * 0.95:
            return 'Bear Market'
        else:
            return 'Sideways/Choppy'

    def backtest(self, df, signals):
        """
        Backtest with adaptive exits

        Same exit logic as master audit for fair comparison
        """
        trades = []

        for sig in signals:
            entry_bar = sig['bar_index']
            entry_price = sig['price']
            regime = sig['regime']

            # ATR-based stop
            atr_col = 'atr' if 'atr' in df.columns else 'ATR_14'
            atr = df[atr_col].iloc[entry_bar]
            stop_distance = atr * self.atr_multipliers.get(regime, 2.0)
            current_stop = entry_price - stop_distance

            # Track position
            peak_price = entry_price
            remaining_size = 1.0
            total_pnl = 0.0

            # Search for exit
            for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                close = df['close'].iloc[i]

                # Update peak
                if close > peak_price:
                    peak_price = close

                # Trailing stop
                profit_pct = (peak_price - entry_price) / entry_price
                if profit_pct > self.trailing_activation:
                    trail_distance = stop_distance * (1 - profit_pct * 0.5)
                    new_stop = peak_price - trail_distance
                    current_stop = max(current_stop, new_stop)

                # Check stop hit
                if low <= current_stop:
                    exit_price = current_stop
                    pnl = (exit_price - entry_price) / entry_price * remaining_size
                    total_pnl += pnl
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': total_pnl,
                        'bars_held': i - entry_bar,
                        'exit_reason': 'STOP',
                        'regime': regime,
                        'entry_reason': sig.get('reason', 'Unknown')
                    })
                    break

                # Partial exits at TP levels
                for idx, (tp_pct, exit_pct) in enumerate(zip(self.tp_levels, self.partial_exits)):
                    if high >= entry_price * tp_pct and remaining_size > 0:
                        exit_size = exit_pct
                        exit_price = entry_price * tp_pct
                        pnl = (exit_price - entry_price) / entry_price * exit_size
                        total_pnl += pnl
                        remaining_size -= exit_size

                        if remaining_size <= 0.01:
                            trades.append({
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl_pct': total_pnl,
                                'bars_held': i - entry_bar,
                                'exit_reason': 'TP',
                                'regime': regime,
                                'entry_reason': sig.get('reason', 'Unknown')
                            })
                            break
            else:
                # Max holding period reached
                if remaining_size > 0:
                    exit_price = df['close'].iloc[min(entry_bar + 500, len(df) - 1)]
                    pnl = (exit_price - entry_price) / entry_price * remaining_size
                    total_pnl += pnl
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': total_pnl,
                        'bars_held': min(500, len(df) - entry_bar - 1),
                        'exit_reason': 'MAX_HOLD',
                        'regime': regime,
                        'entry_reason': sig.get('reason', 'Unknown')
                    })

        return trades

    def calculate_metrics(self, trades):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_return': 0.0,
                'avg_bars_held': 0.0
            }

        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': np.mean(wins) if wins else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'total_return': sum(pnls),
            'avg_bars_held': np.mean([t['bars_held'] for t in trades])
        }

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(asset, timeframe):
    """
    Load OHLCV data and add indicators

    Uses ONLY raw data (data/raw/*.parquet)
    Features files are not compatible (no 'rsi' column)
    """
    # Try raw data with underscore (BTC_USDT_15m.parquet format)
    raw_path = DATA_DIR / f"{asset}_USDT_{timeframe}.parquet"
    if raw_path.exists():
        try:
            df = pd.read_parquet(raw_path)
            print(f"   ✅ Loaded raw data: {len(df)} bars")

            # Add indicators
            df = add_indicators(df)
            return df
        except Exception as e:
            print(f"   ❌ Failed to load {raw_path}: {e}")
            return None

    # Try without underscore (BTCUSDT_15m.parquet format)
    raw_path2 = DATA_DIR / f"{asset}USDT_{timeframe}.parquet"
    if raw_path2.exists():
        try:
            df = pd.read_parquet(raw_path2)
            print(f"   ✅ Loaded raw data: {len(df)} bars")

            # Add indicators
            df = add_indicators(df)
            return df
        except Exception as e:
            print(f"   ❌ Failed to load {raw_path2}: {e}")
            return None

    print(f"   ❌ No data found for {asset} {timeframe}")
    print(f"      Tried: {raw_path}, {raw_path2}")
    return None

def add_indicators(df):
    """Add RSI and ATR indicators"""
    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"   ❌ Missing OHLCV columns: {missing}")
        return df

    # Add RSI if missing
    if 'rsi' not in df.columns and 'RSI_14' not in df.columns:
        try:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))

            rsi_valid = df['rsi'].notna().sum()
            print(f"   ✅ Added RSI ({rsi_valid} valid values)")
        except Exception as e:
            print(f"   ❌ Failed to add RSI: {e}")
            return df
    else:
        print(f"   ℹ️  RSI already exists")

    # Add ATR if missing
    if 'atr' not in df.columns and 'ATR_14' not in df.columns:
        try:
            # Simple ATR calculation
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()

            atr_valid = df['atr'].notna().sum()
            print(f"   ✅ Added ATR ({atr_valid} valid values)")
        except Exception as e:
            print(f"   ❌ Failed to add ATR: {e}")
    else:
        print(f"   ℹ️  ATR already exists")

    return df

# ============================================================================
# MAIN BACKTEST LOOP
# ============================================================================

def run_backtest():
    """Run backtest on all 56 combinations"""

    print("\n" + "="*100)
    print("STARTING BACKTEST")
    print("="*100)

    # Initialize engine
    engine = HybridBacktestEngine(
        ml_threshold=ML_THRESHOLD,
        enable_ml_filter=ENABLE_ML_FILTER,
        enable_crisis_gate=ENABLE_CRISIS_GATE
    )

    results = []
    success_count = 0
    fail_count = 0

    # Test each combination
    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            combo = f"{asset}_{timeframe}"
            print(f"\n{'='*100}")
            print(f"Testing: {combo}")
            print(f"{'='*100}")

            # Load data
            df = load_data(asset, timeframe)
            if df is None:
                fail_count += 1
                continue

            # Generate signals
            print(f"   Generating signals...")
            signals = engine.generate_signals(df)
            print(f"   ✅ Generated {len(signals)} signals")

            if len(signals) == 0:
                print(f"   ⚠️  No signals generated - skipping")
                fail_count += 1
                continue

            # Backtest
            print(f"   Backtesting...")
            trades = engine.backtest(df, signals)
            print(f"   ✅ Executed {len(trades)} trades")

            # Calculate metrics
            metrics = engine.calculate_metrics(trades)

            # Store results
            result = {
                'asset': asset,
                'timeframe': timeframe,
                'combination': combo,
                'data_bars': len(df),
                'signals_generated': len(signals),
                **metrics
            }
            results.append(result)
            success_count += 1

            # Print summary
            print(f"\n   RESULTS:")
            print(f"   Trades:        {metrics['total_trades']}")
            print(f"   Win Rate:      {metrics['win_rate']:.1%}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Total Return:  {metrics['total_return']:.1%}")

    # ========================================
    # SUMMARY STATISTICS
    # ========================================

    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"Tested: {success_count} / {len(ASSETS) * len(TIMEFRAMES)} combinations")
    print(f"Failed: {fail_count}")

    if results:
        df_results = pd.DataFrame(results)

        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")
        print(f"Total trades:        {df_results['total_trades'].sum():,}")
        print(f"Avg win rate:        {df_results['win_rate'].mean():.1%}")
        print(f"Avg profit factor:   {df_results['profit_factor'].mean():.2f}")
        print(f"Avg return:          {df_results['total_return'].mean():.1%}")

        print(f"\n{'='*100}")
        print("TOP 10 PERFORMERS")
        print(f"{'='*100}")
        top10 = df_results.nlargest(10, 'total_return')
        for idx, row in top10.iterrows():
            print(f"{row['combination']:20} | Trades: {row['total_trades']:4} | WR: {row['win_rate']:5.1%} | PF: {row['profit_factor']:5.2f} | Return: {row['total_return']:7.1%}")

        # Save results
        output_file = OUTPUT_DIR / 'hybrid_backtest_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {output_file}")

        # Save CSV
        csv_file = OUTPUT_DIR / 'hybrid_backtest_results.csv'
        df_results.to_csv(csv_file, index=False)
        print(f"✅ CSV saved to: {csv_file}")

        # Print filtering statistics
        print(f"\n{'='*100}")
        print("HYBRID SYSTEM FILTERING STATISTICS")
        print(f"{'='*100}")
        engine.entry_system.print_statistics()

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    run_backtest()
