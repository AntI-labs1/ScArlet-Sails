"""
Day 4: Generate sample patterns for RAG index.
Creates 10+ patterns from actual backtest data.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data():
    """Load feature data."""
    path = Path('data/features/BTC_USDT_15m_features.parquet')
    return pd.read_parquet(path)


def extract_pattern(df, idx, pattern_type, direction, outcome):
    """Extract a single pattern from data."""
    row = df.iloc[idx]
    
    # Get timestamp
    if hasattr(df.index, 'name') and df.index.name:
        timestamp = str(df.index[idx])
    else:
        timestamp = str(idx)
    
    # Extract key features
    features = {}
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close', 'volume']:
            features[col] = float(row[col])
        elif 'rsi' in col.lower() or 'atr' in col.lower() or 'bb' in col.lower():
            if pd.notna(row[col]):
                features[col] = float(row[col])
    
    # Calculate return (4 bars forward)
    if idx + 4 < len(df):
        future_return = (df.iloc[idx + 4]['close'] / row['close'] - 1) * 100
    else:
        future_return = 0
    
    pattern = {
        "id": f"pattern_{idx}_{pattern_type}",
        "timestamp": timestamp,
        "coin": "BTC",
        "timeframe": "15m",
        "pattern_type": pattern_type,
        "direction": direction,
        "entry_price": float(row['close']),
        "features": features,
        "outcome": {
            "result": outcome,
            "return_pct": float(future_return),
            "bars_held": 4,
        },
        "context": {
            "market_regime": "normal",
            "volatility": "medium",
        },
        "created_at": datetime.now().isoformat(),
    }
    
    return pattern


def find_pattern_candidates(df, predictions_path):
    """Find good pattern candidates based on model predictions."""
    # Load predictions
    pred = pd.read_parquet(predictions_path)
    
    # Align with data (predictions are from test set = last 30%)
    n_pred = len(pred)
    start_idx = len(df) - n_pred
    
    candidates = []
    
    # Find high-confidence winning trades
    high_conf_wins = pred[(pred['y_pred'] > 0.75) & (pred['returns'] > 0.002)]
    for i, row in high_conf_wins.head(5).iterrows():
        candidates.append({
            "idx": start_idx + i,
            "type": "momentum_breakout",
            "direction": "long",
            "outcome": "win",
            "confidence": row['y_pred'],
        })
    
    # Find moderate confidence wins
    mod_conf_wins = pred[(pred['y_pred'] > 0.65) & (pred['y_pred'] <= 0.75) & (pred['returns'] > 0.001)]
    for i, row in mod_conf_wins.head(3).iterrows():
        candidates.append({
            "idx": start_idx + i,
            "type": "trend_continuation",
            "direction": "long",
            "outcome": "win",
            "confidence": row['y_pred'],
        })
    
    # Find losing trades for learning
    losses = pred[(pred['y_pred'] > 0.70) & (pred['returns'] < -0.002)]
    for i, row in losses.head(3).iterrows():
        candidates.append({
            "idx": start_idx + i,
            "type": "false_breakout",
            "direction": "long",
            "outcome": "loss",
            "confidence": row['y_pred'],
        })
    
    # Find reversal patterns
    reversals = pred[(pred['y_pred'] < 0.35) & (pred['returns'] < -0.001)]
    for i, row in reversals.head(2).iterrows():
        candidates.append({
            "idx": start_idx + i,
            "type": "reversal",
            "direction": "short",
            "outcome": "win",
            "confidence": 1 - row['y_pred'],
        })
    
    return candidates


def main():
    print("=" * 60)
    print("DAY 4: RAG PATTERN GENERATOR")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")
    
    # Find candidates
    print("\nFinding pattern candidates...")
    candidates = find_pattern_candidates(df, 'models/xgboost_v3_btc_15m_predictions.parquet')
    print(f"Found {len(candidates)} candidates")
    
    # Extract patterns
    print("\nExtracting patterns...")
    patterns = []
    for c in candidates:
        try:
            pattern = extract_pattern(
                df, 
                c['idx'], 
                c['type'], 
                c['direction'], 
                c['outcome']
            )
            pattern['model_confidence'] = c['confidence']
            patterns.append(pattern)
            print(f"  ✅ {pattern['id']}: {c['type']} ({c['outcome']})")
        except Exception as e:
            print(f"  ❌ Failed at idx {c['idx']}: {e}")
    
    # Save to library.json
    output_path = Path('rag/patterns/library.json')
    library = {
        "version": "1.0",
        "patterns": patterns,
        "last_updated": datetime.now().isoformat(),
        "total_count": len(patterns),
        "stats": {
            "wins": sum(1 for p in patterns if p['outcome']['result'] == 'win'),
            "losses": sum(1 for p in patterns if p['outcome']['result'] == 'loss'),
            "pattern_types": list(set(p['pattern_type'] for p in patterns)),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(library, f, indent=2)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total patterns: {len(patterns)}")
    print(f"Wins: {library['stats']['wins']}")
    print(f"Losses: {library['stats']['losses']}")
    print(f"Pattern types: {library['stats']['pattern_types']}")
    print(f"\nSaved to: {output_path}")
    
    # Also save individual pattern files
    for p in patterns:
        pattern_file = Path(f"rag/patterns/{p['id']}.json")
        with open(pattern_file, 'w') as f:
            json.dump(p, f, indent=2)
    
    print(f"Individual patterns saved to: rag/patterns/")


if __name__ == "__main__":
    main()