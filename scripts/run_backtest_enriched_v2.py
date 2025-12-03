"""
Backtest the Logistic Regression model trained on enriched features (54).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.honest_backtest_v2 import HonestBacktestV2


def main():
    print("=" * 80)
    print("BACKTEST: LOGISTIC REGRESSION ON ENRICHED FEATURES (54)")
    print("=" * 80)

    project_root = os.path.dirname(os.path.dirname(__file__))

    # --- PATHS ---
    model_path = os.path.join(project_root, "models", "logistic_enriched_v2.pth")
    scaler_path = os.path.join(project_root, "models", "scaler_enriched_v2.pkl")
    metadata_path = os.path.join(project_root, "models", "logistic_enriched_v2_metadata.json")
    X_test_path = os.path.join(project_root, "models", "X_test_enriched_v2.pt")
    y_test_path = os.path.join(project_root, "models", "y_test_enriched_v2.pt")
    ohlcv_path = os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet")
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # --- [1/6] LOAD MODEL ---
    print("\n[1/6] Loading trained model...")
    # Import the model class
    from models.logistic_baseline import LogisticBaseline  # Assuming it's the same class

    # Determine input dimension from the test data
    # We'll load the test data first to get the shape
    # X_test_for_shape = torch.load(X_test_path, weights_only=False)
    # input_dim = X_test_for_shape.shape[1]
    # del X_test_for_shape # Free memory
    # Hardcode based on our knowledge
    input_dim = 54

    model = LogisticBaseline(input_dim=input_dim)
    try:
        # Use weights_only=False for now, but acknowledge the warning
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        model.eval()
        print(f"✅ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # --- [2/6] LOAD METADATA (for threshold) ---
    print("\n[2/6] Loading model metadata...")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        THRESHOLD = metadata.get("optimal_threshold", 0.5)  # Default to 0.5 if not found
        print(f"✅ Optimal threshold loaded: {THRESHOLD}")
    except FileNotFoundError:
        print(
            f"⚠️  Warning: Metadata file not found at {metadata_path}. Using default threshold 0.5."
        )
        THRESHOLD = 0.5
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding metadata JSON: {e}. Using default threshold 0.5.")
        THRESHOLD = 0.5

    # --- [3/6] LOAD TEST DATA & SCALER ---
    print("\n[3/6] Loading test data and scaler...")
    try:
        # Load test features (2D)
        X_test = torch.load(X_test_path, weights_only=False)
        y_test = torch.load(y_test_path, weights_only=False)
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")

        # Load scaler
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")

        # Scale test data
        X_test_np = X_test.numpy()
        X_test_scaled = scaler.transform(X_test_np)
        print("✅ Test data scaled")

    except FileNotFoundError as e:
        print(f"❌ Error loading data/scaler: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading data/scaler: {e}")
        sys.exit(1)

    # --- [4/6] GENERATE PREDICTIONS ---
    print("\n[4/6] Generating predictions...")
    try:
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_pred_proba = model.predict_proba(X_test_tensor)

        # Apply threshold
        y_pred_signals = (y_pred_proba[:, 1] >= THRESHOLD).astype(int)
        n_signals = np.sum(y_pred_signals)
        print(f"Threshold: {THRESHOLD}")
        print(f"BUY signals generated: {n_signals} ({n_signals / len(y_pred_signals) * 100:.2f}%)")

        if n_signals == 0:
            print("❌ No BUY signals generated. Exiting.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)

    # --- [5/6] LOAD OHLCV DATA ---
    print("\n[5/6] Loading OHLCV data for backtest...")
    try:
        ohlcv_full = pd.read_parquet(ohlcv_path)
        print(f"  Full OHLCV shape: {ohlcv_full.shape}")

        # Align OHLCV data with test predictions
        # The test set corresponds to the last len(X_test) bars of the processed data.
        # The original split was based on chronological order.
        # We need to ensure the OHLCV slice matches the test period used for X_test generation.
        # Based on previous scripts (e.g., prepare_data_enriched_fixed.py), the test set
        # is derived from the end of the full dataset.
        # Let's assume the test set indices in the full OHLCV are from
        # len(ohlcv_full) - len(X_test) onwards.
        test_start_idx = len(ohlcv_full) - len(X_test)
        ohlcv_test_subset = ohlcv_full.iloc[test_start_idx:].reset_index(drop=True)

        if len(ohlcv_test_subset) != len(X_test):
            print(
                f"⚠️  OHLCV test length ({len(ohlcv_test_subset)}) != X_test length ({len(X_test)}). Aligning..."
            )
            min_len = min(len(ohlcv_test_subset), len(X_test))
            ohlcv_test_subset = ohlcv_test_subset.iloc[:min_len].reset_index(drop=True)
            # Also truncate X_test_scaled and y_pred_signals to match
            X_test_scaled = X_test_scaled[:min_len]
            y_pred_signals = y_pred_signals[:min_len]
            y_test = y_test[:min_len]  # If needed for comparison
            print(f"   Aligned to {min_len} rows")

        print(f"✅ OHLCV test data aligned: {ohlcv_test_subset.shape}")

    except FileNotFoundError:
        print(f"❌ Error: OHLCV file not found at {ohlcv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading OHLCV data: {e}")
        sys.exit(1)

    # --- [6/6] RUN BACKTEST ---
    print("\n[6/6] Initializing and running Honest Backtest V2...")

    # Use the correct parameters for HonestBacktestV2
    # These should ideally come from a config or be consistent with previous optimizations
    # For now, we use standard/good defaults. If you have optimized TP/SL for this specific model,
    # load them from a config file.
    backtest = HonestBacktestV2(
        initial_capital=100000.0,
        commission=0.001,  # 0.1% per trade (maker/taker combined estimate)
        slippage=0.0005,  # 0.05% per trade
        position_size_pct=0.95,  # Use 95% of capital per trade
        take_profit=0.03,  # 3% take profit
        stop_loss=0.012,  # 1.2% stop loss (from earlier optimization for a similar model)
        max_hold_bars=288,  # 3 days for 15m bars
        cooldown_bars=10,  # 10 bars cooldown
    )

    print(f"Backtest config: TP={backtest.take_profit*100:.1f}%, SL={backtest.stop_loss*100:.1f}%")

    try:
        # Run the backtest
        metrics = backtest.run(ohlcv_test_subset, y_pred_signals)
        print("\n✅ Backtest completed successfully!")

        # Print the detailed report
        backtest.print_report()

        # Save trades
        if hasattr(backtest, "trades") and backtest.trades:
            trades_df = pd.DataFrame(backtest.trades)
            trades_path = os.path.join(reports_dir, "logistic_enriched_v2_trades.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"\n💾 Trades saved to {trades_path}")

        # Save plot
        plot_path = os.path.join(reports_dir, "logistic_enriched_v2_backtest_equity.png")
        backtest.plot_results(save_path=plot_path)
        print(f"📊 Equity curve plot saved to {plot_path}")

    except Exception as e:
        print(f"❌ An error occurred during backtest execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ BACKTEST PROCESS FINISHED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
