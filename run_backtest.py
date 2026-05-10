#!/usr/bin/env python3
"""
Scarlet Sails Backtesting Framework - Entry Point

Usage:
------
# Single asset backtest
python run_backtest.py --strategy rsi --coin BTC --timeframe 15m

# With date range
python run_backtest.py --strategy combined --coin ENA --timeframe 15m --start 2024-01-01

# Multiple coins
python run_backtest.py --strategy rsi --coins BTC ETH SOL ENA --timeframe 15m

# List available options
python run_backtest.py --help
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from core.backtest_engine import BacktestEngine, BacktestConfig
from core.data_loader import AVAILABLE_COINS, AVAILABLE_TIMEFRAMES
from core.metrics_calculator import MetricsCalculator
from visualization.plotter import BacktestPlotter

from strategies import (
    SimpleRSIStrategy,
    SimpleMAStrategy,
    SimpleBollingerStrategy,
    CombinedStrategy,
    RuleBasedStrategy,
    HybridStrategy,
)
from core.ood_detector import OODDetector


# Strategy mapping
STRATEGIES = {
    'rsi': SimpleRSIStrategy,
    'ma': SimpleMAStrategy,
    'bollinger': SimpleBollingerStrategy,
    'combined': CombinedStrategy,
    'rule_based': RuleBasedStrategy,
    'hybrid': HybridStrategy,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scarlet Sails Backtesting Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --strategy rsi --coin BTC --timeframe 15m
  python run_backtest.py --strategy combined --coin ENA --timeframe 15m --start 2024-01-01
  python run_backtest.py --strategy hybrid --coins BTC ETH SOL --timeframe 15m
        """
    )
    
    # Strategy
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=list(STRATEGIES.keys()),
        default='rsi',
        help='Trading strategy to use'
    )
    
    # Asset selection
    parser.add_argument(
        '--coin', '-c',
        type=str,
        default='BTC',
        help='Single coin to backtest (e.g., BTC, ETH, ENA)'
    )
    
    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        help='Multiple coins for comparison (e.g., BTC ETH SOL)'
    )
    
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        choices=AVAILABLE_TIMEFRAMES,
        default='15m',
        help='Timeframe (15m, 1h, 4h, 1d)'
    )
    
    # Date range
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    # Capital
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital (default: 10000)'
    )
    
    # Risk settings
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.02,
        help='Stop loss percentage (default: 0.02 = 2%%)'
    )
    
    parser.add_argument(
        '--no-stop-loss',
        action='store_true',
        help='Disable stop loss'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='backtest_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    # Data directory
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Path to data directory'
    )
    
    return parser.parse_args()


def run_single_backtest(
    strategy_name: str,
    coin: str,
    timeframe: str,
    start_date: str = None,
    end_date: str = None,
    capital: float = 10000,
    stop_loss: float = 0.02,
    use_stop_loss: bool = True,
    output_dir: str = 'backtest_results',
    verbose: bool = True,
    generate_plots: bool = True,
    data_dir: str = 'data/raw',
):
    """Run backtest on single coin."""
    
    print(f"\n{'='*60}")
    print(f"SCARLET SAILS BACKTESTING")
    print(f"{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Coin: {coin}")
    print(f"Timeframe: {timeframe}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Stop Loss: {stop_loss*100:.1f}%" if use_stop_loss else "Stop Loss: Disabled")
    print(f"{'='*60}\n")
    
    # Create strategy
    strategy_class = STRATEGIES.get(strategy_name)
    if strategy_class is None:
        print(f"Error: Unknown strategy '{strategy_name}'")
        print(f"Available: {list(STRATEGIES.keys())}")
        return None

    # Initialize OOD detector for strategies that need it
    ood_detector = None
    if strategy_name == 'combined':
        ood_detector = OODDetector()

    # Create strategy
    if strategy_name == 'combined':
        strategy = strategy_class(ood_detector=ood_detector)
    else:
        strategy = strategy_class()
    
    # Create config
    config = BacktestConfig(
        initial_capital=capital,
        commission=0.001,  # 0.1%
        slippage=0.0005,   # 0.05%
        use_stop_loss=use_stop_loss,
        stop_loss_pct=stop_loss,
        verbose=verbose,
        save_results=True,
        output_dir=output_dir,
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Run backtest
    try:
        result = engine.run(
            strategy=strategy,
            coin=coin,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nMake sure data file exists at: {data_dir}/{coin}_USDT_{timeframe}.parquet")
        return None
    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Generate plots
    if generate_plots and result:
        print("\nGenerating plots...")
        plotter = BacktestPlotter(output_dir)
        
        # Generate full report
        trades_df = result.equity_curve.copy()
        if result.trades:
            from core.trade_logger import TradeLogger
            logger = TradeLogger()
            for t in result.trades:
                logger.add_trade(t)
            trades_df = logger.to_dataframe()
        
        plotter.generate_full_report(
            equity_curve=result.equity_curve,
            trades_df=trades_df,
            strategy_name=strategy_name,
            coin=coin,
            timeframe=timeframe,
        )
    
    return result


def run_multi_asset_backtest(
    strategy_name: str,
    coins: list,
    timeframe: str,
    **kwargs
):
    """Run backtest on multiple coins and compare."""
    
    results = {}
    
    for coin in coins:
        print(f"\n{'─'*40}")
        print(f"Backtesting {coin}...")
        print(f"{'─'*40}")
        
        result = run_single_backtest(
            strategy_name=strategy_name,
            coin=coin,
            timeframe=timeframe,
            generate_plots=False,  # Generate comparison plot instead
            **kwargs
        )
        
        if result:
            results[coin] = result
    
    # Generate comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("MULTI-ASSET COMPARISON")
        print(f"{'='*60}")
        
        # Comparison table
        metrics_list = [r.metrics for r in results.values() if r.metrics]
        if metrics_list:
            comparison = MetricsCalculator.compare_strategies(metrics_list)
            print("\n" + comparison.to_string(index=False))
        
        # Comparison plot
        output_dir = kwargs.get('output_dir', 'backtest_results')
        plotter = BacktestPlotter(output_dir)
        
        equity_curves = {
            f"{coin}_{timeframe}": r.equity_curve 
            for coin, r in results.items()
        }
        
        plotter.plot_strategy_comparison(
            equity_curves,
            title=f'{strategy_name} Strategy - Multi-Asset Comparison',
            filename=f'{strategy_name}_comparison_{timeframe}.png'
        )
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check for multi-asset mode
    if args.coins:
        # Validate coins
        invalid = [c for c in args.coins if c not in AVAILABLE_COINS]
        if invalid:
            print(f"Error: Invalid coins: {invalid}")
            print(f"Available: {AVAILABLE_COINS}")
            sys.exit(1)
        
        results = run_multi_asset_backtest(
            strategy_name=args.strategy,
            coins=args.coins,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            capital=args.capital,
            stop_loss=args.stop_loss,
            use_stop_loss=not args.no_stop_loss,
            output_dir=args.output,
            verbose=not args.quiet,
            data_dir=args.data_dir,
        )
    else:
        # Single asset mode
        if args.coin not in AVAILABLE_COINS:
            print(f"Error: Invalid coin: {args.coin}")
            print(f"Available: {AVAILABLE_COINS}")
            sys.exit(1)
        
        result = run_single_backtest(
            strategy_name=args.strategy,
            coin=args.coin,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            capital=args.capital,
            stop_loss=args.stop_loss,
            use_stop_loss=not args.no_stop_loss,
            output_dir=args.output,
            verbose=not args.quiet,
            generate_plots=not args.no_plots,
            data_dir=args.data_dir,
        )
        
        if result is None:
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()