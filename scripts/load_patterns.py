#!/usr/bin/env python3
"""
ScArlet-Sails Pattern Loader CLI

Загрузка паттернов в RAG систему через командную строку.

Usage:
    # Из файла
    python scripts/load_patterns.py --file patterns.json
    
    # Из директории
    python scripts/load_patterns.py --dir patterns/
    
    # С валидацией
    python scripts/load_patterns.py --file patterns.json --validate
    
    # Только проверка (без загрузки)
    python scripts/load_patterns.py --file patterns.json --dry-run
    
    # Статистика
    python scripts/load_patterns.py --stats
    
    # Перестроить индекс
    python scripts/load_patterns.py --rebuild-index
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.pattern_loader import PatternLoader, LoadResult
from rag.pattern_validator import validate_pattern_file, BatchValidationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Печать заголовка."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_load_result(result: LoadResult) -> None:
    """Печать результата загрузки."""
    print()
    print(f"  Total patterns:  {result.total_patterns}")
    print(f"  Loaded:          {result.loaded_count} ✓")
    print(f"  Skipped:         {result.skipped_count}")
    print(f"  Errors:          {result.error_count}")
    print(f"  Success rate:    {result.success_rate:.0%}")
    
    if result.warnings:
        print()
        print("  Warnings:")
        for w in result.warnings[:5]:
            print(f"    ⚠ {w}")
        if len(result.warnings) > 5:
            print(f"    ... and {len(result.warnings) - 5} more")
    
    if result.errors:
        print()
        print("  Errors:")
        for e in result.errors[:5]:
            print(f"    ✗ {e}")
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more")
    
    print()


def print_validation_result(result: BatchValidationResult) -> None:
    """Печать результата валидации."""
    print()
    print(f"  Total:    {result.total}")
    print(f"  Valid:    {result.valid_count} ✓")
    print(f"  Invalid:  {result.invalid_count} ✗")
    print(f"  Rate:     {result.success_rate:.0%}")
    
    if result.warnings:
        print()
        print("  Warnings:")
        for w in result.warnings[:10]:
            print(f"    ⚠ {w}")
    
    if result.errors:
        print()
        print("  Errors:")
        for e in result.errors[:10]:
            print(f"    ✗ [{e.pattern_id}] {e.field}: {e.message}")
    
    print()


def print_stats(stats: dict) -> None:
    """Печать статистики библиотеки."""
    print()
    print(f"  Total patterns: {stats.get('total_patterns', 0)}")
    
    if stats.get('outcomes'):
        print()
        print("  Outcomes:")
        for outcome, count in stats['outcomes'].items():
            print(f"    {outcome:<12} {count}")
    
    if stats.get('directions'):
        print()
        print("  Directions:")
        for direction, count in stats['directions'].items():
            print(f"    {direction:<12} {count}")
    
    if stats.get('categories'):
        print()
        print("  Categories:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            print(f"    {cat:<25} {count}")
    
    if stats.get('metadata'):
        meta = stats['metadata']
        print()
        print("  Metadata:")
        print(f"    Last updated: {meta.get('last_updated', 'N/A')}")
        print(f"    Version:      {meta.get('version', 'N/A')}")
    
    print()


def load_from_file(args) -> int:
    """Загрузить из файла."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1
    
    print_header(f"LOADING: {file_path.name}")
    
    # Только валидация?
    if args.validate_only:
        result = validate_pattern_file(file_path, strict=args.strict)
        print_validation_result(result)
        return 0 if result.valid_count == result.total else 1
    
    # Загрузка
    loader = PatternLoader(
        patterns_dir=args.patterns_dir,
        validate=args.validate,
        strict=args.strict,
        skip_duplicates=not args.allow_duplicates,
    )
    
    result = loader.load_file(file_path, save=not args.dry_run)
    
    print_load_result(result)
    
    if args.dry_run:
        print("  (Dry run - nothing saved)")
    
    return 0 if result.error_count == 0 else 1


def load_from_directory(args) -> int:
    """Загрузить из директории."""
    dir_path = Path(args.dir)
    
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return 1
    
    print_header(f"LOADING FROM: {dir_path}")
    
    loader = PatternLoader(
        patterns_dir=args.patterns_dir,
        validate=args.validate,
        strict=args.strict,
        skip_duplicates=not args.allow_duplicates,
    )
    
    result = loader.load_directory(
        dir_path,
        recursive=args.recursive,
        save=not args.dry_run,
    )
    
    print(f"  Files found: {result.total_files}")
    print_load_result(result)
    
    if args.dry_run:
        print("  (Dry run - nothing saved)")
    
    return 0 if result.error_count == 0 else 1


def show_stats(args) -> int:
    """Показать статистику."""
    print_header("PATTERN LIBRARY STATISTICS")
    
    loader = PatternLoader(patterns_dir=args.patterns_dir)
    stats = loader.get_stats()
    
    if not stats.get('exists'):
        print("\n  Library not found. Load some patterns first.")
        print()
        return 1
    
    print_stats(stats)
    return 0


def rebuild_index(args) -> int:
    """Перестроить FAISS индекс."""
    print_header("REBUILDING RAG INDEX")
    
    try:
        from rag.hybrid_retriever import HybridRetriever
        
        print("\n  Loading retriever...")
        retriever = HybridRetriever()
        
        print("  Rebuilding index...")
        count = retriever.rebuild_index(verbose=True)
        
        print(f"\n  ✓ Index rebuilt: {count} patterns")
        print()
        return 0
        
    except ImportError as e:
        print(f"\n  Error: Cannot import HybridRetriever: {e}")
        return 1
    except Exception as e:
        print(f"\n  Error: {e}")
        return 1


def create_example(args) -> int:
    """Создать пример файла паттернов."""
    print_header("CREATING EXAMPLE FILE")
    
    example_patterns = [
        {
            "id": "ma50_bounce_001",
            "name": "MA50 Bounce with RSI Confirmation",
            "direction": "long",
            "outcome": "win",
            "pnl_pct": 3.2,
            "category": "trend_continuation",
            "description": "Price touched MA50 from above after 3 red candles. RSI at 32 showed oversold. Volume 1.2x average confirmed buyer interest. Entered at bounce candle close.",
            "entry": {
                "indicators": {
                    "rsi": 32,
                    "price_to_ma50_pct": -0.5,
                    "volume_ratio": 1.2
                }
            },
            "context": {
                "symbol": "BTC_USDT",
                "timeframe": "4h",
                "regime": "normal",
                "trend": "bullish"
            },
            "lessons": [
                "MA50 bounce works well in bullish trend",
                "RSI confirmation improves win rate",
                "Wait for bounce candle close before entry"
            ],
            "tags": ["ma50", "bounce", "oversold", "trend_continuation"]
        },
        {
            "id": "false_breakout_001",
            "name": "Failed Resistance Breakout",
            "direction": "short",
            "outcome": "win",
            "pnl_pct": 2.8,
            "category": "false_breakout",
            "description": "Price broke above resistance on low volume, immediately rejected. RSI divergence warned of weakness. Shorted the rejection candle.",
            "entry": {
                "indicators": {
                    "rsi": 72,
                    "volume_ratio": 0.7
                }
            },
            "context": {
                "symbol": "BTC_USDT",
                "timeframe": "4h",
                "regime": "normal"
            },
            "lessons": [
                "Low volume breakouts often fail",
                "RSI divergence = warning sign",
                "Wait for rejection confirmation"
            ],
            "tags": ["false_breakout", "resistance", "divergence", "short"]
        },
        {
            "id": "momentum_loss_001",
            "name": "Momentum Exhaustion Long",
            "direction": "long",
            "outcome": "loss",
            "pnl_pct": -2.1,
            "category": "trend_reversal",
            "description": "Entered long on apparent support, but momentum was exhausted. Price continued down through support.",
            "context": {
                "symbol": "ETH_USDT",
                "timeframe": "4h",
                "regime": "high_vol"
            },
            "lessons": [
                "Don't catch falling knives in high vol",
                "Wait for momentum to confirm reversal",
                "Support levels fail in strong downtrends"
            ],
            "tags": ["support", "failed", "high_vol", "lesson"]
        }
    ]
    
    output_file = Path(args.output or "example_patterns.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(example_patterns, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Created: {output_file}")
    print(f"  Patterns: {len(example_patterns)}")
    print()
    print("  Load with:")
    print(f"    python scripts/load_patterns.py --file {output_file}")
    print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ScArlet-Sails Pattern Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load from file
    python scripts/load_patterns.py --file my_patterns.json
    
    # Load from directory
    python scripts/load_patterns.py --dir patterns/
    
    # Validate only (no loading)
    python scripts/load_patterns.py --file patterns.json --validate-only
    
    # Dry run (validate + show what would be loaded)
    python scripts/load_patterns.py --file patterns.json --dry-run
    
    # Show library statistics
    python scripts/load_patterns.py --stats
    
    # Rebuild FAISS index
    python scripts/load_patterns.py --rebuild-index
    
    # Create example file
    python scripts/load_patterns.py --example
        """
    )
    
    # Input source
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--file', type=str, help='Load from JSON/CSV file')
    source.add_argument('--dir', type=str, help='Load from directory')
    source.add_argument('--stats', action='store_true', help='Show library statistics')
    source.add_argument('--rebuild-index', action='store_true', help='Rebuild FAISS index')
    source.add_argument('--example', action='store_true', help='Create example patterns file')
    
    # Options
    parser.add_argument('--patterns-dir', type=str, default='rag/patterns',
                        help='Directory for pattern library (default: rag/patterns)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate patterns before loading (default: True)')
    parser.add_argument('--no-validate', action='store_false', dest='validate',
                        help='Skip validation')
    parser.add_argument('--strict', action='store_true',
                        help='Strict validation (warnings become errors)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate, do not load')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and process, but do not save')
    parser.add_argument('--allow-duplicates', action='store_true',
                        help='Allow duplicate pattern IDs')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Recursively scan directories (default: True)')
    parser.add_argument('--output', type=str, help='Output file for --example')
    
    args = parser.parse_args()
    
    # Dispatch
    if args.file:
        return load_from_file(args)
    elif args.dir:
        return load_from_directory(args)
    elif args.stats:
        return show_stats(args)
    elif args.rebuild_index:
        return rebuild_index(args)
    elif args.example:
        return create_example(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())