"""Command Line Interface for RAG pattern extraction."""

import argparse
import sys
from pathlib import Path

from rag.extractor import PatternExtractor
from rag.config import RAGConfig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract trading patterns from cryptocurrency data'
    )

    parser.add_argument(
        '--coin',
        type=str,
        required=True,
        help=f'Coin symbol (supported: {", ".join(RAGConfig.COINS)})'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        required=True,
        help=f'Timeframe (supported: {", ".join(RAGConfig.TIMEFRAMES)})'
    )

    parser.add_argument(
        '--pattern-type',
        type=str,
        default='bullish',
        choices=['bullish', 'bearish', 'consolidation'],
        help='Type of pattern to extract'
    )

    parser.add_argument(
        '--data-path',
        type=Path,
        default=None,
        help='Custom path to data file (optional)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename for patterns (optional)'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show pattern statistics'
    )

    args = parser.parse_args()

    try:
        # Initialize extractor
        print(f"Initializing pattern extractor for {args.coin} {args.timeframe}...")
        extractor = PatternExtractor(args.coin, args.timeframe)

        # Load data
        print("Loading data...")
        data = extractor.load_data(args.data_path)
        print(f"Loaded {len(data)} candles")

        # Extract patterns
        print(f"Extracting {args.pattern_type} patterns...")
        patterns = extractor.extract_patterns(args.pattern_type)
        print(f"Found {len(patterns)} patterns")

        # Save patterns
        if patterns:
            output_path = extractor.save_patterns(args.output)
            print(f"Patterns saved to: {output_path}")

            # Show statistics
            if args.stats:
                stats = extractor.get_statistics()
                print("\nPattern Statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        else:
            print("No patterns found")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
