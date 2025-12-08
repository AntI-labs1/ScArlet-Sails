#!/usr/bin/env python3
"""
Build RAG Vector Index

Run this after adding new patterns to create/update the FAISS index.

Usage:
    python scripts/build_rag_index.py           # Full rebuild
    python scripts/build_rag_index.py --update  # Incremental update
    python scripts/build_rag_index.py --stats   # Show statistics
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import PatternVectorStore
from rag.retriever import RAGRetriever


def main():
    parser = argparse.ArgumentParser(description="Build RAG Vector Index")
    parser.add_argument('--update', action='store_true', help='Incremental update only')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--min-wbox', type=float, default=0.0, help='Minimum W_box threshold')
    parser.add_argument('--patterns-dir', default='rag/patterns', help='Patterns directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  SCARLET SAILS — RAG INDEX BUILDER")
    print("="*60)
    
    if args.stats:
        retriever = RAGRetriever(patterns_dir=args.patterns_dir)
        stats = retriever.get_stats()
        
        print(f"\n📊 INDEX STATISTICS:")
        print(f"   Total patterns: {stats['index']['total']}")
        print(f"   W_box avg: {stats['index']['w_box_avg']:.3f}")
        print(f"   W_box range: [{stats['index']['w_box_min']:.3f}, {stats['index']['w_box_max']:.3f}]")
        
        print(f"\n📈 BY COIN:")
        for coin, count in stats['index'].get('by_coin', {}).items():
            print(f"   {coin}: {count}")
        
        print(f"\n⏰ BY TIMEFRAME:")
        for tf, count in stats['index'].get('by_timeframe', {}).items():
            print(f"   {tf}: {count}")
        
        if stats['outcomes'].get('total_patterns', 0) > 0:
            print(f"\n🎯 OUTCOME STATISTICS:")
            print(f"   Patterns with trades: {stats['outcomes']['patterns_with_trades']}")
            print(f"   Total trades: {stats['outcomes']['total_trades']}")
            print(f"   Avg win rate: {stats['outcomes']['avg_win_rate']:.1%}")
            print(f"   Avg PnL: {stats['outcomes']['avg_pnl']:.2f}%")
        
        return
    
    store = PatternVectorStore(
        patterns_dir=args.patterns_dir,
        min_w_box=args.min_wbox,
    )
    
    if args.update:
        print(f"\n🔄 Updating index (incremental)...")
        added = store.update_index(verbose=True)
        print(f"\n✅ Added {added} new patterns")
    else:
        print(f"\n🔨 Building full index...")
        if args.min_wbox > 0:
            print(f"   (W_box threshold: {args.min_wbox})")
        count = store.build_from_directory(verbose=True)
        print(f"\n✅ Indexed {count} patterns")
    
    print(f"\n📁 Files created:")
    print(f"   {args.patterns_dir}/embeddings.faiss")
    print(f"   {args.patterns_dir}/metadata.pkl")
    print(f"   {args.patterns_dir}/index_config.json")


if __name__ == "__main__":
    main()
