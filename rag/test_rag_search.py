#!/usr/bin/env python3
"""
Test RAG Search

Test the retrieval system with sample queries.

Usage:
    python scripts/test_rag_search.py
    python scripts/test_rag_search.py --coin BTC --tf 1h
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import RAGRetriever


def main():
    parser = argparse.ArgumentParser(description="Test RAG Search")
    parser.add_argument('--coin', default='BTC', help='Coin to search')
    parser.add_argument('--tf', default='1h', help='Timeframe')
    parser.add_argument('--direction', default='long', choices=['long', 'short'])
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  SCARLET SAILS — RAG SEARCH TEST")
    print("="*60)
    
    # Initialize retriever
    print("\n📥 Initializing retriever...")
    retriever = RAGRetriever(use_multi_hyde=True)
    
    # Create test state
    test_state = {
        'symbol': args.coin,
        'timeframe': args.tf,
        'direction': args.direction,
        'indicators': {
            'rsi_zscore': -1.2,
            'rsi_low': True,
            'trend_up': True,
            'vol_low': True,
            'volume_zscore': 1.5,
            'div_rsi_bullish': True,
        },
        'box': {
            'touches_support': 3,
            'touches_resistance': 2,
            'box_range_pct': 3.5,
        },
        'w_box': {
            'W_box': 0.65,
        }
    }
    
    print(f"\n🔍 TEST QUERY:")
    print(f"   Symbol: {test_state['symbol']}")
    print(f"   Timeframe: {test_state['timeframe']}")
    print(f"   Direction: {test_state['direction']}")
    print(f"   RSI: oversold (z=-1.2)")
    print(f"   Trend: up")
    print(f"   Volume: high (z=1.5)")
    print(f"   Divergence: bullish")
    print(f"   W_box: 0.65")
    
    # Search
    print(f"\n🔎 Searching (Multi-HyDE)...")
    context = retriever.build_council_context(test_state, top_k=args.top_k)
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Similar patterns: {len(context['similar_patterns'])}")
    print(f"   Sample size: {context['sample_size']}")
    print(f"   Historical win rate: {context['historical_win_rate']:.1%}")
    print(f"   Historical avg PnL: {context['historical_avg_pnl']:.2f}%")
    print(f"   Recommendation: {context['recommendation']}")
    print(f"   Confidence: {context['confidence']:.2f}")
    
    if context['similar_patterns']:
        print(f"\n🔗 SIMILAR PATTERNS:")
        for i, p in enumerate(context['similar_patterns'][:5], 1):
            print(f"\n   {i}. {p['pattern_id']}")
            print(f"      Similarity: {p['similarity']:.3f}")
            print(f"      W_box: {p.get('w_box', 'N/A')}")
            
            if p.get('historical_performance'):
                perf = p['historical_performance']
                print(f"      Win rate: {perf['win_rate']:.1%} ({perf['wins']}/{perf['total_trades']})")
                print(f"      Avg PnL: {perf['avg_pnl']:.2f}%")
            
            if p.get('hypothesis_matches', 1) > 1:
                print(f"      ⭐ Multi-hypothesis match ({p['hypothesis_matches']})")
    else:
        print("\n   ⚠️ No patterns found. Run: python scripts/build_rag_index.py")


if __name__ == "__main__":
    main()
