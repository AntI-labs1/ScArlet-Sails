#!/usr/bin/env python3
"""
RAG SQL ROUTER - Query audit results in natural language

USAGE:
    python rag_sql_router.py "Show all BTC trades where RSI < 25"
    python rag_sql_router.py "What's the best performing asset on 15m?"
    python rag_sql_router.py "Compare Rule-Based vs ML on SOL"

VERSION: 1MVP (Simple pattern matching)
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


class SimpleRAGRouter:
    """
    Simple RAG SQL Router for analyzing audit results

    VERSION 1MVP: Pattern matching (no LLM yet)
    VERSION 2MVP: Full RAG with embeddings + LLM
    """

    def __init__(self, audit_file: Path):
        """Load audit results"""
        with open(audit_file, 'r') as f:
            self.audit = json.load(f)

        self.models = ['rule_based', 'ml', 'hybrid']

        print(f"✅ Loaded audit data: {audit_file}")
        print(f"   Models: {', '.join(self.models)}")

    def query(self, natural_query: str) -> Dict:
        """
        Convert natural language query to filters and return results

        Examples:
            "Show all BTC trades" → filter by asset=BTC
            "What's the best performing asset?" → sort by PF, show top 1
            "Compare Rule-Based vs ML" → show both models side by side
        """
        query_lower = natural_query.lower()

        # Detect intent
        intent = self._detect_intent(query_lower)

        # Extract entities
        assets = self._extract_assets(query_lower)
        timeframes = self._extract_timeframes(query_lower)
        models = self._extract_models(query_lower)
        metrics = self._extract_metrics(query_lower)

        # Execute based on intent
        if intent == 'show':
            return self._show(assets, timeframes, models, metrics)
        elif intent == 'compare':
            return self._compare(assets, timeframes, models, metrics)
        elif intent == 'best':
            return self._best(assets, timeframes, models, metrics)
        elif intent == 'worst':
            return self._worst(assets, timeframes, models, metrics)
        elif intent == 'summary':
            return self._summary(models)
        else:
            return {'error': f'Unknown intent: {intent}'}

    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        if any(word in query for word in ['show', 'display', 'list', 'get']):
            return 'show'
        elif any(word in query for word in ['compare', 'vs', 'versus', 'diff', 'difference']):
            return 'compare'
        elif any(word in query for word in ['best', 'top', 'highest', 'maximum']):
            return 'best'
        elif any(word in query for word in ['worst', 'bottom', 'lowest', 'minimum']):
            return 'worst'
        elif any(word in query for word in ['summary', 'overview', 'total']):
            return 'summary'
        else:
            return 'show'  # Default

    def _extract_assets(self, query: str) -> List[str]:
        """Extract asset names from query"""
        assets = []
        asset_names = ["BTC", "ETH", "LTC", "LINK", "ALGO", "HBAR", "SOL",
                      "DOT", "AVAX", "UNI", "ENA", "ONDO", "SUI", "LDO"]

        for asset in asset_names:
            if asset.lower() in query:
                assets.append(asset)

        return assets if assets else asset_names  # All if none specified

    def _extract_timeframes(self, query: str) -> List[str]:
        """Extract timeframes from query"""
        timeframes = []
        tf_patterns = {
            '15m': ['15m', '15 minute', 'fifteen minute'],
            '1h': ['1h', '1 hour', 'one hour', 'hourly'],
            '4h': ['4h', '4 hour', 'four hour'],
            '1d': ['1d', '1 day', 'one day', 'daily']
        }

        for tf, patterns in tf_patterns.items():
            if any(p in query for p in patterns):
                timeframes.append(tf)

        return timeframes if timeframes else ['15m', '1h', '4h', '1d']  # All if none

    def _extract_models(self, query: str) -> List[str]:
        """Extract model names from query"""
        models = []

        if any(word in query for word in ['rule', 'rsi', 'simple']):
            models.append('rule_based')
        if any(word in query for word in ['ml', 'machine learning', 'xgboost']):
            models.append('ml')
        if any(word in query for word in ['hybrid', 'combined']):
            models.append('hybrid')

        return models if models else self.models  # All if none specified

    def _extract_metrics(self, query: str) -> List[str]:
        """Extract metrics to show"""
        metrics = []

        if 'trade' in query or 'signal' in query:
            metrics.append('total_trades')
        if 'win' in query or 'wr' in query:
            metrics.append('win_rate')
        if 'profit' in query or 'pf' in query:
            metrics.append('profit_factor')
        if 'ood' in query or 'distribution' in query:
            metrics.append('ood_ratio')

        # Default: show all key metrics
        if not metrics:
            metrics = ['total_trades', 'win_rate', 'profit_factor']

        return metrics

    def _show(self, assets: List[str], timeframes: List[str],
              models: List[str], metrics: List[str]) -> Dict:
        """Show filtered results"""
        results = {}

        for model in models:
            results[model] = []

            for asset in assets:
                for tf in timeframes:
                    combo = f"{asset}_{tf}"

                    if combo in self.audit[model]:
                        data = self.audit[model][combo]

                        if 'error' not in data:
                            row = {'combo': combo}
                            for metric in metrics:
                                row[metric] = data.get(metric, 0)
                            results[model].append(row)

        return results

    def _compare(self, assets: List[str], timeframes: List[str],
                 models: List[str], metrics: List[str]) -> Dict:
        """Compare models side by side"""
        comparison = []

        for asset in assets:
            for tf in timeframes:
                combo = f"{asset}_{tf}"
                row = {'combo': combo}

                for model in models:
                    if combo in self.audit[model]:
                        data = self.audit[model][combo]

                        if 'error' not in data:
                            for metric in metrics:
                                key = f"{model}_{metric}"
                                row[key] = data.get(metric, 0)

                comparison.append(row)

        return {'comparison': comparison}

    def _best(self, assets: List[str], timeframes: List[str],
              models: List[str], metrics: List[str]) -> Dict:
        """Find best performing combinations"""
        all_results = []

        for model in models:
            for asset in assets:
                for tf in timeframes:
                    combo = f"{asset}_{tf}"

                    if combo in self.audit[model]:
                        data = self.audit[model][combo]

                        if 'error' not in data and data.get('total_trades', 0) > 0:
                            all_results.append({
                                'model': model,
                                'combo': combo,
                                'total_trades': data.get('total_trades', 0),
                                'win_rate': data.get('win_rate', 0),
                                'profit_factor': data.get('profit_factor', 0)
                            })

        # Sort by profit factor (default best metric)
        sorted_results = sorted(all_results, key=lambda x: x['profit_factor'], reverse=True)

        return {
            'best': sorted_results[:10],
            'total_analyzed': len(all_results)
        }

    def _worst(self, assets: List[str], timeframes: List[str],
               models: List[str], metrics: List[str]) -> Dict:
        """Find worst performing combinations"""
        best_result = self._best(assets, timeframes, models, metrics)

        # Reverse the sorting
        all_results = best_result['best']
        all_results.reverse()

        return {
            'worst': all_results[-10:] if len(all_results) > 10 else all_results,
            'total_analyzed': best_result['total_analyzed']
        }

    def _summary(self, models: List[str]) -> Dict:
        """Get summary statistics for models"""
        summary = {}

        for model in models:
            combos = self.audit[model]

            # Filter out errors
            valid = [c for c in combos.values() if 'error' not in c and c.get('available')]

            if valid:
                total_trades = sum(c.get('total_trades', 0) for c in valid)
                avg_wr = sum(c.get('win_rate', 0) for c in valid if c.get('total_trades', 0) > 0) / len([c for c in valid if c.get('total_trades', 0) > 0])
                avg_pf = sum(c.get('profit_factor', 0) for c in valid if c.get('total_trades', 0) > 0) / len([c for c in valid if c.get('total_trades', 0) > 0])

                summary[model] = {
                    'available_combos': len(valid),
                    'total_trades': total_trades,
                    'avg_win_rate': round(avg_wr, 2),
                    'avg_profit_factor': round(avg_pf, 2)
                }

                if model == 'ml':
                    avg_ood = sum(c.get('ood_ratio', 0) for c in valid) / len(valid)
                    summary[model]['avg_ood_ratio'] = round(avg_ood * 100, 2)

        return summary


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python rag_sql_router.py \"your query here\"")
        print("\nExamples:")
        print("  python rag_sql_router.py \"Show all BTC trades\"")
        print("  python rag_sql_router.py \"What's the best performing asset?\"")
        print("  python rag_sql_router.py \"Compare Rule-Based vs ML\"")
        print("  python rag_sql_router.py \"Show summary\"")
        return

    query = sys.argv[1]

    # Load audit
    audit_file = REPORTS_DIR / "comprehensive_model_audit.json"

    if not audit_file.exists():
        print(f"❌ Audit file not found: {audit_file}")
        print("   Run comprehensive_model_audit.py first!")
        return

    # Create router
    router = SimpleRAGRouter(audit_file)

    # Execute query
    print(f"\n{'='*100}")
    print(f"QUERY: {query}")
    print(f"{'='*100}\n")

    results = router.query(query)

    # Pretty print results
    print(json.dumps(results, indent=2))

    print(f"\n{'='*100}")
    print("QUERY COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
