"""
RAG Health Check & Monitoring
=============================

Automated health checks for RAG system:
- Index integrity
- Latency benchmarks
- Data quality
- Alerts on degradation

Usage:
    from rag.health_check import RAGHealthCheck
    
    checker = RAGHealthCheck()
    report = checker.full_check()
    
    if not report['healthy']:
        print("ALERT:", report['issues'])
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import json
import statistics


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'value': self.value,
            'threshold': self.threshold,
            'message': self.message,
        }


@dataclass
class HealthReport:
    """Full health report."""
    healthy: bool
    checks: List[HealthCheckResult]
    issues: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'healthy': self.healthy,
            'checks': [c.to_dict() for c in self.checks],
            'issues': self.issues,
            'timestamp': self.timestamp,
        }


class RAGHealthCheck:
    """
    Health monitoring for RAG system.
    """
    
    # Thresholds
    MAX_LATENCY_MS = 200  # Maximum acceptable retrieval latency
    MIN_PATTERNS = 10     # Minimum patterns for useful RAG
    MAX_INDEX_AGE_HOURS = 168  # Maximum index age (1 week)
    MIN_WIN_RATE_VARIANCE = 0.1  # Alert if win rate varies too much
    
    def __init__(self, patterns_dir: str = "rag/patterns"):
        """Initialize health checker."""
        self.patterns_dir = Path(patterns_dir)
        self._retriever = None
    
    @property
    def retriever(self):
        """Lazy load retriever for testing."""
        if self._retriever is None:
            try:
                from .hybrid_retriever import HybridRetriever
                self._retriever = HybridRetriever()
            except Exception as e:
                print(f"⚠️ Could not load retriever: {e}")
        return self._retriever
    
    def full_check(self) -> HealthReport:
        """
        Run all health checks.
        
        Returns:
            HealthReport with all results
        """
        from datetime import datetime
        
        checks = []
        issues = []
        
        # Check 1: Index exists
        result = self.check_index_exists()
        checks.append(result)
        if not result.passed:
            issues.append(result.message)
        
        # Check 2: Pattern count
        result = self.check_pattern_count()
        checks.append(result)
        if not result.passed:
            issues.append(result.message)
        
        # Check 3: Retrieval latency
        result = self.check_latency()
        checks.append(result)
        if not result.passed:
            issues.append(result.message)
        
        # Check 4: Index freshness
        result = self.check_index_freshness()
        checks.append(result)
        if not result.passed:
            issues.append(result.message)
        
        # Check 5: Outcomes data
        result = self.check_outcomes_data()
        checks.append(result)
        if not result.passed:
            issues.append(result.message)
        
        # Overall health
        critical_passed = all(
            c.passed for c in checks 
            if c.name in ['index_exists', 'pattern_count']
        )
        all_passed = all(c.passed for c in checks)
        
        return HealthReport(
            healthy=critical_passed,
            checks=checks,
            issues=issues,
            timestamp=datetime.now().isoformat(),
        )
    
    def check_index_exists(self) -> HealthCheckResult:
        """Check if vector index exists."""
        index_file = self.patterns_dir / "embeddings.faiss"
        metadata_file = self.patterns_dir / "metadata.pkl"
        
        index_exists = index_file.exists()
        metadata_exists = metadata_file.exists()
        
        passed = index_exists and metadata_exists
        
        return HealthCheckResult(
            name='index_exists',
            passed=passed,
            value={'index': index_exists, 'metadata': metadata_exists},
            threshold={'index': True, 'metadata': True},
            message="Index files exist" if passed else "Missing index files. Run: python scripts/build_rag_index.py",
        )
    
    def check_pattern_count(self) -> HealthCheckResult:
        """Check number of indexed patterns."""
        pattern_files = list(self.patterns_dir.glob("*.json"))
        pattern_files = [f for f in pattern_files if f.name not in ['outcomes.json', 'library.json', 'index_config.json']]
        
        count = len(pattern_files)
        passed = count >= self.MIN_PATTERNS
        
        return HealthCheckResult(
            name='pattern_count',
            passed=passed,
            value=count,
            threshold=self.MIN_PATTERNS,
            message=f"Found {count} patterns" if passed else f"Only {count} patterns. Need at least {self.MIN_PATTERNS}.",
        )
    
    def check_latency(self) -> HealthCheckResult:
        """Check retrieval latency."""
        if self.retriever is None:
            return HealthCheckResult(
                name='latency',
                passed=False,
                value=None,
                threshold=self.MAX_LATENCY_MS,
                message="Could not load retriever for latency test",
            )
        
        # Test state
        test_state = {
            'symbol': 'BTC',
            'timeframe': '1h',
            'indicators': {
                'rsi_zscore': -0.5,
                'volume_zscore': 0.5,
                'trend_up': True,
            }
        }
        
        # Run multiple times
        latencies = []
        for _ in range(5):
            start = time.time()
            try:
                self.retriever.retrieve(test_state, top_k=5, use_cache=False)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            except Exception as e:
                return HealthCheckResult(
                    name='latency',
                    passed=False,
                    value=None,
                    threshold=self.MAX_LATENCY_MS,
                    message=f"Retrieval failed: {e}",
                )
        
        avg_latency = statistics.mean(latencies)
        passed = avg_latency < self.MAX_LATENCY_MS
        
        return HealthCheckResult(
            name='latency',
            passed=passed,
            value=round(avg_latency, 2),
            threshold=self.MAX_LATENCY_MS,
            message=f"Avg latency: {avg_latency:.2f}ms" if passed else f"Latency too high: {avg_latency:.2f}ms > {self.MAX_LATENCY_MS}ms",
        )
    
    def check_index_freshness(self) -> HealthCheckResult:
        """Check when index was last updated."""
        config_file = self.patterns_dir / "index_config.json"
        
        if not config_file.exists():
            return HealthCheckResult(
                name='index_freshness',
                passed=False,
                value=None,
                threshold=f"{self.MAX_INDEX_AGE_HOURS} hours",
                message="No index config found. Index may be stale.",
            )
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            from datetime import datetime
            updated_at = datetime.fromisoformat(config.get('updated_at', '2000-01-01'))
            age_hours = (datetime.now() - updated_at).total_seconds() / 3600
            
            passed = age_hours < self.MAX_INDEX_AGE_HOURS
            
            return HealthCheckResult(
                name='index_freshness',
                passed=passed,
                value=round(age_hours, 1),
                threshold=self.MAX_INDEX_AGE_HOURS,
                message=f"Index is {age_hours:.1f} hours old" if passed else f"Index is {age_hours:.1f} hours old. Consider rebuilding.",
            )
            
        except Exception as e:
            return HealthCheckResult(
                name='index_freshness',
                passed=False,
                value=None,
                threshold=f"{self.MAX_INDEX_AGE_HOURS} hours",
                message=f"Could not read index config: {e}",
            )
    
    def check_outcomes_data(self) -> HealthCheckResult:
        """Check if outcomes are being recorded."""
        outcomes_file = self.patterns_dir / "outcomes.json"
        
        if not outcomes_file.exists():
            return HealthCheckResult(
                name='outcomes_data',
                passed=True,  # Not critical
                value=0,
                threshold="Any",
                message="No outcomes recorded yet. This is OK for new systems.",
            )
        
        try:
            with open(outcomes_file, 'r') as f:
                outcomes = json.load(f)
            
            count = len(outcomes)
            
            return HealthCheckResult(
                name='outcomes_data',
                passed=True,
                value=count,
                threshold="Any",
                message=f"Found {count} patterns with outcome data.",
            )
            
        except Exception as e:
            return HealthCheckResult(
                name='outcomes_data',
                passed=False,
                value=None,
                threshold="Any",
                message=f"Could not read outcomes: {e}",
            )
    
    def print_report(self, report: Optional[HealthReport] = None):
        """Print formatted health report."""
        if report is None:
            report = self.full_check()
        
        print("\n" + "=" * 60)
        print("  RAG HEALTH CHECK REPORT")
        print("=" * 60)
        
        status = "✅ HEALTHY" if report.healthy else "❌ UNHEALTHY"
        print(f"\nStatus: {status}")
        print(f"Time: {report.timestamp}")
        
        print("\nChecks:")
        for check in report.checks:
            icon = "✅" if check.passed else "❌"
            print(f"  {icon} {check.name}: {check.message}")
        
        if report.issues:
            print("\n⚠️ Issues:")
            for issue in report.issues:
                print(f"  - {issue}")
        
        print("\n" + "=" * 60)


# =============================================================================
# CLI RUNNER
# =============================================================================

def main():
    """Run health check from command line."""
    checker = RAGHealthCheck()
    checker.print_report()


if __name__ == "__main__":
    main()
