#!/usr/bin/env python3
"""
🔍 DATA VALIDATION UTILITY

Инструменты для валидации данных во всей системе.
Используется как CLI инструмент и как библиотека.

Usage:
    python validate_data.py data/market_data.json
    python validate_data.py --check-all
    python validate_data.py --source binance --since 2024-01-01
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.canonical_pipeline import CanonicalPipeline, ValidationError


class DataValidator:
    """
    🔍 Utility for validating data files and directories
    """
    
    def __init__(self, verbose: bool = False):
        self.pipeline = CanonicalPipeline()
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self, verbose: bool):
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - [%(levelname)s] - %(message)s'
        )
    
    def validate_file(self, filepath: Path) -> Dict:
        """
        Validate a single JSON file
        """
        self.logger.info(f"📄 Validating: {filepath}")
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            
            results = {
                "file": str(filepath),
                "total": len(data),
                "valid": 0,
                "invalid": 0,
                "errors": []
            }
            
            for idx, item in enumerate(data):
                try:
                    self.pipeline.validate(item)
                    results["valid"] += 1
                except ValidationError as e:
                    results["invalid"] += 1
                    results["errors"].append({
                        "index": idx,
                        "error": str(e),
                        "data": item
                    })
            
            self.logger.info(
                f"✅ Valid: {results['valid']}/{results['total']} | "
                f"❌ Invalid: {results['invalid']}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {filepath}: {e}")
            return {
                "file": str(filepath),
                "error": str(e)
            }
    
    def validate_directory(self, dirpath: Path, pattern: str = "*.json") -> List[Dict]:
        """
        Validate all JSON files in a directory
        """
        self.logger.info(f"📂 Validating directory: {dirpath}")
        
        files = list(dirpath.glob(pattern))
        self.logger.info(f"Found {len(files)} files")
        
        results = []
        for filepath in files:
            result = self.validate_file(filepath)
            results.append(result)
        
        return results
    
    def check_recent_data(self, data_dir: Path, hours: int = 24) -> Dict:
        """
        Check data freshness - ensure recent data exists
        """
        self.logger.info(f"⌛ Checking data from last {hours} hours")
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        results = {
            "cutoff": cutoff.isoformat(),
            "fresh_files": [],
            "stale_files": [],
            "missing_data": False
        }
        
        for filepath in data_dir.glob("*.json"):
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            
            if mtime >= cutoff:
                results["fresh_files"].append({
                    "file": str(filepath),
                    "modified": mtime.isoformat()
                })
            else:
                results["stale_files"].append({
                    "file": str(filepath),
                    "modified": mtime.isoformat(),
                    "age_hours": (datetime.now() - mtime).total_seconds() / 3600
                })
        
        if not results["fresh_files"]:
            results["missing_data"] = True
            self.logger.warning(f"⚠️ No fresh data found in last {hours} hours!")
        
        return results
    
    def validate_source(self, source: str, data_dir: Path) -> Dict:
        """
        Validate data from a specific source
        """
        self.logger.info(f"🔍 Validating source: {source}")
        
        pattern = f"*{source}*.json"
        files = list(data_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"⚠️ No files found for source: {source}")
            return {"source": source, "files": 0}
        
        results = {
            "source": source,
            "files": len(files),
            "validations": []
        }
        
        for filepath in files:
            result = self.validate_file(filepath)
            results["validations"].append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Data Validation Utility"
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="File or directory to validate"
    )
    
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all data directories"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        help="Validate data from specific source"
    )
    
    parser.add_argument(
        "--check-recent",
        type=int,
        metavar="HOURS",
        help="Check for data from last N hours"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save validation report to file"
    )
    
    args = parser.parse_args()
    
    validator = DataValidator(verbose=args.verbose)
    
    results = None
    
    # Выполняем запрошенную операцию
    if args.check_all:
        data_dir = ROOT_DIR / "data"
        if data_dir.exists():
            results = validator.validate_directory(data_dir)
        else:
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
    
    elif args.source:
        data_dir = ROOT_DIR / "data"
        if data_dir.exists():
            results = validator.validate_source(args.source, data_dir)
        else:
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
    
    elif args.check_recent:
        data_dir = ROOT_DIR / "data"
        if data_dir.exists():
            results = validator.check_recent_data(data_dir, args.check_recent)
        else:
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
    
    elif args.path:
        if args.path.is_file():
            results = validator.validate_file(args.path)
        elif args.path.is_dir():
            results = validator.validate_directory(args.path)
        else:
            print(f"❌ Path not found: {args.path}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)
    
    # Вывод результатов
    if results:
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Сохраняем в файл, если указан
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Report saved to: {args.output}")
    
    # Pipeline stats
    print("\n" + "="*60)
    print("📊 PIPELINE STATISTICS")
    print("="*60)
    stats = validator.pipeline.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
