"""
SCARLET SAILS — health check entry point.

Канонический CLI бэктеста: `python run_backtest.py --help`.
Этот файл — только проверка системного состояния (конфиг, модели, данные).
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ScarletSails:
    """System health checker."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        logger.info("=" * 80)
        logger.info("SCARLET SAILS — HEALTH CHECK")
        logger.info("=" * 80)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path(__file__).parent
        logger.info(f"Config: {config_path}")
        logger.info(f"Assets: {len(self.config['data']['assets'])}")
        logger.info(f"Timeframes: {len(self.config['data']['timeframes'])}")

    def health_check(self) -> bool:
        logger.info("")
        logger.info("Checks:")

        data_dir = self.project_root / self.config["data"]["data_dir"]
        xgb_path = self.project_root / self.config["models"]["xgboost"]["model_path"]

        checks = {
            "Config loaded": self.config is not None,
            "Models directory": (self.project_root / "models").exists(),
            "Data directory": data_dir.exists(),
            "XGBoost model": xgb_path.exists(),
            "Feature engine (v2)": (self.project_root / "core" / "feature_engine_v2.py").exists(),
            "Rule-based v2": (self.project_root / "strategies" / "rule_based_v2.py").exists(),
            "XGBoost ML v3": (self.project_root / "strategies" / "xgboost_ml_v3.py").exists(),
        }

        for name, status in checks.items():
            mark = "PASS" if status else "FAIL"
            logger.info(f"  [{mark}] {name}")

        all_pass = all(checks.values())
        logger.info("=" * 80)
        logger.info("HEALTHY" if all_pass else "DEGRADED — fix failed checks before running backtests")
        return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scarlet Sails — health check. For backtests use run_backtest.py.",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    system = ScarletSails(config_path=args.config)
    ok = system.health_check()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
