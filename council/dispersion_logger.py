"""DISPERSION LOGGER для Council"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class DispersionSnapshot:
    timestamp: str
    p_rb: float
    p_ml: float
    dispersion: float
    regime: str

class DispersionLogger:
    def __init__(self, output_dir: str = "analysis/dispersion_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[DispersionSnapshot] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_snapshot(self, p_rb: float, p_ml: float, regime: str):
        dispersion = abs(p_rb - p_ml)
        snapshot = DispersionSnapshot(
            timestamp=datetime.now().isoformat(),
            p_rb=p_rb, p_ml=p_ml, dispersion=dispersion, regime=regime
        )
        self.snapshots.append(snapshot)
    
    deave_session(self):
        output_file = self.output_dir / f"dispersion_{self.session_id}.json"
        data = {'snapshots': [asdict(s) for s in self.snapshots]}
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        return output_file
