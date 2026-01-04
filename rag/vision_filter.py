"""VISION FILTER для LLM Chart Analysis (stub)"""
from typing import Dict, Tuple

class VisionFilter:
    def __init__(self, model_name: str = "qwen2.5-vl-7b"):
        self.model_name = model_name
        print(f"⚠️ VisionFilter initialized (stub mode)")
    
    def analyze_chart(self, df) -> Dict:
        return {'pattern': 'unknown', 'confidence': 0, 'error': True}
    
    def filter_signals(self, df, p_rb: float, p_ml: float) -> Tuple[bool, float, str]:
        return True, 1.0, "Vision filter not active"
