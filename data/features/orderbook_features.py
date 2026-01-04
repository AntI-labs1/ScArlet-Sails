"""ORDER BOOK L2 MICROSTRUCTURE FEATURES"""
import pandas as pd
import numpy as np
from typing import Dict

class OrderBookFeatures:
    def __init__(self, n_levels: int = 10):
        self.n_levels = n_levels
    
    def calculate_obi(self, bids: list, asks: list) -> float:
        bid_vol = sum([size for _, size in bids[:self.n_levels]])
        ask_vol = sum([size for _, size in asks[:self.n_levels]])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
    
    def extract_all_features(self, orderbook: Dict) -> Dict[str, float]:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        return {'ob_imbalance': self.calculate_obi(bids, asks)}
