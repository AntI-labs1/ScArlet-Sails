"""
RL Advisor wrapper for Hybrid strategy integration.

Takes a trained DQNAgent and exposes a scalar V_rl(S) in [0,1]
that can be plugged into the Hybrid strategy as γ · V_rl(S).
"""
from typing import Dict, Optional

import numpy as np
import torch
import pandas as pd

from rl.dqn import DQNAgent
from rl.trading_environment import normalize_price_features, normalize_volume_features


class RLAdvisor:
    def __init__(self, agent: DQNAgent, device: Optional[str] = None):
        self.agent = agent
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agent.policy_net.to(self.device)

    def _market_state_to_vector(self, market_state: Dict) -> np.ndarray:
        if market_state is None:
            return np.zeros(self.agent.state_dim, dtype=np.float32)

        if 'state_vector' in market_state:
            vec = np.array(market_state['state_vector'], dtype=np.float32)
            if vec.shape[0] == self.agent.state_dim:
                return vec

        close_window = pd.Series(market_state.get('close_window', []))
        volume_window = pd.Series(market_state.get('volume_window', []))

        price_feats = normalize_price_features(close_window)
        vol_feats = normalize_volume_features(volume_window)

        state = np.zeros(self.agent.state_dim, dtype=np.float32)
        state[0] = price_feats['latest_ret']
        state[1] = price_feats['mean_ret']
        state[2] = price_feats['vol'] + vol_feats['z_vol']
        state[3] = price_feats['trend']
        state[4] = price_feats['rel_price']

        state[5] = float(market_state.get('position', 0.0))
        state[6] = float(market_state.get('unrealized_pnl', 0.0))
        state[7] = float(market_state.get('equity_change', 0.0))
        state[8] = float(market_state.get('drawdown', 0.0))

        volatility = abs(price_feats['vol'])
        if volatility < 0.015:
            state[9:12] = [1.0, 0.0, 0.0]
        elif volatility < 0.03:
            state[9:12] = [0.0, 1.0, 0.0]
        else:
            state[9:12] = [0.0, 0.0, 1.0]

        state = np.clip(state, -10.0, 10.0)
        return state

    def get_value(self, market_state: Dict) -> float:
        state_vec = self._market_state_to_vector(market_state)
        with torch.no_grad():
            state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.agent.policy_net(state_tensor)
            probs = torch.softmax(q_values, dim=1)
            enter_idx = 1  # ENTER_LONG action
            if probs.numel() <= enter_idx:
                return 0.0
            return float(probs[0, enter_idx].clamp(0.0, 1.0).item())

    def calculate_rl_component(self, market_state: Dict) -> float:
        return self.get_value(market_state)
