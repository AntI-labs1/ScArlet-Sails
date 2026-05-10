"""
Simple Strategies for Scarlet Sails Backtesting

These are baseline strategies for testing the backtester.
Production strategies are in separate files.
"""

import pandas as pd
import numpy as np
from typing import Optional


class SimpleRSIStrategy:
    """
    RSI Mean Reversion Strategy with Trend Filter.
    
    Buy when:
    - RSI < oversold_threshold
    - Price above EMA (uptrend)
    
    Sell when:
    - RSI > overbought_threshold
    - OR price below EMA (trend reversal)
    
    Optimized for crypto 15m timeframe.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 35,  # Less extreme for crypto
        overbought: float = 65,  # Less extreme for crypto
        ema_period: int = 50,  # Trend filter
        use_trend_filter: bool = True,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.ema_period = ema_period
        self.use_trend_filter = use_trend_filter
    
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI + Trend Filter.
        
        Returns
        -------
        pd.Series
            1 = Buy, -1 = Sell, 0 = Hold
        """
        close = df['close']
        rsi = self._calculate_rsi(close)
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        
        signals = pd.Series(0, index=df.index)
        
        in_position = False
        entry_price = 0.0
        
        for i in range(len(df)):
            if pd.isna(rsi.iloc[i]) or pd.isna(ema.iloc[i]):
                continue
            
            price = close.iloc[i]
            current_rsi = rsi.iloc[i]
            in_uptrend = price > ema.iloc[i]
            
            if not in_position:
                # Buy conditions
                rsi_oversold = current_rsi < self.oversold
                trend_ok = in_uptrend if self.use_trend_filter else True
                
                if rsi_oversold and trend_ok:
                    signals.iloc[i] = 1
                    in_position = True
                    entry_price = price
            else:
                # Sell conditions
                rsi_overbought = current_rsi > self.overbought
                trend_broken = not in_uptrend if self.use_trend_filter else False
                
                # Take profit: 3% gain
                take_profit = price > entry_price * 1.03
                
                # Stop loss: 2% loss (handled by backtester, but add trend exit)
                
                if rsi_overbought or trend_broken or take_profit:
                    signals.iloc[i] = -1
                    in_position = False
        
        return signals


class SimpleMAStrategy:
    """
    Simple Moving Average Crossover Strategy.
    
    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.
    """
    
    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossover."""
        fast_ma = df['close'].rolling(window=self.fast_period).mean()
        slow_ma = df['close'].rolling(window=self.slow_period).mean()
        
        signals = pd.Series(0, index=df.index)
        
        in_position = False
        
        for i in range(1, len(df)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            
            # Crossover detection
            prev_above = fast_ma.iloc[i-1] > slow_ma.iloc[i-1]
            curr_above = fast_ma.iloc[i] > slow_ma.iloc[i]
            
            if not in_position and not prev_above and curr_above:
                # Golden cross - buy
                signals.iloc[i] = 1
                in_position = True
            elif in_position and prev_above and not curr_above:
                # Death cross - sell
                signals.iloc[i] = -1
                in_position = False
        
        return signals


class SimpleBollingerStrategy:
    """
    Simple Bollinger Bands Mean Reversion Strategy.
    
    Buy when price touches lower band.
    Sell when price touches upper band.
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ):
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands."""
        close = df['close']
        
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        
        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std
        
        signals = pd.Series(0, index=df.index)
        
        in_position = False
        
        for i in range(len(df)):
            if pd.isna(lower.iloc[i]) or pd.isna(upper.iloc[i]):
                continue
            
            price = close.iloc[i]
            
            if not in_position and price <= lower.iloc[i]:
                signals.iloc[i] = 1  # Buy at lower band
                in_position = True
            elif in_position and price >= upper.iloc[i]:
                signals.iloc[i] = -1  # Sell at upper band
                in_position = False
        
        return signals


class CombinedStrategy:
    """
    Combined strategy using multiple indicators.
    
    Buy when:
    - RSI < 40 (not extreme oversold)
    - Price above slow MA (uptrend)
    - Price near lower Bollinger Band
    
    Sell when:
    - RSI > 60
    - Or price below slow MA
    """
    
    def __init__(
        self,
        ood_detector=None,
        rsi_period: int = 14,
        ma_period: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        self.ood_detector = ood_detector
        self.rsi_period = rsi_period
        self.ma_period = ma_period
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using multiple indicators."""
        if self.ood_detector and self.ood_detector.is_safe_mode:
            return pd.Series(0, index=df.index) # Silence

        close = df['close']
        
        # Calculate indicators
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Moving Average
        ma = close.rolling(window=self.ma_period).mean()
        
        # Bollinger Bands
        bb_ma = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_lower = bb_ma - self.bb_std * bb_std
        bb_upper = bb_ma + self.bb_std * bb_std
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        in_position = False
        
        for i in range(len(df)):
            # Skip if indicators not ready
            if pd.isna(rsi.iloc[i]) or pd.isna(ma.iloc[i]) or pd.isna(bb_lower.iloc[i]):
                continue
            
            price = close.iloc[i]
            
            # Buy conditions
            buy_rsi = rsi.iloc[i] < 40
            buy_trend = price > ma.iloc[i]
            buy_bb = price < bb_lower.iloc[i] * 1.02  # Within 2% of lower band
            
            # Sell conditions
            sell_rsi = rsi.iloc[i] > 60
            sell_trend = price < ma.iloc[i]
            
            if not in_position and buy_rsi and buy_trend and buy_bb:
                signals.iloc[i] = 1
                in_position = True
            elif in_position and (sell_rsi or sell_trend):
                signals.iloc[i] = -1
                in_position = False
        
        return signals


# Placeholder for production strategies
class RuleBasedStrategy:
    """
    Simplified Rule-Based Strategy (adapted from rule_based_v2.py)
    
    Works with OHLCV data (5 columns).
    OPTIMIZED: Pre-calculates all indicators once.
    
    Formula:
    P_rb(S) = W_opportunity(S) · ∏ᵢ Iᵢ(S) - C_fixed - R_penalty(S)
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_lower: float = 20,
        rsi_upper: float = 80,
        ema_fast: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        signal_threshold: float = 0.05,
    ):
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.ema_fast = ema_fast
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.C_fixed = commission + slippage
        self.signal_threshold = signal_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on P_rb formula.
        OPTIMIZED: Pre-calculates all indicators.
        """
        close = df['close']
        volume = df['volume']
        n = len(df)
        
        # Pre-calculate ALL indicators ONCE
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # EMA
        ema = close.ewm(span=self.ema_fast, adjust=False).mean()
        
        # Volume MA
        vol_ma = volume.rolling(14).mean()
        
        # Bollinger Bands
        bb_ma = close.rolling(self.bb_period).mean()
        bb_std_val = close.rolling(self.bb_period).std()
        bb_upper = bb_ma + self.bb_std * bb_std_val
        bb_lower = bb_ma - self.bb_std * bb_std_val
        
        # Volatility (rolling std of returns)
        returns = close.pct_change()
        volatility = returns.rolling(50).std()
        
        # W_opportunity (volatility + volume score)
        vol_score = (volatility / 0.05).clip(0, 1)
        volume_ratio = (volume / vol_ma).clip(0, 2) / 2
        W_opportunity = 0.5 * vol_score + 0.5 * volume_ratio
        
        # Technical filters
        I1_rsi = ((rsi > self.rsi_lower) & (rsi < self.rsi_upper)).astype(int)
        I2_ema = (close > ema).astype(int)
        I3_volume = (volume > vol_ma).astype(int)
        
        bb_range = bb_upper - bb_lower
        bb_position = (close - bb_lower) / bb_range.replace(0, np.inf)
        I4_bb = ((bb_position > 0.25) & (bb_position < 0.75)).astype(int)
        
        filters_product = I1_rsi * I2_ema * I3_volume * I4_bb
        
        # Risk penalty
        risk_penalty = (volatility * 2.0).clip(0, 0.5)
        
        # Calculate P_rb for all bars
        P_rb = W_opportunity * filters_product - self.C_fixed - risk_penalty
        P_rb = P_rb.fillna(-1)
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        in_position = False
        
        for idx in range(self.bb_period, n):
            if not in_position:
                if P_rb.iloc[idx] > self.signal_threshold:
                    signals.iloc[idx] = 1
                    in_position = True
            else:
                if P_rb.iloc[idx] < -0.1 or filters_product.iloc[idx] == 0:
                    signals.iloc[idx] = -1
                    in_position = False
        
        return signals


class HybridStrategy:
    """
    Simplified Hybrid Strategy - OPTIMIZED
    
    Combines RSI + MA + BB signals with adaptive weighting.
    Works with OHLCV data (5 columns).
    Pre-calculates all indicators once for speed.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        ma_fast: int = 9,
        ma_slow: int = 21,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        
        # Adaptive weights
        self.alpha = 0.4  # Trend weight
        self.beta = 0.4   # Mean reversion weight
        self.gamma = 0.2  # Momentum weight
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate hybrid trading signals.
        OPTIMIZED: Pre-calculates all indicators once.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        n = len(df)
        
        # Pre-calculate ALL indicators
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        ma_fast = close.rolling(self.ma_fast).mean()
        ma_slow = close.rolling(self.ma_slow).mean()
        
        # Bollinger Bands
        bb_ma = close.rolling(self.bb_period).mean()
        bb_std_val = close.rolling(self.bb_period).std()
        bb_upper = bb_ma + self.bb_std * bb_std_val
        bb_lower = bb_ma - self.bb_std * bb_std_val
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # Pre-calculate scores
        trend_score = np.where(ma_fast > ma_slow, 1.0, -1.0)
        
        rsi_score = np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0))
        bb_score = np.where(close < bb_lower, 1.0, np.where(close > bb_upper, -1.0, 0.0))
        mean_rev_score = 0.5 * rsi_score + 0.5 * bb_score
        
        momentum_score = np.where(close > ma_slow, 1.0, -1.0)
        
        # Hybrid score
        P_hyb = (self.alpha * trend_score + 
                 self.beta * mean_rev_score + 
                 self.gamma * momentum_score)
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        
        start_idx = max(self.ma_slow, self.bb_period)
        
        for i in range(start_idx, n):
            price = close.iloc[i]
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else price * 0.02
            
            if not in_position:
                # Buy: positive hybrid score with trend confirmation
                if P_hyb[i] > 0.3 and trend_score[i] > 0:
                    signals.iloc[i] = 1
                    in_position = True
                    entry_price = price
                    entry_idx = i
            else:
                # Sell conditions
                take_profit = price > entry_price + 2 * current_atr
                stop_loss = price < entry_price - 1.5 * current_atr
                signal_exit = P_hyb[i] < -0.3
                
                if take_profit or stop_loss or signal_exit:
                    signals.iloc[i] = -1
                    in_position = False
        
        return signals