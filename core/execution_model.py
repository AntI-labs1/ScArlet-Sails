"""
Execution Model with Asset Tiering for Realistic Slippage Calculation.

Classifies 14 coins by liquidity tiers and calculates realistic fill prices
based on market depth and asset tier.

Tier 1 (BTC, ETH): Slippage 3 bps (0.03%)
Tier 2 (SOL, AVAX, DOT, LINK, UNI, LTC): Slippage 10 bps (0.10%)
Tier 3 (ALGO, HBAR, SUI, LDO): Slippage 50 bps (0.50%)
Tier 4 (ENA, ONDO): Slippage 150 bps (1.50%)

Author: Scarlet Sails Team
"""

from enum import Enum
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AssetTier(Enum):
    """Liquidity tier classification."""
    TIER_1 = "tier_1"  # Highest liquidity
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"  # Lowest liquidity


# Asset tier mapping (14 coins)
ASSET_TIERS: Dict[str, AssetTier] = {
    # Tier 1: Highest liquidity (3 bps = 0.03%)
    'BTC': AssetTier.TIER_1,
    'ETH': AssetTier.TIER_1,
    
    # Tier 2: High liquidity (10 bps = 0.10%)
    'SOL': AssetTier.TIER_2,
    'AVAX': AssetTier.TIER_2,
    'DOT': AssetTier.TIER_2,
    'LINK': AssetTier.TIER_2,
    'UNI': AssetTier.TIER_2,
    'LTC': AssetTier.TIER_2,
    
    # Tier 3: Medium liquidity (50 bps = 0.50%)
    'ALGO': AssetTier.TIER_3,
    'HBAR': AssetTier.TIER_3,
    'SUI': AssetTier.TIER_3,
    'LDO': AssetTier.TIER_3,
    
    # Tier 4: Low liquidity (150 bps = 1.50%)
    'ENA': AssetTier.TIER_4,
    'ONDO': AssetTier.TIER_4,
}

# Slippage in basis points (bps) per tier
TIER_SLIPPAGE_BPS: Dict[AssetTier, float] = {
    AssetTier.TIER_1: 3.0,    # 0.03%
    AssetTier.TIER_2: 10.0,   # 0.10%
    AssetTier.TIER_3: 50.0,   # 0.50%
    AssetTier.TIER_4: 150.0,  # 1.50%
}

# Commission (same for all tiers, typically 0.1%)
DEFAULT_COMMISSION = 0.001  # 0.1%


class ExecutionModel:
    """
    Realistic execution model with asset tiering.
    
    Calculates fill prices and costs based on:
    - Asset liquidity tier
    - Order direction (buy/sell)
    - Order size (for future: size-dependent slippage)
    """
    
    def __init__(self, commission: float = DEFAULT_COMMISSION):
        """
        Initialize execution model.
        
        Args:
            commission: Trading commission rate (default 0.1%)
        """
        self.commission = commission
        self.asset_tiers = ASSET_TIERS.copy()
        self.tier_slippage = TIER_SLIPPAGE_BPS.copy()
        
        logger.info(f"ExecutionModel initialized with commission={commission:.4f}")
    
    def get_asset_tier(self, coin: str) -> AssetTier:
        """
        Get liquidity tier for asset.
        
        Args:
            coin: Asset symbol (e.g., 'BTC', 'ETH')
        
        Returns:
            AssetTier enum
        """
        coin_upper = coin.upper()
        tier = self.asset_tiers.get(coin_upper, AssetTier.TIER_3)  # Default to Tier 3
        
        if coin_upper not in self.asset_tiers:
            logger.warning(f"Unknown asset '{coin}', defaulting to Tier 3")
        
        return tier
    
    def get_slippage_bps(self, coin: str) -> float:
        """
        Get slippage in basis points for asset.
        
        Args:
            coin: Asset symbol
        
        Returns:
            Slippage in basis points (e.g., 3.0 = 0.03%)
        """
        tier = self.get_asset_tier(coin)
        return self.tier_slippage[tier]
    
    def get_slippage_pct(self, coin: str) -> float:
        """
        Get slippage as percentage for asset.
        
        Args:
            coin: Asset symbol
        
        Returns:
            Slippage as decimal (e.g., 0.0003 = 0.03%)
        """
        bps = self.get_slippage_bps(coin)
        return bps / 10000.0
    
    def calculate_fill_price(
        self,
        coin: str,
        reference_price: float,
        direction: int,
        order_size: Optional[float] = None,
    ) -> float:
        """
        Calculate realistic fill price based on asset tier.
        
        Args:
            coin: Asset symbol
            reference_price: Reference price (e.g., close price)
            direction: 1 for buy (long), -1 for sell (short)
            order_size: Optional order size (for future: size-dependent slippage)
        
        Returns:
            Fill price after slippage
        """
        slippage_pct = self.get_slippage_pct(coin)
        
        if direction == 1:
            # Buy: pay more (worse fill)
            fill_price = reference_price * (1 + slippage_pct)
        elif direction == -1:
            # Sell: receive less (worse fill)
            fill_price = reference_price * (1 - slippage_pct)
        else:
            # Neutral: no slippage
            fill_price = reference_price
        
        return fill_price
    
    def calculate_execution_costs(
        self,
        coin: str,
        reference_price: float,
        position_size: float,
        direction: int,
    ) -> Dict[str, float]:
        """
        Calculate total execution costs (commission + slippage).
        
        Args:
            coin: Asset symbol
            reference_price: Reference price
            position_size: Position size in units
            direction: 1 for buy, -1 for sell
        
        Returns:
            Dict with:
                - fill_price: Actual fill price
                - commission: Commission cost
                - slippage_cost: Slippage cost
                - total_cost: Total execution cost
        """
        position_value = position_size * reference_price
        
        # Commission
        commission = position_value * self.commission
        
        # Slippage
        fill_price = self.calculate_fill_price(coin, reference_price, direction)
        slippage_cost = abs(fill_price - reference_price) * position_size
        
        # Total cost
        total_cost = commission + slippage_cost
        
        return {
            'fill_price': fill_price,
            'commission': commission,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'slippage_bps': self.get_slippage_bps(coin),
            'slippage_pct': self.get_slippage_pct(coin),
        }
    
    def get_tier_summary(self) -> Dict[str, list]:
        """Get summary of assets by tier."""
        summary = {
            'tier_1': [],
            'tier_2': [],
            'tier_3': [],
            'tier_4': [],
        }
        
        for coin, tier in self.asset_tiers.items():
            tier_name = tier.value
            summary[tier_name].append(coin)
        
        return summary


# Convenience function for quick integration
def create_execution_model(commission: float = DEFAULT_COMMISSION) -> ExecutionModel:
    """
    Factory function to create execution model.
    
    Args:
        commission: Trading commission rate
    
    Returns:
        ExecutionModel instance
    """
    return ExecutionModel(commission=commission)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("EXECUTION MODEL - ASSET TIERING")
    print("=" * 60)
    
    model = ExecutionModel()
    
    # Test different assets
    test_assets = ['BTC', 'ETH', 'SOL', 'ALGO', 'ENA']
    reference_price = 50000.0
    position_size = 1.0
    
    print("\nSlippage by Asset:")
    print("-" * 60)
    for coin in test_assets:
        tier = model.get_asset_tier(coin)
        slippage_bps = model.get_slippage_bps(coin)
        slippage_pct = model.get_slippage_pct(coin)
        
        print(f"{coin:6s} | Tier: {tier.value:6s} | Slippage: {slippage_bps:6.1f} bps ({slippage_pct*100:5.2f}%)")
    
    print("\nExecution Costs (Buy $50k position):")
    print("-" * 60)
    for coin in test_assets:
        costs = model.calculate_execution_costs(
            coin=coin,
            reference_price=reference_price,
            position_size=position_size,
            direction=1,
        )
        
        print(f"{coin:6s} | Fill: ${costs['fill_price']:10.2f} | "
              f"Commission: ${costs['commission']:6.2f} | "
              f"Slippage: ${costs['slippage_cost']:6.2f} | "
              f"Total: ${costs['total_cost']:6.2f}")
    
    print("\nAsset Tier Summary:")
    print("-" * 60)
    summary = model.get_tier_summary()
    for tier_name, coins in summary.items():
        if coins:
            print(f"{tier_name}: {', '.join(coins)}")
    
    print("\n" + "=" * 60)

