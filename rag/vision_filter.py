"""
Vision Filter with Ollama Qwen2.5-VL Integration.

Uses local Qwen2.5-VL model via Ollama to analyze candlestick charts
and provide "common sense" validation of trading patterns.

Functionality:
- Renders last 100 candles using mplfinance
- Sends screenshot to Ollama Qwen2.5-VL
- Asks: "Подтверждаешь ли ты паттерн продолжения тренда?"
- If LLM says "No" → position multiplier 0.5x

Author: Scarlet Sails Team
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
import requests
import base64
import io
import os
import tempfile

logger = logging.getLogger(__name__)

# Ollama API endpoint (default local)
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_VISION_MODEL', 'qwen2.5-vl:latest')


class VisionFilter:
    """
    Vision-based pattern validation using Qwen2.5-VL via Ollama.
    
    Analyzes candlestick charts to validate trading patterns
    and adjust position sizing based on LLM confidence.
    """
    
    def __init__(
        self,
        ollama_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        n_candles: int = 100,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize vision filter.
        
        Args:
            ollama_url: Ollama API base URL
            model_name: Model name (e.g., 'qwen2.5-vl:latest')
            n_candles: Number of candles to render
            temp_dir: Temporary directory for chart images
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.n_candles = n_candles
        
        # Create temp directory for charts
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / 'scarlet_sails_charts'
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"VisionFilter initialized: "
            f"model={model_name}, n_candles={n_candles}, "
            f"ollama_url={ollama_url}"
        )
    
    def render_candlestick_chart(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Render last N candles as candlestick chart.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
            output_path: Optional output path (auto-generated if None)
        
        Returns:
            Path to saved chart image
        """
        # Get last N candles
        chart_df = df.tail(self.n_candles).copy()
        
        # Ensure proper column names for mplfinance
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in chart_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.temp_dir / f'chart_{timestamp}.png'
        
        # Render candlestick chart
        try:
            mpf.plot(
                chart_df,
                type='candle',
                volume=True,
                style='charles',
                savefig=dict(
                    fname=str(output_path),
                    dpi=150,
                    bbox_inches='tight',
                ),
                show_nontrading=False,
                warn_too_much_data=False,
            )
            
            logger.debug(f"Chart rendered: {output_path}")
            return Path(output_path)
        
        except Exception as e:
            logger.error(f"Failed to render chart: {e}")
            raise
    
    def render_multi_scale_diptych(
        self,
        df_15m: pd.DataFrame,
        df_4h: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Render diptych (two-panel chart) with 15m and 4h timeframes.
        
        Args:
            df_15m: 15-minute OHLCV DataFrame
            df_4h: 4-hour OHLCV DataFrame
            output_path: Optional output path
        
        Returns:
            Path to saved chart image
        """
        # Get last N candles for each timeframe
        chart_15m = df_15m.tail(self.n_candles).copy()
        # For 4h, use fewer candles to show longer period
        chart_4h = df_4h.tail(self.n_candles // 4).copy()
        
        # Ensure proper column names
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for df_name, df_data in [('15m', chart_15m), ('4h', chart_4h)]:
            if not all(col in df_data.columns for col in required_cols):
                raise ValueError(f"DataFrame {df_name} must have columns: {required_cols}")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.temp_dir / f'diptych_{timestamp}.png'
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Top panel: 15m timeframe
        ax1 = plt.subplot(2, 1, 1)
        mpf.plot(
            chart_15m,
            type='candle',
            volume=True,
            style='charles',
            ax=ax1,
            show_nontrading=False,
            warn_too_much_data=False,
        )
        ax1.set_title('15m Timeframe - Pattern Detail', fontsize=12, fontweight='bold')
        
        # Bottom panel: 4h timeframe
        ax2 = plt.subplot(2, 1, 2)
        mpf.plot(
            chart_4h,
            type='candle',
            volume=True,
            style='charles',
            ax=ax2,
            show_nontrading=False,
            warn_too_much_data=False,
        )
        ax2.set_title('4h Timeframe - Trend Context', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Diptych rendered: {output_path}")
        return Path(output_path)
    
    def image_to_base64(self, image_path: Path) -> str:
        """
        Convert image file to base64 string.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def query_ollama_vision(
        self,
        image_path: Path,
        prompt: str = "Подтверждаешь ли ты паттерн продолжения тренда? Ответь только 'Yes' или 'No'.",
    ) -> str:
        """
        Send image and prompt to Ollama vision model.
        
        Args:
            image_path: Path to chart image
            prompt: Text prompt for LLM
        
        Returns:
            LLM response text
        """
        # Convert image to base64
        image_base64 = self.image_to_base64(image_path)
        
        # Prepare request
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get('response', '').strip()
            
            logger.debug(f"Ollama response: {answer}")
            return answer
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            # Fallback: return neutral response
            return "Yes"
    
    def validate_pattern(
        self,
        df: pd.DataFrame,
        pattern_type: str = "trend_continuation",
        df_4h: Optional[pd.DataFrame] = None,
    ) -> Dict[str, any]:
        """
        Validate trading pattern using multi-scale vision analysis.
        
        Args:
            df: 15m OHLCV DataFrame
            pattern_type: Type of pattern to validate
            df_4h: Optional 4h OHLCV DataFrame for trend context
        
        Returns:
            Dict with:
                - confirmed: bool (True if pattern confirmed)
                - confidence: float (0-1)
                - position_multiplier: float (0.5 if rejected, 1.0 if confirmed)
                - llm_response: str
        """
        # Render chart(s)
        try:
            if df_4h is not None:
                # Multi-scale diptych
                chart_path = self.render_multi_scale_diptych(df, df_4h)
            else:
                # Single timeframe (backward compatibility)
                chart_path = self.render_candlestick_chart(df)
        except Exception as e:
            logger.error(f"Failed to render chart: {e}")
            # Fallback: assume pattern confirmed
            return {
                'confirmed': True,
                'confidence': 0.5,
                'position_multiplier': 1.0,
                'llm_response': 'Error rendering chart',
                'error': str(e),
            }
        
        # Prepare enhanced prompt based on pattern type
        prompts = {
            'trend_continuation': (
                "Ты видишь два графика: верхний (15m) показывает детали паттерна, "
                "нижний (4h) показывает общий тренд. "
                "Подтверждает ли структура 4h текущий лонг-сигнал на 15m? "
                "Проверь согласованность трендов между таймфреймами. "
                "Ответь только 'Yes' (если тренды согласованы) или 'No' (если противоречат)."
            ),
            'reversal': (
                "Ты видишь два графика: верхний (15m) показывает детали паттерна разворота, "
                "нижний (4h) показывает общий тренд. "
                "Подтверждает ли структура 4h паттерн разворота на 15m? "
                "Проверь согласованность сигналов между таймфреймами. "
                "Ответь только 'Yes' или 'No'."
            ),
            'breakout': (
                "Ты видишь два графика: верхний (15m) показывает пробой, "
                "нижний (4h) показывает контекст тренда. "
                "Подтверждает ли структура 4h пробой на 15m? "
                "Проверь силу и направление пробоя на обоих таймфреймах. "
                "Ответь только 'Yes' или 'No'."
            ),
        }
        
        prompt = prompts.get(pattern_type, prompts['trend_continuation'])
        
        # Query Ollama
        try:
            llm_response = self.query_ollama_vision(chart_path, prompt)
        except Exception as e:
            logger.error(f"Failed to query Ollama: {e}")
            # Fallback: assume pattern confirmed
            return {
                'confirmed': True,
                'confidence': 0.5,
                'position_multiplier': 1.0,
                'llm_response': f'Error: {str(e)}',
            }
        
        # Parse response
        response_lower = llm_response.lower()
        confirmed = 'yes' in response_lower or 'да' in response_lower
        rejected = 'no' in response_lower or 'нет' in response_lower
        
        if rejected:
            position_multiplier = 0.5
            confidence = 0.3
        elif confirmed:
            position_multiplier = 1.0
            confidence = 0.7
        else:
            # Ambiguous response - neutral
            position_multiplier = 0.75
            confidence = 0.5
        
        # Clean up temp file
        try:
            chart_path.unlink()
        except Exception:
            pass
        
        return {
            'confirmed': confirmed and not rejected,
            'confidence': confidence,
            'position_multiplier': position_multiplier,
            'llm_response': llm_response,
        }
    
    def apply_vision_filter(
        self,
        base_position_size: float,
        df: pd.DataFrame,
        pattern_type: str = "trend_continuation",
        df_4h: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, Dict[str, any]]:
        """
        Apply vision filter to position sizing with multi-scale validation.
        
        Args:
            base_position_size: Base position size
            df: 15m OHLCV DataFrame
            pattern_type: Pattern type to validate
            df_4h: Optional 4h OHLCV DataFrame for trend context
        
        Returns:
            Tuple of (adjusted_position_size, validation_result)
        """
        validation = self.validate_pattern(df, pattern_type, df_4h=df_4h)
        
        adjusted_size = base_position_size * validation['position_multiplier']
        
        return adjusted_size, validation


# Convenience function
def create_vision_filter(
    ollama_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> VisionFilter:
    """
    Factory function to create vision filter.
    
    Args:
        ollama_url: Optional Ollama URL (uses env var or default)
        model_name: Optional model name (uses env var or default)
    
    Returns:
        VisionFilter instance
    """
    return VisionFilter(
        ollama_url=ollama_url or OLLAMA_BASE_URL,
        model_name=model_name or OLLAMA_MODEL,
    )


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("VISION FILTER - OLLAMA QWEN2.5-VL INTEGRATION")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    
    # Generate sample OHLCV data
    close_prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 200)))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, 200)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, 200))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, 200))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, 200)
    }, index=dates)
    
    print(f"\nSample data: {len(df)} bars")
    
    # Initialize vision filter
    try:
        filter_obj = VisionFilter()
        
        # Test pattern validation
        print("\nValidating pattern...")
        result = filter_obj.validate_pattern(df, pattern_type='trend_continuation')
        
        print(f"\nValidation Result:")
        print(f"  Confirmed: {result['confirmed']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Position Multiplier: {result['position_multiplier']:.2f}")
        print(f"  LLM Response: {result['llm_response']}")
        
        # Test position sizing
        base_size = 1000.0
        adjusted_size, validation = filter_obj.apply_vision_filter(
            base_size, df, pattern_type='trend_continuation'
        )
        
        print(f"\nPosition Sizing:")
        print(f"  Base Size: ${base_size:.2f}")
        print(f"  Adjusted Size: ${adjusted_size:.2f}")
        print(f"  Multiplier: {validation['position_multiplier']:.2f}x")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running and Qwen2.5-VL model is installed:")
        print("  ollama pull qwen2.5-vl:latest")
    
    print("\n" + "=" * 60)
