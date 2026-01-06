import pytest
import numpy as np
import pandas as pd
from core.risk.rolling_dispersion import RollingDispersionCalculator

def get_mult_value(result):
    """
    Helper to extract float multiplier from whatever update() returns.
    Updated based on Error Log: Looking for 'confidence_multiplier'.
    """
    # 1. Точное совпадение с объектом из лога
    if hasattr(result, 'confidence_multiplier'):
        return float(result.confidence_multiplier)
        
    # 2. Альтернативное имя
    if hasattr(result, 'multiplier'):
        return float(result.multiplier)
        
    # 3. Если это словарь
    if isinstance(result, dict):
        return float(result.get('confidence_multiplier', result.get('multiplier', 0.0)))
        
    # 4. Если просто число
    try:
        return float(result)
    except:
        raise ValueError(f"Cannot extract multiplier from {type(result)}: {result}")

def test_dispersion_logic_inversion():
    calc = RollingDispersionCalculator(window=20)
    print("\n🧪 STARTING DISPERSION TEST (Final Fix)...")

    # --- СЦЕНАРИЙ 1: СОГЛАСИЕ (AGREEMENT) ---
    calc.reset()
    last_mult_agreed = 0.0
    print("   Feeding AGREED data (p=0.9, p=0.9)...")
    
    # Кормим данными, чтобы заполнить окно
    for i in range(50):
        row = {'p_rb': 0.9, 'p_ml': 0.9, 'p_hyb': 0.9}
        res = calc.update(pd.Series(row))
        last_mult_agreed = get_mult_value(res)
        
    print(f"   📊 Result (Agreed): {last_mult_agreed:.4f}")

    # --- СЦЕНАРИЙ 2: ХАОС (CHAOS) ---
    calc.reset()
    last_mult_chaos = 0.0
    print("   Feeding CHAOS data (p=0.1, p=0.9)...")
    
    for i in range(50):
        row = {'p_rb': 0.1, 'p_ml': 0.9, 'p_hyb': 0.5}
        res = calc.update(pd.Series(row))
        last_mult_chaos = get_mult_value(res)
        
    print(f"   📊 Result (Chaos):  {last_mult_chaos:.4f}")

    # --- ДИАГНОСТИКА ---
    if last_mult_agreed < last_mult_chaos:
        print("\n❌ DIAGNOSIS: Logic is STILL INVERTED.")
        print("   Chaos (High Risk) is getting MORE money than Agreement.")
        print("   We need to apply the math fix.")
    else:
        print("\n✅ DIAGNOSIS: Logic is CORRECT.")
        print("   Agreement gets more money. Risk is managed.")

    # --- ПРОВЕРКА (ASSERTIONS) ---
    assert last_mult_agreed > last_mult_chaos, \
        f"Logic Error: Agreed ({last_mult_agreed:.4f}) <= Chaos ({last_mult_chaos:.4f})"

    assert last_mult_agreed >= 0.8, "Agreed multiplier too low"
    assert last_mult_chaos <= 0.7, "Chaos multiplier too high"

if __name__ == "__main__":
    try:
        test_dispersion_logic_inversion()
        print("\n✅ TEST PASSED SUCCESS")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n⚠️ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
