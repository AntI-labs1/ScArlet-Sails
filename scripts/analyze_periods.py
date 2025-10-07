import pandas as pd
import matplotlib.pyplot as plt

# Загрузить данные и посмотреть на периоды
df = pd.read_parquet('data/raw/BTC_USDT_15m.parquet')

# Разбить на периоды по 5000 samples
period_size = 5000
n_periods = len(df) // period_size

results = []
for i in range(n_periods):
    start = i * period_size
    end = start + period_size
    period_df = df.iloc[start:end]
    
    # Рассчитать метрики периода
    returns = (period_df['close'].iloc[-1] - period_df['close'].iloc[0]) / period_df['close'].iloc[0]
    volatility = period_df['close'].pct_change().std()
    up_days = (period_df['close'].diff() > 0).mean()
    
    results.append({
        'period': i,
        'start_date': period_df.index[0],
        'end_date': period_df.index[-1],
        'total_return': returns,
        'volatility': volatility,
        'up_ratio': up_days
    })

periods_df = pd.DataFrame(results)
print(periods_df)

# Визуализация
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].bar(periods_df['period'], periods_df['total_return'])
axes[0].set_title('Total Return by Period')
axes[0].axhline(y=0, color='r', linestyle='--')

axes[1].bar(periods_df['period'], periods_df['up_ratio'])
axes[1].set_title('UP Days Ratio by Period')
axes[1].axhline(y=0.5, color='r', linestyle='--')

axes[2].bar(periods_df['period'], periods_df['volatility'])
axes[2].set_title('Volatility by Period')

plt.tight_layout()
plt.savefig('reports/period_analysis.png')
plt.show()

# Найти bull/bear периоды
bull_periods = periods_df[periods_df['total_return'] > 0.1]
bear_periods = periods_df[periods_df['total_return'] < -0.1]

print(f"\n🐂 Bull periods (return >10%): {len(bull_periods)}")
print(f"🐻 Bear periods (return <-10%): {len(bear_periods)}")
print(f"🦀 Sideways periods: {len(periods_df) - len(bull_periods) - len(bear_periods)}")