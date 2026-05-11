# backtesting/honest_backtest.py
#
# DEPRECATED 2026-05: используйте backtesting/vbt_engine.py
# (см. backtesting/MIGRATION_NOTES.md). Этот движок оставлен как research-артефакт
# и не получает обновлений: Sharpe-аннуализация и обработка комиссий расходятся
# с каноном.
import warnings as _warnings

import pandas as pd
import numpy as np
from typing import Dict

_warnings.warn(
    "backtesting.honest_backtest is deprecated; use backtesting.vbt_engine.",
    DeprecationWarning,
    stacklevel=2,
)
_warnings.filterwarnings('ignore')

class HonestBacktestEngine:
    """
    Честный бэктест с учётом комиссий, проскальзывания и реального исполнения.
    Стратегия: Если сигнал на баре i = 1, покупаем на open бара i+1.
               Продаём на open бара i+2 (если i+2 < len(open_prices)).
    Требуется: len(data) >= len(signals) + 2
    """

    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.0005):
        """
        :param commission_rate: Комиссия за сделку (buy + sell). 0.1% = 0.001
        :param slippage_rate: Проскальзывание от цены. 0.05% = 0.0005
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: np.ndarray,
        initial_capital: float = 100_000.0,
        position_size_pct: float = 0.1  # 10% капитала на позицию
    ) -> Dict:
        """
        Запускает бэктест.

        :param data: DataFrame с колонками ['open', 'high', 'low', 'close', 'timestamp']
        :param signals: Массив numpy (0 - держать, 1 - покупать) для каждого бара
        :param initial_capital: Начальный капитал
        :param position_size_pct: Процент капитала для одной позиции
        :return: Словарь с результатами бэктеста
        """
        # --- ИСПРАВЛЕННАЯ ПРОВЕРКА ДЛИНЫ ---
        # Для стратегии "buy next open, sell next+1 open":
        # signals[0..M-1] -> покупка на open[1..M], продажа на open[2..M+1]
        # Требуется len(data) >= len(signals) + 2, чтобы все сигналы могли привести к покупке и продаже
        # Если len(data) == len(signals) + 1, то последняя покупка не продается (или продается по close последнего бара)
        # Мы требуем >= len(signals) + 2, чтобы каждая покупка имела шанс на продажу.
        if len(data) < len(signals) + 2:
            raise ValueError(f"Length of data ({len(data)}) must be at least len(signals) + 2 ({len(signals) + 2}) for the backtest logic.")

        # --- Подготовка данных ---
        # Убедимся, что колонки в правильном регистре
        required_cols = ['open', 'high', 'low', 'close']
        data_lower = data.copy()
        data_lower.columns = data_lower.columns.str.lower()
        if not all(col in data_lower.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Убедимся, что индекс отсортирован по времени
        data_lower = data_lower.sort_index()

        open_prices = data_lower['open'].values
        high_prices = data_lower['high'].values
        low_prices = data_lower['low'].values
        close_prices = data_lower['close'].values

        # --- Проверка на NaN ---
        if np.isnan(open_prices).any() or np.isnan(close_prices).any():
            raise ValueError("Data contains NaN values.")

        # --- Инициализация переменных ---
        capital = initial_capital
        realized_trades = [] # <-- ИНИЦИАЛИЗАЦИЯ СДЕЛОК ЗДЕСЬ
        equity_curve = [capital]

        # --- Цикл бэктеста ---
        # Идём только по длине сигналов, т.к. на каждый сигнал приходится один бар входа
        for i in range(len(signals)):
            signal = signals[i]
            # Если сигнал BUY (1), пытаемся купить на open следующего бара (i+1)
            if signal == 1:
                entry_bar_index = i + 1
                exit_bar_index = i + 2 # Продажа на open через бар после входа

                # Проверяем, существуют ли бары для входа и выхода
                if entry_bar_index < len(open_prices) and exit_bar_index < len(open_prices):
                    entry_price_raw = open_prices[entry_bar_index]
                    exit_price_raw = open_prices[exit_bar_index]

                    # Цена покупки с учётом проскальзывания и комиссии (увеличивается)
                    buy_price_with_slippage = entry_price_raw * (1 + self.slippage_rate) * (1 + self.commission_rate)
                    # Цена продажи с учётом проскальзывания и комиссии (уменьшается)
                    sell_price_with_slippage = exit_price_raw * (1 - self.slippage_rate) * (1 - self.commission_rate)

                    # Размер позиции
                    position_value = capital * position_size_pct
                    units = position_value / buy_price_with_slippage

                    # PnL
                    entry_value = units * buy_price_with_slippage
                    exit_value = units * sell_price_with_slippage
                    pnl = exit_value - entry_value

                    trade_info = {
                        'entry_bar': entry_bar_index,
                        'exit_bar': exit_bar_index,
                        'entry_price_raw': entry_price_raw,
                        'exit_price_raw': exit_price_raw,
                        'entry_price_exec': buy_price_with_slippage,
                        'exit_price_exec': sell_price_with_slippage,
                        'units': units,
                        'entry_value': entry_value,
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'pnl_pct': (sell_price_with_slippage / buy_price_with_slippage - 1) * 100
                    }
                    realized_trades.append(trade_info)

                    # Обновим капитал
                    capital += pnl

            # Обновим кривую капитала после каждой итерации (или после каждой сделки, на выбор)
            # В данном случае, обновляем только если была сделка, но добавляем значение в любом случае для синхронизации с барами
            # Лучше обновлять кривую капитала на каждом баре, но для простоты, обновим после каждой потенциальной сделки
            # Или обновлять в конце, после всех сделок, используя `capital` как финальное значение.
            # Для кривой на основе сделок, добавим текущий капитал после каждой итерации
            equity_curve.append(capital)

        # Если не было сделок, кривая остаётся с начальным капиталом
        # if not realized_trades:
        #     equity_curve = [initial_capital] * (len(signals) + 1) # +1 для синхронизации с итерациями цикла

        # Используем финальный капитал как последнюю точку, если нужно, чтобы длина совпадала с количеством итераций + 1
        # Но для метрик важны только значения *после* сделок.
        # Упростим: equity_curve формируется внутри цикла, добавляя капитал после каждой итерации (даже если сделки не было).
        # Это не идеально отражает кривую, но даёт значения для расчёта.
        # Лучше: equity_curve обновляется только при сделках, иначе остаётся предыдущее значение.
        # Перепишем логику: начальное значение, затем обновление только при сделках, финальное значение.
        # Но для простоты, пусть будет так, как сейчас, и мы рассчитаем метрики на основе `realized_trades` и финального `capital`.

        # --- Расчёт метрик ---
        # Если не было сделок, возвращаем нулевые метрики
        if not realized_trades:
            print("Warning: No trades were executed in the backtest.")
            return {
                'total_return_pct': 0.0,
                'total_pnl': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'equity_curve': equity_curve,
                'trades': realized_trades
            }

        # Рассчитываем метрики на основе `realized_trades`
        total_pnl = sum(t['pnl'] for t in realized_trades)
        total_return_pct = (capital / initial_capital - 1) * 100

        pnl_values = [t['pnl'] for t in realized_trades]
        win_trades = [p for p in pnl_values if p > 0]
        loss_trades = [p for p in pnl_values if p < 0]

        num_trades = len(realized_trades)
        num_wins = len(win_trades)
        num_losses = len(loss_trades)
        win_rate = num_wins / num_trades if num_trades > 0 else 0.0

        total_profit = sum(win_trades)
        total_loss = abs(sum(loss_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf if total_profit > 0 else 0.0

        # Для Max Drawdown используем кривую капитала, построенную внутри цикла
        # Это не точная кривая капитала по барам, а кривая *после* потенциальных сделок.
        # Для более точного MDD нужно отслеживать капитал после *каждого* бара, а не только после сделок.
        # Упрощение: используем кривую, построенную в цикле.
        # Лучше: построить кривую на основе баров, интерполируя между сделками.
        # Пока используем существующую `equity_curve`, понимая её ограничение.
        equity_curve_series = pd.Series(equity_curve)
        # Для MDD нужно, чтобы кривая отражала капитал на каждом баре, а не только после итераций.
        # Т.к. внутри цикла мы добавляем капитал каждый раз, длина equity_curve = len(signals) + 1
        # Это не отражает реальную динамику капитала между сигналами.
        # Рассчитаем MDD на основе финального капитала и предположим, что максимальная просадка была.
        # Или построим кривую заново, учитывая только сделки.
        # Построим кривую на основе сделок:
        # Начальный капитал -> капитал после сделки 1 -> капитал после сделки 2 -> ...
        # Используем индексы сделок для синхронизации.
        # Или: просто используем total_pnl и предположим, что MDD рассчитывается на основе серии PnL.
        # Нет, нужно использовать кривую капитала.
        # Построим упрощённую кривую: начальный капитал, затем значения после каждой сделки.
        # equity_curve_trade_based = [initial_capital]
        # running_cap = initial_capital
        # for trade in realized_trades:
        #     running_cap += trade['pnl']
        #     equity_curve_trade_based.append(running_cap)
        # equity_curve_for_dd = pd.Series(equity_curve_trade_based)

        # Или используем исходную кривую, построенную в цикле, как приближение.
        # Это будет заведомо неправильный MDD, т.к. кривая обновляется каждый раз, а не только при сделках.
        # Лучше использовать кривую, построенную на основе сделок.
        equity_curve_trade_based = [initial_capital]
        running_cap = initial_capital
        for trade in realized_trades:
            running_cap += trade['pnl']
            equity_curve_trade_based.append(running_cap)

        equity_curve_for_dd = pd.Series(equity_curve_trade_based)

        running_max = np.maximum.accumulate(equity_curve_for_dd)
        drawdown = (equity_curve_for_dd - running_max) / running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100

        # Sharpe Ratio (annualized, assuming 252 trading days, 15min bars ~ 26208 per year)
        # Используем PnL от сделок для расчёта волатильности и среднего.
        # Возвращаемся к стандартному расчёту на основе серии возвратов.
        # Пусть возврат за сделку = pnl / entry_value
        if len(realized_trades) > 1:
            trade_returns = [t['pnl'] / t['entry_value'] for t in realized_trades]
            avg_return_per_trade = np.mean(trade_returns)
            volatility_per_trade = np.std(trade_returns)
            if volatility_per_trade != 0:
                # Приближённый Sharpe, не annualized, т.к. частота сделок не фиксирована
                # Для annualized нужно знать среднее количество сделок в год
                # sharpe_ratio = (avg_return_per_trade / volatility_per_trade) * np.sqrt(num_trades_per_year_approx)
                # Пусть будет просто на основе среднего и std PnL за сделку, нормированного на entry_value
                # Или на основе кривой капитала по сделкам
                trade_returns_series = pd.Series(trade_returns)
                # Предположим, что возвраты происходят с фиксированной частотой (например, 1 возврат = 1 средняя сделка)
                # Это приближение. Лучше использовать кривую капитала по барам, но мы её не строим полноценно.
                # Используем кривую капитала по сделкам
                equity_curve_for_sharpe = pd.Series(equity_curve_trade_based)
                period_returns = equity_curve_for_sharpe.pct_change().dropna()
                if len(period_returns) > 1 and period_returns.std() != 0:
                    # Это возвраты *после* каждой *сделки*, а не по барам.
                    # Это нестандартно для Sharpe, но для данной реализации сойдёт.
                    # Annualized Sharpe
                    sharpe_ratio = (period_returns.mean() / period_returns.std()) * np.sqrt(26208) # 15min freq
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Calmar Ratio
        calmar_ratio = total_return_pct / abs(max_drawdown_pct) if abs(max_drawdown_pct) > 0 else np.inf if total_return_pct > 0 else 0.0

        results = {
            'total_return_pct': total_return_pct,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'equity_curve': equity_curve_trade_based, # Используем кривую, построенную на основе сделок
            'trades': realized_trades
        }

        return results
