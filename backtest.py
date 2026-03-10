"""Backtest engine with comprehensive performance metrics.

Major enhancement over the original notebook which only had simple profit/loss.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from universe import get_limit_ratio


class BacktestEngine:
    """Backtest engine for strategy evaluation.

    Buy at close price on signal day.
    Sell: if next-day high hits target, sell at target price; otherwise sell at next-day close.
    Includes commission and slippage.
    """

    def __init__(self,
                 commission: float = None,
                 slippage: float = None,
                 shares: int = None):
        self.commission = commission or config.COMMISSION_RATE
        self.slippage = slippage or config.SLIPPAGE_RATE
        self.shares = shares or config.SHARES_PER_TRADE

    def run(self,
            screen_func,
            stock_data: dict[str, pd.DataFrame],
            days: list[str],
            target_pct: float = None,
            stop_loss_pct: float = None) -> pd.DataFrame:
        """Run backtest for a given screen function over specified days.

        Args:
            screen_func: screening function(stock_data, date) -> list[str]
            stock_data: dict of code -> DataFrame with indicators
            days: list of trading day strings
            target_pct: target profit percentage (e.g., 0.03 for 3%)
            stop_loss_pct: optional stop-loss percentage (e.g., -0.05 for -5%).
                If next-day Low breaches stop price, sell at stop price.
                Priority: stop-loss checked first (using Low), then target (using High).

        Returns:
            DataFrame with daily trade records.
        """
        if target_pct is None:
            target_pct = config.DEFAULT_TARGET_PCT

        records = []

        for j in range(len(days) - 1):
            buy_date = days[j]
            sell_date = days[j + 1]

            # BUG FIX: original notebook used 'today' instead of days[j]
            # in the bottom_activity backtest
            selected = screen_func(stock_data, buy_date)

            for code in selected:
                df = stock_data[code]
                if pd.Timestamp(buy_date) not in df.index:
                    continue
                if pd.Timestamp(sell_date) not in df.index:
                    continue

                cost_price = df.loc[pd.Timestamp(buy_date), 'Close']
                target_price = cost_price * (1 + target_pct)
                next_high = df.loc[pd.Timestamp(sell_date), 'High']
                next_low = df.loc[pd.Timestamp(sell_date), 'Low']
                next_close = df.loc[pd.Timestamp(sell_date), 'Close']

                # Apply slippage to buy
                actual_buy = cost_price * (1 + self.slippage)

                # Determine sell price with stop-loss support
                hit_stop = False
                if stop_loss_pct is not None:
                    stop_price = cost_price * (1 + stop_loss_pct)
                    if next_low <= stop_price:
                        # Stop-loss triggered (assume hit before target)
                        actual_sell = stop_price * (1 - self.slippage)
                        hit_target = False
                        hit_stop = True

                if not hit_stop:
                    if next_high >= target_price:
                        actual_sell = target_price * (1 - self.slippage)
                        hit_target = True
                    else:
                        actual_sell = next_close * (1 - self.slippage)
                        hit_target = False

                # Check limit-up hit
                ratio = get_limit_ratio(code)
                limit_up_price = round(cost_price * (1 + ratio), 2)
                hit_limit_up = next_high >= limit_up_price * (1 - config.LIMIT_TOLERANCE)

                # Compute P&L
                capital = actual_buy * self.shares
                commission_cost = capital * self.commission * 2  # buy + sell
                gross_pnl = (actual_sell - actual_buy) * self.shares
                net_pnl = gross_pnl - commission_cost
                pnl_pct = net_pnl / capital if capital > 0 else 0

                records.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'code': code,
                    'buy_price': actual_buy,
                    'sell_price': actual_sell,
                    'target_price': target_price,
                    'hit_target': hit_target,
                    'hit_limit_up': hit_limit_up,
                    'capital': capital,
                    'gross_pnl': gross_pnl,
                    'commission': commission_cost,
                    'net_pnl': net_pnl,
                    'pnl_pct': pnl_pct,
                })

        return pd.DataFrame(records)

    def generate_report(self, results: pd.DataFrame) -> dict:
        """Generate comprehensive performance report with per-day aggregation.

        Each trading day is treated independently: daily capital, daily P&L,
        and daily return are computed, then summary stats are derived from
        the daily return series.
        """
        empty_report = {
            'total_trades': 0,
            'win_rate': 0,
            'limit_up_rate': 0,
            'profit_factor': 0,
            'cumulative_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'total_pnl': 0,
            'avg_daily_capital': 0,
            'daily_details': pd.DataFrame(),
        }
        if results.empty:
            return empty_report

        # --- Per-day aggregation ---
        grouped = results.groupby('buy_date')
        daily = pd.DataFrame({
            'capital': grouped['capital'].sum(),
            'pnl': grouped['net_pnl'].sum(),
            'trades': grouped['net_pnl'].count(),
            'wins': grouped['net_pnl'].apply(lambda x: (x > 0).sum()),
            'limit_ups': grouped['hit_limit_up'].sum(),
            'codes': grouped['code'].apply(lambda x: ','.join(x)),
        })
        daily['win_rate'] = daily['wins'] / daily['trades']
        daily['return'] = daily['pnl'] / daily['capital']
        daily.index.name = 'date'

        # --- Summary stats from daily return series ---
        daily_returns = daily['return']
        n_days = len(daily_returns)

        total_trades = int(daily['trades'].sum())
        total_wins = int(daily['wins'].sum())
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        total_limit_ups = int(daily['limit_ups'].sum())
        limit_up_rate = total_limit_ups / total_trades if total_trades > 0 else 0

        total_profit = results.loc[results['net_pnl'] > 0, 'net_pnl'].sum()
        total_loss = abs(results.loc[results['net_pnl'] <= 0, 'net_pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        total_pnl = results['net_pnl'].sum()
        avg_daily_capital = daily['capital'].mean()

        # Cumulative return via compounding daily returns
        cumulative_return = (1 + daily_returns).prod() - 1

        # Annualized return (compound, ~250 trading days/year)
        if n_days > 0:
            annualized_return = (1 + cumulative_return) ** (250 / n_days) - 1
        else:
            annualized_return = 0

        # Sharpe / Sortino from daily return series
        if n_days > 1:
            mean_ret = daily_returns.mean()
            std_ret = daily_returns.std()
            sharpe_ratio = (mean_ret / std_ret * np.sqrt(250)) if std_ret > 0 else 0

            downside = daily_returns[daily_returns < 0]
            down_std = downside.std() if len(downside) > 1 else 0
            sortino_ratio = (mean_ret / down_std * np.sqrt(250)) if down_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Max drawdown from cumulative equity curve
        equity = (1 + daily_returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'limit_up_rate': limit_up_rate,
            'profit_factor': profit_factor,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_pnl': total_pnl,
            'avg_daily_capital': avg_daily_capital,
            'daily_details': daily,
        }


def print_report(report: dict) -> None:
    """Pretty-print a backtest report with per-day details."""
    # Daily details
    daily = report.get('daily_details')
    if daily is not None and not daily.empty:
        print()
        for date, row in daily.iterrows():
            print(f"  {date}  本金:{row['capital']:.2f}  "
                  f"收益:{row['pnl']:.2f}  "
                  f"胜率:{row['win_rate']:.1%}  "
                  f"收益率:{row['return']:.2%}  "
                  f"涨停:{int(row['limit_ups'])}/{int(row['trades'])}  "
                  f"({row['codes']})")

    # Summary
    print("\n" + "=" * 50)
    print("  BACKTEST REPORT")
    print("=" * 50)
    print(f"  Total Trades:       {report['total_trades']}")
    print(f"  Win Rate:           {report['win_rate']:.2%}")
    print(f"  Limit-Up Rate:      {report['limit_up_rate']:.2%}")
    print(f"  Profit Factor:      {report['profit_factor']:.2f}")
    print(f"  Cumulative Return:  {report['cumulative_return']:.2%}")
    print(f"  Annualized Return:  {report['annualized_return']:.2%}")
    print(f"  Max Drawdown:       {report['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:       {report['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:      {report['sortino_ratio']:.2f}")
    print(f"  Total P&L:          {report['total_pnl']:.2f}")
    print(f"  Avg Daily Capital:  {report['avg_daily_capital']:.2f}")
    print("=" * 50 + "\n")
