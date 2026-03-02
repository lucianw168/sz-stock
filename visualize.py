"""Visualization module: K-line charts, equity curves, drawdown plots."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False


def plot_candlestick(stock_data: dict[str, pd.DataFrame],
                     code: str,
                     tail: int = 40) -> None:
    """Multi-panel candlestick chart.

    Panel 0: Price candles
    Panel 1: ADX / ADXR / +DM / -DM
    Panel 2: CRSI / RSI-6 / RSI-12
    Panel 3: OBV
    """
    if not HAS_MPLFINANCE:
        print("mplfinance not installed. Run: pip install mplfinance")
        return

    if code not in stock_data:
        print(f"Stock code {code} not found in data.")
        return

    df = stock_data[code].tail(tail)

    addplots = [
        mpf.make_addplot(df['ADX'], color='green', panel=1),
        mpf.make_addplot(df['ADXR'], color='orange', panel=1),
        mpf.make_addplot(df['-DM'], color='blue', panel=1),
        mpf.make_addplot(df['+DM'], color='red', panel=1),
        mpf.make_addplot(df['CRSI'], color='purple', panel=2),
        mpf.make_addplot(df['Rsi'], color='orange', panel=2),
        mpf.make_addplot(df['Rsi12'], color='blue', panel=2),
        mpf.make_addplot(df['OBV'], color='red', panel=3),
    ]

    mpf.plot(
        df, type='candle', style='sas',
        title=f'{code} Candlestick Chart',
        ylabel='Price',
        volume=False,
        addplot=addplots,
        panel_ratios=(3, 1, 2, 2),
    )


def plot_equity_curve(results: pd.DataFrame) -> None:
    """Plot cumulative equity curve from backtest results."""
    if results.empty:
        print("No trades to plot.")
        return

    daily = results.groupby('buy_date')['pnl_pct'].mean()
    cumulative = (1 + daily).cumprod()

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(cumulative)), cumulative.values, 'b-', linewidth=1.5)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Equity Curve')
    plt.xlabel('Trading Day')
    plt.ylabel('Cumulative Return')
    plt.xticks(range(0, len(cumulative), max(1, len(cumulative) // 10)),
               [cumulative.index[i] for i in range(0, len(cumulative), max(1, len(cumulative) // 10))],
               rotation=45)
    plt.tight_layout()
    plt.show()


def plot_drawdown(results: pd.DataFrame) -> None:
    """Plot drawdown chart from backtest results."""
    if results.empty:
        print("No trades to plot.")
        return

    daily = results.groupby('buy_date')['pnl_pct'].mean()
    cumulative = (1 + daily).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    plt.figure(figsize=(12, 4))
    plt.fill_between(range(len(drawdown)), drawdown.values, 0, color='red', alpha=0.3)
    plt.plot(range(len(drawdown)), drawdown.values, 'r-', linewidth=1)
    plt.title('Drawdown')
    plt.xlabel('Trading Day')
    plt.ylabel('Drawdown')
    plt.xticks(range(0, len(drawdown), max(1, len(drawdown) // 10)),
               [drawdown.index[i] for i in range(0, len(drawdown), max(1, len(drawdown) // 10))],
               rotation=45)
    plt.tight_layout()
    plt.show()
