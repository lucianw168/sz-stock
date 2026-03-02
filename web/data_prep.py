"""Data preparation: parquet -> dict/JSON for Jinja2 templates."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd

import config
import downloader
import indicators
import screener
from backtest import BacktestEngine


# Strategy descriptions: which signals compose each screen
SCREEN_DESCRIPTIONS = {
    '超卖': 'CCI 穿越 -100 + 底部逃逸 + CCI 深度超卖 + 日涨幅 < 3%',
    '当日金叉买入': 'UOS 穿越 65 + 非涨停 + RSI 金叉',
    '底部异动': '成交量激增 + 底部逃逸',
    '震荡指标': 'ADX 穿越 ADXR + 底部逃逸',
    '近日上涨': 'ADX 穿越 ADXR + +DM > -DM + 非涨停',
    'CCI快进出': 'CCI 穿越 -100 + 底部逃逸 + CCI 深度超卖 + 日涨幅 < 3%',
    'OBV涨停梦': 'OBV 突破 + 成交量激增 + CCI > 50 + 日涨幅 > 2%（仅主板）',
    'OBV底部突破': 'OBV 突破 + 成交量激增 + 底部逃逸',
    '形态识别': '横盘缩量涨停延迟回收 — AI从历史涨停事件学习的多日形态模式，'
               '横盘→缩量涨停→阴线→第3天回收确认 = 买入信号',
}


def load_stock_data():
    """Load processed parquet and compute indicators.

    Returns:
        dict[str, DataFrame] with indicators computed, or None if no data.
    """
    stock_data = downloader.load_processed()
    if not stock_data:
        return None

    print("Computing indicators for web...")
    for code in list(stock_data.keys()):
        stock_data[code] = indicators.compute_all(stock_data[code])

    return stock_data


def run_screen_for_date(stock_data, date):
    """Run all screens for a given date.

    Returns:
        dict[str, list[str]] mapping screen name -> list of stock codes.
    """
    results = {}
    for name, func in screener.ALL_SCREENS.items():
        codes = func(stock_data, date)
        results[name] = codes
    return results


def get_pattern_signals_for_date(stock_data, date):
    """Get detailed pattern signal info for a given date.

    Returns:
        list of dicts with pattern match details (code, explanation, confidence, etc.)
    """
    from pattern_screen import get_latest_pattern_signals
    signals = get_latest_pattern_signals()
    # Filter to signals matching this date
    date_ts = date.replace('-', '')
    return [s for s in signals if s.get('signal_date') == date_ts]


def list_available_dates(stock_data):
    """Get sorted list of all trading dates from the data.

    Returns:
        list[str] of date strings.
    """
    sample = next(iter(stock_data.values()))
    dates = sorted(sample.index.strftime('%Y-%m-%d').tolist())
    return dates


def get_stock_indicators(stock_data, code, date):
    """Extract key indicator values for a stock on a given date.

    Returns:
        dict with indicator values, or None if data unavailable.
    """
    if code not in stock_data:
        return None
    df = stock_data[code]
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return None

    row = df.loc[ts]
    prev_close = df['Close'].shift(1).loc[ts] if ts in df.index else None
    change_pct = ((row['Close'] - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0

    return {
        'code': code,
        'close': round(float(row['Close']), 2),
        'change_pct': round(float(change_pct), 2),
        'volume': int(row['Volume']),
        'rsi6': round(float(row.get('Rsi', 0)), 1) if pd.notna(row.get('Rsi')) else '-',
        'rsi12': round(float(row.get('Rsi12', 0)), 1) if pd.notna(row.get('Rsi12')) else '-',
        'crsi': round(float(row.get('CRSI', 0)), 1) if pd.notna(row.get('CRSI')) else '-',
        'cci': round(float(row.get('CCI', 0)), 1) if pd.notna(row.get('CCI')) else '-',
        'obv': int(row.get('OBV', 0)) if pd.notna(row.get('OBV')) else '-',
        'adx': round(float(row.get('ADX', 0)), 1) if pd.notna(row.get('ADX')) else '-',
        'adxr': round(float(row.get('ADXR', 0)), 1) if pd.notna(row.get('ADXR')) else '-',
        'vol_7d': round(float(row.get('Volume percentage 7d', 0)), 1) if pd.notna(row.get('Volume percentage 7d')) else '-',
    }


def prepare_candlestick_data(stock_data, code, tail=60):
    """Convert stock data to ECharts K-line format.

    Returns:
        dict with dates, ohlc, volume, and indicator arrays for ECharts.
    """
    if code not in stock_data:
        return None
    df = stock_data[code].tail(tail).copy()

    dates = df.index.strftime('%Y-%m-%d').tolist()
    ohlc = df[['Open', 'Close', 'Low', 'High']].values.tolist()
    ohlc = [[round(v, 2) for v in row] for row in ohlc]
    volume = df['Volume'].fillna(0).astype(int).tolist()

    def safe_list(col):
        if col in df.columns:
            return [round(float(v), 2) if pd.notna(v) else None for v in df[col]]
        return [None] * len(dates)

    return {
        'dates': dates,
        'ohlc': ohlc,
        'volume': volume,
        'adx': safe_list('ADX'),
        'adxr': safe_list('ADXR'),
        'plus_dm': safe_list('+DM'),
        'minus_dm': safe_list('-DM'),
        'crsi': safe_list('CRSI'),
        'rsi6': safe_list('Rsi'),
        'rsi12': safe_list('Rsi12'),
        'obv': [int(v) if pd.notna(v) else None for v in df.get('OBV', pd.Series([0]*len(df)))],
    }


def prepare_equity_curve(daily_details):
    """Convert backtest daily details to ECharts equity curve format.

    Args:
        daily_details: DataFrame from BacktestEngine.generate_report()['daily_details']

    Returns:
        dict with dates, equity, drawdown arrays.
    """
    if daily_details is None or daily_details.empty:
        return {'dates': [], 'equity': [], 'drawdown': []}

    dates = daily_details.index.tolist()
    returns = daily_details['return']
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max)

    return {
        'dates': dates,
        'equity': [round(float(v), 4) for v in equity],
        'drawdown': [round(float(v), 4) for v in drawdown],
    }


def run_all_backtests(stock_data, days):
    """Run backtests for all 8 strategies.

    Args:
        stock_data: dict[str, DataFrame] with indicators.
        days: list[str] of trading day strings.

    Returns:
        dict[str, dict] mapping screen name -> {report, results, equity_data}.
    """
    engine = BacktestEngine()
    all_reports = {}

    for name, screen_func in screener.ALL_SCREENS.items():
        print(f"  Backtesting: {name}...")
        results = engine.run(screen_func, stock_data, days, target_pct=config.DEFAULT_TARGET_PCT)
        report = engine.generate_report(results)
        equity_data = prepare_equity_curve(report.get('daily_details'))

        # Build trade detail list for the template
        trade_details = []
        if not results.empty:
            for _, row in results.iterrows():
                # Compute actual close-to-close return (not backtest simulation)
                code = row['code']
                buy_date = row['buy_date']
                sell_date = row['sell_date']
                actual_return = 0.0
                buy_close = 0.0
                sell_close = 0.0
                if code in stock_data:
                    df = stock_data[code]
                    buy_ts = pd.Timestamp(buy_date)
                    sell_ts = pd.Timestamp(sell_date)
                    if buy_ts in df.index and sell_ts in df.index:
                        buy_close = float(df.loc[buy_ts, 'Close'])
                        sell_close = float(df.loc[sell_ts, 'Close'])
                        if buy_close > 0:
                            actual_return = (sell_close / buy_close - 1) * 100

                trade_details.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'code': code,
                    'buy_price': round(buy_close, 2),
                    'sell_price': round(sell_close, 2),
                    'pnl_pct': round(actual_return, 2),
                    'hit_target': bool(row['hit_target']),
                    'hit_limit_up': bool(row['hit_limit_up']),
                })

        all_reports[name] = {
            'report': {
                'total_trades': report['total_trades'],
                'win_rate': round(report['win_rate'] * 100, 1),
                'limit_up_rate': round(report['limit_up_rate'] * 100, 1),
                'profit_factor': round(report['profit_factor'], 2),
                'cumulative_return': round(report['cumulative_return'] * 100, 1),
                'annualized_return': round(report['annualized_return'] * 100, 1),
                'max_drawdown': round(report['max_drawdown'] * 100, 1),
                'sharpe_ratio': round(report['sharpe_ratio'], 2),
                'sortino_ratio': round(report['sortino_ratio'], 2),
                'total_pnl': round(report['total_pnl'], 2),
            },
            'equity_data': equity_data,
            'trade_details': trade_details,
        }

    return all_reports
