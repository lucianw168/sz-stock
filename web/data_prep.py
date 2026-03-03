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
import strategies as sig
from backtest import BacktestEngine


# Strategy descriptions: which signals compose each screen
SCREEN_DESCRIPTIONS = {
    'CCI快进出': 'CCI 穿越 -100 + 底部逃逸 + CCI 深度超卖 + 日涨幅 < 3% + 价格贴近EMA5',
    'OBV涨停梦': 'OBV 突破 + 成交量激增 + CCI > 50 + 日涨幅 > 2%（仅主板）',
    'OBV波段': 'OBV 突破 + 成交量激增 + CCI > 50 + 日涨幅 > 2% + ADX < 40 + 底部逃逸',
    '形态识别': '横盘缩量涨停延迟回收 — AI从历史涨停事件学习的多日形态模式，'
               '横盘→缩量涨停→阴线→第3天回收确认 = 买入信号',
}

TRADING_METHODS = {
    'CCI快进出': '信号日收盘买入 → 次日高点达5%目标价卖出，否则次日收盘卖出',
    'OBV涨停梦': '信号日收盘买入 → 次日高点达5%目标价卖出，否则次日收盘卖出',
    'OBV波段': '信号日收盘买入 → 次日高点达5%目标价卖出，否则次日收盘卖出',
    '形态识别': '信号确认后择机买入（不限于当日收盘），持有至目标价或止损',
}

# Maps strategy name -> list of signal function names used in that strategy
STRATEGY_SIGNALS = {
    'CCI快进出': [
        'signal_cci_cross_neg100',
        'signal_escape_bottom',
        'signal_cci_deep_oversold',
        'signal_daily_gain_lt3',
        'signal_ema5_proximity',
    ],
    'OBV涨停梦': [
        'signal_obv_breakout',
        'signal_volume_surge',
        'signal_cci_momentum_floor',
        'signal_daily_gain_gt2',
    ],
    'OBV波段': [
        'signal_obv_breakout',
        'signal_volume_surge',
        'signal_cci_momentum_floor',
        'signal_daily_gain_gt2',
        'signal_adx_below_max',
        'signal_escape_bottom',
    ],
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


def get_signal_diagnostics(stock_data, code, date, strategy_name):
    """Extract actual indicator values vs thresholds for each signal in a strategy.

    Returns:
        dict[signal_name -> {label, items: [{name, value, threshold, op}]}]
        or empty dict if strategy not in STRATEGY_SIGNALS.
    """
    if strategy_name not in STRATEGY_SIGNALS:
        return {}
    if code not in stock_data:
        return {}

    df = stock_data[code]
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return {}

    idx = df.index.get_loc(ts)
    result = {}

    for sig_name in STRATEGY_SIGNALS[strategy_name]:
        items = []

        if sig_name == 'signal_cci_cross_neg100':
            cci_today = float(df['CCI'].iloc[idx]) if pd.notna(df['CCI'].iloc[idx]) else None
            cci_yest = float(df['CCI'].iloc[idx - 1]) if idx > 0 and pd.notna(df['CCI'].iloc[idx - 1]) else None
            items = [
                {'name': 'CCI(今日)', 'value': round(cci_today, 1) if cci_today is not None else '-',
                 'threshold': str(config.CCI_CROSS_LEVEL), 'op': '>'},
                {'name': 'CCI(昨日)', 'value': round(cci_yest, 1) if cci_yest is not None else '-',
                 'threshold': str(config.CCI_CROSS_LEVEL), 'op': '<'},
            ]
            result[sig_name] = {'label': 'CCI穿越-100', 'items': items}

        elif sig_name == 'signal_escape_bottom':
            if idx >= config.ESCAPE_BOTTOM_SHORT_WINDOW and idx >= config.ESCAPE_BOTTOM_LONG_WINDOW:
                short_pct = abs(float(df['Close'].iloc[idx]) / float(df['Close'].iloc[idx - config.ESCAPE_BOTTOM_SHORT_WINDOW]) - 1)
                long_pct = float(df['Close'].iloc[idx]) / float(df['Close'].iloc[idx - config.ESCAPE_BOTTOM_LONG_WINDOW]) - 1
            else:
                short_pct = None
                long_pct = None
            items = [
                {'name': '15日距离%', 'value': f'{short_pct * 100:.1f}%' if short_pct is not None else '-',
                 'threshold': f'{config.ESCAPE_BOTTOM_SHORT_PCT * 100:.0f}%', 'op': '<='},
                {'name': '55日深度%', 'value': f'{long_pct * 100:.1f}%' if long_pct is not None else '-',
                 'threshold': f'{config.ESCAPE_BOTTOM_LONG_PCT * 100:.0f}%', 'op': '<='},
            ]
            result[sig_name] = {'label': '底部逃逸', 'items': items}

        elif sig_name == 'signal_cci_deep_oversold':
            def _cci(offset):
                i = idx - offset
                if 0 <= i < len(df) and pd.notna(df['CCI'].iloc[i]):
                    return round(float(df['CCI'].iloc[i]), 1)
                return '-'
            items = [
                {'name': 'CCI(t-1)', 'value': _cci(1), 'threshold': str(config.CCI_CROSS_LEVEL), 'op': '<'},
                {'name': 'CCI(t-2)', 'value': _cci(2), 'threshold': str(config.CCI_DEEP_LEVEL), 'op': '<'},
                {'name': 'CCI(t-3)', 'value': _cci(3), 'threshold': str(config.CCI_DEEP_LEVEL), 'op': '<'},
                {'name': 'CCI(t-7)', 'value': _cci(6), 'threshold': f'{config.CCI_DEEP_BEFORE7D_LOWER}~{config.CCI_DEEP_BEFORE7D_UPPER}', 'op': '区间'},
            ]
            result[sig_name] = {'label': 'CCI深度超卖', 'items': items}

        elif sig_name == 'signal_daily_gain_lt3':
            prev_close = float(df['Close'].iloc[idx - 1]) if idx > 0 else None
            cur_close = float(df['Close'].iloc[idx])
            gain = (cur_close / prev_close - 1) * 100 if prev_close and prev_close > 0 else None
            items = [
                {'name': '日涨幅%', 'value': f'{gain:.2f}%' if gain is not None else '-',
                 'threshold': f'{config.DAILY_GAIN_LT3_PCT * 100:.0f}%', 'op': '<'},
            ]
            result[sig_name] = {'label': '日涨幅<3%', 'items': items}

        elif sig_name == 'signal_ema5_proximity':
            close = float(df['Close'].iloc[idx])
            ema5 = float(df['EMA'].iloc[idx]) if pd.notna(df['EMA'].iloc[idx]) else None
            prox = abs(close / ema5 - 1) * 100 if ema5 and ema5 > 0 else None
            items = [
                {'name': '|Close/EMA5-1|%', 'value': f'{prox:.2f}%' if prox is not None else '-',
                 'threshold': f'{config.EMA5_PROXIMITY_PCT * 100:.0f}%', 'op': '<'},
                {'name': 'EMA5', 'value': round(ema5, 2) if ema5 is not None else '-'},
            ]
            result[sig_name] = {'label': '价格贴近EMA5', 'items': items}

        elif sig_name == 'signal_obv_breakout':
            obv = float(df['OBV'].iloc[idx]) if pd.notna(df['OBV'].iloc[idx]) else None
            window = config.OBV_CONSOLIDATION_WINDOW
            if idx >= window:
                obv_max_30 = float(df['OBV'].iloc[idx - window:idx].max())
            else:
                obv_max_30 = None
            items = [
                {'name': 'OBV', 'value': int(obv) if obv is not None else '-'},
                {'name': f'{window}日最高OBV', 'value': int(obv_max_30) if obv_max_30 is not None else '-'},
                {'name': '突破量', 'value': int(obv - obv_max_30) if obv is not None and obv_max_30 is not None else '-',
                 'threshold': '> 0', 'op': '>'},
            ]
            result[sig_name] = {'label': 'OBV突破', 'items': items}

        elif sig_name == 'signal_volume_surge':
            vol_7d = float(df['Volume percentage 7d'].iloc[idx]) if pd.notna(df['Volume percentage 7d'].iloc[idx]) else None
            vol_3d = float(df['Volume percentage 3d'].iloc[idx]) if pd.notna(df['Volume percentage 3d'].iloc[idx]) else None
            items = [
                {'name': '量比7d%', 'value': f'{vol_7d:.1f}' if vol_7d is not None else '-',
                 'threshold': f'{config.VOLUME_7D_LOWER}~{config.VOLUME_7D_UPPER}', 'op': '区间'},
                {'name': '量比3d%', 'value': f'{vol_3d:.1f}' if vol_3d is not None else '-',
                 'threshold': str(config.VOLUME_3D_UPPER), 'op': '<'},
            ]
            result[sig_name] = {'label': '成交量激增', 'items': items}

        elif sig_name == 'signal_cci_momentum_floor':
            cci = float(df['CCI'].iloc[idx]) if pd.notna(df['CCI'].iloc[idx]) else None
            items = [
                {'name': 'CCI', 'value': round(cci, 1) if cci is not None else '-',
                 'threshold': str(config.CCI_MOMENTUM_FLOOR), 'op': '>'},
            ]
            result[sig_name] = {'label': 'CCI动量底线', 'items': items}

        elif sig_name == 'signal_daily_gain_gt2':
            prev_close = float(df['Close'].iloc[idx - 1]) if idx > 0 else None
            cur_close = float(df['Close'].iloc[idx])
            gain = (cur_close / prev_close - 1) * 100 if prev_close and prev_close > 0 else None
            items = [
                {'name': '日涨幅%', 'value': f'{gain:.2f}%' if gain is not None else '-',
                 'threshold': f'{config.DAILY_GAIN_GT2_PCT * 100:.0f}%', 'op': '>'},
            ]
            result[sig_name] = {'label': '日涨幅>2%', 'items': items}

        elif sig_name == 'signal_adx_below_max':
            adx = float(df['ADX'].iloc[idx]) if pd.notna(df['ADX'].iloc[idx]) else None
            items = [
                {'name': 'ADX', 'value': round(adx, 1) if adx is not None else '-',
                 'threshold': str(config.ADX_BUY_MAX), 'op': '<'},
            ]
            result[sig_name] = {'label': 'ADX趋势过滤', 'items': items}

    return result


def compute_motive_label(stock_data, code, date):
    """Classify OBV strategy motive based on price-volume characteristics.

    Returns:
        str: one of '出货', '试盘', '吸筹', '待确认'
    """
    if code not in stock_data:
        return '待确认'

    df = stock_data[code]
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return '待确认'

    idx = df.index.get_loc(ts)
    row = df.iloc[idx]

    # Position within 55-day range (0% = low, 100% = high)
    if idx >= 55:
        high_55 = df['Close'].iloc[idx - 55:idx + 1].max()
        low_55 = df['Close'].iloc[idx - 55:idx + 1].min()
        if high_55 > low_55:
            position_pct = (float(row['Close']) - low_55) / (high_55 - low_55) * 100
        else:
            position_pct = 50
    else:
        position_pct = 50

    # Volume ratio vs 20-day average
    if idx >= 20:
        avg_vol_20 = df['Volume'].iloc[idx - 20:idx].mean()
        vol_ratio = float(row['Volume']) / avg_vol_20 if avg_vol_20 > 0 else 1
    else:
        vol_ratio = 1

    # Upper shadow ratio
    high = float(row['High'])
    close = float(row['Close'])
    open_ = float(row['Open'])
    low = float(row['Low'])
    body_top = max(close, open_)
    full_range = high - low
    upper_shadow_ratio = (high - body_top) / full_range if full_range > 0 else 0

    # Daily amplitude
    amplitude = (high - low) / float(row['Close']) * 100 if close > 0 else 0

    # OBV 5-day trend
    if idx >= 5:
        obv_5d_rising = float(df['OBV'].iloc[idx]) > float(df['OBV'].iloc[idx - 5])
    else:
        obv_5d_rising = False

    # Classification rules
    if position_pct >= 80 and vol_ratio >= 3 and upper_shadow_ratio > 0.3:
        return '出货'
    if vol_ratio < 2 and upper_shadow_ratio > 0.5:
        return '试盘'
    if position_pct <= 30 and amplitude < 3 and obv_5d_rising:
        return '吸筹'
    return '待确认'


def compute_failure_analysis(stock_data, trade_details, n=10):
    """Analyze the most recent losing trades for common failure patterns.

    Args:
        stock_data: dict[str, DataFrame]
        trade_details: list of trade detail dicts from run_all_backtests
        n: number of recent losses to analyze

    Returns:
        dict with failure pattern statistics, or None if no losses.
    """
    # Filter to losing trades
    losses = [t for t in trade_details if t['pnl_pct'] < 0]
    if not losses:
        return None

    # Take last n losses (most recent)
    recent = losses[-n:]

    long_upper_shadow = 0
    next_day_gap_down = 0
    high_volume_retreat = 0
    bearish_close = 0
    detail_trades = []

    for t in recent:
        code = t['code']
        buy_date = t['buy_date']
        sell_date = t['sell_date']

        if code not in stock_data:
            continue
        df = stock_data[code]
        buy_ts = pd.Timestamp(buy_date)
        sell_ts = pd.Timestamp(sell_date)

        if buy_ts not in df.index or sell_ts not in df.index:
            continue

        buy_row = df.loc[buy_ts]
        sell_row = df.loc[sell_ts]

        # Upper shadow ratio on buy day
        high = float(buy_row['High'])
        close = float(buy_row['Close'])
        open_ = float(buy_row['Open'])
        low = float(buy_row['Low'])
        body_top = max(close, open_)
        full_range = high - low
        upper_shadow = (high - body_top) / full_range if full_range > 0 else 0

        # Next day gap = sell_day open vs buy_day close
        gap_pct = (float(sell_row['Open']) / close - 1) * 100 if close > 0 else 0

        # Volume ratio on sell day
        sell_idx = df.index.get_loc(sell_ts)
        if sell_idx >= 7:
            avg_vol = df['Volume'].iloc[sell_idx - 7:sell_idx].mean()
            sell_vol_ratio = float(sell_row['Volume']) / avg_vol if avg_vol > 0 else 1
        else:
            sell_vol_ratio = 1
        sell_bearish = float(sell_row['Close']) < float(sell_row['Open'])

        flags = []
        if upper_shadow > 0.5:
            long_upper_shadow += 1
            flags.append('上影线过长')
        if gap_pct < -1:
            next_day_gap_down += 1
            flags.append('次日低开')
        if sell_vol_ratio > 1.5 and sell_bearish:
            high_volume_retreat += 1
            flags.append('放量回落')
        if sell_bearish:
            bearish_close += 1
            flags.append('收阴')

        detail_trades.append({
            'buy_date': buy_date,
            'sell_date': sell_date,
            'code': code,
            'pnl_pct': t['pnl_pct'],
            'upper_shadow': round(upper_shadow, 2),
            'gap_pct': round(gap_pct, 2),
            'sell_vol_ratio': round(sell_vol_ratio, 2),
            'flags': flags,
        })

    total = len(recent)
    parts = []
    if long_upper_shadow:
        parts.append(f'{long_upper_shadow}次上影线过长')
    if next_day_gap_down:
        parts.append(f'{next_day_gap_down}次次日低开')
    if high_volume_retreat:
        parts.append(f'{high_volume_retreat}次放量回落')
    if bearish_close:
        parts.append(f'{bearish_close}次收阴')
    summary = f'最近{total}次失败中，' + '，'.join(parts) if parts else f'最近{total}次失败无明显共同特征'

    return {
        'total_losses': total,
        'long_upper_shadow': long_upper_shadow,
        'next_day_gap_down': next_day_gap_down,
        'high_volume_retreat': high_volume_retreat,
        'bearish_close': bearish_close,
        'summary': summary,
        'trades': detail_trades,
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
    """Run backtests for all strategies.

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

        failure_analysis = compute_failure_analysis(stock_data, trade_details)

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
            'failure_analysis': failure_analysis,
        }

    return all_reports
