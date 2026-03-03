"""Composite signal screening + daily stock selection.

Retained strategies: 形态识别, OBV涨停梦, CCI快进出, OBV波段.
Removed: 超卖(=CCI快进出重复), 当日金叉买入, 近日上涨, 底部异动, OBV底部突破.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
import strategies as sig
from pattern_screen import screen_pattern
from universe import get_limit_ratio


def _exclude_limit_up(stock_data, codes, date):
    """Exclude stocks that hit limit-up on the given date (cannot buy at close)."""
    ts = pd.Timestamp(date)
    result = []
    for code in codes:
        df = stock_data[code]
        if ts not in df.index:
            continue
        idx = df.index.get_loc(ts)
        if idx == 0:
            result.append(code)
            continue
        prev_close = float(df['Close'].iloc[idx - 1])
        cur_close = float(df['Close'].iloc[idx])
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if cur_close < limit_price * (1 - config.LIMIT_TOLERANCE):
            result.append(code)
    return result


def _filter_on_date(stock_data: dict[str, pd.DataFrame],
                    date: str,
                    signal_funcs: list) -> list[str]:
    """Return stock codes where ALL signal functions are True on the given date.

    Args:
        stock_data: dict of code -> DataFrame with indicators computed.
        date: target date string (e.g., '2025-10-27').
        signal_funcs: list of signal functions, each taking a DataFrame and
                      returning a boolean Series.

    Returns:
        List of stock codes that pass all filters on the given date.
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        if ts not in df.index:
            continue
        all_pass = True
        for func in signal_funcs:
            signal = func(df)
            if ts not in signal.index or not signal.loc[ts]:
                all_pass = False
                break
        if all_pass:
            result.append(code)

    return result


def screen_cci_quick(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """CCI快进出: CCI超卖反弹 + 底部确认 + 价格贴近EMA5均线."""
    return _filter_on_date(stock_data, date, [
        sig.signal_cci_cross_neg100,
        sig.signal_escape_bottom,
        sig.signal_cci_deep_oversold,
        sig.signal_daily_gain_lt3,
        sig.signal_ema5_proximity,
    ])


def screen_obv_momentum(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """OBV涨停梦: OBV breakout + volume surge + momentum filters (main board only)."""
    codes = _filter_on_date(stock_data, date, [
        sig.signal_obv_breakout,
        sig.signal_volume_surge,
        sig.signal_cci_momentum_floor,
        sig.signal_daily_gain_gt2,
    ])
    # Main board only: exclude ChiNext (300/301) which have 20% limit-up
    codes = [c for c in codes if not c.startswith(('300', '301'))]
    # Exclude stocks at limit-up (cannot buy at close)
    return _exclude_limit_up(stock_data, codes, date)


def screen_obv_wave(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """OBV波段: OBV涨停梦 + ADX趋势过滤 + 底部确认."""
    codes = _filter_on_date(stock_data, date, [
        sig.signal_obv_breakout,
        sig.signal_volume_surge,
        sig.signal_cci_momentum_floor,
        sig.signal_daily_gain_gt2,
        sig.signal_adx_below_max,
        sig.signal_escape_bottom,
    ])
    # Exclude stocks at limit-up (cannot buy at close)
    return _exclude_limit_up(stock_data, codes, date)


# Registry of all screens with display names
ALL_SCREENS = {
    'CCI快进出': screen_cci_quick,
    'OBV涨停梦': screen_obv_momentum,
    'OBV波段': screen_obv_wave,
    '形态识别': screen_pattern,
}


def make_screen(signal_funcs: list):
    """Create a screen function from an arbitrary list of signal functions."""
    def dynamic_screen(stock_data, date):
        return _filter_on_date(stock_data, date, signal_funcs)
    names = [f.__name__ for f in signal_funcs]
    dynamic_screen.__name__ = f"screen_{'_'.join(names)}"
    return dynamic_screen


def run_all_screens(stock_data: dict[str, pd.DataFrame],
                    date: str) -> dict[str, list[str]]:
    """Run all composite screens and return results."""
    results = {}
    for name, func in ALL_SCREENS.items():
        codes = func(stock_data, date)
        results[name] = codes
        print(f"{name}: {codes}")
    return results
