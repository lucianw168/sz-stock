"""Composite signal screening + daily stock selection.

Encapsulates the 7 composite screens from the original notebook.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import strategies as sig
from pattern_screen import screen_pattern


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


def screen_oversold(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """超卖: CCI crosses -100 + escape bottom + CCI deep oversold + daily gain < 3%."""
    return _filter_on_date(stock_data, date, [
        sig.signal_cci_cross_neg100,
        sig.signal_escape_bottom,
        sig.signal_cci_deep_oversold,
        sig.signal_daily_gain_lt3,
    ])


def screen_golden_cross(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """当日金叉买入: UOS cross 65 + no limit up + RSI golden cross."""
    return _filter_on_date(stock_data, date, [
        sig.signal_uos_cross65,
        sig.signal_no_limit_up,
        sig.signal_rsi_golden_cross,
    ])


def screen_bottom_activity(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """底部异动: Volume surge + escape bottom."""
    return _filter_on_date(stock_data, date, [
        sig.signal_volume_surge,
        sig.signal_escape_bottom,
    ])


def screen_oscillation(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """震荡指标: ADX crosses ADXR + escape bottom."""
    return _filter_on_date(stock_data, date, [
        sig.signal_adx_cross_adxr,
        sig.signal_escape_bottom,
    ])


def screen_recent_uptrend(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """近日上涨: ADX crosses ADXR + DM positive + no limit up."""
    return _filter_on_date(stock_data, date, [
        sig.signal_adx_cross_adxr,
        sig.signal_dm_positive,
        sig.signal_no_limit_up,
    ])


def screen_cci_quick(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """CCI快进出: CCI crosses -100 + escape bottom + CCI deep oversold + daily gain < 3%."""
    return _filter_on_date(stock_data, date, [
        sig.signal_cci_cross_neg100,
        sig.signal_escape_bottom,
        sig.signal_cci_deep_oversold,
        sig.signal_daily_gain_lt3,
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
    return [c for c in codes if not c.startswith(('300', '301'))]


def screen_obv_wave(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """OBV波段: OBV涨停梦 + ADX<40 filter for wave trading."""
    return _filter_on_date(stock_data, date, [
        sig.signal_obv_breakout,
        sig.signal_volume_surge,
        sig.signal_cci_momentum_floor,
        sig.signal_daily_gain_gt2,
        sig.signal_adx_below_max,
    ])


def screen_obv_bottom_breakout(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """OBV底部突破: OBV breakout + volume surge + escape bottom."""
    return _filter_on_date(stock_data, date, [
        sig.signal_obv_breakout,
        sig.signal_volume_surge,
        sig.signal_escape_bottom,
    ])


# Registry of all screens with display names
ALL_SCREENS = {
    '超卖': screen_oversold,
    '当日金叉买入': screen_golden_cross,
    '底部异动': screen_bottom_activity,
    '震荡指标': screen_oscillation,
    '近日上涨': screen_recent_uptrend,
    'CCI快进出': screen_cci_quick,
    'OBV涨停梦': screen_obv_momentum,
    'OBV波段': screen_obv_wave,
    'OBV底部突破': screen_obv_bottom_breakout,
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
