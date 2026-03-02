"""Trading strategy signal definitions.

All signals follow a unified interface: signal_xxx(df) -> pd.Series[bool]
Each function returns a boolean Series aligned with the DataFrame's index.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

import config


# ============================================================
# CRSI-based signals
# ============================================================

def signal_crsi_cross50(df: pd.DataFrame) -> pd.Series:
    """CRSI crosses above 50 today (was below 50 yesterday)."""
    return (
        (df['CRSI'] > 50) &
        (df['CRSI'] < config.CRSI_CROSS50_UPPER) &
        (df['CRSI'].shift(1) < 50)
    )


def signal_crsi_above50(df: pd.DataFrame) -> pd.Series:
    """CRSI in the 50-58 range."""
    return (
        (df['CRSI'] > config.CRSI_ABOVE50_LOWER) &
        (df['CRSI'] < config.CRSI_ABOVE50_UPPER)
    )


def signal_crsi_below25(df: pd.DataFrame) -> pd.Series:
    """CRSI drops below 25 while previous 2 days were above 50."""
    return (
        (df['CRSI'] < config.CRSI_BELOW25_THRESHOLD) &
        (df['CRSI'].shift(1) > 50) &
        (df['CRSI'].shift(2) > 50)
    )


def signal_crsi_decrease(df: pd.DataFrame) -> pd.Series:
    """CRSI drops by more than 15 in one day."""
    return (df['CRSI'] - df['CRSI'].shift(1)) < config.CRSI_DECREASE_DELTA


def signal_crsi_cross20(df: pd.DataFrame) -> pd.Series:
    """CRSI crosses above 20 today (was below 20 yesterday)."""
    return (
        (df['CRSI'] > config.CRSI_CROSS20_THRESHOLD) &
        (df['CRSI'].shift(1) < config.CRSI_CROSS20_THRESHOLD) &
        (df['CRSI'] < config.CRSI_CROSS50_UPPER)
    )


# ============================================================
# RSI-based signals
# ============================================================

def signal_rsi_golden_cross(df: pd.DataFrame) -> pd.Series:
    """RSI-6 crosses above RSI-12 (golden cross)."""
    yesterday_diff = df['Rsi'].shift(1) - df['Rsi12'].shift(1)
    today_diff = df['Rsi'] - df['Rsi12']
    return (yesterday_diff < 0) & (today_diff > 0)


def signal_rsi_strengthening(df: pd.DataFrame) -> pd.Series:
    """RSI gap is narrowing for 3 consecutive days (approaching golden cross)."""
    diff_2d = df['Rsi'].shift(2) - df['Rsi12'].shift(2)
    diff_1d = df['Rsi'].shift(1) - df['Rsi12'].shift(1)
    diff_0d = df['Rsi'] - df['Rsi12']

    return (
        (diff_0d < 0) &
        ((diff_1d - diff_2d) > config.RSI_STRENGTHENING_DELTA) &
        ((diff_0d - diff_1d) > config.RSI_STRENGTHENING_DELTA)
    )


def signal_rsi_declining(df: pd.DataFrame) -> pd.Series:
    """RSI gap is widening (bearish divergence)."""
    diff_2d = df['Rsi'].shift(2) - df['Rsi12'].shift(2)
    diff_1d = df['Rsi'].shift(1) - df['Rsi12'].shift(1)
    diff_0d = df['Rsi'] - df['Rsi12']

    return (
        (diff_0d < 0) &
        ((diff_1d - diff_2d) < 0) &
        ((diff_0d - diff_1d) < 0)
    )


# ============================================================
# Volume-based signals
# ============================================================

def signal_volume_surge(df: pd.DataFrame) -> pd.Series:
    """Volume 7-day ratio is elevated but 3-day ratio is low (building up)."""
    return (
        (df['Volume percentage 7d'] > config.VOLUME_7D_LOWER) &
        (df['Volume percentage 7d'] < config.VOLUME_7D_UPPER) &
        (df['Volume percentage 3d'] < config.VOLUME_3D_UPPER)
    )


def signal_obv_breakout(df: pd.DataFrame) -> pd.Series:
    """OBV breaks above consolidation range."""
    window = config.OBV_CONSOLIDATION_WINDOW
    rolling_max = df['OBV'].rolling(window).max()
    rolling_min = df['OBV'].rolling(window).min()

    is_consolidating = (df['OBV'] >= rolling_min) & (df['OBV'] <= rolling_max)
    consolidation_periods = is_consolidating.rolling(window).sum() >= window

    return (df['OBV'] > rolling_max.shift(2)) & consolidation_periods


def signal_uos_cross65(df: pd.DataFrame) -> pd.Series:
    """UOS crosses above 65."""
    return (
        (df['UOS'].shift(1) < config.UOS_CROSS_LEVEL) &
        (df['UOS'] > config.UOS_CROSS_LEVEL)
    )


# ============================================================
# Other signals
# ============================================================

def signal_consecutive_decline(df: pd.DataFrame) -> pd.Series:
    """Two consecutive declining days."""
    return (
        (df['Close'].shift(1) - df['Close'].shift(2) < 0) &
        (df['Close'] - df['Close'].shift(1) < 0)
    )


def signal_escape_bottom(df: pd.DataFrame) -> pd.Series:
    """Price near 15-day level (within 8%) but down 20%+ from 55 days ago."""
    short_pct = (df['Close'] / df['Close'].shift(config.ESCAPE_BOTTOM_SHORT_WINDOW) - 1).abs()
    long_pct = df['Close'] / df['Close'].shift(config.ESCAPE_BOTTOM_LONG_WINDOW) - 1

    return (
        (short_pct <= config.ESCAPE_BOTTOM_SHORT_PCT) &
        (long_pct <= config.ESCAPE_BOTTOM_LONG_PCT)
    )


def signal_pct_rank_spike(df: pd.DataFrame) -> pd.Series:
    """Percentage rank spike: current >= 60, recent was >= 85, now <= 30."""
    return (
        (df['Percentage rank'] >= config.PCT_RANK_SPIKE_THRESHOLD) &
        (
            (df['Percentage rank'].shift(2) >= config.PCT_RANK_SPIKE_HIGH) |
            (df['Percentage rank'].shift(1) >= config.PCT_RANK_SPIKE_HIGH)
        ) &
        (df['Percentage rank'] <= config.PCT_RANK_SPIKE_LOW)
    )


def signal_no_limit_up(df: pd.DataFrame) -> pd.Series:
    """Today's gain is less than 5% (not a limit-up day)."""
    return (df['Close'] / df['Close'].shift(1) - 1) < config.NO_LIMIT_UP_PCT


def signal_avoid_high(df: pd.DataFrame) -> pd.Series:
    """Price hasn't risen more than 15% in the last 7 days."""
    return (df['Close'] / df['Close'].shift(config.AVOID_HIGH_WINDOW) - 1) <= config.AVOID_HIGH_PCT


def signal_adxr_above25(df: pd.DataFrame) -> pd.Series:
    """ADXR is above 25 (active trend)."""
    return df['ADXR'] > config.ADXR_ABOVE_THRESHOLD


def signal_adx_cross_adxr(df: pd.DataFrame) -> pd.Series:
    """ADX crosses above ADXR (trend strengthening)."""
    yesterday = df['ADX'].shift(1) - df['ADXR'].shift(1)
    today = df['ADX'] - df['ADXR']
    return (yesterday < 0) & (today > 0)


def signal_dm_positive(df: pd.DataFrame) -> pd.Series:
    """+DM is greater than -DM (bullish directional movement)."""
    return df['+DM'] > df['-DM']


def signal_cci_cross_neg100(df: pd.DataFrame) -> pd.Series:
    """CCI crosses above -100 from below."""
    return (
        (df['CCI'].shift(1) < config.CCI_CROSS_LEVEL) &
        (df['CCI'] > config.CCI_CROSS_LEVEL)
    )


def signal_cci_deep_oversold(df: pd.DataFrame) -> pd.Series:
    """CCI deeply oversold: yesterday and 2-3 days ago below -110, 7 days ago between -100 and -70."""
    return (
        (df['CCI'].shift(1) < config.CCI_CROSS_LEVEL) &
        (df['CCI'].shift(2) < config.CCI_DEEP_LEVEL) &
        (df['CCI'].shift(3) < config.CCI_DEEP_LEVEL) &
        (df['CCI'].shift(6) > config.CCI_DEEP_BEFORE7D_LOWER) &
        (df['CCI'].shift(6) < config.CCI_DEEP_BEFORE7D_UPPER)
    )


def signal_ema5_proximity(df: pd.DataFrame) -> pd.Series:
    """Price is within 3% of EMA-5."""
    return (df['Close'] / df['EMA'] - 1).abs() < config.EMA5_PROXIMITY_PCT


def signal_sideways(df: pd.DataFrame) -> pd.Series:
    """Sideways consolidation: price barely moved over 5, 15, and 30 days."""
    d5 = (df['Close'] / df['Close'].shift(5) - 1).abs() < config.SIDEWAYS_5D_PCT
    d15 = df['Close'].shift(5) / df['Close'].shift(15) - 1 < config.SIDEWAYS_15D_PCT
    d30 = df['Close'].shift(10) / df['Close'].shift(30) - 1 < config.SIDEWAYS_30D_PCT
    return d5 & d15 & d30


def signal_daily_gain_gt5(df: pd.DataFrame) -> pd.Series:
    """Today's gain > 5%."""
    return (df['Close'] / df['Close'].shift(1) - 1) > config.DAILY_GAIN_GT5_PCT


def signal_daily_gain_lt3(df: pd.DataFrame) -> pd.Series:
    """Today's gain < 3%."""
    return (df['Close'] / df['Close'].shift(1) - 1) < config.DAILY_GAIN_LT3_PCT


def signal_cci_momentum_floor(df: pd.DataFrame) -> pd.Series:
    """CCI above momentum floor."""
    return df['CCI'] > config.CCI_MOMENTUM_FLOOR


def signal_daily_gain_gt2(df: pd.DataFrame) -> pd.Series:
    """Today's gain > 2%."""
    return (df['Close'] / df['Close'].shift(1) - 1) > config.DAILY_GAIN_GT2_PCT


def signal_adx_below_max(df: pd.DataFrame) -> pd.Series:
    """ADX below buy threshold (filter out overextended trends)."""
    return df['ADX'] < config.ADX_BUY_MAX




# ============================================================
# Signal registry: name -> function
# ============================================================
SIGNAL_REGISTRY = {
    'crsi_cross50':        signal_crsi_cross50,
    'crsi_above50':        signal_crsi_above50,
    'crsi_below25':        signal_crsi_below25,
    'crsi_decrease':       signal_crsi_decrease,
    'crsi_cross20':        signal_crsi_cross20,
    'rsi_golden_cross':    signal_rsi_golden_cross,
    'rsi_strengthening':   signal_rsi_strengthening,
    'rsi_declining':       signal_rsi_declining,
    'volume_surge':        signal_volume_surge,
    'obv_breakout':        signal_obv_breakout,
    'uos_cross65':         signal_uos_cross65,
    'consecutive_decline': signal_consecutive_decline,
    'escape_bottom':       signal_escape_bottom,
    'pct_rank_spike':      signal_pct_rank_spike,
    'no_limit_up':         signal_no_limit_up,
    'avoid_high':          signal_avoid_high,
    'adxr_above25':        signal_adxr_above25,
    'adx_cross_adxr':      signal_adx_cross_adxr,
    'dm_positive':         signal_dm_positive,
    'cci_cross_neg100':    signal_cci_cross_neg100,
    'cci_deep_oversold':   signal_cci_deep_oversold,
    'ema5_proximity':      signal_ema5_proximity,
    'sideways':            signal_sideways,
    'daily_gain_gt5':      signal_daily_gain_gt5,
    'daily_gain_lt3':      signal_daily_gain_lt3,
    'cci_momentum_floor':  signal_cci_momentum_floor,
    'daily_gain_gt2':      signal_daily_gain_gt2,
    'adx_below_max':       signal_adx_below_max,
}

# ============================================================
# Signal -> config parameter dependencies
# ============================================================
SIGNAL_PARAMS = {
    'crsi_cross50':        ['CRSI_CROSS50_UPPER'],
    'crsi_above50':        ['CRSI_ABOVE50_LOWER', 'CRSI_ABOVE50_UPPER'],
    'crsi_below25':        ['CRSI_BELOW25_THRESHOLD'],
    'crsi_decrease':       ['CRSI_DECREASE_DELTA'],
    'crsi_cross20':        ['CRSI_CROSS20_THRESHOLD', 'CRSI_CROSS50_UPPER'],
    'rsi_golden_cross':    [],
    'rsi_strengthening':   ['RSI_STRENGTHENING_DELTA'],
    'rsi_declining':       [],
    'volume_surge':        ['VOLUME_7D_LOWER', 'VOLUME_7D_UPPER', 'VOLUME_3D_UPPER'],
    'obv_breakout':        ['OBV_CONSOLIDATION_WINDOW'],
    'uos_cross65':         ['UOS_CROSS_LEVEL'],
    'consecutive_decline': [],
    'escape_bottom':       ['ESCAPE_BOTTOM_SHORT_WINDOW', 'ESCAPE_BOTTOM_SHORT_PCT',
                            'ESCAPE_BOTTOM_LONG_WINDOW', 'ESCAPE_BOTTOM_LONG_PCT'],
    'pct_rank_spike':      ['PCT_RANK_SPIKE_THRESHOLD', 'PCT_RANK_SPIKE_HIGH',
                            'PCT_RANK_SPIKE_LOW'],
    'no_limit_up':         ['NO_LIMIT_UP_PCT'],
    'avoid_high':          ['AVOID_HIGH_PCT', 'AVOID_HIGH_WINDOW'],
    'adxr_above25':        ['ADXR_ABOVE_THRESHOLD'],
    'adx_cross_adxr':      [],
    'dm_positive':         [],
    'cci_cross_neg100':    ['CCI_CROSS_LEVEL'],
    'cci_deep_oversold':   ['CCI_CROSS_LEVEL', 'CCI_DEEP_LEVEL',
                            'CCI_DEEP_BEFORE7D_LOWER', 'CCI_DEEP_BEFORE7D_UPPER'],
    'ema5_proximity':      ['EMA5_PROXIMITY_PCT'],
    'sideways':            ['SIDEWAYS_5D_PCT', 'SIDEWAYS_15D_PCT', 'SIDEWAYS_30D_PCT'],
    'daily_gain_gt5':      ['DAILY_GAIN_GT5_PCT'],
    'daily_gain_lt3':      ['DAILY_GAIN_LT3_PCT'],
    'cci_momentum_floor':  ['CCI_MOMENTUM_FLOOR'],
    'daily_gain_gt2':      ['DAILY_GAIN_GT2_PCT'],
    'adx_below_max':       ['ADX_BUY_MAX'],
}
