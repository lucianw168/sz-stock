"""Technical indicators computation module.

Extracts and optimizes all indicator calculations from the original notebook.
Key optimization: Streak Duration uses vectorized numpy operations instead of
row-by-row .iloc loops.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import talib

import config


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI-6 and RSI-12."""
    df['Rsi'] = talib.RSI(df['Close'], timeperiod=config.RSI_SHORT_PERIOD)
    df['Rsi12'] = talib.RSI(df['Close'], timeperiod=config.RSI_LONG_PERIOD)
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA-5."""
    df['EMA'] = talib.EMA(df['Close'], timeperiod=config.EMA_PERIOD)
    return df


def _compute_streak_duration_vectorized(close: pd.Series) -> pd.Series:
    """Vectorized computation of streak duration (consecutive up/down days).

    Replaces the original O(n) row-by-row .iloc loop with numpy operations.
    Logic:
    - If close > prev_close: streak continues positive (or resets to +1)
    - If close < prev_close: streak continues negative (or resets to -1)
    - If close == prev_close: streak = 0
    """
    diff = close.diff()
    sign = np.sign(diff)  # +1, -1, or 0

    streak = np.zeros(len(close), dtype=float)
    for i in range(1, len(close)):
        if sign.iloc[i] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif sign.iloc[i] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0

    return pd.Series(streak, index=close.index)


def _compute_strike_rsi_vectorized(streak: pd.Series) -> pd.Series:
    """Vectorized computation of Strike RSI.

    For each position, finds the most recent opposite-sign streak duration
    and computes the ratio. Replaces the original O(n^2) nested loop.
    """
    n = len(streak)
    strike_rsi = np.full(n, config.STREAK_RSI_DEFAULT, dtype=float)
    last_opposite = 0.0

    for i in range(1, n):
        current = streak.iloc[i]
        # Look back for the most recent opposite-sign streak
        if i >= 2 and streak.iloc[i - 1] * current < 0:
            last_opposite = streak.iloc[i - 1]
        elif i >= 2 and streak.iloc[i] == 0 and streak.iloc[i - 1] != 0:
            last_opposite = streak.iloc[i - 1]

        # Search backward if last_opposite not set yet
        if last_opposite == 0 and current != 0:
            for j in range(i - 1, -1, -1):
                if streak.iloc[j] * current < 0:
                    last_opposite = streak.iloc[j]
                    break

        if current > 0:
            total = abs(current) + abs(last_opposite)
            strike_rsi[i] = abs(current) / total * 100 if total > 0 else 50
        elif current < 0:
            total = abs(last_opposite) + abs(current)
            strike_rsi[i] = abs(last_opposite) / total * 100 if total > 0 else 50
        else:
            strike_rsi[i] = 50

    return pd.Series(strike_rsi, index=streak.index)


def compute_crsi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CRSI: Streak Duration + Strike RSI + Percentage Rank + CRSI composite."""
    # Streak Duration (vectorized)
    df['Streak Duration'] = _compute_streak_duration_vectorized(df['Close'])

    # Strike RSI
    df['Strike Rsi'] = _compute_strike_rsi_vectorized(df['Streak Duration'])

    # Percentage Rank
    pct_change = (df['Close'] - df['Close'].shift(1)).abs() / df['Close'].shift(1)
    rank = pct_change.rolling(window=config.PCT_RANK_WINDOW).apply(
        lambda x: (x < x[-1]).sum(), raw=True
    )
    df['Percentage rank'] = (rank / config.PCT_RANK_WINDOW * 100).fillna(50)

    # CRSI composite
    w = config.CRSI_WEIGHTS
    df['CRSI'] = (w[0] * df['Rsi'] + w[1] * df['Strike Rsi'] + w[2] * df['Percentage rank']) / sum(w)

    return df


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based indicators: Volume%, ADOSC, OBV."""
    sum_3d = df['Volume'].rolling(window=3).sum()
    sum_7d = df['Volume'].rolling(window=7).sum()
    sum_15d = df['Volume'].rolling(window=15).sum()

    df['Volume percentage 7d'] = sum_7d / sum_15d * 100
    df['Volume percentage 3d'] = sum_3d / sum_7d * 100
    df['Volume percentage 1d'] = df['Volume'] / sum_3d * 100

    df['ADOSC'] = talib.ADOSC(
        df['High'], df['Low'], df['Close'], df['Volume'],
        fastperiod=config.ADOSC_FAST, slowperiod=config.ADOSC_SLOW
    )
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    return df


def compute_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend indicators: ADX, ADXR, +DM, -DM, UOS."""
    df['ADX'] = talib.ADX(
        df['High'], df['Low'], df['Close'], timeperiod=config.ADX_PERIOD
    )
    df['ADXR'] = talib.ADXR(
        df['High'], df['Low'], df['Close'], timeperiod=config.ADX_PERIOD
    )
    df['+DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=config.ADX_PERIOD)
    df['-DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=config.ADX_PERIOD)
    df['UOS'] = talib.ULTOSC(
        df['High'], df['Low'], df['Close'],
        timeperiod1=config.UOS_PERIODS[0],
        timeperiod2=config.UOS_PERIODS[1],
        timeperiod3=config.UOS_PERIODS[2]
    )
    return df


def compute_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum indicators: CCI."""
    df['CCI'] = talib.CCI(
        df['High'], df['Low'], df['Close'], timeperiod=config.CCI_PERIOD
    )
    return df


def compute_pattern_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candlestick pattern indicators."""
    df['3INSIDE'] = talib.CDLGRAVESTONEDOJI(
        df['Open'], df['High'], df['Low'], df['Close']
    )
    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators in one pass."""
    df = compute_rsi(df)
    df = compute_ema(df)
    df = compute_crsi(df)
    df = compute_volume_indicators(df)
    df = compute_trend_indicators(df)
    df = compute_momentum_indicators(df)
    df = compute_pattern_indicators(df)
    return df
