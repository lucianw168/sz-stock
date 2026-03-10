"""Composite signal screening + daily stock selection.

Strategies: OBV涨停梦, 形态识别, 涨停接力, 跳空涨停基因, 缩量突破, 群龙夺宝.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
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


def _had_recent_limit_up(df: pd.DataFrame, date: str, code: str,
                          lookback: int = config.RELAY_RECENT_LU_DAYS) -> bool:
    """Check if stock had a limit-up event within the last `lookback` trading days."""
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return False

    idx = df.index.get_loc(ts)
    limit_ratio = get_limit_ratio(code)

    # Look back up to `lookback` trading days (excluding today)
    start = max(1, idx - lookback)
    for i in range(start, idx):
        prev_close = float(df['Close'].iloc[i - 1])
        cur_close = float(df['Close'].iloc[i])
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if cur_close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            return True
    return False


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


def screen_limit_relay(stock_data: dict[str, pd.DataFrame], date: str) -> list[str]:
    """涨停接力: Recent limit-up + strong daily gain + close near high + CCI momentum.

    Data-mined strategy based on analysis of 10,000+ limit-up events:
    Stocks with a recent limit-up (within 5 trading days) that show strong
    momentum today (≥5% gain, close in top 10% of range, CCI>50) have
    ~14-20% probability of hitting limit-up the next day (vs 1.5% base rate).

    Key discovery: close_position ≥ 0.9 is the critical discriminator,
    boosting hit rate from ~12% to ~15-20% by selecting stocks with the
    strongest closing action (minimal upper shadow = no selling pressure).

    Filters:
    1. Main board only (10% limit-up, exclude ChiNext 300/301)
    2. Daily gain ≥ 5% (strong momentum)
    3. Not at limit-up today (can still buy at close)
    4. Close position ≥ 90% of day's range (strongest close)
    5. CCI > 50 (momentum confirmation)
    6. Had a limit-up within last 5 trading days (涨停基因)
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        # Main board only
        if code.startswith(('300', '301')):
            continue
        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < config.RELAY_RECENT_LU_DAYS + 1:
            continue

        close = float(df['Close'].iloc[idx])
        prev_close = float(df['Close'].iloc[idx - 1])

        # Daily gain >= threshold
        daily_ret = close / prev_close - 1
        if daily_ret < config.RELAY_MIN_DAILY_GAIN:
            continue

        # Not at limit-up today (can't buy)
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            continue

        # Close position in top 10% of day's range
        high = float(df['High'].iloc[idx])
        low = float(df['Low'].iloc[idx])
        day_range = high - low
        if day_range <= 0:
            continue
        close_pos = (close - low) / day_range
        if close_pos < config.RELAY_CLOSE_POSITION_MIN:
            continue

        # CCI momentum floor
        if 'CCI' in df.columns:
            cci_val = float(df['CCI'].iloc[idx])
            if cci_val < config.RELAY_CCI_MIN:
                continue

        # Recent limit-up within N trading days
        if not _had_recent_limit_up(df, date, code):
            continue

        result.append(code)

    return result


def _is_gap_up_hold(df, idx):
    """Check if day at idx is a gap-up hold: open > prev high, low > prev high, bullish."""
    if idx < 1:
        return False
    o = float(df['Open'].iloc[idx])
    h_prev = float(df['High'].iloc[idx - 1])
    low = float(df['Low'].iloc[idx])
    close = float(df['Close'].iloc[idx])
    return o > h_prev and low > h_prev and close > o


def _has_vol_floor_rising(df, idx):
    """Check if volume floor is rising over 3 periods of N days."""
    n = config.ACCUM_VOL_FLOOR_DAYS
    periods = config.ACCUM_VOL_FLOOR_PERIODS
    total = n * periods
    if idx < total:
        return False
    vol = df['Volume'].values
    mins = []
    for p in range(periods):
        start = idx - total + p * n
        end = start + n
        mins.append(min(vol[start:end]))
    return all(mins[i] < mins[i + 1] for i in range(len(mins) - 1))


def _has_staircase(df, idx):
    """Check for N consecutive days of higher closes AND higher lows."""
    n = config.ACCUM_STAIRCASE_DAYS
    if idx < n:
        return False
    c = df['Close'].values
    l = df['Low'].values
    for j in range(n - 1):
        d = idx - n + 1 + j
        if c[d + 1] <= c[d] or l[d + 1] <= l[d]:
            return False
    return True


def _has_vol_price_sync(df, idx):
    """Check for N consecutive days of rising close AND rising volume."""
    n = config.ACCUM_VOL_PRICE_SYNC_DAYS
    if idx < n:
        return False
    c = df['Close'].values
    v = df['Volume'].values
    for j in range(n - 1):
        d = idx - n + 1 + j
        if c[d + 1] <= c[d] or v[d + 1] <= v[d]:
            return False
    return True


def _is_breaking_n_day_high(df, idx):
    """Check if today's close is the highest in N days."""
    w = config.ACCUM_BREAKOUT_HIGH_WINDOW
    if idx < w:
        return False
    c = df['Close'].values
    return c[idx] >= max(c[idx - w:idx])


def screen_accumulation_breakout(stock_data: dict[str, pd.DataFrame],
                                  date: str) -> list[str]:
    """蓄势突破: Multi-day accumulation patterns + gap-up hold breakout.

    Data-mined strategy based on 30+ pattern templates tested against
    10,000+ limit-up events. Identifies two-phase setups:

    Phase 1 - Accumulation (2+ patterns required for confirmation):
      a) 量能地板抬升: Volume trough rises over 3 consecutive periods
      b) 台阶上涨: 4+ days of higher closes & higher lows
      c) 量价齐升: 3+ days of rising close AND rising volume
      d) 突破N日新高: Close breaks above 20-day high

    Phase 2 - Breakout trigger (required):
      跳空高开不回补: Open > yesterday's high, low > yesterday's high,
      close > open (5.09% single LU rate, strongest single pattern found)

    Requiring 2+ accumulation patterns increases LU rate to ~7%
    with PF approaching 1.0 (200-day backtest: 7.0% LU, PF=0.92).

    Additional filters:
    - Main board only (exclude ChiNext 300/301)
    - Not at limit-up today
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        # Main board only
        if code.startswith(('300', '301')):
            continue
        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < 25:
            continue

        # Phase 2: Gap-up hold (required)
        if not _is_gap_up_hold(df, idx):
            continue

        # Not at limit-up today
        close = float(df['Close'].iloc[idx])
        prev_close = float(df['Close'].iloc[idx - 1])
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            continue

        # Phase 1: At least 2 accumulation patterns for confirmation
        pattern_count = sum([
            _has_vol_floor_rising(df, idx),
            _has_staircase(df, idx),
            _has_vol_price_sync(df, idx),
            _is_breaking_n_day_high(df, idx),
        ])
        if pattern_count < 2:
            continue

        result.append(code)

    return result


def _count_limit_ups(df, code, idx, window):
    """Count limit-up events in the last `window` trading days (excluding today)."""
    limit_ratio = get_limit_ratio(code)
    c = df['Close'].values
    count = 0
    start = max(1, idx - window)
    for i in range(start, idx):
        lp = round(c[i - 1] * (1 + limit_ratio), 2)
        if c[i] >= lp * (1 - config.LIMIT_TOLERANCE):
            count += 1
    return count


def screen_monster_volume(stock_data: dict[str, pd.DataFrame],
                          date: str) -> list[str]:
    """妖股放量: Frequent limit-up history + volume explosion + low selling pressure.

    Data-mined from 2884 stocks × 730 days with 70/30 train/test validation.
    Core finding: stocks with 3+ limit-ups in 20 days that show volume explosion
    (3x of 3-day average) with minimal upper shadow have 20-26% next-day
    limit-up probability (vs 1.5% base rate = 13-16x lift).

    Top validated combos:
      lu_3_20d + vol_3d>3x + upper_shadow<0.05  →  26.5% test LU (16.5x lift)
      lu_3_20d + vol_3d>3x + is_20d_high        →  22.0% test LU (13.7x lift)
      lu_3_20d + vol_5d>3x + CCI>100            →  19.5% test LU (12.1x lift)

    Conditions:
    1. Main board only (10% limit-up, exclude ChiNext 300/301)
    2. 3+ limit-ups in last 20 trading days (妖股 DNA)
    3. Today's volume >= 3x of 3-day moving average (volume explosion)
    4. Upper shadow <= 10% of day's range (low selling pressure)
    5. Not at limit-up today (can still buy)
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        # Main board only
        if code.startswith(('300', '301')):
            continue
        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < 25:
            continue

        c = df['Close'].values
        o = df['Open'].values
        h = df['High'].values
        lo = df['Low'].values
        v = df['Volume'].values

        close = float(c[idx])
        prev_close = float(c[idx - 1])

        # Not at limit-up today (can't buy)
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            continue

        # Limit-up count in last 20 trading days >= threshold
        lu_count = _count_limit_ups(df, code, idx, 20)
        if lu_count < config.MONSTER_LU_COUNT_20D:
            continue

        # Volume explosion: today's volume >= 3x of 3-day moving average
        if idx < 3:
            continue
        vol_avg_3d = np.mean(v[idx - 3:idx])
        if vol_avg_3d <= 0 or v[idx] / vol_avg_3d < config.MONSTER_VOL_RATIO_3D_MIN:
            continue

        # Upper shadow filter: (high - max(close, open)) / range <= threshold
        day_range = h[idx] - lo[idx]
        if day_range <= 0:
            continue
        upper_shadow = (h[idx] - max(close, o[idx])) / day_range
        if upper_shadow > config.MONSTER_UPPER_SHADOW_MAX:
            continue

        result.append(code)

    return result


def _had_recent_limit_up_n(df: pd.DataFrame, code: str, idx: int,
                           lookback: int) -> bool:
    """Check if stock had a limit-up event within the last `lookback` trading days."""
    limit_ratio = get_limit_ratio(code)
    c = df['Close'].values
    start = max(1, idx - lookback)
    for i in range(start, idx):
        lp = round(float(c[i - 1]) * (1 + limit_ratio), 2)
        if float(c[i]) >= lp * (1 - config.LIMIT_TOLERANCE):
            return True
    return False


def screen_gap_lu_dna(stock_data: dict[str, pd.DataFrame],
                      date: str) -> list[str]:
    """跳空涨停基因: Gap-up hold + strong close + recent limit-up DNA.

    PF-optimized strategy (the FIRST to achieve PF>1 on both train AND test).
    Data-mined from 2884 stocks × 730 days with 70/30 train/test validation.

    Core insight: gap-up hold (open>prev high, low>prev high, close>open) is
    the strongest single-day pattern (5.09% LU rate). Combined with recent
    limit-up DNA and strong close position, it becomes PROFITABLE:

    At 7% target:
      close_pos≥0.7 + lu_in_10d: Train PF 1.27, Test PF 1.32, LU 17.3%/16.3%
      close_pos≥0.8 + lu_in_10d: Train PF 1.58, Test PF 1.31, LU 22.1%/17.1%

    Conditions:
    1. Main board only (exclude ChiNext 300/301)
    2. Gap-up hold: open > prev high, low > prev high, close > open
    3. Close position >= 70% of day's range (strong close, minimal selling)
    4. Had a limit-up within last 10 trading days (涨停基因)
    5. Not at limit-up today (can buy at close)
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        # Main board only
        if code.startswith(('300', '301')):
            continue
        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < config.GAP_LU_RECENT_DAYS + 1:
            continue

        c = df['Close'].values
        o = df['Open'].values
        h = df['High'].values
        lo = df['Low'].values

        close = float(c[idx])
        prev_close = float(c[idx - 1])
        prev_high = float(h[idx - 1])

        # Not at limit-up today
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            continue

        # Gap-up hold: open > prev high, low > prev high, close > open
        open_price = float(o[idx])
        low = float(lo[idx])
        if not (open_price > prev_high and low > prev_high and close > open_price):
            continue

        # Close position >= threshold
        high = float(h[idx])
        day_range = high - low
        if day_range <= 0:
            continue
        close_pos = (close - low) / day_range
        if close_pos < config.GAP_LU_CLOSE_POS_MIN:
            continue

        # Recent limit-up within N trading days
        if not _had_recent_limit_up_n(df, code, idx, config.GAP_LU_RECENT_DAYS):
            continue

        result.append(code)

    return result


def screen_squeeze_breakout(stock_data: dict[str, pd.DataFrame],
                            date: str) -> list[str]:
    """缩量突破: Volume squeeze accumulation → breakout above recent high.

    Strongest PF strategy found via data mining (Test PF 1.41@7% with NOT_new_high).
    Identifies institutional accumulation (quiet low-volume pullback) followed
    by a decisive volume breakout above 10-day high with strong close.

    Phase 1 - Squeeze (look back up to 20 days for consecutive quiet days):
      - Volume ≤ 70% of 5-day moving average (institutions accumulating quietly)
      - Daily return ≤ 0% (pulling back or flat, shaking out weak hands)
      - At least 1 consecutive squeeze day required

    Phase 2 - Breakout (today):
      - Volume ≥ 1.5x of 5-day moving average (institutions buying aggressively)
      - Close > 10-day closing high (decisive breakout)
      - Close position ≥ 80% of day's range (strong close, no selling pressure)
      - CCI > 50 (momentum confirmation)
      - NOT at 60-day closing high (避免冲高回落, validated by 游资趋势)

    Validation (486-day backtest, 70/30 split, 7% target):
      Base + CCI>50: Train PF=1.45, Test PF=2.39 (strong but sparse)
      + NOT_new_high: filters overextended breakouts, improves consistency

    Additional filters:
    - Main board only (exclude ChiNext 300/301)
    - Not at limit-up today (can buy at close)
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        # Main board only
        if code.startswith(('300', '301')):
            continue
        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < config.BREAKOUT_HIGH_WINDOW + config.SQUEEZE_LOOKBACK:
            continue

        c = df['Close'].values
        o = df['Open'].values
        h = df['High'].values
        lo = df['Low'].values
        v = df['Volume'].values.astype(float)

        close = float(c[idx])
        prev_close = float(c[idx - 1])

        # Not at limit-up today
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            continue

        # --- Phase 2: Breakout checks (cheap to compute, filter first) ---

        # Volume breakout: today's volume >= 1.5x of 5-day average
        vol_ma5 = np.mean(v[idx - 5:idx])
        if vol_ma5 <= 0 or v[idx] / vol_ma5 < config.BREAKOUT_VOL_RATIO_MIN:
            continue

        # Break 10-day closing high
        high_window = c[idx - config.BREAKOUT_HIGH_WINDOW:idx]
        if close <= np.max(high_window):
            continue

        # Close position >= 80%
        day_range = h[idx] - lo[idx]
        if day_range <= 0:
            continue
        close_pos = (close - lo[idx]) / day_range
        if close_pos < config.BREAKOUT_CLOSE_POS_MIN:
            continue

        # CCI > 50
        if 'CCI' in df.columns:
            cci_val = df['CCI'].iloc[idx]
            if pd.isna(cci_val) or float(cci_val) < config.BREAKOUT_CCI_MIN:
                continue

        # NOT at 20-day closing high (避免冲高回落)
        nh_window = config.BREAKOUT_NOT_NEW_HIGH_DAYS
        if idx >= nh_window:
            high_20d = np.max(c[idx - nh_window:idx])
            if close >= high_20d:
                continue

        # --- Phase 1: Squeeze check (look back for quiet accumulation) ---
        squeeze_days = 0
        for lb in range(1, config.SQUEEZE_LOOKBACK + 1):
            li = idx - lb
            if li < 5:
                break
            # 5-day vol average at that point
            lb_vol_ma5 = np.mean(v[li - 5:li])
            if lb_vol_ma5 <= 0:
                break
            vr = v[li] / lb_vol_ma5
            dr = (c[li] / c[li - 1] - 1) if c[li - 1] > 0 else 0

            if vr <= config.SQUEEZE_VOL_RATIO_MAX and dr <= 0:
                squeeze_days += 1
            else:
                break  # Must be consecutive

        if squeeze_days < config.SQUEEZE_MIN_DAYS:
            continue

        result.append(code)

    return result


def screen_dragon_treasure(stock_data: dict[str, pd.DataFrame],
                            date: str) -> list[str]:
    """群龙夺宝: Multiple limit-ups in consolidation platform + volume breakout.

    多个涨停板聚集在同一横盘平台内，阳成团阴分散（阳日均量>阴日均量），
    红肥绿瘦（阳线实体>阴线实体），放量突破颈线（≥2x平台均量）。

    Data-mined from 2884 stocks × 486 days with 70/30 train/test validation.
    Strict version: N=208, Test PF@7%=1.11, 5d LU=30%, avg max gain 9.4%

    Conditions:
    1. Main board only (exclude ChiNext 300/301)
    2. Consolidation platform ≥15 days, price range ≤25%
    3. ≥2 limit-ups within the platform (群龙)
    4. 阳成团阴分散: avg up-day volume > avg down-day volume
    5. 红肥绿瘦: avg up-day body > avg down-day body
    6. Breakout: close > platform high (突破颈线)
    7. Volume: breakout volume ≥ 2x platform average (倍量过颈线)
    8. Daily gain ≥ 5%
    9. Not at limit-up today (can buy at close)
    """
    ts = pd.Timestamp(date)
    result = []

    for code, df in stock_data.items():
        # Main board only
        if code.startswith(('300', '301')):
            continue
        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < 30:
            continue

        c = df['Close'].values
        o = df['Open'].values
        v = df['Volume'].values.astype(float)

        close = float(c[idx])
        prev_close = float(c[idx - 1])
        if prev_close <= 0:
            continue

        # Daily gain >= 5%
        daily_ret = close / prev_close - 1
        if daily_ret < config.DRAGON_MIN_DAILY_GAIN:
            continue

        # Not at limit-up today
        limit_ratio = get_limit_ratio(code)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
            continue

        # Find consolidation platform (expand from 15 to max 90 days)
        best_consol = None
        for lookback in range(config.DRAGON_CONSOL_MIN_DAYS, min(idx, 91)):
            seg_c = c[idx - lookback:idx]
            if len(seg_c) < config.DRAGON_CONSOL_MIN_DAYS:
                break
            c_min, c_max = float(seg_c.min()), float(seg_c.max())
            if c_min <= 0:
                break
            price_range = (c_max - c_min) / c_min
            if price_range <= config.DRAGON_CONSOL_MAX_RANGE:
                best_consol = (lookback, c_max, np.mean(v[idx - lookback:idx]))
            else:
                break

        if best_consol is None:
            continue

        lb, platform_high, avg_vol = best_consol
        seg_start = idx - lb

        # Breakout: close > platform high
        if close <= platform_high:
            continue

        # Volume breakout: vol >= 2x platform average
        if avg_vol <= 0 or v[idx] / avg_vol < config.DRAGON_BVR_MIN:
            continue

        # Count limit-ups in platform
        lu_count = 0
        for j in range(seg_start + 1, idx):
            if c[j - 1] > 0:
                lp = round(float(c[j - 1]) * (1 + limit_ratio), 2)
                if float(c[j]) >= lp * (1 - config.LIMIT_TOLERANCE):
                    lu_count += 1
        if lu_count < config.DRAGON_MIN_LU_COUNT:
            continue

        # 阳成团阴分散 + 红肥绿瘦
        up_vols, down_vols = [], []
        up_bodies, down_bodies = [], []
        for j in range(seg_start + 1, idx):
            if c[j - 1] <= 0:
                continue
            ret_j = c[j] / c[j - 1] - 1
            if ret_j > 0:
                up_vols.append(v[j])
                up_bodies.append(abs(c[j] - o[j]))
            elif ret_j < 0:
                down_vols.append(v[j])
                down_bodies.append(abs(c[j] - o[j]))

        avg_up_vol = np.mean(up_vols) if up_vols else 0
        avg_down_vol = np.mean(down_vols) if down_vols else 1
        if avg_down_vol <= 0 or avg_up_vol / avg_down_vol <= 1.0:
            continue

        avg_up_body = np.mean(up_bodies) if up_bodies else 0
        avg_down_body = np.mean(down_bodies) if down_bodies else 1
        if avg_down_body <= 0 or avg_up_body / avg_down_body <= 1.0:
            continue

        result.append(code)

    return result


# Registry of all screens with display names
ALL_SCREENS = {
    'OBV涨停梦': screen_obv_momentum,
    '形态识别': screen_pattern,
    '涨停接力': screen_limit_relay,
    '跳空涨停基因': screen_gap_lu_dna,
    '缩量突破': screen_squeeze_breakout,
    '群龙夺宝': screen_dragon_treasure,
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
