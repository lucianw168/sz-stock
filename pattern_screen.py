"""Pattern detection screen: 投资笔记 形态识别策略.

横盘≥22天 → 缩量涨停突破 → 倍量阴线洗盘 → 回收确认 + 无下影线

Based on data mining of 2884 stocks × 486 days:
- 66 signals, 27% 5-day limit-up rate, avg max gain 10%
- Key validated conditions: 非峰值量, 倍量阴线, 无下影线
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

import numpy as np
import pandas as pd

import config
from universe import get_limit_ratio

logger = logging.getLogger(__name__)

_latest_signals = []


def screen_pattern(stock_data, date):
    """Screener-compatible: returns list of codes with pattern signals."""
    signals = run_pattern_scan(stock_data, date)
    return [s['code'] for s in signals]


def run_pattern_scan(stock_data, date):
    """Run pattern detection for a specific date.

    Args:
        stock_data: dict of code -> DataFrame (processed format with indicators)
        date: date string 'YYYY-MM-DD'

    Returns:
        list of signal dicts with pattern match details
    """
    global _latest_signals
    signals = _scan_date(stock_data, date)
    _latest_signals = signals
    return signals


def get_latest_pattern_signals():
    """Get full signal details from the most recent scan."""
    return _latest_signals


# ======================================================================
# Core detection logic
# ======================================================================

def _is_limit_up(code, close, prev_close):
    """Check if close price is at limit-up."""
    lr = get_limit_ratio(code)
    lp = round(prev_close * (1 + lr), 2)
    return close >= lp * (1 - config.LIMIT_TOLERANCE)


def _find_consolidation(c, v, b_idx):
    """Find consolidation period before breakout day.

    Looks backwards from b_idx for a stretch of low-volatility trading.
    Returns dict with days/high/low/range/avg_vol/max_vol, or None.
    """
    min_days = config.PATTERN_CONSOL_MIN_DAYS
    max_range = config.PATTERN_CONSOL_MAX_RANGE

    best = None
    for lookback in range(min_days, min(b_idx, 61)):
        seg_c = c[b_idx - lookback:b_idx]
        seg_v = v[b_idx - lookback:b_idx]

        if len(seg_c) < min_days:
            break

        c_min, c_max = seg_c.min(), seg_c.max()
        if c_min <= 0:
            break

        price_range = (c_max - c_min) / c_min
        if price_range <= max_range:
            best = {
                'days': lookback,
                'high': c_max,
                'low': c_min,
                'range': price_range,
                'avg_vol': float(np.mean(seg_v)),
                'max_vol': float(np.max(seg_v)),
            }
        else:
            break

    return best


def _scan_date(stock_data, date):
    """Scan all stocks for 投资笔记 pattern on a specific date.

    Today = Day C (confirmation). Looks back for:
    - Day B: limit-up breakout after consolidation (缩量涨停)
    - Day P: bearish pullback between B and C (倍量阴线)
    - Day C: recovery above pullback close + no lower shadow (回收+无下影线)
    """
    signals = []
    ts = pd.Timestamp(date)
    date_ts = date.replace('-', '')  # YYYYMMDD for web generator compat

    for code, df in stock_data.items():
        # Main board only (exclude ChiNext 300/301)
        if code.startswith(('300', '301')):
            continue

        if ts not in df.index:
            continue

        idx = df.index.get_loc(ts)
        if idx < config.PATTERN_CONSOL_MIN_DAYS + 10:
            continue

        c = df['Close'].values
        o = df['Open'].values
        h = df['High'].values
        lo = df['Low'].values
        v = df['Volume'].values.astype(float)

        # --- Day C checks (today) ---
        today_c = float(c[idx])
        today_o = float(o[idx])
        today_h = float(h[idx])
        today_l = float(lo[idx])

        if today_c <= 0:
            continue

        # Must not be limit-up (can't buy at close)
        if idx > 0:
            prev = float(c[idx - 1])
            if prev > 0 and _is_limit_up(code, today_c, prev):
                continue

        # Lower shadow check (投资笔记: 无下影线最关键)
        day_range = today_h - today_l
        if day_range > 0:
            lower_shadow = (min(today_o, today_c) - today_l) / day_range
        else:
            lower_shadow = 0

        if lower_shadow > config.PATTERN_CONFIRM_MAX_SHADOW:
            continue

        # --- Look back for Day B (breakout) ---
        max_span = config.PATTERN_MAX_SPAN_DAYS
        found = False

        for b_offset in range(2, max_span + 1):
            b_idx = idx - b_offset
            if b_idx < config.PATTERN_CONSOL_MIN_DAYS + 1:
                continue

            b_prev = float(c[b_idx - 1])
            if b_prev <= 0:
                continue

            # Day B: must be limit-up
            if not _is_limit_up(code, float(c[b_idx]), b_prev):
                continue

            # Consolidation before Day B
            consol = _find_consolidation(c, v, b_idx)
            if consol is None:
                continue

            # Must break consolidation high
            if c[b_idx] <= consol['high']:
                continue

            # Not peak volume (投资笔记: 涨停成交量越少越好)
            if config.PATTERN_NOT_PEAK_VOL and v[b_idx] >= consol['max_vol']:
                continue

            # --- Find pullback day (Day P) between B and C ---
            p_idx_best = None

            for p_idx in range(b_idx + 1, idx):
                # Must be bearish candle (阴线)
                if c[p_idx] >= o[p_idx]:
                    continue

                # Pullback volume (投资笔记: 倍量阴线)
                p_vol_ratio = v[p_idx] / v[b_idx] if v[b_idx] > 0 else 0
                if p_vol_ratio < config.PATTERN_PULLBACK_VOL_MIN:
                    continue

                # Drop not too severe
                p_drop = (c[p_idx] - c[b_idx]) / c[b_idx]
                if p_drop < config.PATTERN_PULLBACK_MAX_DROP:
                    continue

                p_idx_best = p_idx
                break

            if p_idx_best is None:
                continue

            # Day C must recover above Day P close (回收)
            if today_c <= c[p_idx_best]:
                continue

            # --- All conditions met ---
            b_vol_ratio = v[b_idx] / consol['avg_vol'] if consol['avg_vol'] > 0 else 0
            p_vol_r = v[p_idx_best] / v[b_idx] if v[b_idx] > 0 else 0
            p_drop_f = (c[p_idx_best] - c[b_idx]) / c[b_idx]

            vol_desc = '缩量' if b_vol_ratio < 1.0 else f'{b_vol_ratio:.1f}倍量'

            signal = {
                'code': code,
                'ts_code': f'{code}.SZ',
                'pattern_name': 'txt_pattern',
                'display_name': '横盘缩量涨停回收',
                'signal_date': date_ts,
                'signal_price': today_c,
                'consol_days': consol['days'],
                'breakout_vol_ratio': round(b_vol_ratio, 2),
                'breakout_seal_quality': 0.0,
                'pullback_vol_ratio': round(p_vol_r, 2),
                'pullback_drop_pct': round(p_drop_f, 4),
                'confidence': 0.27,  # 27% 5-day limit-up rate from mining
                'explanation': (
                    f"{consol['days']}天横盘(波动{consol['range']*100:.0f}%)后"
                    f"{vol_desc}涨停后{p_vol_r:.1f}倍量阴线洗盘后回收，"
                    f"无下影线(5日涨停率27%)"
                ),
                'phase_dates': {
                    'breakout': str(df.index[b_idx].date()).replace('-', ''),
                    'pullback': str(df.index[p_idx_best].date()).replace('-', ''),
                    'confirmation': date_ts,
                },
            }
            signals.append(signal)
            found = True
            break

        # (found breaks the b_offset loop for this stock)

    return signals
