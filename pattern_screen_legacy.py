"""蛰伏涨停 strategy: state-machine pattern detection from learned library.

横盘缩量涨停延迟回收 — data-mined from 32,997 limit-up events.
Consolidation → shrink-volume limit-up (supply exhaustion) → pullback → recovery.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

import pandas as pd

import config
from pattern_library import PatternLibrary, PatternPhase
from universe import get_limit_ratio

logger = logging.getLogger(__name__)

# Config defaults (these were in config.py before the 投资笔记 rewrite)
_PATTERN_LIBRARY_PATH = getattr(
    config, 'PATTERN_LIBRARY_PATH',
    os.path.join(config.MODEL_DIR, 'pattern_library.pkl'))
_PATTERN_WARMUP_DAYS = getattr(config, 'PATTERN_WARMUP_DAYS', 60)


# ======================================================================
# Inline pattern detector (adapted from limit_up/pattern_detector.py)
# Works with tushare-format data: lowercase columns, trade_date string
# ======================================================================

class _PatternState:
    """Tracking state for one pattern on one stock."""
    __slots__ = [
        'pattern_name', 'phase', 'phase_dates',
        'consol_start_idx', 'consol_days', 'consol_high', 'consol_low',
        'consol_avg_vol', 'breakout_date', 'breakout_close', 'breakout_vol',
        'breakout_vol_ratio', 'breakout_seal_quality',
        'pullback_date', 'pullback_open', 'pullback_close', 'pullback_low',
        'pullback_vol_ratio', 'pullback_days_waited',
        'confirm_days_waited',
    ]

    def __init__(self, pattern_name):
        self.pattern_name = pattern_name
        self.reset()

    def reset(self):
        self.phase = PatternPhase.IDLE
        self.phase_dates = {}
        self.consol_start_idx = -1
        self.consol_days = 0
        self.consol_high = 0.0
        self.consol_low = float('inf')
        self.consol_avg_vol = 0.0
        self.breakout_date = ''
        self.breakout_close = 0.0
        self.breakout_vol = 0.0
        self.breakout_vol_ratio = 0.0
        self.breakout_seal_quality = 0.0
        self.pullback_date = ''
        self.pullback_open = 0.0
        self.pullback_close = 0.0
        self.pullback_low = float('inf')
        self.pullback_vol_ratio = 0.0
        self.pullback_days_waited = 0
        self.confirm_days_waited = 0


class PatternDetector:
    """Scans all stocks daily via state machines. Works with tushare format data."""

    def __init__(self, library, daily_data, limit_events=None):
        """
        Args:
            library: PatternLibrary with learned templates
            daily_data: {ts_code: DataFrame} with tushare columns
            limit_events: combined limit-up events DataFrame
        """
        self.library = library
        self.daily_data = daily_data
        self.limit_events = limit_events
        self.stock_states = {}  # ts_code -> {pattern_name: _PatternState}

        self.active_templates = [
            t for t in library.list_templates()
            if t.params.success_rate >= t.min_confidence
            and PatternPhase.PULLBACK in t.phases
            and PatternPhase.CONFIRMATION in t.phases
        ]

        # Limit event lookup
        self._limit_lookup = {}
        if limit_events is not None and not limit_events.empty:
            for _, row in limit_events.iterrows():
                key = (row.get('ts_code', ''), str(row.get('trade_date', '')))
                self._limit_lookup[key] = row

    def initialize_states(self):
        for code in self.daily_data:
            states = {}
            for t in self.active_templates:
                states[t.name] = _PatternState(t.name)
            self.stock_states[code] = states

    def warmup(self, dates):
        print(f'  Warming up pattern detector over {len(dates)} days...')
        for i, date in enumerate(dates):
            self.scan_date(date, emit_signals=False)
            if (i + 1) % 20 == 0:
                print(f'    Warmup: {i+1}/{len(dates)} days')

    def scan_date(self, date, emit_signals=True):
        """Scan all stocks for the given date. Returns list of signal dicts."""
        signals = []
        date_str = str(date)

        for code, states in self.stock_states.items():
            daily = self.daily_data.get(code)
            if daily is None or daily.empty:
                continue

            row_mask = daily['trade_date'] == date_str
            if not row_mask.any():
                continue

            row_idx = daily.index[row_mask][0]
            row = daily.loc[row_idx]

            if row_idx < 60:
                continue

            for template in self.active_templates:
                state = states[template.name]
                result = self._update_state(
                    code, daily, row_idx, row, date_str, template, state)
                if result is not None and emit_signals:
                    signals.append(result)

        return signals

    def _update_state(self, code, daily, row_idx, row, date_str,
                      template, state):
        import numpy as np
        params = template.params
        close = row['close']
        open_price = row['open']
        high = row['high']
        low = row['low']
        vol = row['vol']

        # Check limit-up
        is_limit_up = False
        if row_idx >= 1:
            prev_close = daily.iloc[row_idx - 1]['close']
            ratio = get_limit_ratio(code.split('.')[0])
            limit_price = round(prev_close * (1 + ratio), 2)
            is_limit_up = close >= limit_price * (1 - config.LIMIT_TOLERANCE)

        # IDLE -> CONSOLIDATION
        if state.phase == PatternPhase.IDLE:
            if PatternPhase.CONSOLIDATION in template.phases:
                consol = self._detect_consolidation(daily, row_idx, params)
                if consol is not None:
                    state.phase = PatternPhase.CONSOLIDATION
                    state.consol_days = consol['days']
                    state.consol_high = consol['high']
                    state.consol_low = consol['low']
                    state.consol_avg_vol = consol['avg_vol']
                    state.phase_dates['consolidation'] = date_str

        # CONSOLIDATION -> BREAKOUT
        elif state.phase == PatternPhase.CONSOLIDATION:
            if is_limit_up:
                vol_ratio = vol / state.consol_avg_vol if state.consol_avg_vol > 0 else 1.0
                seal_quality = self._compute_seal_quality(code, date_str, row)
                vol_ok = params.breakout_vol_ratio_min <= vol_ratio <= params.breakout_vol_ratio_max
                seal_ok = seal_quality >= params.breakout_seal_quality_min
                breaks_high = close > state.consol_high

                if vol_ok and seal_ok and breaks_high:
                    state.phase = PatternPhase.BREAKOUT_LIMIT_UP
                    state.breakout_date = date_str
                    state.breakout_close = close
                    state.breakout_vol = vol
                    state.breakout_vol_ratio = vol_ratio
                    state.breakout_seal_quality = seal_quality
                    state.phase_dates['breakout'] = date_str
            elif close < state.consol_low * 0.97:
                state.reset()
            else:
                consol = self._detect_consolidation(daily, row_idx, params)
                if consol is None:
                    state.reset()
                else:
                    state.consol_days = consol['days']
                    state.consol_high = consol['high']
                    state.consol_low = consol['low']
                    state.consol_avg_vol = consol['avg_vol']

        # BREAKOUT -> PULLBACK
        elif state.phase == PatternPhase.BREAKOUT_LIMIT_UP:
            state.pullback_days_waited += 1
            is_bearish = close < open_price

            if is_bearish:
                pb_vol_ratio = vol / state.breakout_vol if state.breakout_vol > 0 else 0
                drop_pct = (close - state.breakout_close) / state.breakout_close
                vol_ok = pb_vol_ratio >= params.pullback_vol_ratio_min
                drop_ok = drop_pct >= params.pullback_max_drop_pct

                if vol_ok and drop_ok:
                    state.phase = PatternPhase.PULLBACK
                    state.pullback_date = date_str
                    state.pullback_open = open_price
                    state.pullback_close = close
                    state.pullback_low = low
                    state.pullback_vol_ratio = pb_vol_ratio
                    state.phase_dates['pullback'] = date_str
                elif not drop_ok:
                    state.reset()
            elif is_limit_up:
                state.reset()

            if state.pullback_days_waited > params.pullback_max_days:
                state.reset()

        # PULLBACK -> SIGNAL
        elif state.phase == PatternPhase.PULLBACK:
            state.confirm_days_waited += 1
            recovers = close > state.pullback_open * (1 + params.confirm_recovery_pct)

            if recovers and not is_limit_up:
                state.phase = PatternPhase.SIGNAL
                state.phase_dates['confirmation'] = date_str
                signal = self._create_signal(
                    code, date_str, close, template, state)
                return signal

            if low < state.pullback_low:
                state.pullback_low = low

            if state.confirm_days_waited > params.confirm_max_days:
                state.reset()
            if close < state.consol_low * 0.97:
                state.reset()
            if recovers and is_limit_up:
                state.reset()

        elif state.phase in (PatternPhase.SIGNAL, PatternPhase.FAILED):
            state.reset()

        return None

    def _detect_consolidation(self, daily, row_idx, params):
        import numpy as np
        min_days = params.consol_min_days
        max_range = params.consol_max_range_pct

        if row_idx < min_days:
            return None

        best = None
        for lookback in range(min_days, min(row_idx, 60) + 1):
            segment = daily.iloc[row_idx - lookback:row_idx]
            closes = segment['close'].values
            vols = segment['vol'].values

            if len(closes) < min_days:
                break

            c_min = closes.min()
            c_max = closes.max()
            if c_min <= 0:
                break

            price_range = (c_max - c_min) / c_min
            if price_range <= max_range:
                best = {
                    'days': lookback,
                    'high': c_max,
                    'low': c_min,
                    'range': price_range,
                    'avg_vol': np.mean(vols),
                }
            else:
                break

        return best

    def _compute_seal_quality(self, code, date_str, row):
        score = 0.0
        event = self._limit_lookup.get((code, date_str))
        if event is not None:
            fd_amount = event.get('fd_amount', 0)
            open_times = event.get('open_times', 0)
            first_time = str(event.get('first_time', ''))

            if pd.notna(fd_amount) and fd_amount > 0:
                score += 0.4
            if open_times == 0:
                score += 0.3
            elif open_times <= 1:
                score += 0.15
            try:
                ft = int(str(first_time).replace(':', '').replace('.', ''))
                if ft <= 100000:
                    score += 0.2
                elif ft <= 110000:
                    score += 0.1
            except (ValueError, TypeError):
                pass
        else:
            if row['high'] > row['low']:
                close_pos = (row['close'] - row['low']) / (row['high'] - row['low'])
                score = 0.3 + 0.4 * close_pos
            else:
                score = 0.5

        return min(score, 1.0)

    def _create_signal(self, code, date_str, price, template, state):
        p = template.params
        parts = []
        if state.consol_days > 0:
            parts.append(f'{state.consol_days}天横盘')
        if state.breakout_vol_ratio > 0:
            vol_desc = '缩量' if state.breakout_vol_ratio < 1.0 else f'{state.breakout_vol_ratio:.1f}倍量'
            parts.append(f'{vol_desc}涨停')
        if state.pullback_vol_ratio > 0:
            parts.append(f'{state.pullback_vol_ratio:.1f}倍量阴线回踩')
        parts.append(f'历史胜率{p.success_rate:.0%}')

        explanation = '后'.join(parts[:3])
        if len(parts) > 3:
            explanation += '，' + parts[-1]

        # Convert ts_code to plain code for sz project
        plain_code = code.split('.')[0]

        signal = {
            'ts_code': code,
            'code': plain_code,
            'pattern_name': template.name,
            'display_name': template.display_name,
            'signal_date': date_str,
            'signal_price': price,
            'confidence': p.success_rate,
            'explanation': f'{template.display_name} - {explanation}',
            'consol_days': state.consol_days,
            'breakout_vol_ratio': state.breakout_vol_ratio,
            'breakout_seal_quality': state.breakout_seal_quality,
            'pullback_vol_ratio': state.pullback_vol_ratio,
            'pullback_drop_pct': (
                (state.pullback_close - state.breakout_close) / state.breakout_close
                if state.breakout_close > 0 else 0
            ),
            'phase_dates': dict(state.phase_dates),
        }

        state.reset()
        return signal


# ======================================================================
# Screener-compatible interface
# ======================================================================

_detector = None
_detector_date = None
_latest_signals = []


def _ensure_detector(stock_data_sz_format):
    """Lazy-initialize the pattern detector from processed data."""
    global _detector, _detector_date

    if _detector is not None:
        return _detector

    # Load pattern library
    lib_path = _PATTERN_LIBRARY_PATH
    if not os.path.exists(lib_path):
        # Try importing from limit_up project
        limit_up_lib = os.path.normpath(os.path.join(
            config.BASE_DIR, '..', '..', 'muti_factor', 'limit_up',
            'data', 'models', 'pattern_library.pkl'
        ))
        if os.path.exists(limit_up_lib):
            import shutil
            shutil.copy2(limit_up_lib, lib_path)
            print(f"  Copied pattern library from limit_up project")
        else:
            logger.warning(f"Pattern library not found: {lib_path}")
            return None

    library = PatternLibrary.load(lib_path)
    if not library.list_templates():
        logger.warning("Pattern library has no active templates.")
        return None

    print(f"  Pattern library: {library.summary()}")

    # Load tushare-format daily data
    import downloader
    daily_tushare = downloader.load_daily_tushare_format()
    if not daily_tushare:
        logger.warning("No tushare-format daily data for pattern detection.")
        return None

    # Load limit events
    limit_events = downloader.load_limit_events()

    # Build detector
    _detector = PatternDetector(library, daily_tushare, limit_events)
    _detector.initialize_states()

    return _detector


def run_pattern_scan(stock_data, date):
    """Run pattern detection for a specific date.

    Args:
        stock_data: dict of code -> DataFrame (sz format, needed to init detector)
        date: date string in 'YYYY-MM-DD' format

    Returns:
        list of signal dicts with pattern match details
    """
    global _latest_signals, _detector_date

    detector = _ensure_detector(stock_data)
    if detector is None:
        return []

    # Convert date format: YYYY-MM-DD -> YYYYMMDD
    date_ts = date.replace('-', '')

    # On first call or new date range, do warmup
    if _detector_date is None:
        # Get available dates from data
        sample_code = next(iter(detector.daily_data))
        sample_df = detector.daily_data[sample_code]
        all_dates = sorted(sample_df['trade_date'].unique())

        # Find warmup range
        target_idx = None
        for i, d in enumerate(all_dates):
            if d >= date_ts:
                target_idx = i
                break

        if target_idx is not None and target_idx > _PATTERN_WARMUP_DAYS:
            warmup_dates = all_dates[target_idx - _PATTERN_WARMUP_DAYS:target_idx]
            detector.warmup(warmup_dates)

    signals = detector.scan_date(date_ts, emit_signals=True)
    _latest_signals = signals
    _detector_date = date_ts

    return signals


def screen_pattern(stock_data, date):
    """Screener-compatible function: returns list of stock codes with pattern signals.

    Also stores full signal details in _latest_signals for the web generator.
    """
    signals = run_pattern_scan(stock_data, date)
    codes = [s['code'] for s in signals]
    return codes


def get_latest_pattern_signals():
    """Get full signal details from the most recent scan."""
    return _latest_signals
