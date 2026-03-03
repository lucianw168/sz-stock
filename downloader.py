"""Data download module: tushare API with local parquet caching.

Replaces yfinance with tushare for more complete Chinese A-stock data.
Output format is backward-compatible: capitalized columns (Open, High, Low,
Close, Volume) with DatetimeIndex.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import logging
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tushare as ts

import config

logger = logging.getLogger(__name__)


# ======================================================================
# Tushare downloader with caching
# ======================================================================

class TushareDownloader:
    """Tushare Pro API wrapper with parquet caching and rate limiting."""

    def __init__(self, token=None):
        token = token or config.TUSHARE_TOKEN
        if not token:
            raise ValueError('TUSHARE_TOKEN not set.')
        self.pro = ts.pro_api(token)
        self._last_call = 0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < config.API_CALL_INTERVAL:
            time.sleep(config.API_CALL_INTERVAL - elapsed)
        self._last_call = time.time()

    @staticmethod
    def _cache_path(api_name, key):
        d = os.path.join(config.RAW_DIR, api_name)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f'{key}.parquet')

    def _load_cache(self, api_name, key):
        path = self._cache_path(api_name, key)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def _save_cache(self, df, api_name, key):
        if df is not None and not df.empty:
            path = self._cache_path(api_name, key)
            df.to_parquet(path, index=False)

    def _load_or_fetch(self, api_name, key, fetch_fn):
        cached = self._load_cache(api_name, key)
        if cached is not None and not cached.empty:
            return cached
        self._throttle()
        try:
            df = fetch_fn()
        except Exception as e:
            logger.warning(f'API call failed ({api_name}/{key}): {e}')
            return pd.DataFrame()
        if df is not None and not df.empty:
            self._save_cache(df, api_name, key)
        return df if df is not None else pd.DataFrame()

    def _incremental_fetch(self, api_name, key, fetch_fn,
                           date_col, start_date, end_date):
        cached = self._load_cache(api_name, key)
        if cached is not None and not cached.empty and date_col in cached.columns:
            cached_max = str(cached[date_col].max())
            if cached_max >= end_date:
                return cached
            start_date = str(int(cached_max) + 1)

        self._throttle()
        try:
            new_df = fetch_fn(start_date, end_date)
        except Exception as e:
            logger.warning(f'Incremental fetch failed ({api_name}/{key}): {e}')
            return cached if cached is not None else pd.DataFrame()

        if new_df is None or new_df.empty:
            return cached if cached is not None else pd.DataFrame()

        if cached is not None and not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[date_col], keep='last')
            combined = combined.sort_values(date_col).reset_index(drop=True)
        else:
            combined = new_df

        self._save_cache(combined, api_name, key)
        return combined

    # --- API methods ---

    def fetch_stock_list(self):
        def _fetch():
            return self.pro.stock_basic(
                exchange='SZSE', list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
        return self._load_or_fetch('stock_basic', 'SZSE', _fetch)

    def fetch_trade_calendar(self, start_year, end_year):
        key = f'{start_year}_{end_year}'
        def _fetch():
            return self.pro.trade_cal(
                exchange='SZSE',
                start_date=f'{start_year}0101',
                end_date=f'{end_year}1231',
                fields='cal_date,is_open,pretrade_date'
            )
        return self._load_or_fetch('trade_cal', key, _fetch)

    def fetch_daily(self, ts_code, start_date, end_date):
        key = ts_code.replace('.', '_')
        def _fetch(s, e):
            return self.pro.daily(ts_code=ts_code, start_date=s, end_date=e)
        return self._incremental_fetch(
            'daily', key, _fetch, 'trade_date', start_date, end_date
        )

    def fetch_limit_list(self, trade_date):
        key = f'date_{trade_date}'
        def _fetch():
            return self.pro.limit_list_d(trade_date=trade_date)
        return self._load_or_fetch('limit_list_d', key, _fetch)

    def get_trading_dates(self, start_date, end_date):
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        cal = self.fetch_trade_calendar(start_year, end_year)
        if cal.empty:
            return []
        cal = cal[cal['is_open'] == 1]
        dates = cal['cal_date'].astype(str).tolist()
        return sorted([d for d in dates if start_date <= d <= end_date])


# ======================================================================
# Module-level interface (backward-compatible with old yfinance API)
# ======================================================================

_downloader = None


def _get_downloader():
    global _downloader
    if _downloader is None:
        _downloader = TushareDownloader()
    return _downloader


def _ts_code_to_code(ts_code):
    """Convert '000001.SZ' to '000001'."""
    return ts_code.split('.')[0]


def _code_to_ts_code(code):
    """Convert '000001' to '000001.SZ'."""
    return f'{code}.SZ'


def _apply_forward_adjustment(df):
    """Apply forward adjustment (前复权) using pre_close to detect ex-right dates.

    On ex-dividend/split dates, pre_close differs from the previous day's close.
    We use this gap to compute cumulative adjustment factors, anchoring the latest
    date at factor=1.0 so current prices remain unchanged.

    This avoids needing a separate adj_factor API call.
    """
    if 'pre_close' not in df.columns or 'close' not in df.columns:
        return df
    if len(df) < 2:
        return df

    df = df.sort_values('trade_date').copy()
    close_vals = df['close'].values.astype(float)
    pre_close_vals = df['pre_close'].values.astype(float)

    # ratio[i] = pre_close[i] / close[i-1]   (1.0 normally, <1.0 on ex-right)
    ratio = np.ones(len(df))
    mask = close_vals[:-1] != 0
    ratio[1:] = np.where(mask, pre_close_vals[1:] / close_vals[:-1], 1.0)

    # Ignore tiny floating-point noise
    ratio = np.where(np.abs(ratio - 1.0) > 0.001, ratio, 1.0)

    # Reverse cumulative product: rev_cumprod[i] = ratio[i] * ratio[i+1] * ... * ratio[N-1]
    rev_cumprod = np.cumprod(ratio[::-1])[::-1]

    # factors[i] = product of ratio[j] for j from i+1 to N-1
    # => factors[i] = rev_cumprod[i+1],  factors[N-1] = 1.0
    factors = np.ones(len(df))
    factors[:-1] = rev_cumprod[1:]

    # Apply to OHLC prices (volume stays unadjusted — same as yfinance auto_adjust)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col].values * factors

    return df


def _convert_daily_to_sz_format(df):
    """Convert tushare daily DataFrame to sz project format.

    Input: tushare columns (open, high, low, close, vol, trade_date YYYYMMDD)
    Output: capitalized columns (Open, High, Low, Close, Volume) with DatetimeIndex,
            with forward price adjustment (前复权) applied.
    """
    if df is None or df.empty:
        return None

    out = _apply_forward_adjustment(df)
    out.index = pd.to_datetime(out['trade_date'], format='%Y%m%d')
    out = out.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'vol': 'Volume',
    })
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    extra = [c for c in ['amount', 'pre_close', 'change', 'pct_chg'] if c in out.columns]
    out = out[cols + extra]
    out.sort_index(inplace=True)
    return out


# ======================================================================
# Public download functions
# ======================================================================

def download_full(start_date=None):
    """Full history download for all SZSE stocks via tushare.

    Returns:
        dict mapping stock code to DataFrame (sz format).
    """
    dl = _get_downloader()

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=config.LOOKBACK_DAYS)).strftime('%Y%m%d')
    else:
        start_date = start_date.replace('-', '')

    end_date = datetime.now().strftime('%Y%m%d')

    # Get stock list
    stock_list = dl.fetch_stock_list()
    if stock_list.empty:
        print("Failed to fetch stock list.")
        return {}

    ts_codes = sorted(stock_list['ts_code'].tolist())
    stock_data = {}
    failed = []

    print(f"Downloading daily data: {start_date} to {end_date}, "
          f"{len(ts_codes)} stocks...")
    for i, ts_code in enumerate(ts_codes):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(ts_codes)}")
        raw = dl.fetch_daily(ts_code, start_date, end_date)
        if raw is not None and not raw.empty:
            code = _ts_code_to_code(ts_code)
            converted = _convert_daily_to_sz_format(raw)
            if converted is not None:
                stock_data[code] = converted
        else:
            failed.append(ts_code)

    print(f"Downloaded {len(stock_data)} stocks, {len(failed)} failed.")

    # Download limit-up events for pattern detection
    print("Downloading limit-up event data...")
    trading_dates = dl.get_trading_dates(start_date, end_date)
    for i, td in enumerate(trading_dates):
        if (i + 1) % 50 == 0:
            print(f"  Limit list: {i + 1}/{len(trading_dates)}")
        dl.fetch_limit_list(td)
    print(f"Downloaded limit list for {len(trading_dates)} trading dates.")

    # Save combined processed data
    _save_raw(stock_data)
    save_processed(stock_data)
    return stock_data


def download_incremental(days=None):
    """Incremental download of recent data.

    Returns:
        dict mapping stock code to DataFrame (sz format).
    """
    dl = _get_downloader()

    if days is None:
        days = config.INCREMENTAL_DAYS

    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')

    stock_list = dl.fetch_stock_list()
    if stock_list.empty:
        print("No stock list available.")
        return {}

    ts_codes = sorted(stock_list['ts_code'].tolist())
    stock_data = {}

    print(f"Incremental download: last {days} days ({start_date} to {end_date})...")
    for i, ts_code in enumerate(ts_codes):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(ts_codes)}")
        raw = dl.fetch_daily(ts_code, start_date, end_date)
        if raw is not None and not raw.empty:
            code = _ts_code_to_code(ts_code)
            converted = _convert_daily_to_sz_format(raw)
            if converted is not None:
                stock_data[code] = converted

    # Update limit list
    trading_dates = dl.get_trading_dates(start_date, end_date)
    for td in trading_dates:
        dl.fetch_limit_list(td)

    print(f"Downloaded {len(stock_data)} stocks.")
    return stock_data


def merge_incremental(existing, incremental):
    """Merge incremental data into existing data by date index."""
    merged = {}
    all_codes = set(list(existing.keys()) + list(incremental.keys()))

    for code in all_codes:
        old = existing.get(code)
        new = incremental.get(code)
        if old is None and new is not None:
            merged[code] = new
        elif old is not None and new is None:
            merged[code] = old
        else:
            combined = pd.concat([old, new])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            merged[code] = combined

    return merged


def quality_check(stock_data):
    """Flag anomalous rows in the data."""
    for code, df in stock_data.items():
        anomaly = (
            (df['High'] < df[['Open', 'Close']].max(axis=1)) |
            (df['Close'] <= 0) |
            (df['Open'] <= 0) |
            (df['Volume'] < 0)
        )
        df['is_anomaly'] = anomaly.astype(int)
    return stock_data


def _save_raw(stock_data):
    """Save combined raw data to parquet."""
    if not stock_data:
        return
    frames = []
    for code, df in stock_data.items():
        tmp = df.copy()
        tmp['ticker'] = code
        frames.append(tmp)
    combined = pd.concat(frames)
    path = os.path.join(config.RAW_DIR, "full_history.parquet")
    combined.to_parquet(path)
    print(f"Saved raw data to {path}")


def save_processed(stock_data):
    """Save processed data to parquet."""
    if not stock_data:
        return
    frames = []
    for code, df in stock_data.items():
        tmp = df.copy()
        tmp['ticker'] = code
        frames.append(tmp)
    combined = pd.concat(frames)
    path = os.path.join(config.PROCESSED_DIR, "ohlcv_sz.parquet")
    combined.to_parquet(path)
    print(f"Saved processed data to {path}")


def load_processed():
    """Load processed data, return as dict of DataFrames.

    Tries combined parquet first, then falls back to raw tushare cache.
    """
    # Try combined processed file
    path = os.path.join(config.PROCESSED_DIR, 'ohlcv_sz.parquet')
    if os.path.exists(path):
        combined = pd.read_parquet(path)
        stock_data = {}
        for code, group in combined.groupby('ticker'):
            df = group.drop(columns=['ticker'])
            df.sort_index(inplace=True)
            stock_data[code] = df
        return stock_data

    # Fallback: load from raw tushare cache
    daily_dir = os.path.join(config.RAW_DIR, 'daily')
    if not os.path.exists(daily_dir):
        return None

    stock_data = {}
    files = sorted(f for f in os.listdir(daily_dir) if f.endswith('.parquet'))
    print(f"Loading from raw cache: {len(files)} files...")
    for f in files:
        raw = pd.read_parquet(os.path.join(daily_dir, f))
        if raw.empty:
            continue
        ts_code = (raw['ts_code'].iloc[0] if 'ts_code' in raw.columns
                   else f.replace('.parquet', '').replace('_', '.'))
        code = _ts_code_to_code(ts_code)
        converted = _convert_daily_to_sz_format(raw)
        if converted is not None:
            stock_data[code] = converted

    if stock_data:
        print(f"Loaded {len(stock_data)} stocks from raw cache.")
        # Build processed file for faster future loads
        save_processed(stock_data)

    return stock_data if stock_data else None


def load_limit_events():
    """Load all cached limit-up events as a single DataFrame.

    Used by pattern detection for seal quality analysis.
    """
    limit_dir = os.path.join(config.RAW_DIR, 'limit_list_d')
    if not os.path.exists(limit_dir):
        return pd.DataFrame()

    frames = []
    for f in sorted(os.listdir(limit_dir)):
        if f.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(limit_dir, f))
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_daily_tushare_format():
    """Load daily data in tushare native format (lowercase columns, trade_date string).

    Used by pattern detector which expects tushare format.
    Returns: dict[ts_code, DataFrame]
    """
    daily_dir = os.path.join(config.RAW_DIR, 'daily')
    if not os.path.exists(daily_dir):
        return {}

    stock_data = {}
    files = sorted(f for f in os.listdir(daily_dir) if f.endswith('.parquet'))
    for f in files:
        raw = pd.read_parquet(os.path.join(daily_dir, f))
        if raw.empty:
            continue
        ts_code = (raw['ts_code'].iloc[0] if 'ts_code' in raw.columns
                   else f.replace('.parquet', '').replace('_', '.'))
        raw = raw.sort_values('trade_date').reset_index(drop=True)
        raw['trade_date'] = raw['trade_date'].astype(str)
        stock_data[ts_code] = raw

    return stock_data


# ======================================================================
# Import data from limit_up project (avoids re-downloading)
# ======================================================================

def import_from_limit_up(limit_up_raw_dir=None):
    """Import cached tushare data from the limit_up project.

    Args:
        limit_up_raw_dir: Path to limit_up/data/raw directory.
    """
    if limit_up_raw_dir is None:
        # Default: relative path from sz project to limit_up project
        limit_up_raw_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'muti_factor', 'limit_up', 'data', 'raw'
        )
        limit_up_raw_dir = os.path.normpath(limit_up_raw_dir)

    if not os.path.exists(limit_up_raw_dir):
        print(f"Source not found: {limit_up_raw_dir}")
        return False

    for subdir in ['daily', 'limit_list_d', 'stock_basic', 'trade_cal']:
        src = os.path.join(limit_up_raw_dir, subdir)
        dst = os.path.join(config.RAW_DIR, subdir)
        if not os.path.exists(src):
            continue

        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            files = [f for f in os.listdir(src) if f.endswith('.parquet')]
            copied = 0
            for f in files:
                src_file = os.path.join(src, f)
                dst_file = os.path.join(dst, f)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied += 1
            print(f"  Imported {subdir}: {copied} new files "
                  f"({len(files)} total)")
        else:
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  Imported {subdir}")

    # Also copy stock_list.parquet if exists
    for extra_file in ['stock_list.parquet']:
        src = os.path.join(limit_up_raw_dir, extra_file)
        dst = os.path.join(config.RAW_DIR, extra_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Imported {extra_file}")

    print("Import complete.")
    return True


# ======================================================================
# Real-time data (intraday)
# ======================================================================

def _fetch_rt_tushare():
    """Fetch real-time daily K-line for all SZSE stocks via pro.rt_k().

    Returns:
        dict[str, dict]: code -> {Open, High, Low, Close, Volume, pre_close}
        Volume is converted from 股 to 手 to match pro.daily() units.
    """
    try:
        dl = _get_downloader()
    except ValueError as e:
        print(f"  Tushare not configured: {e}")
        return {}
    dl._throttle()

    try:
        # Main board (0*) + ChiNext (3*)
        df = dl.pro.rt_k(ts_code='0*.SZ,3*.SZ')
    except Exception as e:
        logger.warning(f'rt_k API call failed: {e}')
        print(f"  Tushare rt_k failed: {e}")
        return {}

    if df is None or df.empty:
        return {}

    today = datetime.now().strftime('%Y-%m-%d')
    result = {}

    for _, row in df.iterrows():
        ts_code = row['ts_code']
        code = _ts_code_to_code(ts_code)

        if pd.isna(row['close']) or row['close'] <= 0:
            continue
        if pd.isna(row['open']) or row['open'] <= 0:
            continue

        result[code] = {
            'date': today,
            'Open': float(row['open']),
            'High': float(row['high']),
            'Low': float(row['low']),
            'Close': float(row['close']),
            'Volume': float(row['vol']) / 100,  # 股 → 手 (match daily API)
            'amount': float(row['amount']) / 1000 if pd.notna(row.get('amount')) else 0,
            'pre_close': float(row['pre_close']),
        }

    return result


def _fetch_rt_yfinance(stock_codes):
    """Fetch real-time data via yfinance as fallback (~15 min delay).

    Args:
        stock_codes: iterable of stock codes (e.g., ['000001', '000002'])

    Returns:
        dict[str, dict] — same format as _fetch_rt_tushare().
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. pip install yfinance")
        return {}

    codes = list(stock_codes)
    tickers = [f"{c}.SZ" for c in codes]
    today = datetime.now().strftime('%Y-%m-%d')
    result = {}

    BATCH = 500
    total_batches = (len(tickers) + BATCH - 1) // BATCH

    for b in range(total_batches):
        batch = tickers[b * BATCH : (b + 1) * BATCH]
        print(f"    yfinance batch {b + 1}/{total_batches} "
              f"({len(batch)} tickers)...")

        try:
            df = yf.download(
                batch, period='1d', progress=False,
                threads=True, group_by='ticker',
            )
        except Exception as e:
            print(f"    yfinance download error: {e}")
            continue

        if df is None or df.empty:
            continue

        multi_ticker = len(batch) > 1

        for ticker in batch:
            code = ticker.split('.')[0]
            try:
                if multi_ticker:
                    ohlcv = df[ticker].dropna()
                else:
                    ohlcv = df.dropna()

                if ohlcv.empty:
                    continue

                row = ohlcv.iloc[-1]
                close_val = float(row['Close'])
                if close_val <= 0:
                    continue

                result[code] = {
                    'date': today,
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': close_val,
                    'Volume': float(row['Volume']) / 100,  # 股 → 手
                    'amount': 0,
                    'pre_close': 0,  # yfinance 不提供，inject_realtime 会回退
                }
            except (KeyError, IndexError, TypeError):
                continue

    return result


def fetch_realtime(stock_codes=None):
    """Fetch real-time quotes: try Tushare rt_k first, fall back to yfinance.

    Args:
        stock_codes: iterable of codes for yfinance fallback.
                     If None, yfinance fallback is skipped.

    Returns:
        (dict[str, dict], source_str)
    """
    data = _fetch_rt_tushare()
    if data:
        return data, 'tushare'

    if stock_codes is None:
        return {}, 'none'

    print("  Tushare unavailable, falling back to yfinance (≈15 min delay)...")
    data = _fetch_rt_yfinance(stock_codes)
    if data:
        return data, 'yfinance'

    return {}, 'none'


def inject_realtime(stock_data, realtime):
    """Inject real-time data as today's row into each stock's DataFrame.

    If today's row already exists it is replaced with the latest quote.
    When pre_close is missing (yfinance fallback), it is derived from
    the last historical close.

    Args:
        stock_data: dict[code, DataFrame] — historical daily data
        realtime: dict[code, dict] from fetch_realtime()

    Returns:
        (stock_data, injected_count)
    """
    today = datetime.now().strftime('%Y-%m-%d')
    today_ts = pd.Timestamp(today)

    injected = 0
    for code, rt in realtime.items():
        if code not in stock_data:
            continue

        df = stock_data[code]

        # Derive pre_close from last historical close when missing
        pre_close = rt.get('pre_close', 0)
        if pre_close <= 0:
            hist = df[df.index < today_ts]
            if not hist.empty:
                pre_close = float(hist['Close'].iloc[-1])

        new_row = pd.DataFrame({
            'Open': [rt['Open']],
            'High': [rt['High']],
            'Low': [rt['Low']],
            'Close': [rt['Close']],
            'Volume': [rt['Volume']],
        }, index=[today_ts])

        # Carry optional columns
        if 'amount' in df.columns:
            new_row['amount'] = rt.get('amount', 0)
        if 'pre_close' in df.columns:
            new_row['pre_close'] = pre_close
        if 'change' in df.columns:
            new_row['change'] = rt['Close'] - pre_close if pre_close > 0 else 0
        if 'pct_chg' in df.columns:
            new_row['pct_chg'] = (
                (rt['Close'] - pre_close) / pre_close * 100
                if pre_close > 0 else 0
            )

        # Drop existing today row, append new one
        if today_ts in df.index:
            df = df.drop(today_ts)

        stock_data[code] = pd.concat([df, new_row]).sort_index()
        injected += 1

    return stock_data, injected
