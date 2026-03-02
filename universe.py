"""Stock universe definition for SZ market.

Dynamically loads from tushare stock list cache.
Falls back to hardcoded list if cache not available.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def _load_sz_codes():
    """Load SZSE stock codes from tushare cache or raw daily files."""
    import pandas as pd

    # Try tushare stock_basic cache
    stock_list_path = os.path.join(config.RAW_DIR, 'stock_basic', 'SZSE.parquet')
    if os.path.exists(stock_list_path):
        df = pd.read_parquet(stock_list_path)
        if not df.empty and 'ts_code' in df.columns:
            codes = sorted(df['ts_code'].str.split('.').str[0].tolist())
            return codes

    # Fallback: scan raw daily cache directory
    daily_dir = os.path.join(config.RAW_DIR, 'daily')
    if os.path.exists(daily_dir):
        files = [f for f in os.listdir(daily_dir) if f.endswith('.parquet')]
        codes = sorted(f.replace('.parquet', '').replace('_SZ', '')
                       for f in files)
        if codes:
            return codes

    return []


sz_codes = _load_sz_codes()


def is_chinext(code: str) -> bool:
    """Check if a stock code belongs to ChiNext board (300/301 prefix, 20% limit)."""
    return code.startswith('300') or code.startswith('301')


def get_limit_ratio(code: str) -> float:
    """Return the limit-up ratio for a given stock code."""
    return config.CHINEXT_LIMIT_RATIO if is_chinext(code) else config.MAIN_BOARD_LIMIT_RATIO
