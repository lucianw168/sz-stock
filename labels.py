"""Label construction for ML: limit-up (涨停) labels."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from universe import is_chinext, get_limit_ratio


def compute_limit_up_price(prev_close: float | pd.Series,
                           code: str) -> float | pd.Series:
    """Compute the limit-up price based on board type.

    Main board: 10%, ChiNext (300/301): 20%.
    Price is rounded to 2 decimal places (A-share convention).
    """
    ratio = get_limit_ratio(code)
    return np.round(prev_close * (1 + ratio), 2)


def build_labels(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Build t+1 limit-up labels.

    label=1 when both next-day close and high reach the limit-up price.
    Uses a tolerance of 0.1% (configurable via config.LIMIT_TOLERANCE).
    """
    prev_close = df['Close']
    limit_price = compute_limit_up_price(prev_close, code)

    # t+1 data
    next_close = df['Close'].shift(-1)
    next_high = df['High'].shift(-1)

    tolerance = config.LIMIT_TOLERANCE
    close_hit = next_close >= limit_price * (1 - tolerance)
    high_hit = next_high >= limit_price * (1 - tolerance)

    df['limit_up_price'] = limit_price
    df['label'] = (close_hit & high_hit).astype(int)
    # Last row has no next-day data
    df.loc[df.index[-1], 'label'] = np.nan

    return df


def build_all_labels(stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build labels for all stocks and optionally save features."""
    for code, df in stock_data.items():
        stock_data[code] = build_labels(df, code)
    return stock_data


def save_features(stock_data: dict[str, pd.DataFrame]) -> None:
    """Save features with labels to parquet."""
    frames = []
    for code, df in stock_data.items():
        tmp = df.copy()
        tmp['ticker'] = code
        frames.append(tmp)
    combined = pd.concat(frames)
    path = os.path.join(config.FEATURES_DIR, "features_with_labels.parquet")
    combined.to_parquet(path)
    print(f"Saved features with labels to {path}")
