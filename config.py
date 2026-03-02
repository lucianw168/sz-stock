"""Global configuration for SZ quantitative trading system."""

import os
from datetime import datetime, timedelta

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
MODEL_DIR = os.path.join(DATA_DIR, "models")

for _d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, PREDICTIONS_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)

# --- Tushare ---
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
API_CALL_INTERVAL = 0.12  # seconds between API calls (500/min limit)

# --- Date configuration ---
# Auto-derive from system time; override via CLI args if needed
TODAY = datetime.now().strftime("%Y-%m-%d")
TOMORROW = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

# --- Download parameters ---
LOOKBACK_DAYS = 730  # ~2 years
INCREMENTAL_DAYS = 60

# --- Limit-up ratios ---
MAIN_BOARD_LIMIT_RATIO = 0.10   # Main board: 10%
CHINEXT_LIMIT_RATIO = 0.20      # ChiNext (300/301): 20%
LIMIT_TOLERANCE = 0.001         # 0.1% tolerance for limit-up detection

# --- Pattern system ---
PATTERN_LIBRARY_PATH = os.path.join(MODEL_DIR, 'pattern_library.pkl')
PATTERN_WARMUP_DAYS = 60
PATTERN_STOP_LOSS_PCT = -0.05   # wider stop for pattern trades
PATTERN_MAX_SIGNALS_PER_DAY = 3

# --- Strategy parameters (centralized, no more hardcoding) ---
# CRSI thresholds
CRSI_CROSS50_UPPER = 80
CRSI_ABOVE50_LOWER = 50
CRSI_ABOVE50_UPPER = 58
CRSI_BELOW25_THRESHOLD = 25
CRSI_DECREASE_DELTA = -15
CRSI_CROSS20_THRESHOLD = 20

# RSI thresholds
RSI_GOLDEN_CROSS_DELTA = 4
RSI_STRENGTHENING_DELTA = 1

# Volume thresholds
VOLUME_7D_LOWER = 52
VOLUME_7D_UPPER = 60
VOLUME_3D_UPPER = 40
OBV_CONSOLIDATION_WINDOW = 30

# UOS threshold
UOS_CROSS_LEVEL = 65

# Escape bottom
ESCAPE_BOTTOM_SHORT_WINDOW = 15
ESCAPE_BOTTOM_SHORT_PCT = 0.08
ESCAPE_BOTTOM_LONG_WINDOW = 55
ESCAPE_BOTTOM_LONG_PCT = -0.2

# Percentage rank spike
PCT_RANK_SPIKE_THRESHOLD = 60
PCT_RANK_SPIKE_HIGH = 85
PCT_RANK_SPIKE_LOW = 30

# Price filters
NO_LIMIT_UP_PCT = 0.05
AVOID_HIGH_PCT = 0.15
AVOID_HIGH_WINDOW = 7

# ADX/ADXR
ADXR_ABOVE_THRESHOLD = 25

# CCI
CCI_CROSS_LEVEL = -100
CCI_DEEP_LEVEL = -110
CCI_DEEP_BEFORE7D_LOWER = -100
CCI_DEEP_BEFORE7D_UPPER = -70

# EMA proximity
EMA5_PROXIMITY_PCT = 0.03

# Sideways
SIDEWAYS_5D_PCT = 0.03
SIDEWAYS_15D_PCT = 0.03
SIDEWAYS_30D_PCT = 0.06

# Daily gain filters
DAILY_GAIN_GT5_PCT = 0.05
DAILY_GAIN_LT3_PCT = 0.03
DAILY_GAIN_GT2_PCT = 0.02

# CCI momentum floor (for OBV涨停梦 filter)
CCI_MOMENTUM_FLOOR = 50

# ADX buy filter: reject stocks with overextended trends
ADX_BUY_MAX = 40

# --- Backtest parameters ---
COMMISSION_RATE = 0.0003    # 万三
SLIPPAGE_RATE = 0.001       # 0.1%
STAMP_TAX_RATE = 0.001      # 印花税 千一 (卖出时收取)
DEFAULT_TARGET_PCT = 0.03   # Default 3% target
SHARES_PER_TRADE = 100      # Shares per trade

# --- Indicator parameters ---
RSI_SHORT_PERIOD = 6
RSI_LONG_PERIOD = 12
EMA_PERIOD = 5
STREAK_RSI_DEFAULT = 50
PCT_RANK_WINDOW = 20
CRSI_WEIGHTS = (3, 1, 1)    # RSI weight, Strike RSI weight, Pct Rank weight
ADOSC_FAST = 3
ADOSC_SLOW = 10
ADX_PERIOD = 8
CCI_PERIOD = 14
UOS_PERIODS = (7, 14, 28)
