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
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '2ed1b1d65a86d49ad17dc9af9f523fa4f6b5aa9880e2715db2da43c2')
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

# --- 形态识别 strategy parameters (投资笔记) ---
# 横盘≥22天 → 缩量涨停突破 → 倍量阴线洗盘 → 回收确认 + 无下影线
# Mining: 66 signals/486days, 27% 5-day limit-up rate, avg max gain 10%
PATTERN_CONSOL_MIN_DAYS = 22       # Min consolidation days (投资笔记: ≥1月 ≈ 22 trading days)
PATTERN_CONSOL_MAX_RANGE = 0.20    # Max price range during consolidation
PATTERN_NOT_PEAK_VOL = True        # Breakout vol < consolidation peak vol (缩量涨停)
PATTERN_PULLBACK_VOL_MIN = 1.5     # Pullback vol ≥ 1.5x breakout vol (倍量阴线)
PATTERN_PULLBACK_MAX_DROP = -0.10  # Max drop from breakout close
PATTERN_CONFIRM_MAX_SHADOW = 0.05  # Max lower shadow on confirm day (无下影线)
PATTERN_MAX_SPAN_DAYS = 2          # Days from breakout to confirmation (2 = strict 投资笔记)

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
ESCAPE_BOTTOM_SHORT_WINDOW = 18
ESCAPE_BOTTOM_SHORT_PCT = 0.05
ESCAPE_BOTTOM_LONG_WINDOW = 50
ESCAPE_BOTTOM_LONG_PCT = -0.15

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
CCI_CROSS_LEVEL = -120
CCI_DEEP_LEVEL = -130
CCI_DEEP_BEFORE7D_LOWER = -120
CCI_DEEP_BEFORE7D_UPPER = -80

# EMA proximity
EMA5_PROXIMITY_PCT = 0.03

# Sideways
SIDEWAYS_5D_PCT = 0.03
SIDEWAYS_15D_PCT = 0.03
SIDEWAYS_30D_PCT = 0.06

# Daily gain filters
DAILY_GAIN_GT5_PCT = 0.05
DAILY_GAIN_LT3_PCT = 0.02
DAILY_GAIN_GT2_PCT = 0.02

# CCI momentum floor (for OBV涨停梦 filter)
CCI_MOMENTUM_FLOOR = 50

# ADX buy filter: reject stocks with overextended trends
ADX_BUY_MAX = 40

# --- 涨停接力 strategy parameters ---
RELAY_RECENT_LU_DAYS = 5       # Must have limit-up within N trading days
RELAY_MIN_DAILY_GAIN = 0.05    # Minimum daily return (5%)
RELAY_CLOSE_POSITION_MIN = 0.9 # Close in top 10% of day's range (key discriminator)
RELAY_CCI_MIN = 50             # CCI momentum floor

# --- 蓄势突破 strategy parameters ---
# Core: gap-up hold (open > prev_high, low > prev_high, close > open)
# Enhancement: at least one multi-day accumulation pattern required
ACCUM_VOL_FLOOR_PERIODS = 3    # Volume floor rising: 3 periods of 3 days each
ACCUM_VOL_FLOOR_DAYS = 3       # Days per period
ACCUM_STAIRCASE_DAYS = 4       # Min consecutive higher-close + higher-low days
ACCUM_VOL_PRICE_SYNC_DAYS = 3  # Min consecutive rising close + rising volume
ACCUM_BREAKOUT_HIGH_WINDOW = 20  # N-day high breakout window

# --- 妖股放量 strategy parameters ---
# Data-mined from 2884 stocks with train/test validation (2025-07-28 split)
# Core finding: stocks with 3+ limit-ups in 20 days + volume explosion have
# 20-26% next-day limit-up probability (10-16x lift over 1.5% base rate)
MONSTER_LU_COUNT_20D = 3       # Min limit-up count in last 20 trading days
MONSTER_VOL_RATIO_3D_MIN = 3.0 # Today's volume >= 3x of 3-day moving average
MONSTER_UPPER_SHADOW_MAX = 0.10 # Max upper shadow ratio (tighter = higher LU%)

# --- 跳空涨停基因 strategy parameters ---
# Data-mined from 2884 stocks × 730 days, PF-optimized with train/test validation.
# Core: gap-up hold (strongest single pattern, 5.09% base LU) + recent LU DNA
# + strong close position = 17-22% LU rate AND PF>1 (profitable).
# Best results at 7% target:
#   close_pos≥0.7 + lu_in_10d: Train PF 1.27, Test PF 1.32, LU 17.3%/16.3%
#   close_pos≥0.8 + lu_in_10d: Train PF 1.58, Test PF 1.31, LU 22.1%/17.1%
GAP_LU_CLOSE_POS_MIN = 0.7    # Min close position (close-low)/(high-low)
GAP_LU_RECENT_DAYS = 10       # Must have limit-up within N trading days
GAP_LU_TARGET_PCT = 0.07      # Optimal target: 7% (higher than default 5%)

# --- 缩量突破 strategy parameters ---
# Data-mined from 2884 stocks × 486 days, PF-optimized with train/test validation.
# Core pattern: volume squeeze (institutional accumulation) → volume breakout above
# 10-day high with strong close = strongest PF strategy found (Test PF 2.39@7%).
# Phase 1 - Squeeze: vol ≤ 70% of 5d avg + non-positive return (quiet accumulation)
# Phase 2 - Breakout: vol ≥ 1.5x of 5d avg + close > 10d high + close_pos ≥ 0.8
SQUEEZE_VOL_RATIO_MAX = 0.7    # Squeeze day: volume ≤ 70% of 5-day average
SQUEEZE_MIN_DAYS = 1           # Min consecutive squeeze days before breakout
SQUEEZE_LOOKBACK = 20          # Max days to look back for squeeze period
BREAKOUT_VOL_RATIO_MIN = 1.5   # Breakout day: volume ≥ 1.5x of 5-day average
BREAKOUT_HIGH_WINDOW = 10      # Break N-day closing high
BREAKOUT_CLOSE_POS_MIN = 0.8   # Close position ≥ 80% of day's range
BREAKOUT_CCI_MIN = 50          # CCI > 50 (momentum confirmation)
BREAKOUT_NOT_NEW_HIGH_DAYS = 60  # Reject if close is at N-day high (避免冲高回落)
BREAKOUT_TARGET_PCT = 0.07     # Optimal target: 7%

# --- 群龙夺宝 strategy parameters ---
# 多个涨停板聚集在同一横盘平台 + 阳成团阴分散 + 红肥绿瘦 + 放量突破颈线
# Data-mined from 2884 stocks × 486 days, train/test validation.
# Strict version: N=208, Test PF@7%=1.11, 5d LU=30%, avg max gain 9.4%
# Sub-filter: +涨停≥3 Test PF@7%=2.05(N=18), +bvr≥3 Test PF@7%=1.37(N=31)
DRAGON_CONSOL_MIN_DAYS = 15    # Min consolidation days (横盘平台)
DRAGON_CONSOL_MAX_RANGE = 0.25 # Max price range during consolidation (≤25%)
DRAGON_MIN_LU_COUNT = 2        # Min limit-up count in platform (群龙: ≥2涨停)
DRAGON_MIN_DAILY_GAIN = 0.05   # Min daily gain on breakout day (≥5%)
DRAGON_BVR_MIN = 2.0           # Breakout volume ratio: vol ≥ 2x platform avg (倍量过颈线)
DRAGON_TARGET_PCT = 0.07       # Optimal target: 7%

# --- Backtest parameters ---
COMMISSION_RATE = 0.0003    # 万三
SLIPPAGE_RATE = 0.001       # 0.1%
STAMP_TAX_RATE = 0.001      # 印花税 千一 (卖出时收取)
DEFAULT_TARGET_PCT = 0.05   # Default 5% target (optimized from 3%)
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
