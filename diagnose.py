"""Feature diagnosis tool for strategy signal analysis.

Collects indicator values at each trade's buy date, splits into limit-up vs
non-limit-up groups, and performs statistical comparison to identify the most
discriminating features.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

import config
from backtest import BacktestEngine
from universe import is_chinext, get_limit_ratio


# Raw indicator columns produced by indicators.compute_all()
RAW_FEATURES = [
    'Rsi', 'Rsi12', 'EMA',
    'Streak Duration', 'Strike Rsi', 'Percentage rank', 'CRSI',
    'Volume percentage 7d', 'Volume percentage 3d', 'Volume percentage 1d',
    'ADOSC', 'OBV',
    'ADX', 'ADXR', '+DM', '-DM', 'UOS',
    'CCI', '3INSIDE',
]


def compute_derived_features(df: pd.DataFrame, code: str, buy_date: str) -> dict:
    """Compute derived features for one trade at buy_date.

    Args:
        df: stock DataFrame with indicators (DatetimeIndex).
        code: stock code string.
        buy_date: date string of the buy signal.

    Returns:
        dict of feature_name -> value.
    """
    ts = pd.Timestamp(buy_date)
    idx = df.index.get_loc(ts)
    row = df.iloc[idx]
    features = {}

    # daily return
    if idx >= 1:
        prev_close = df.iloc[idx - 1]['Close']
        features['daily_return'] = (row['Close'] - prev_close) / prev_close
    else:
        features['daily_return'] = np.nan

    # intraday amplitude
    features['intraday_amplitude'] = (row['High'] - row['Low']) / row['Close']

    # upper shadow ratio
    body_top = max(row['Open'], row['Close'])
    features['upper_shadow_ratio'] = (row['High'] - body_top) / (row['High'] - row['Low']) \
        if row['High'] != row['Low'] else 0.0

    # close position in range
    features['close_position_in_range'] = (row['Close'] - row['Low']) / (row['High'] - row['Low']) \
        if row['High'] != row['Low'] else 0.5

    # RSI gap (RSI6 - RSI12) short-term momentum
    features['rsi_gap'] = row['Rsi'] - row['Rsi12'] if pd.notna(row['Rsi']) and pd.notna(row['Rsi12']) else np.nan

    # RSI gap delta (1-day change)
    if idx >= 1:
        prev = df.iloc[idx - 1]
        prev_gap = prev['Rsi'] - prev['Rsi12'] if pd.notna(prev['Rsi']) and pd.notna(prev['Rsi12']) else np.nan
        curr_gap = features['rsi_gap']
        features['rsi_gap_delta'] = curr_gap - prev_gap if pd.notna(curr_gap) and pd.notna(prev_gap) else np.nan
    else:
        features['rsi_gap_delta'] = np.nan

    # ADX-ADXR spread (trend acceleration)
    features['adx_adxr_spread'] = row['ADX'] - row['ADXR'] \
        if pd.notna(row['ADX']) and pd.notna(row['ADXR']) else np.nan

    # DM spread (+DM minus -DM)
    features['dm_spread'] = row['+DM'] - row['-DM'] \
        if pd.notna(row['+DM']) and pd.notna(row['-DM']) else np.nan

    # OBV 5-day momentum
    if idx >= 5:
        features['obv_momentum_5d'] = row['OBV'] - df.iloc[idx - 5]['OBV']
    else:
        features['obv_momentum_5d'] = np.nan

    # volume ratio vs 5-day average
    if idx >= 5:
        avg_vol_5d = df.iloc[idx - 5:idx]['Volume'].mean()
        features['volume_ratio_5d'] = row['Volume'] / avg_vol_5d if avg_vol_5d > 0 else np.nan
    else:
        features['volume_ratio_5d'] = np.nan

    # pct from 20-day high
    if idx >= 20:
        high_20d = df.iloc[idx - 20:idx]['High'].max()
        features['pct_from_20d_high'] = (row['Close'] - high_20d) / high_20d
    else:
        features['pct_from_20d_high'] = np.nan

    # 5-day cumulative return
    if idx >= 5:
        close_5d_ago = df.iloc[idx - 5]['Close']
        features['return_5d'] = (row['Close'] - close_5d_ago) / close_5d_ago
    else:
        features['return_5d'] = np.nan

    # is ChiNext (20% limit vs 10%)
    features['is_chinext'] = 1.0 if is_chinext(code) else 0.0

    return features


def collect_trade_features(stock_data: dict[str, pd.DataFrame],
                           results_df: pd.DataFrame) -> pd.DataFrame:
    """Collect raw + derived features for every trade in backtest results.

    Args:
        stock_data: dict code -> DataFrame with indicators.
        results_df: backtest results DataFrame (from BacktestEngine.run).

    Returns:
        DataFrame indexed like results_df, with feature columns + hit_limit_up.
    """
    rows = []
    for _, trade in results_df.iterrows():
        code = trade['code']
        buy_date = trade['buy_date']
        df = stock_data[code]
        ts = pd.Timestamp(buy_date)

        if ts not in df.index:
            continue

        row_data = {
            'buy_date': buy_date,
            'code': code,
            'hit_limit_up': trade['hit_limit_up'],
        }

        # Raw features
        for col in RAW_FEATURES:
            if col in df.columns:
                val = df.loc[ts, col]
                row_data[col] = val if pd.notna(val) else np.nan
            else:
                row_data[col] = np.nan

        # Derived features
        derived = compute_derived_features(df, code, buy_date)
        row_data.update(derived)

        rows.append(row_data)

    return pd.DataFrame(rows)


def _cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def compare_groups(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compare limit-up vs non-limit-up groups across all features.

    Args:
        features_df: DataFrame from collect_trade_features.

    Returns:
        DataFrame with columns: feature, lu_mean, nlu_mean, diff, cohens_d,
        t_pvalue, u_pvalue, significance. Sorted by |cohens_d| descending.
    """
    lu = features_df[features_df['hit_limit_up'] == True]
    nlu = features_df[features_df['hit_limit_up'] == False]

    feature_cols = [c for c in features_df.columns if c not in ('buy_date', 'code', 'hit_limit_up')]

    records = []
    for feat in feature_cols:
        g1 = lu[feat].dropna()
        g2 = nlu[feat].dropna()

        if len(g1) < 2 or len(g2) < 2:
            continue

        lu_mean = g1.mean()
        nlu_mean = g2.mean()
        diff = lu_mean - nlu_mean
        d = _cohens_d(g1, g2)

        # Welch t-test
        try:
            _, t_p = stats.ttest_ind(g1, g2, equal_var=False)
        except Exception:
            t_p = np.nan

        # Mann-Whitney U test
        try:
            _, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        except Exception:
            u_p = np.nan

        # Significance based on the smaller p-value
        p = min(t_p, u_p) if pd.notna(t_p) and pd.notna(u_p) else (t_p if pd.notna(t_p) else u_p)
        if pd.isna(p):
            sig = ''
        elif p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        records.append({
            'feature': feat,
            'lu_mean': lu_mean,
            'nlu_mean': nlu_mean,
            'diff': diff,
            'cohens_d': d,
            't_pvalue': t_p,
            'u_pvalue': u_p,
            'significance': sig,
        })

    comp = pd.DataFrame(records)
    if not comp.empty:
        comp = comp.sort_values('cohens_d', key=abs, ascending=False).reset_index(drop=True)
    return comp


def print_diagnosis(comp_df: pd.DataFrame, lu_df: pd.DataFrame, nlu_df: pd.DataFrame) -> None:
    """Print formatted diagnosis output to terminal.

    Args:
        comp_df: comparison DataFrame from compare_groups.
        lu_df: limit-up subset of features DataFrame.
        nlu_df: non-limit-up subset of features DataFrame.
    """
    n_lu = len(lu_df)
    n_nlu = len(nlu_df)
    n_total = n_lu + n_nlu

    print(f"\n{'=' * 90}")
    print(f"  FEATURE DIAGNOSIS: {n_lu} limit-up / {n_nlu} non-limit-up ({n_total} total)")
    print(f"{'=' * 90}")

    if comp_df.empty:
        print("  No features to compare.")
        return

    # Header
    header = (f"{'Feature':<28} {'涨停均值':>10} {'非涨停均值':>10} {'差值':>10} "
              f"{'Cohen d':>9} {'t p-val':>9} {'U p-val':>9} {'显著':>4}")
    print(header)
    print('-' * 90)

    for _, row in comp_df.iterrows():
        sig = row['significance']
        line = (f"{row['feature']:<28} {row['lu_mean']:>10.4f} {row['nlu_mean']:>10.4f} "
                f"{row['diff']:>10.4f} {row['cohens_d']:>9.3f} "
                f"{row['t_pvalue']:>9.4f} {row['u_pvalue']:>9.4f} {sig:>4}")
        print(line)

    # Detail section for limit-up trades
    detail_cols = ['Rsi', 'CCI', 'Volume percentage 7d', 'ADX', 'OBV',
                   'daily_return', 'close_position_in_range', 'volume_ratio_5d',
                   'obv_momentum_5d']
    # Only show columns that exist
    detail_cols = [c for c in detail_cols if c in lu_df.columns]

    print(f"\n{'=' * 90}")
    print(f"  LIMIT-UP TRADE DETAILS ({n_lu} trades)")
    print(f"{'=' * 90}")

    for _, trade in lu_df.iterrows():
        parts = [f"{trade['buy_date']}  {trade['code']}"]
        for col in detail_cols:
            val = trade[col]
            if pd.notna(val):
                parts.append(f"{col}={val:.2f}")
        print('  ' + '  '.join(parts))


def run_diagnosis(stock_data: dict[str, pd.DataFrame],
                  screen_func,
                  screen_name: str,
                  days: list[str],
                  target_pct: float = None) -> None:
    """Main entry point: run backtest, collect features, compare groups, print.

    Args:
        stock_data: dict code -> DataFrame with indicators.
        screen_func: screen function(stock_data, date) -> list[str].
        screen_name: display name for the screen.
        days: list of trading day strings.
        target_pct: target profit pct (default from config).
    """
    print(f"\nRunning backtest for '{screen_name}'...")
    engine = BacktestEngine()
    results = engine.run(screen_func, stock_data, days, target_pct=target_pct)

    if results.empty:
        print("No trades found. Nothing to diagnose.")
        return

    n_lu = results['hit_limit_up'].sum()
    n_total = len(results)
    print(f"Found {n_total} trades, {n_lu} hit limit-up ({n_lu/n_total:.1%})")

    print("Collecting features...")
    features_df = collect_trade_features(stock_data, results)

    if features_df.empty:
        print("No feature data collected.")
        return

    lu_df = features_df[features_df['hit_limit_up'] == True]
    nlu_df = features_df[features_df['hit_limit_up'] == False]

    print("Comparing groups...")
    comp_df = compare_groups(features_df)
    print_diagnosis(comp_df, lu_df, nlu_df)
