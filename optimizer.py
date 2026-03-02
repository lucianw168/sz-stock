"""Parameter optimization engine with walk-forward validation.

Two-phase grid search:
  Phase 1: Signal combination search (default params, walk-forward)
  Phase 2: Parameter + trading rule tuning for top combos (walk-forward)

Objective function: win_rate * avg_return

Performance: precomputes all signal boolean Series once per stock,
then uses fast index lookups during backtesting instead of
recomputing rolling operations on every screen call.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from itertools import combinations, product
import time

import numpy as np
import pandas as pd

import config
from strategies import SIGNAL_REGISTRY, SIGNAL_PARAMS
from backtest import BacktestEngine


# ============================================================
# Utilities
# ============================================================

class ConfigOverride:
    """Context manager to temporarily override config attributes."""

    def __init__(self, overrides: dict):
        self.overrides = overrides
        self.originals = {}

    def __enter__(self):
        for key, value in self.overrides.items():
            self.originals[key] = getattr(config, key)
            setattr(config, key, value)
        return self

    def __exit__(self, *exc):
        for key, value in self.originals.items():
            setattr(config, key, value)
        return False


# ============================================================
# Signal cache — the key performance optimization
# ============================================================

class SignalCache:
    """Precompute and cache signal boolean Series for all stocks.

    Instead of recomputing rolling operations on every screen call,
    signals are computed once and looked up by date during backtesting.
    """

    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.cache = {}   # signal_name -> {code: pd.Series[bool]}

    def ensure(self, signal_names):
        """Precompute any signals not yet in cache."""
        for name in signal_names:
            if name in self.cache:
                continue
            func = SIGNAL_REGISTRY[name]
            self.cache[name] = {
                code: func(df) for code, df in self.stock_data.items()
            }

    def invalidate(self, signal_names):
        """Remove cached results (call before recomputing with new config)."""
        for name in signal_names:
            self.cache.pop(name, None)

    def precompute_selections(self, signal_names, all_days):
        """Precompute which stocks are selected on each day.

        Returns a screen function where each call is O(1) dict lookup.
        """
        selections = {}
        for day in all_days:
            ts = pd.Timestamp(day)
            selected = []
            for code in self.stock_data:
                all_pass = True
                for name in signal_names:
                    series = self.cache[name].get(code)
                    if series is None or ts not in series.index or not series.loc[ts]:
                        all_pass = False
                        break
                if all_pass:
                    selected.append(code)
            if selected:
                selections[day] = selected

        def precomputed_screen(stock_data, date):
            return selections.get(date, [])

        return precomputed_screen


# ============================================================
# Parameter grid definitions
# ============================================================

PARAM_GRIDS = {
    'CRSI_CROSS50_UPPER':         [70, 75, 80, 85, 90],
    'CRSI_ABOVE50_LOWER':         [45, 48, 50, 52, 55],
    'CRSI_ABOVE50_UPPER':         [55, 58, 62, 65],
    'CRSI_BELOW25_THRESHOLD':     [20, 25, 30],
    'CRSI_DECREASE_DELTA':        [-20, -15, -12, -10],
    'CRSI_CROSS20_THRESHOLD':     [15, 18, 20, 22, 25],
    'RSI_STRENGTHENING_DELTA':    [0.5, 1, 1.5, 2],
    'VOLUME_7D_LOWER':            [45, 48, 52, 55, 58],
    'VOLUME_7D_UPPER':            [55, 58, 60, 63, 65],
    'VOLUME_3D_UPPER':            [35, 38, 40, 43, 45],
    'OBV_CONSOLIDATION_WINDOW':   [20, 25, 30, 35, 40],
    'UOS_CROSS_LEVEL':            [55, 60, 65, 70],
    'ESCAPE_BOTTOM_SHORT_WINDOW': [10, 12, 15, 18, 20],
    'ESCAPE_BOTTOM_SHORT_PCT':    [0.05, 0.06, 0.08, 0.10, 0.12],
    'ESCAPE_BOTTOM_LONG_WINDOW':  [45, 50, 55, 60, 65],
    'ESCAPE_BOTTOM_LONG_PCT':     [-0.25, -0.22, -0.20, -0.18, -0.15],
    'PCT_RANK_SPIKE_THRESHOLD':   [50, 55, 60, 65],
    'PCT_RANK_SPIKE_HIGH':        [80, 85, 90],
    'PCT_RANK_SPIKE_LOW':         [25, 28, 30, 33, 35],
    'NO_LIMIT_UP_PCT':            [0.04, 0.05, 0.06, 0.07],
    'AVOID_HIGH_PCT':             [0.10, 0.12, 0.15, 0.18, 0.20],
    'AVOID_HIGH_WINDOW':          [5, 7, 10],
    'ADXR_ABOVE_THRESHOLD':       [20, 22, 25, 28, 30],
    'CCI_CROSS_LEVEL':            [-120, -110, -100, -90],
    'CCI_DEEP_LEVEL':             [-130, -120, -110, -100],
    'CCI_DEEP_BEFORE7D_LOWER':    [-120, -110, -100, -90],
    'CCI_DEEP_BEFORE7D_UPPER':    [-80, -70, -60, -50],
    'EMA5_PROXIMITY_PCT':         [0.02, 0.025, 0.03, 0.035, 0.04],
    'SIDEWAYS_5D_PCT':            [0.02, 0.025, 0.03, 0.035, 0.04],
    'SIDEWAYS_15D_PCT':           [0.02, 0.025, 0.03, 0.035, 0.04],
    'SIDEWAYS_30D_PCT':           [0.04, 0.05, 0.06, 0.07, 0.08],
    'DAILY_GAIN_GT5_PCT':         [0.04, 0.045, 0.05, 0.055, 0.06],
    'DAILY_GAIN_LT3_PCT':         [0.02, 0.025, 0.03, 0.035, 0.04],
}

TARGET_PCT_GRID = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]

# Signal pairs that should not appear together
SIGNAL_CONFLICTS = [
    ('rsi_golden_cross', 'rsi_declining'),
    ('daily_gain_gt5', 'daily_gain_lt3'),
    ('crsi_cross50', 'crsi_below25'),
]


# ============================================================
# Walk-forward validation
# ============================================================

class WalkForwardValidator:
    """Expanding-window walk-forward validator."""

    def __init__(self, min_train_days: int = 120, test_window: int = 20):
        self.min_train_days = min_train_days
        self.test_window = test_window
        self.engine = BacktestEngine()

    def generate_folds(self, all_days):
        """Generate (train_days, test_days) pairs with expanding window."""
        n = len(all_days)
        folds = []
        test_start = self.min_train_days
        while test_start + self.test_window <= n:
            folds.append((
                all_days[:test_start],
                all_days[test_start:test_start + self.test_window],
            ))
            test_start += self.test_window
        return folds

    def evaluate(self, screen_func, stock_data, all_days, target_pct=None):
        """Run walk-forward evaluation. Returns aggregated metrics dict."""
        if target_pct is None:
            target_pct = config.DEFAULT_TARGET_PCT

        folds = self.generate_folds(all_days)
        if not folds:
            return self._empty()

        total_trades = 0
        weighted_wr = 0.0
        weighted_ret = 0.0
        fold_details = []

        for train_days, test_days in folds:
            # Train: quick viability check
            train_res = self.engine.run(screen_func, stock_data, train_days, target_pct)
            if len(train_res) < 3:
                continue

            # Test: out-of-sample performance
            test_res = self.engine.run(screen_func, stock_data, test_days, target_pct)
            test_rpt = self.engine.generate_report(test_res)
            n = test_rpt['total_trades']

            if n == 0:
                fold_details.append({
                    'test_start': test_days[0], 'test_end': test_days[-1],
                    'trades': 0, 'win_rate': 0, 'avg_return': 0,
                })
                continue

            avg_ret = test_res['pnl_pct'].mean()
            fold_details.append({
                'test_start': test_days[0], 'test_end': test_days[-1],
                'trades': n, 'win_rate': test_rpt['win_rate'], 'avg_return': avg_ret,
            })
            total_trades += n
            weighted_wr += test_rpt['win_rate'] * n
            weighted_ret += avg_ret * n

        if total_trades == 0:
            return self._empty()

        wr = weighted_wr / total_trades
        ret = weighted_ret / total_trades
        return {
            'avg_win_rate': wr,
            'avg_return': ret,
            'objective': wr * ret,
            'total_trades': total_trades,
            'n_folds': len(folds),
            'n_folds_active': sum(1 for f in fold_details if f['trades'] > 0),
            'fold_details': fold_details,
        }

    def _empty(self):
        return {
            'avg_win_rate': 0, 'avg_return': 0, 'objective': -999,
            'total_trades': 0, 'n_folds': 0, 'n_folds_active': 0,
            'fold_details': [],
        }


# ============================================================
# Signal combination generator
# ============================================================

def generate_signal_combos(focus_signals=None, min_signals=2, max_signals=4):
    """Generate candidate signal combinations."""
    focus = focus_signals or []
    for s in focus:
        if s not in SIGNAL_REGISTRY:
            raise ValueError(f"Unknown signal: {s}")

    remaining = [s for s in SIGNAL_REGISTRY if s not in focus]

    combos = []
    for size in range(min_signals, max_signals + 1):
        n_extra = size - len(focus)
        if n_extra < 0:
            continue
        if n_extra == 0:
            combos.append(tuple(sorted(focus)))
            continue
        for extra in combinations(remaining, n_extra):
            combo = tuple(sorted(focus + list(extra)))
            has_conflict = any(a in combo and b in combo for a, b in SIGNAL_CONFLICTS)
            if not has_conflict:
                combos.append(combo)

    return combos


# ============================================================
# Parameter grid helpers
# ============================================================

def get_param_grid(signal_names):
    """Collect tunable parameters for a set of signals (deduplicated)."""
    params = {}
    for sig in signal_names:
        for p in SIGNAL_PARAMS.get(sig, []):
            if p not in params and p in PARAM_GRIDS:
                params[p] = PARAM_GRIDS[p]
    return params


def enumerate_param_combos(param_grid, max_combos=5000):
    """Generate all parameter value combinations from a grid dict."""
    if not param_grid:
        return [{}]
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
        if len(combos) >= max_combos:
            break
    return combos


def get_affected_signals(param_overrides, signal_names):
    """Determine which signals are affected by a set of param overrides."""
    affected = set()
    override_keys = set(param_overrides.keys())
    for sig in signal_names:
        sig_params = set(SIGNAL_PARAMS.get(sig, []))
        if sig_params & override_keys:
            affected.add(sig)
    return affected


# ============================================================
# Optimizer
# ============================================================

class Optimizer:
    """Two-phase parameter optimization engine.

    Phase 1: search signal combinations with default params.
    Phase 2: grid-search params + target_pct for top combos.

    Uses SignalCache for O(1) signal lookups during backtesting.
    """

    def __init__(self, stock_data, focus_signals=None,
                 min_signals=2, max_signals=4,
                 train_window=120, test_window=20, top_n=5):
        self.stock_data = stock_data
        self.focus_signals = focus_signals
        self.min_signals = min_signals
        self.max_signals = max_signals
        self.top_n = top_n

        # Derive trading days
        sample_df = next(iter(stock_data.values()))
        self.all_days = sorted(sample_df.index.strftime('%Y-%m-%d').tolist())

        self.validator = WalkForwardValidator(
            min_train_days=train_window,
            test_window=test_window,
        )

        self.signal_cache = SignalCache(stock_data)
        self.phase1_results = []
        self.phase2_results = []

    def run(self):
        """Execute both phases. Returns ranked results list."""
        print("=" * 60)
        print("  PARAMETER OPTIMIZATION")
        print(f"  {len(self.all_days)} trading days  |  "
              f"{len(self.stock_data)} stocks")
        print(f"  walk-forward: train>={self.validator.min_train_days}d, "
              f"test={self.validator.test_window}d")
        print("=" * 60)

        self._phase1()
        if self.phase1_results:
            self._phase2()

        final = sorted(self.phase2_results, key=lambda r: r['objective'], reverse=True)
        self._print_results(final)
        return final

    # ---- Phase 1: signal combination search ----

    def _phase1(self):
        print(f"\n--- Phase 1: Signal Combination Search ---")

        # Precompute ALL signals with default params
        print(f"  Precomputing {len(SIGNAL_REGISTRY)} signals for "
              f"{len(self.stock_data)} stocks...")
        t0 = time.time()
        self.signal_cache.ensure(list(SIGNAL_REGISTRY.keys()))
        print(f"  Done in {time.time() - t0:.1f}s")

        combos = generate_signal_combos(
            self.focus_signals, self.min_signals, self.max_signals,
        )
        print(f"  Evaluating {len(combos)} candidate combinations...")

        results = []
        for i, combo in enumerate(combos):
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(combos)}")

            screen = self.signal_cache.precompute_selections(
                list(combo), self.all_days,
            )
            metrics = self.validator.evaluate(
                screen, self.stock_data, self.all_days,
            )

            if metrics['total_trades'] < 5:
                continue

            results.append({
                'signals': combo,
                'params': {},
                'target_pct': config.DEFAULT_TARGET_PCT,
                **metrics,
            })

        results.sort(key=lambda r: r['objective'], reverse=True)
        self.phase1_results = results[:self.top_n]

        print(f"\n  Phase 1 done. {len(results)} viable combos found.")
        print(f"  Top {len(self.phase1_results)}:")
        for i, r in enumerate(self.phase1_results):
            sigs = ' + '.join(r['signals'])
            print(f"    {i+1}. [{r['objective']:.6f}] wr={r['avg_win_rate']:.1%} "
                  f"ret={r['avg_return']:.4f} trades={r['total_trades']}  {sigs}")

    # ---- Phase 2: parameter + trading rule tuning ----

    def _phase2(self):
        print(f"\n--- Phase 2: Parameter Tuning ---")

        all_results = []
        for rank, p1 in enumerate(self.phase1_results):
            combo = p1['signals']
            combo_list = list(combo)
            print(f"\n  [{rank+1}/{len(self.phase1_results)}] Tuning: {' + '.join(combo)}")

            param_grid = get_param_grid(combo_list)
            param_combos = enumerate_param_combos(param_grid)
            total = len(param_combos) * len(TARGET_PCT_GRID)

            # Subsample if too large
            if total > 1000:
                max_pc = 1000 // len(TARGET_PCT_GRID)
                if max_pc < len(param_combos):
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(param_combos), max_pc, replace=False)
                    param_combos = [param_combos[i] for i in sorted(idx)]
                    total = len(param_combos) * len(TARGET_PCT_GRID)

            print(f"    {len(param_grid)} tunable params  |  "
                  f"{len(param_combos)} param combos x {len(TARGET_PCT_GRID)} targets "
                  f"= {total} evaluations")

            done = 0
            last_params = None

            for params in param_combos:
                # Recompute only signals affected by param changes
                if params != last_params:
                    if params:
                        affected = get_affected_signals(params, combo_list)
                        if affected:
                            self.signal_cache.invalidate(affected)
                            with ConfigOverride(params):
                                self.signal_cache.ensure(affected)
                    elif last_params:
                        affected = get_affected_signals(last_params, combo_list)
                        if affected:
                            self.signal_cache.invalidate(affected)
                            self.signal_cache.ensure(affected)
                    last_params = params

                # Precompute selections once per param combo (O(days*stocks))
                # then each target_pct evaluation uses O(1) screen lookups
                screen = self.signal_cache.precompute_selections(
                    combo_list, self.all_days,
                )

                for tgt in TARGET_PCT_GRID:
                    done += 1
                    if done % 200 == 0:
                        print(f"    ... {done}/{total}")

                    metrics = self.validator.evaluate(
                        screen, self.stock_data, self.all_days, target_pct=tgt,
                    )

                    if metrics['total_trades'] < 5:
                        continue

                    all_results.append({
                        'signals': combo,
                        'params': params.copy(),
                        'target_pct': tgt,
                        **metrics,
                    })

            # Restore cache to default params for next combo
            if last_params:
                affected = get_affected_signals(last_params, combo_list)
                if affected:
                    self.signal_cache.invalidate(affected)
                    self.signal_cache.ensure(affected)

        all_results.sort(key=lambda r: r['objective'], reverse=True)
        self.phase2_results = all_results
        print(f"\n  Phase 2 done. {len(all_results)} valid configurations.")

    # ---- Results display ----

    def _print_results(self, results):
        top = results[:20]
        if not top:
            print("\n  No valid configurations found.")
            return

        print("\n" + "=" * 70)
        print("  OPTIMIZATION RESULTS (Top 20)")
        print("=" * 70)

        for i, r in enumerate(top):
            print(f"\n  Rank {i+1}:")
            print(f"    Signals:    {' + '.join(r['signals'])}")
            print(f"    Target:     {r['target_pct']:.1%}")
            print(f"    Objective:  {r['objective']:.6f}  "
                  f"(wr={r['avg_win_rate']:.2%} x ret={r['avg_return']:.4f})")
            print(f"    Trades:     {r['total_trades']}  "
                  f"(folds: {r['n_folds_active']}/{r['n_folds']})")
            if r['params']:
                print(f"    Params:")
                for k, v in sorted(r['params'].items()):
                    default = getattr(config, k)
                    marker = " *" if v != default else ""
                    print(f"      {k}: {v}{marker}")

        print("\n" + "=" * 70)
        print("  (* = differs from default)")
        print("=" * 70)


# ============================================================
# Entry point
# ============================================================

def run_optimization(stock_data, focus_signals=None,
                     min_signals=2, max_signals=4,
                     train_window=120, test_window=20, top_n=5):
    """Main entry point for the optimization system."""
    t0 = time.time()

    optimizer = Optimizer(
        stock_data=stock_data,
        focus_signals=focus_signals,
        min_signals=min_signals,
        max_signals=max_signals,
        train_window=train_window,
        test_window=test_window,
        top_n=top_n,
    )
    results = optimizer.run()

    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.1f}s.")
    return results
