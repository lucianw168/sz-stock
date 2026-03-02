"""Pipeline orchestration: init full build + daily run."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

import pandas as pd

import config
import downloader
import indicators
import labels
import screener
from pattern_screen import run_pattern_scan, get_latest_pattern_signals


class Pipeline:
    """Main pipeline orchestrating the two workflows."""

    def __init__(self):
        self.stock_data: dict[str, pd.DataFrame] = {}

    def init_full_build(self, start_date: str = None) -> dict[str, pd.DataFrame]:
        """Workflow 1: Full history build (one-time).

        1. Full download -> data/raw/full_history.parquet
        2. Data quality check -> data/processed/ohlcv_sz.parquet
        3. Compute all technical indicators
        4. Build labels
        """
        print("=" * 60)
        print("  FULL BUILD: Initializing...")
        print("=" * 60)

        # Step 1: Download
        print("\n[1/4] Downloading full history...")
        self.stock_data = downloader.download_full(start_date)

        # Step 2: Quality check + save processed
        print("\n[2/4] Quality check and processing...")
        self.stock_data = downloader.quality_check(self.stock_data)
        downloader.save_processed(self.stock_data)

        # Step 3: Compute indicators
        print("\n[3/4] Computing technical indicators...")
        for code in list(self.stock_data.keys()):
            self.stock_data[code] = indicators.compute_all(self.stock_data[code])
        print(f"  Computed indicators for {len(self.stock_data)} stocks.")

        # Step 4: Build labels
        print("\n[4/4] Building labels...")
        self.stock_data = labels.build_all_labels(self.stock_data)
        labels.save_features(self.stock_data)

        print("\nFull build complete!")
        return self.stock_data

    def run_daily(self, date: str = None) -> dict[str, list[str]]:
        """Workflow 2: Daily run.

        1. Incremental download (last 60 days)
        2. Merge with existing data
        3. Generate features
        4. Run all screens -> output stock picks
        5. Save prediction snapshot
        """
        if date is None:
            date = config.TODAY

        print("=" * 60)
        print(f"  DAILY RUN: {date}")
        print("=" * 60)

        # Step 1: Incremental download
        print("\n[1/5] Incremental download...")
        incremental = downloader.download_incremental()

        # Step 2: Merge with existing
        print("\n[2/5] Merging data...")
        existing = downloader.load_processed()
        if existing:
            self.stock_data = downloader.merge_incremental(existing, incremental)
        else:
            self.stock_data = incremental
        self.stock_data = downloader.quality_check(self.stock_data)
        downloader.save_processed(self.stock_data)

        # Step 3: Compute indicators
        print("\n[3/5] Computing indicators...")
        for code in list(self.stock_data.keys()):
            self.stock_data[code] = indicators.compute_all(self.stock_data[code])

        # Step 4: Run all screens
        print(f"\n[4/5] Running screens for {date}...")
        results = screener.run_all_screens(self.stock_data, date)

        # Step 5: Save prediction snapshot
        print("\n[5/5] Saving prediction snapshot...")
        self._save_prediction(results, date)

        # Save features
        self._save_daily_features(date)

        print(f"\nDaily run for {date} complete!")
        return results

    def run_screen_only(self, date: str = None) -> dict[str, list[str]]:
        """Run screens on existing data without downloading."""
        if date is None:
            date = config.TODAY

        print(f"Running screens for {date} on existing data...")

        # Load from processed if not in memory
        if not self.stock_data:
            self.stock_data = downloader.load_processed()
            if not self.stock_data:
                print("No processed data found. Run 'init' or 'daily' first.")
                return {}

        # Compute indicators if needed
        sample_code = next(iter(self.stock_data))
        if 'CRSI' not in self.stock_data[sample_code].columns:
            print("Computing indicators...")
            for code in list(self.stock_data.keys()):
                self.stock_data[code] = indicators.compute_all(self.stock_data[code])

        results = screener.run_all_screens(self.stock_data, date)
        return results

    def _save_prediction(self, results: dict[str, list[str]], date: str) -> None:
        """Save daily prediction snapshot."""
        rows = []
        for screen_name, codes in results.items():
            for code in codes:
                rows.append({'screen': screen_name, 'code': code, 'date': date})

        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(config.PREDICTIONS_DIR, f"{date}.parquet")
            df.to_parquet(path, index=False)
            print(f"  Saved prediction to {path}")
        else:
            print("  No predictions to save.")

    def run_pattern_scan(self, date: str = None):
        """Run pattern detection scan for a specific date.

        Loads data if needed, runs pattern detector, prints results.
        """
        if date is None:
            date = config.TODAY

        print("=" * 60)
        print(f"  PATTERN SCAN: {date}")
        print("=" * 60)

        # Load data if needed
        if not self.stock_data:
            self.stock_data = downloader.load_processed()
            if not self.stock_data:
                print("No processed data. Run 'init' first.")
                return []

        print(f"\n  Loaded {len(self.stock_data)} stocks.")
        print("  Running pattern detection...")

        signals = run_pattern_scan(self.stock_data, date)

        if signals:
            print(f"\n  Found {len(signals)} pattern signals:")
            print("  " + "-" * 56)
            for s in signals:
                print(f"  {s['code']}  {s['explanation']}")
                print(f"         胜率: {s['confidence']:.0%}  "
                      f"价格: {s['signal_price']:.2f}  "
                      f"日期: {s['signal_date']}")
            print("  " + "-" * 56)
        else:
            print(f"\n  No pattern signals for {date}.")

        return signals

    def import_from_limit_up(self, limit_up_raw_dir=None):
        """Import tushare data from the limit_up project."""
        print("=" * 60)
        print("  IMPORTING DATA FROM LIMIT_UP PROJECT")
        print("=" * 60)
        success = downloader.import_from_limit_up(limit_up_raw_dir)
        if success:
            print("\nBuilding processed data from imported cache...")
            self.stock_data = downloader.load_processed()
            if self.stock_data:
                print(f"Loaded {len(self.stock_data)} stocks.")
                self.stock_data = downloader.quality_check(self.stock_data)
                downloader.save_processed(self.stock_data)
                print("Done!")
        return success

    def _save_daily_features(self, date: str) -> None:
        """Save today's feature snapshot."""
        frames = []
        ts = pd.Timestamp(date)
        for code, df in self.stock_data.items():
            if ts in df.index:
                row = df.loc[[ts]].copy()
                row['ticker'] = code
                frames.append(row)

        if frames:
            combined = pd.concat(frames)
            path = os.path.join(config.FEATURES_DIR, "features_today.parquet")
            combined.to_parquet(path)
            print(f"  Saved daily features to {path}")
