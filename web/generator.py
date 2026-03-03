"""Static site generator for the SZ stock screening system."""

import os
import shutil
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import screener
from web.data_prep import (
    SCREEN_DESCRIPTIONS,
    TRADING_METHODS,
    load_stock_data,
    run_screen_for_date,
    list_available_dates,
    get_stock_indicators,
    prepare_candlestick_data,
    run_all_backtests,
    get_pattern_signals_for_date,
    get_signal_diagnostics,
    compute_motive_label,
)


class WebGenerator:
    """Generate a static website from SZ stock screening data."""

    def __init__(self, output_dir='site'):
        self.output_dir = output_dir
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.static_dir = os.path.join(os.path.dirname(__file__), 'static')
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.strategy_names = list(screener.ALL_SCREENS.keys())
        self.generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Data holders
        self.stock_data = None
        self.all_dates = []
        self.target_dates = []  # dates to generate pages for
        self.screen_results_by_date = {}  # date -> {screen_name: [codes]}
        self.pattern_signals_by_date = {}  # date -> [signal_dicts]
        self.backtest_reports = {}  # screen_name -> {report, equity_data, trade_details}
        self.all_selected_codes = set()  # all codes selected by any strategy on any date
        self.stock_appearances = {}  # code -> [{date, screen}]

    def build(self, date=None, stock_data=None):
        """Full site build.

        Args:
            date: build for a specific date (default: last 30 days).
            stock_data: pre-loaded dict[code, DataFrame] with indicators
                        already computed.  When provided, skips disk loading.
        """
        print("=" * 60)
        print("  Building static website...")
        print("=" * 60)

        if stock_data is not None:
            self.stock_data = stock_data
            print(f"\n[1/7] Using pre-loaded data ({len(stock_data)} stocks)")
        else:
            self._load_data()
        self._determine_dates(date)
        self._run_screens()
        self._run_backtests()
        self._prepare_output_dir()
        self._copy_static()
        self._build_index()
        self._build_history()
        self._build_daily_pages()
        self._build_strategy_pages()
        self._build_stock_pages()

        total_pages = (1 + 1 + len(self.target_dates) +
                       len(self.strategy_names) + len(self.all_selected_codes))
        print(f"\nDone! Generated {total_pages} pages in {self.output_dir}/")

    def _load_data(self):
        """Load parquet data and compute indicators."""
        print("\n[1/7] Loading stock data...")
        self.stock_data = load_stock_data()
        if not self.stock_data:
            raise RuntimeError("No processed data. Run 'python run.py init' or 'daily' first.")
        print(f"  Loaded {len(self.stock_data)} stocks.")

    def _determine_dates(self, date=None):
        """Determine which dates to build pages for."""
        print("\n[2/7] Determining dates...")
        self.all_dates = list_available_dates(self.stock_data)

        if date:
            # Build for a single specified date + surrounding dates
            self.target_dates = [date]
        else:
            # Use last 30 trading days
            self.target_dates = self.all_dates[-30:]

        print(f"  Building pages for {len(self.target_dates)} dates.")

    def _run_screens(self):
        """Run all screens for each target date."""
        print("\n[3/7] Running screens...")
        for d in self.target_dates:
            results = run_screen_for_date(self.stock_data, d)
            self.screen_results_by_date[d] = results

            # Collect pattern signal details
            pattern_signals = get_pattern_signals_for_date(self.stock_data, d)
            self.pattern_signals_by_date[d] = pattern_signals

            # Track selected codes and appearances
            for name, codes in results.items():
                for code in codes:
                    self.all_selected_codes.add(code)
                    if code not in self.stock_appearances:
                        self.stock_appearances[code] = []
                    self.stock_appearances[code].append({'date': d, 'screen': name})

        print(f"  {len(self.all_selected_codes)} unique stocks selected.")

    def _run_backtests(self):
        """Run backtests for all strategies."""
        print("\n[4/7] Running backtests...")
        # Use target dates for backtest (need at least 2 days)
        bt_days = self.target_dates if len(self.target_dates) >= 2 else self.all_dates[-60:]
        self.backtest_reports = run_all_backtests(self.stock_data, bt_days)

    def _prepare_output_dir(self):
        """Clean and create output directory structure."""
        print("\n[5/7] Preparing output directory...")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        for subdir in ['days', 'strategy', 'stock', 'static']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def _copy_static(self):
        """Copy static CSS files."""
        src = os.path.join(self.static_dir, 'style.css')
        dst = os.path.join(self.output_dir, 'static', 'style.css')
        shutil.copy2(src, dst)

    def _common_ctx(self, active='', root=''):
        """Common template context."""
        return {
            'strategies': self.strategy_names,
            'generated_at': self.generated_at,
            'active': active,
            'root': root,
        }

    def _write(self, path, content):
        """Write rendered HTML to file."""
        full_path = os.path.join(self.output_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _build_index(self):
        """Generate index.html — today's picks."""
        print("\n[6/7] Building pages...")
        print("  index.html")
        tmpl = self.env.get_template('index.html')

        latest_date = self.target_dates[-1] if self.target_dates else self.all_dates[-1]
        screen_results = self.screen_results_by_date.get(latest_date, {})

        # Calendar: last 30 target dates
        calendar_dates = self.target_dates[-30:]

        # Backtest summaries
        backtest_summaries = {}
        for name, data in self.backtest_reports.items():
            backtest_summaries[name] = data['report']

        pattern_signals = self.pattern_signals_by_date.get(latest_date, [])

        # KPI aggregation
        total_candidates = sum(len(codes) for codes in screen_results.values())

        total_bt_trades = sum(
            r.get('total_trades', 0)
            for r in backtest_summaries.values()
        )

        best_strategy = None
        best_wr = -1
        for sname, rpt in backtest_summaries.items():
            wr = rpt.get('win_rate', 0)
            if wr > best_wr:
                best_wr = wr
                best_strategy = {
                    'name': sname,
                    'win_rate': rpt.get('win_rate', 0),
                    'cumulative_return': rpt.get('cumulative_return', 0),
                    'max_drawdown': rpt.get('max_drawdown', 0),
                }

        html = tmpl.render(
            **self._common_ctx(active='index', root=''),
            date=latest_date,
            screen_results=screen_results,
            calendar_dates=calendar_dates,
            backtest_summaries=backtest_summaries,
            pattern_signals=pattern_signals,
            total_candidates=total_candidates,
            best_strategy=best_strategy,
            total_bt_trades=total_bt_trades,
        )
        self._write('index.html', html)

    def _build_history(self):
        """Generate history.html — date index."""
        print("  history.html")
        tmpl = self.env.get_template('history.html')

        dates_info = []
        for d in reversed(self.target_dates):
            results = self.screen_results_by_date.get(d, {})
            counts = {name: len(codes) for name, codes in results.items()}
            total = sum(counts.values())
            dates_info.append({'date': d, 'counts': counts, 'total': total})

        html = tmpl.render(
            **self._common_ctx(active='history', root=''),
            dates=dates_info,
        )
        self._write('history.html', html)

    def _build_daily_pages(self):
        """Generate days/{date}.html for each target date."""
        tmpl = self.env.get_template('daily.html')

        for i, d in enumerate(self.target_dates):
            print(f"  days/{d}.html")
            results = self.screen_results_by_date.get(d, {})

            # Enrich with indicators, diagnostics, and motive labels
            enriched = {}
            for name, codes in results.items():
                stocks = []
                for code in codes:
                    info = get_stock_indicators(self.stock_data, code, d)
                    if info:
                        info['diagnostics'] = get_signal_diagnostics(
                            self.stock_data, code, d, name)
                        if name in ('OBV涨停梦', 'OBV波段'):
                            info['motive'] = compute_motive_label(
                                self.stock_data, code, d)
                        stocks.append(info)
                enriched[name] = stocks

            prev_date = self.target_dates[i - 1] if i > 0 else None
            next_date = self.target_dates[i + 1] if i < len(self.target_dates) - 1 else None
            pattern_signals = self.pattern_signals_by_date.get(d, [])

            html = tmpl.render(
                **self._common_ctx(root='../'),
                date=d,
                screen_results=enriched,
                prev_date=prev_date,
                next_date=next_date,
                pattern_signals=pattern_signals,
                trading_methods=TRADING_METHODS,
            )
            self._write(f'days/{d}.html', html)

    def _build_strategy_pages(self):
        """Generate strategy/{name}.html for each strategy."""
        tmpl = self.env.get_template('strategy.html')

        for name in self.strategy_names:
            print(f"  strategy/{name}.html")
            bt = self.backtest_reports.get(name, {})

            html = tmpl.render(
                **self._common_ctx(active=f'strategy-{name}', root='../'),
                name=name,
                description=SCREEN_DESCRIPTIONS.get(name, ''),
                report=bt.get('report', {}),
                equity_data=bt.get('equity_data', {'dates': [], 'equity': [], 'drawdown': []}),
                trade_details=bt.get('trade_details', []),
                trading_method=TRADING_METHODS.get(name, ''),
                failure_analysis=bt.get('failure_analysis'),
            )
            self._write(f'strategy/{name}.html', html)

    def _build_stock_pages(self):
        """Generate stock/{code}.html for each selected stock."""
        tmpl = self.env.get_template('stock.html')

        for code in sorted(self.all_selected_codes):
            print(f"  stock/{code}.html")
            chart_data = prepare_candlestick_data(self.stock_data, code, tail=60)
            appearances = sorted(
                self.stock_appearances.get(code, []),
                key=lambda x: x['date'],
                reverse=True,
            )

            html = tmpl.render(
                **self._common_ctx(root='../'),
                code=code,
                chart_data=chart_data or {},
                appearances=appearances,
            )
            self._write(f'stock/{code}.html', html)

        print(f"\n[7/7] Generated {len(self.all_selected_codes)} stock pages.")
