#!/usr/bin/env python
"""Entry point for the SZ quantitative trading system.

Usage:
    python run.py init [--start 2024-01-01]    # Full history build (tushare)
    python run.py import-data                   # Import data from limit_up project
    python run.py daily [--date 2025-10-27]    # Daily run
    python run.py screen [--date 2025-10-27]   # Screens only
    python run.py scan [--date 2025-10-27]     # Pattern scan only
    python run.py agent [--days 60] [--capital 100000]  # Trading agent
    python run.py backtest [--days ...]        # Run backtest
    python run.py diagnose [--screen ...]      # Feature diagnosis
    python run.py plot 002906 [--tail 40]      # K-line chart
    python run.py live [--loop] [--interval 15] # Live real-time screening
    python run.py web [--output site]          # Generate static website
"""

import argparse
import sys
import os

# Add the sz directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline import Pipeline
from backtest import BacktestEngine, print_report
from agent import TradingAgent, print_agent_report
from visualize import plot_candlestick, plot_equity_curve, plot_drawdown
import screener


def cmd_init(args):
    """Initialize full build via tushare."""
    pipeline = Pipeline()
    if args.from_cache:
        pipeline.import_from_limit_up()
    else:
        pipeline.init_full_build(start_date=args.start)


def cmd_import_data(args):
    """Import data from limit_up project."""
    pipeline = Pipeline()
    pipeline.import_from_limit_up(limit_up_raw_dir=args.source)


def cmd_daily(args):
    """Run daily pipeline."""
    if args.date:
        config.TODAY = args.date
    pipeline = Pipeline()
    pipeline.run_daily(date=config.TODAY)


def cmd_screen(args):
    """Run screens only."""
    if args.date:
        config.TODAY = args.date
    pipeline = Pipeline()
    pipeline.run_screen_only(date=config.TODAY)


def cmd_scan(args):
    """Run pattern scan only."""
    date = args.date or config.TODAY
    pipeline = Pipeline()
    pipeline.run_pattern_scan(date=date)


def cmd_backtest(args):
    """Run backtest for all screens."""
    pipeline = Pipeline()

    import downloader
    stock_data = downloader.load_processed()
    if not stock_data:
        print("No processed data. Run 'init' or 'daily' first.")
        return

    import indicators
    print("Computing indicators...")
    for code in list(stock_data.keys()):
        stock_data[code] = indicators.compute_all(stock_data[code])

    if args.days:
        days = args.days.split(',')
    else:
        sample = next(iter(stock_data.values()))
        all_days = sorted(sample.index[-60:].strftime('%Y-%m-%d').tolist())
        days = all_days

    engine = BacktestEngine()
    target = args.target if args.target else config.DEFAULT_TARGET_PCT

    for name, screen_func in screener.ALL_SCREENS.items():
        print(f"\n{'=' * 50}")
        print(f"  Backtesting: {name}")
        print(f"{'=' * 50}")

        results = engine.run(screen_func, stock_data, days, target_pct=target)
        report = engine.generate_report(results)
        print_report(report)

        if not results.empty and args.plot:
            plot_equity_curve(results)
            plot_drawdown(results)


def cmd_agent(args):
    """Run the trading agent simulation."""
    import downloader
    import indicators

    stock_data = downloader.load_processed()
    if not stock_data:
        print("No processed data. Run 'init' or 'daily' first.")
        return

    print("Computing indicators...")
    for code in list(stock_data.keys()):
        stock_data[code] = indicators.compute_all(stock_data[code])

    if args.days_list:
        days = args.days_list.split(',')
    else:
        sample = next(iter(stock_data.values()))
        n = args.days
        all_days = sorted(sample.index[-n:].strftime('%Y-%m-%d').tolist())
        days = all_days

    screen_name = args.screen
    if screen_name not in screener.ALL_SCREENS:
        print(f"Unknown screen '{screen_name}'. "
              f"Available: {', '.join(screener.ALL_SCREENS.keys())}")
        return
    screen_func = screener.ALL_SCREENS[screen_name]

    print(f"\nAgent simulation: {len(days)} trading days, "
          f"initial capital \u00a5{args.capital:,.0f}")
    print(f"Screen: {screen_name}")
    print(f"Period: {days[0]} ~ {days[-1]}\n")

    agent = TradingAgent(stock_data, initial_capital=args.capital,
                         screen_func=screen_func)
    agent.run(days)

    report = agent.generate_report()
    print_agent_report(report)


def cmd_diagnose(args):
    """Run feature diagnosis for a screen strategy."""
    import downloader
    import indicators
    from diagnose import run_diagnosis

    stock_data = downloader.load_processed()
    if not stock_data:
        print("No processed data. Run 'init' or 'daily' first.")
        return

    print("Computing indicators...")
    for code in list(stock_data.keys()):
        stock_data[code] = indicators.compute_all(stock_data[code])

    screen_name = args.screen
    if screen_name not in screener.ALL_SCREENS:
        print(f"Unknown screen '{screen_name}'. "
              f"Available: {', '.join(screener.ALL_SCREENS.keys())}")
        return
    screen_func = screener.ALL_SCREENS[screen_name]

    if args.days:
        days = args.days.split(',')
    else:
        sample = next(iter(stock_data.values()))
        all_days = sorted(sample.index[-60:].strftime('%Y-%m-%d').tolist())
        days = all_days

    target = args.target if args.target else config.DEFAULT_TARGET_PCT
    run_diagnosis(stock_data, screen_func, screen_name, days, target_pct=target)


def cmd_optimize(args):
    """Run parameter optimization."""
    import downloader
    import indicators
    from optimizer import run_optimization

    stock_data = downloader.load_processed()
    if not stock_data:
        print("No processed data. Run 'init' or 'daily' first.")
        return

    print("Computing indicators...")
    for code in list(stock_data.keys()):
        stock_data[code] = indicators.compute_all(stock_data[code])

    focus = args.focus.split(',') if args.focus else None
    run_optimization(
        stock_data=stock_data,
        focus_signals=focus,
        min_signals=args.min_signals,
        max_signals=args.max_signals,
        train_window=args.train_window,
        test_window=args.test_window,
        top_n=args.top_n,
    )


def cmd_live(args):
    """Run live screening with real-time data, optionally looping."""
    import copy
    import time as time_mod
    from datetime import datetime as dt, timedelta

    import downloader
    import indicators

    interval_sec = args.interval * 60

    # Load historical data once
    print("Loading historical data...")
    base_data = downloader.load_processed()
    if not base_data:
        print("No processed data. Run 'init' or 'daily' first.")
        return

    # Pre-compute indicators on historical data (saves time on each loop)
    print("Computing indicators on historical data...")
    for code in list(base_data.keys()):
        base_data[code] = indicators.compute_all(base_data[code])
    print(f"  Ready: {len(base_data)} stocks.\n")

    iteration = 0
    while True:
        iteration += 1
        now = dt.now()
        hhmm = now.hour * 100 + now.minute

        print(f"\n{'='*60}")
        print(f"  Live #{iteration} @ {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Trading hours check (9:15 - 15:05), skip outside unless --force
        if (hhmm < 915 or hhmm > 1505) and not args.force:
            print("  Market closed. Waiting for next interval...")
            if not args.loop:
                break
            time_mod.sleep(interval_sec)
            continue

        # Fetch real-time data (Tushare → yfinance fallback)
        print("  Fetching real-time quotes...")
        realtime, source = downloader.fetch_realtime(
            stock_codes=base_data.keys()
        )
        if not realtime:
            print("  No real-time data returned.")
            if not args.loop:
                break
            time_mod.sleep(interval_sec)
            continue
        print(f"  Got {len(realtime)} stocks via {source}.")

        # Deep copy historical data + inject real-time as today's row
        live_data = copy.deepcopy(base_data)
        live_data, injected = downloader.inject_realtime(live_data, realtime)
        print(f"  Injected real-time for {injected} stocks.")

        # Re-compute indicators (only today's row is new, but indicators
        # are cumulative so we recompute the full series)
        print("  Recomputing indicators...")
        for code in list(live_data.keys()):
            live_data[code] = indicators.compute_all(live_data[code])

        # Run screens
        today = now.strftime('%Y-%m-%d')
        print(f"\n  Screening results for {today}:")
        results = screener.run_all_screens(live_data, today)

        total = sum(len(codes) for codes in results.values())
        print(f"\n  Total candidates: {total}")

        # Generate website if requested
        if args.web:
            print("\n  Generating website...")
            from web.generator import WebGenerator
            gen = WebGenerator(output_dir=args.output)
            gen.build(date=today, stock_data=live_data)

        if not args.loop:
            break

        print(f"\n  Next refresh in {args.interval} min "
              f"({now.strftime('%H:%M')} → "
              f"{(now + timedelta(minutes=args.interval)).strftime('%H:%M')})...")
        time_mod.sleep(interval_sec)


def cmd_web(args):
    """Generate static website."""
    from web.generator import WebGenerator
    gen = WebGenerator(output_dir=args.output)
    gen.build(date=args.date)


def cmd_plot(args):
    """Plot candlestick chart."""
    import downloader
    import indicators

    stock_data = downloader.load_processed()
    if not stock_data:
        print("No processed data. Run 'init' or 'daily' first.")
        return

    code = args.code
    if code not in stock_data:
        print(f"Stock code {code} not found.")
        return

    stock_data[code] = indicators.compute_all(stock_data[code])
    plot_candlestick(stock_data, code, tail=args.tail)


def main():
    parser = argparse.ArgumentParser(
        description='SZ Quantitative Trading System (tushare)'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init
    p_init = subparsers.add_parser('init', help='Initialize full history build')
    p_init.add_argument('--start', type=str, default=None,
                        help='Start date (e.g., 2024-01-01)')
    p_init.add_argument('--from-cache', action='store_true',
                        help='Import from limit_up project cache instead of downloading')

    # import-data
    p_import = subparsers.add_parser('import-data',
                                      help='Import data from limit_up project')
    p_import.add_argument('--source', type=str, default=None,
                          help='Path to limit_up/data/raw directory')

    # daily
    p_daily = subparsers.add_parser('daily', help='Run daily pipeline')
    p_daily.add_argument('--date', type=str, default=None,
                         help='Override today\'s date')

    # screen
    p_screen = subparsers.add_parser('screen', help='Run screens only')
    p_screen.add_argument('--date', type=str, default=None,
                          help='Override today\'s date')

    # scan (pattern detection)
    p_scan = subparsers.add_parser('scan', help='Run pattern detection scan')
    p_scan.add_argument('--date', type=str, default=None,
                        help='Date to scan (YYYY-MM-DD, default: today)')

    # agent
    p_agent = subparsers.add_parser('agent', help='Run trading agent simulation')
    p_agent.add_argument('--days', type=int, default=60,
                         help='Number of recent trading days (default: 60)')
    p_agent.add_argument('--days-list', type=str, default=None,
                         help='Comma-separated list of specific trading days')
    p_agent.add_argument('--capital', type=float, default=100000,
                         help='Initial capital (default: 100000)')
    p_agent.add_argument('--screen', type=str, default='OBV\u6da8\u505c\u68a6',
                         help='Screen strategy name (default: OBV\u6da8\u505c\u68a6)')

    # backtest
    p_bt = subparsers.add_parser('backtest', help='Run backtest')
    p_bt.add_argument('--days', type=str, default=None,
                      help='Comma-separated list of trading days')
    p_bt.add_argument('--target', type=float, default=None,
                      help='Target profit percentage (e.g., 0.03)')
    p_bt.add_argument('--plot', action='store_true',
                      help='Show equity curve and drawdown plots')

    # diagnose
    p_diag = subparsers.add_parser('diagnose',
                                    help='Feature diagnosis for a screen')
    p_diag.add_argument('--screen', type=str, default='OBV\u6da8\u505c\u68a6',
                         help='Screen name (default: OBV\u6da8\u505c\u68a6)')
    p_diag.add_argument('--days', type=str, default=None,
                         help='Comma-separated list of trading days')
    p_diag.add_argument('--target', type=float, default=None,
                         help='Target profit percentage (e.g., 0.03)')

    # optimize
    p_opt = subparsers.add_parser('optimize', help='Run parameter optimization')
    p_opt.add_argument('--focus', type=str, default=None,
                        help='Comma-separated signal names to always include')
    p_opt.add_argument('--min-signals', type=int, default=2,
                        help='Min signals per combo (default: 2)')
    p_opt.add_argument('--max-signals', type=int, default=4,
                        help='Max signals per combo (default: 4)')
    p_opt.add_argument('--train-window', type=int, default=120,
                        help='Min training days (default: 120)')
    p_opt.add_argument('--test-window', type=int, default=20,
                        help='Test window days (default: 20)')
    p_opt.add_argument('--top-n', type=int, default=5,
                        help='Top combos for Phase 2 (default: 5)')

    # plot
    p_plot = subparsers.add_parser('plot', help='Plot candlestick chart')
    p_plot.add_argument('code', type=str, help='Stock code (e.g., 002906)')
    p_plot.add_argument('--tail', type=int, default=40,
                        help='Number of days to show')

    # live
    p_live = subparsers.add_parser('live',
                                    help='Live screening with real-time data')
    p_live.add_argument('--interval', type=int, default=15,
                        help='Refresh interval in minutes (default: 15)')
    p_live.add_argument('--loop', action='store_true',
                        help='Keep looping (default: run once)')
    p_live.add_argument('--web', action='store_true',
                        help='Also regenerate website on each refresh')
    p_live.add_argument('--output', type=str, default='site',
                        help='Website output directory (default: site)')
    p_live.add_argument('--force', action='store_true',
                        help='Run even outside trading hours')

    # web
    p_web = subparsers.add_parser('web', help='Generate static website')
    p_web.add_argument('--date', type=str, default=None,
                       help='Build for a specific date (default: last 30 days)')
    p_web.add_argument('--output', type=str, default='site',
                       help='Output directory (default: site)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'init': cmd_init,
        'import-data': cmd_import_data,
        'daily': cmd_daily,
        'screen': cmd_screen,
        'scan': cmd_scan,
        'agent': cmd_agent,
        'backtest': cmd_backtest,
        'diagnose': cmd_diagnose,
        'optimize': cmd_optimize,
        'live': cmd_live,
        'plot': cmd_plot,
        'web': cmd_web,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
