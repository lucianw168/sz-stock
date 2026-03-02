"""OBV涨停梦 模拟交易 Agent.

Rule-based trading agent that goes beyond mechanical backtesting:
- Evaluates each screened stock with buy criteria (RSI, CCI, volume, anomaly)
- Manages positions with stop-loss, take-profit, trend deterioration, max hold
- Learns limit-up continuation patterns from historical data
- Adapts holding strategy when positions hit limit-up
- Tracks daily NAV, trade logs, and generates performance reports
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import config
from screener import screen_obv_momentum
from universe import get_limit_ratio


@dataclass
class LimitUpExperience:
    """Learned model from historical limit-up continuation patterns."""
    total_limit_ups: int = 0
    total_continuations: int = 0
    by_streak: dict = field(default_factory=dict)      # {streak: [total, continued]}
    by_indicator: dict = field(default_factory=dict)    # {name: {bucket: [total, continued]}}
    failed_drawdowns: list = field(default_factory=list)
    trailing_stop_pct: float = 0.03
    live_total: int = 0
    live_correct: int = 0


class TradingAgent:
    """OBV涨停梦模拟交易 Agent."""

    def __init__(self, stock_data, initial_capital=100000, screen_func=None):
        self.stock_data = stock_data
        self.capital = float(initial_capital)
        self.initial_capital = float(initial_capital)
        self.commission = config.COMMISSION_RATE
        self.slippage = config.SLIPPAGE_RATE
        self.positions = {}   # {code: {buy_price, buy_date, shares, days_held, ...}}
        self.trade_log = []   # completed trades
        self.daily_log = []   # every decision made
        self.daily_nav = []   # daily net asset value
        self.experience = None
        self.screen_func = screen_func or screen_obv_momentum

    # ------------------------------------------------------------------
    # Indicator bucketing for experience model
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_rsi(rsi):
        if rsi < 30: return 'rsi_<30'
        if rsi < 50: return 'rsi_30-50'
        if rsi < 70: return 'rsi_50-70'
        if rsi < 80: return 'rsi_70-80'
        return 'rsi_>=80'

    @staticmethod
    def _bucket_cci(cci):
        if cci < -100: return 'cci_<-100'
        if cci < 0: return 'cci_-100-0'
        if cci < 100: return 'cci_0-100'
        if cci < 200: return 'cci_100-200'
        return 'cci_>=200'

    @staticmethod
    def _bucket_volume(vol):
        if vol < 30: return 'vol_<30'
        if vol < 60: return 'vol_30-60'
        if vol < 100: return 'vol_60-100'
        return 'vol_>=100'

    @staticmethod
    def _bucket_adx(adx):
        if adx < 15: return 'adx_<15'
        if adx < 25: return 'adx_15-25'
        if adx < 40: return 'adx_25-40'
        return 'adx_>=40'

    @staticmethod
    def _bucket_seal(row):
        """封板强度: how firmly locked at limit-up."""
        if row['Close'] >= row['High'] * 0.999:
            return 'seal_full'
        if row['Close'] >= row['High'] * 0.99:
            return 'seal_near'
        return 'seal_broken'

    # ------------------------------------------------------------------
    # Experience model building
    # ------------------------------------------------------------------

    def _build_experience_from_history(self, first_day):
        """Scan all history before first_day to learn limit-up patterns."""
        exp = LimitUpExperience()
        first_ts = pd.Timestamp(first_day)

        for code, df in self.stock_data.items():
            hist = df[df.index < first_ts]
            if len(hist) < 3:
                continue

            ratio = get_limit_ratio(code)

            for i in range(1, len(hist) - 1):
                row = hist.iloc[i]
                prev_row = hist.iloc[i - 1]
                next_row = hist.iloc[i + 1]

                # Check if current day is limit-up
                limit_price = round(prev_row['Close'] * (1 + ratio), 2)
                if row['Close'] < limit_price * (1 - config.LIMIT_TOLERANCE):
                    continue

                exp.total_limit_ups += 1

                # Determine streak length (consecutive limit-ups ending here)
                streak = 1
                j = i - 1
                while j >= 1:
                    pp = hist.iloc[j - 1]
                    cc = hist.iloc[j]
                    lp = round(pp['Close'] * (1 + ratio), 2)
                    if cc['Close'] >= lp * (1 - config.LIMIT_TOLERANCE):
                        streak += 1
                        j -= 1
                    else:
                        break

                # Check if next day continues limit-up
                next_lp = round(row['Close'] * (1 + ratio), 2)
                continued = next_row['Close'] >= next_lp * (1 - config.LIMIT_TOLERANCE)
                if continued:
                    exp.total_continuations += 1

                # Record by streak
                if streak not in exp.by_streak:
                    exp.by_streak[streak] = [0, 0]
                exp.by_streak[streak][0] += 1
                if continued:
                    exp.by_streak[streak][1] += 1

                # Record by indicators
                for bname, bfunc, val in [
                    ('rsi', self._bucket_rsi, row.get('Rsi')),
                    ('cci', self._bucket_cci, row.get('CCI')),
                    ('vol', self._bucket_volume, row.get('Volume percentage 1d')),
                    ('adx', self._bucket_adx, row.get('ADX')),
                ]:
                    if val is not None and not np.isnan(val):
                        bucket = bfunc(val)
                        if bname not in exp.by_indicator:
                            exp.by_indicator[bname] = {}
                        if bucket not in exp.by_indicator[bname]:
                            exp.by_indicator[bname][bucket] = [0, 0]
                        exp.by_indicator[bname][bucket][0] += 1
                        if continued:
                            exp.by_indicator[bname][bucket][1] += 1

                # Seal strength
                seal_bucket = self._bucket_seal(row)
                if 'seal' not in exp.by_indicator:
                    exp.by_indicator['seal'] = {}
                if seal_bucket not in exp.by_indicator['seal']:
                    exp.by_indicator['seal'][seal_bucket] = [0, 0]
                exp.by_indicator['seal'][seal_bucket][0] += 1
                if continued:
                    exp.by_indicator['seal'][seal_bucket][1] += 1

                # Drawdown on failed continuation
                if not continued:
                    drawdown = (next_row['Low'] - row['Close']) / row['Close']
                    exp.failed_drawdowns.append(drawdown)

        # Adaptive trailing stop from 75th percentile of drawdowns
        if exp.failed_drawdowns:
            p25 = np.percentile(exp.failed_drawdowns, 25)
            exp.trailing_stop_pct = max(0.02, min(0.10, abs(p25)))

        self.experience = exp

    # ------------------------------------------------------------------
    # Continuation probability estimation
    # ------------------------------------------------------------------

    def _estimate_continuation_probability(self, code, date, streak):
        """Weighted estimate of limit-up continuation probability."""
        exp = self.experience
        if exp is None or exp.total_limit_ups == 0:
            return 0.0

        base_rate = exp.total_continuations / exp.total_limit_ups

        # Signal 1: Streak-based rate (weight 50%, degrade to 20% if insufficient)
        if streak in exp.by_streak and exp.by_streak[streak][0] >= 5:
            streak_rate = exp.by_streak[streak][1] / exp.by_streak[streak][0]
            streak_weight = 0.5
        else:
            streak_rate = base_rate
            streak_weight = 0.2

        # Signal 2: Indicator-based rates (weight 30%)
        df = self.stock_data.get(code)
        ts = pd.Timestamp(date)
        row = df.loc[ts]

        indicator_rates = []
        for bname, bfunc, val in [
            ('rsi', self._bucket_rsi, row.get('Rsi')),
            ('cci', self._bucket_cci, row.get('CCI')),
            ('vol', self._bucket_volume, row.get('Volume percentage 1d')),
            ('adx', self._bucket_adx, row.get('ADX')),
        ]:
            if val is not None and not np.isnan(val):
                bucket = bfunc(val)
                if bname in exp.by_indicator and bucket in exp.by_indicator[bname]:
                    total, cont = exp.by_indicator[bname][bucket]
                    if total >= 3:
                        indicator_rates.append(cont / total)

        # Seal strength
        seal_bucket = self._bucket_seal(row)
        if 'seal' in exp.by_indicator and seal_bucket in exp.by_indicator['seal']:
            total, cont = exp.by_indicator['seal'][seal_bucket]
            if total >= 3:
                indicator_rates.append(cont / total)

        if indicator_rates:
            indicator_rate = np.mean(indicator_rates)
            indicator_weight = 0.3
        else:
            indicator_rate = 0.0
            indicator_weight = 0.0

        # Signal 3: Base rate with runtime adjustment (weight 20%)
        adjusted_base = base_rate
        if exp.live_total >= 3:
            live_rate = exp.live_correct / exp.live_total
            adjusted_base = 0.7 * base_rate + 0.3 * live_rate
        base_weight = 0.2

        # Weighted combination
        total_weight = streak_weight + indicator_weight + base_weight
        prob = (streak_rate * streak_weight +
                indicator_rate * indicator_weight +
                adjusted_base * base_weight) / total_weight

        return prob

    # ------------------------------------------------------------------
    # Adaptive trailing stop
    # ------------------------------------------------------------------

    def _get_adaptive_trailing_stop(self):
        """Return trailing stop percentage from experience model."""
        if self.experience is not None:
            return self.experience.trailing_stop_pct
        return 0.03

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, days):
        """Execute simulated trading over the given list of trading days."""
        # Build experience model from history before trading starts
        if days:
            self._build_experience_from_history(days[0])
            self._print_experience_summary()

        for date in days:
            # 1. Process existing positions (stop-loss / take-profit / sell)
            self._process_positions(date)

            # 2. Screen candidates
            candidates = self.screen_func(self.stock_data, date)

            # 3. Evaluate each candidate for buy
            for code in candidates:
                # Skip if already holding
                if code in self.positions:
                    continue
                decision, reason = self._decide_buy(code, date)
                self._log_decision(date, code, 'BUY' if decision else 'SKIP', reason)
                if decision:
                    self._execute_buy(code, date)

            # 4. Log HOLD for remaining positions
            for code in list(self.positions):
                self._log_decision(date, code, 'HOLD', '持仓中')

            # 5. Record NAV
            self._record_nav(date)

    def _print_experience_summary(self):
        """Print the experience model summary."""
        exp = self.experience
        if exp is None:
            return

        print("\n" + "=" * 50)
        print("  涨停续板经验模型")
        print("=" * 50)

        if exp.total_limit_ups == 0:
            print("  无历史涨停样本")
            print("=" * 50)
            return

        rate = exp.total_continuations / exp.total_limit_ups
        print(f"  历史涨停样本:   {exp.total_limit_ups} 次")
        print(f"  次日续板:       {exp.total_continuations} 次 ({rate:.1%})")
        print()
        print("  按连板天数:")
        for streak in sorted(exp.by_streak.keys()):
            total, cont = exp.by_streak[streak]
            sr = cont / total if total > 0 else 0
            bar = '█' * int(sr * 20)
            labels = {1: '首板', 2: '2连板', 3: '3连板'}
            label = labels.get(streak, f'{streak}连板')
            print(f"    {label}: {total}次, 续板 {cont:>3}次 ({sr:.1%}) {bar}")

        print(f"\n  自适应追踪止损: {exp.trailing_stop_pct:.1%}")
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # Buy decision
    # ------------------------------------------------------------------

    def _decide_buy(self, code, date):
        """Evaluate whether to buy a screened stock.

        Screener (screen_obv_momentum) already applies signal filters.
        Here we only check data availability and capital.
        """
        df = self.stock_data.get(code)
        if df is None:
            return False, '无数据'

        ts = pd.Timestamp(date)
        if ts not in df.index:
            return False, '无当日数据'

        row = df.loc[ts]
        close = row['Close']

        # Limit-up stocks cannot be bought (封板无法买入)
        idx = df.index.get_loc(ts)
        if idx >= 1:
            prev_close = df.iloc[idx - 1]['Close']
            ratio = get_limit_ratio(code)
            limit_price = round(prev_close * (1 + ratio), 2)
            if close >= limit_price * (1 - config.LIMIT_TOLERANCE):
                return False, f'当日涨停无法买入 ({close:.2f}≈涨停价{limit_price:.2f})'

        # Capital check
        cost = close * (1 + self.slippage) * 100
        total_cost = cost + cost * self.commission
        if total_cost > self.capital:
            return False, f'资金不足 (需{total_cost:.0f}, 余{self.capital:.0f})'

        return True, f'Close={close:.2f}'

    # ------------------------------------------------------------------
    # Sell decision — three regime logic
    # ------------------------------------------------------------------

    def _decide_sell(self, code, date, position):
        """Three-regime sell decision.

        REGIME 1 — Hard stop-loss (always active)
          - Hard stop: Low <= buy_price * 0.97 (3%)

        REGIME 2 — Limit-up mode (when position has touched limit-up)
          - Trailing stop: Low <= peak_price * (1 - trailing_pct)
            (peak uses previous days only — today's High excluded to avoid
             intraday whipsaw on volatile limit-up days)
          - Today still at limit: check continuation prob >= 40%
          - Touched limit but fell back: sell (momentum broken)
          - First day after streak breaks: sell
          - Max hold extended to 8 days

        REGIME 3 — Normal mode (original logic)
          - 8% take-profit, RSI>80, CCI declining 2d, max 5 days

        NOTE: peak_price is updated OUTSIDE this method (in _process_positions)
        after confirming a hold decision, so trailing stop always uses the
        peak from previous days and is not distorted by today's intraday High.
        """
        df = self.stock_data.get(code)
        if df is None:
            return False, '无数据', None

        ts = pd.Timestamp(date)
        if ts not in df.index:
            return False, '无当日数据', None

        row = df.loc[ts]
        buy_price = position['buy_price']
        days_held = position['days_held']
        peak_price = position['peak_price']  # from previous days only

        # Detect today's limit-up status
        idx = df.index.get_loc(ts)
        today_limit_up = False
        today_touched_limit = False
        if idx >= 1:
            prev_close = df.iloc[idx - 1]['Close']
            ratio = get_limit_ratio(code)
            limit_price = round(prev_close * (1 + ratio), 2)
            today_limit_up = row['Close'] >= limit_price * (1 - config.LIMIT_TOLERANCE)
            today_touched_limit = row['High'] >= limit_price * (1 - config.LIMIT_TOLERANCE)

        # Update limit-up tracking
        if today_limit_up:
            position['limit_up_streak'] += 1
            position['in_limit_up_mode'] = True

        # === REGIME 1: Hard stop-loss (always active, 3%) ===
        hard_stop = buy_price * 0.97
        if row['Low'] <= hard_stop:
            return True, f'止损 (Low={row["Low"]:.2f} <= buy×0.97={hard_stop:.2f})', hard_stop

        # === REGIME 2: Limit-up mode ===
        if position['in_limit_up_mode']:
            # Trailing stop (only in limit-up mode, uses previous days' peak)
            trailing_pct = position['trailing_stop_pct']
            trailing_stop = peak_price * (1 - trailing_pct)
            if not today_limit_up and row['Low'] <= trailing_stop:
                return True, (f'追踪止损 (Low={row["Low"]:.2f} <= '
                              f'peak×{1-trailing_pct:.1%}={trailing_stop:.2f})'), trailing_stop

            if today_limit_up:
                prob = self._estimate_continuation_probability(
                    code, date, position['limit_up_streak'])
                if prob >= 0.40:
                    return False, (f'涨停续板持有 (概率{prob:.1%}≥40%, '
                                   f'{position["limit_up_streak"]}连板)'), None
                else:
                    return True, (f'涨停但续板概率低 (概率{prob:.1%}<40%, '
                                  f'{position["limit_up_streak"]}连板)'), row['Close']

            if today_touched_limit and not today_limit_up:
                return True, f'冲板回落 (触及涨停但未封住)', row['Close']

            if not today_limit_up and position['limit_up_streak'] > 0:
                return True, (f'涨停断板 ({position["limit_up_streak"]}'
                              f'连板后首日)'), row['Close']

            if days_held >= 8:
                return True, f'涨停模式持仓超限 ({days_held}天≥8)', row['Close']

            return False, '涨停模式持仓中', None

        # === REGIME 3: Normal mode ===
        # Take-profit
        target_price = buy_price * 1.08
        if row['High'] >= target_price:
            return True, f'止盈 (High={row["High"]:.2f} >= {target_price:.2f})', target_price

        # RSI > 80
        rsi = row.get('Rsi')
        if rsi is not None and not np.isnan(rsi) and rsi > 80:
            return True, f'RSI严重超买 ({rsi:.1f}>80)', row['Close']

        # CCI declining 2 consecutive days
        if idx >= 2:
            cci_today = row.get('CCI')
            cci_1d = df.iloc[idx - 1].get('CCI')
            cci_2d = df.iloc[idx - 2].get('CCI')
            if (cci_today is not None and cci_1d is not None and cci_2d is not None
                    and not np.isnan(cci_today) and not np.isnan(cci_1d)
                    and not np.isnan(cci_2d)):
                if cci_today < cci_1d < cci_2d:
                    return True, (f'CCI连续下降 '
                                  f'({cci_2d:.0f}→{cci_1d:.0f}→{cci_today:.0f})'), row['Close']

        # Max holding period
        if days_held >= 5:
            return True, f'持仓超限 ({days_held}天≥5)', row['Close']

        return False, '继续持有', None

    # ------------------------------------------------------------------
    # Position processing
    # ------------------------------------------------------------------

    def _process_positions(self, date):
        """Process all open positions: check sell conditions."""
        codes_to_sell = []
        ts = pd.Timestamp(date)

        for code, pos in self.positions.items():
            # Increment days held
            pos['days_held'] += 1

            should_sell, reason, sell_price = self._decide_sell(code, date, pos)
            if should_sell:
                codes_to_sell.append((code, sell_price, reason))
            else:
                # Update peak price AFTER confirming hold, so trailing stop
                # in _decide_sell always uses the peak from previous days.
                df = self.stock_data.get(code)
                if df is not None and ts in df.index:
                    high = df.loc[ts, 'High']
                    if high > pos['peak_price']:
                        pos['peak_price'] = high

        for code, sell_price, reason in codes_to_sell:
            self._execute_sell(code, date, sell_price, reason)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_buy(self, code, date):
        """Execute a buy order at close price + slippage."""
        df = self.stock_data[code]
        ts = pd.Timestamp(date)
        close = df.loc[ts, 'Close']

        actual_buy = close * (1 + self.slippage)
        shares = 100
        cost = actual_buy * shares
        commission = cost * self.commission

        self.capital -= (cost + commission)
        self.positions[code] = {
            'buy_price': actual_buy,
            'buy_date': date,
            'shares': shares,
            'days_held': 0,
            'peak_price': actual_buy,
            'limit_up_streak': 0,
            'in_limit_up_mode': False,
            'trailing_stop_pct': self._get_adaptive_trailing_stop(),
        }

    def _execute_sell(self, code, date, sell_price, reason):
        """Execute a sell order and record the trade."""
        pos = self.positions.pop(code)
        actual_sell = sell_price * (1 - self.slippage)
        shares = pos['shares']

        proceeds = actual_sell * shares
        commission = proceeds * self.commission
        stamp_tax = proceeds * config.STAMP_TAX_RATE

        self.capital += (proceeds - commission - stamp_tax)

        buy_cost = pos['buy_price'] * shares
        buy_commission = buy_cost * self.commission
        total_cost = buy_cost + buy_commission
        net_proceeds = proceeds - commission - stamp_tax
        pnl = net_proceeds - total_cost
        pnl_pct = pnl / total_cost

        # Runtime learning for limit-up trades
        if pos['in_limit_up_mode'] and self.experience is not None:
            self.experience.live_total += 1
            if pos['limit_up_streak'] >= 2:
                self.experience.live_correct += 1

        self.trade_log.append({
            'buy_date': pos['buy_date'],
            'sell_date': date,
            'code': code,
            'buy_price': pos['buy_price'],
            'sell_price': actual_sell,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': pos['days_held'],
            'reason': reason,
            'limit_up_streak': pos['limit_up_streak'],
            'in_limit_up_mode': pos['in_limit_up_mode'],
        })

        self._log_decision(date, code, 'SELL', reason, price=actual_sell)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_decision(self, date, code, action, reason, price=None):
        """Record a decision to the daily log."""
        if price is None:
            df = self.stock_data.get(code)
            ts = pd.Timestamp(date)
            if df is not None and ts in df.index:
                price = df.loc[ts, 'Close']
        self.daily_log.append({
            'date': date,
            'code': code,
            'action': action,
            'reason': reason,
            'price': price,
        })

    def _record_nav(self, date):
        """Record daily net asset value (cash + market value of positions)."""
        ts = pd.Timestamp(date)
        market_value = 0.0
        for code, pos in self.positions.items():
            df = self.stock_data.get(code)
            if df is not None and ts in df.index:
                market_value += df.loc[ts, 'Close'] * pos['shares']
            else:
                # Use buy price as fallback
                market_value += pos['buy_price'] * pos['shares']

        nav = self.capital + market_value
        self.daily_nav.append({
            'date': date,
            'capital': self.capital,
            'market_value': market_value,
            'nav': nav,
            'return': nav / self.initial_capital - 1,
        })

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self):
        """Generate a comprehensive trading report."""
        trades = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        nav = pd.DataFrame(self.daily_nav) if self.daily_nav else pd.DataFrame()

        report = {
            'trades': trades,
            'nav': nav,
            'daily_log': pd.DataFrame(self.daily_log),
            'experience': self.experience,
        }

        if trades.empty:
            report.update({
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_nav': self.capital,
                'total_return': 0,
            })
            return report

        # Trade stats
        n_trades = len(trades)
        n_wins = (trades['pnl'] > 0).sum()
        win_rate = n_wins / n_trades

        # Sell reason breakdown
        reason_counts = trades['reason'].apply(
            lambda r: r.split(' ')[0]  # first word: 止损/止盈/RSI/CCI/持仓/涨停/冲板/硬止损/追踪止损
        ).value_counts().to_dict()

        # Limit-up trade statistics
        limit_up_stats = {}
        if 'in_limit_up_mode' in trades.columns:
            lu_trades = trades[trades['in_limit_up_mode'] == True]
            normal_trades = trades[trades['in_limit_up_mode'] == False]
            limit_up_stats = {
                'lu_count': len(lu_trades),
                'lu_win_rate': (lu_trades['pnl'] > 0).mean() if len(lu_trades) > 0 else 0,
                'lu_avg_pnl_pct': lu_trades['pnl_pct'].mean() if len(lu_trades) > 0 else 0,
                'lu_avg_streak': lu_trades['limit_up_streak'].mean() if len(lu_trades) > 0 else 0,
                'normal_count': len(normal_trades),
                'normal_avg_pnl_pct': normal_trades['pnl_pct'].mean() if len(normal_trades) > 0 else 0,
            }

        report.update({
            'total_trades': n_trades,
            'win_rate': win_rate,
            'total_pnl': trades['pnl'].sum(),
            'avg_pnl_pct': trades['pnl_pct'].mean(),
            'avg_days_held': trades['days_held'].mean(),
            'reason_counts': reason_counts,
            'limit_up_stats': limit_up_stats,
        })

        # NAV-based metrics
        if not nav.empty and len(nav) > 1:
            nav_series = nav['nav']
            daily_returns = nav_series.pct_change().dropna()

            # Max drawdown
            running_max = nav_series.cummax()
            drawdown = (nav_series - running_max) / running_max
            max_drawdown = drawdown.min()

            # Sharpe ratio
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(250)
            else:
                sharpe = 0

            report.update({
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'final_nav': nav_series.iloc[-1],
                'total_return': nav_series.iloc[-1] / self.initial_capital - 1,
            })
        else:
            report.update({
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_nav': self.capital,
                'total_return': self.capital / self.initial_capital - 1,
            })

        return report


def print_agent_report(report):
    """Pretty-print the agent trading report."""
    # --- Decision Log ---
    log_df = report.get('daily_log')
    if log_df is not None and not log_df.empty:
        print("\n" + "=" * 70)
        print("  每日决策日志")
        print("=" * 70)
        for _, row in log_df.iterrows():
            action_str = {
                'BUY': '\033[92mBUY \033[0m',
                'SELL': '\033[91mSELL\033[0m',
                'SKIP': '\033[90mSKIP\033[0m',
                'HOLD': '\033[93mHOLD\033[0m',
            }.get(row['action'], row['action'])
            price_str = f"  ¥{row['price']:.2f}" if row['price'] is not None else ""
            print(f"  {row['date']}  {row['code']}  {action_str}{price_str}  {row['reason']}")

    # --- Trade Summary ---
    trades = report.get('trades')
    if trades is not None and not trades.empty:
        print("\n" + "=" * 70)
        print("  交易明细")
        print("=" * 70)
        has_lu = 'limit_up_streak' in trades.columns
        if has_lu:
            print(f"  {'买入日':>10}  {'卖出日':>10}  {'代码':>6}  {'买价':>8}  "
                  f"{'卖价':>8}  {'收益%':>7}  {'天数':>4}  {'连板':>4}  {'原因'}")
        else:
            print(f"  {'买入日':>10}  {'卖出日':>10}  {'代码':>6}  {'买价':>8}  "
                  f"{'卖价':>8}  {'收益%':>7}  {'天数':>4}  {'原因'}")
        print("  " + "-" * 76)
        for _, t in trades.iterrows():
            pnl_color = '\033[92m' if t['pnl'] > 0 else '\033[91m'
            base = (f"  {t['buy_date']:>10}  {t['sell_date']:>10}  {t['code']:>6}  "
                    f"¥{t['buy_price']:>7.2f}  ¥{t['sell_price']:>7.2f}  "
                    f"{pnl_color}{t['pnl_pct']:>+6.2%}\033[0m  "
                    f"{t['days_held']:>4}")
            if has_lu:
                streak = t.get('limit_up_streak', 0)
                streak_str = f"  {streak:>4}" if streak > 0 else "     -"
                base += streak_str
            print(f"{base}  {t['reason']}")

    # --- Limit-up Stats ---
    lu_stats = report.get('limit_up_stats', {})
    if lu_stats and lu_stats.get('lu_count', 0) > 0:
        print("\n" + "=" * 70)
        print("  涨停交易统计")
        print("=" * 70)
        print(f"  涨停交易数:     {lu_stats['lu_count']}")
        print(f"  涨停胜率:       {lu_stats['lu_win_rate']:.2%}")
        print(f"  涨停平均收益:   {lu_stats['lu_avg_pnl_pct']:.2%}")
        print(f"  平均连板天数:   {lu_stats['lu_avg_streak']:.1f}")
        print(f"  普通交易数:     {lu_stats['normal_count']}")
        print(f"  普通平均收益:   {lu_stats['normal_avg_pnl_pct']:.2%}")

    # --- Performance ---
    print("\n" + "=" * 70)
    print("  绩效报告")
    print("=" * 70)
    print(f"  总交易次数:     {report.get('total_trades', 0)}")
    print(f"  胜率:           {report.get('win_rate', 0):.2%}")
    print(f"  总盈亏:         ¥{report.get('total_pnl', 0):.2f}")
    print(f"  平均收益率:     {report.get('avg_pnl_pct', 0):.2%}")
    print(f"  平均持仓天数:   {report.get('avg_days_held', 0):.1f}")
    print(f"  最大回撤:       {report.get('max_drawdown', 0):.2%}")
    print(f"  夏普比率:       {report.get('sharpe_ratio', 0):.2f}")
    print(f"  最终净值:       ¥{report.get('final_nav', 0):.2f}")
    print(f"  总收益率:       {report.get('total_return', 0):.2%}")

    # Reason breakdown
    reason_counts = report.get('reason_counts', {})
    if reason_counts:
        print(f"\n  卖出原因分布:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # --- Experience Model Runtime Stats ---
    exp = report.get('experience')
    if exp is not None and exp.live_total > 0:
        print(f"\n  运行时学习:")
        print(f"    涨停交易观测: {exp.live_total}")
        print(f"    续板成功:     {exp.live_correct}")
        live_rate = exp.live_correct / exp.live_total if exp.live_total > 0 else 0
        print(f"    实际续板率:   {live_rate:.1%}")

    # --- NAV Curve ---
    nav = report.get('nav')
    if nav is not None and not nav.empty:
        print("\n" + "=" * 70)
        print("  净值曲线")
        print("=" * 70)
        for _, row in nav.iterrows():
            bar_len = max(0, int((row['nav'] / report.get('final_nav', row['nav'])) * 30))
            ret_color = '\033[92m' if row['return'] >= 0 else '\033[91m'
            print(f"  {row['date']}  ¥{row['nav']:>10.2f}  "
                  f"{ret_color}{row['return']:>+6.2%}\033[0m  "
                  f"{'█' * bar_len}")

    print("=" * 70 + "\n")
