"""Data preparation: parquet -> dict/JSON for Jinja2 templates."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd

import config
import downloader
import indicators
import screener
import strategies as sig
from backtest import BacktestEngine


# Strategy descriptions: which signals compose each screen
SCREEN_DESCRIPTIONS = {
    'OBV涨停梦': 'OBV 突破 + 成交量激增 + CCI > 50 + 日涨幅 > 2%（仅主板）',
    '形态识别': '横盘≥22天→缩量涨停突破→倍量阴线洗盘→回收确认+无下影线（仅主板）—— '
               '投资笔记核心策略。数据挖掘验证：66信号中27%在5日内涨停，均最大涨幅10%',
    '涨停接力': '近5日有涨停 + 日涨幅≥5% + 收盘位≥90% + CCI>50（仅主板）—— '
               '数据挖掘10,000+涨停事件发现close_position≥0.9是关键区分因子，'
               '将涨停命中率从12%提升至15-20%',
    '跳空涨停基因': '跳空高开不回补 + 收盘位≥70% + 近10日有涨停（仅主板）—— '
                  '首个PF>1的盈利策略。跳空不回补是最强单一K线形态（open>prev_high, '
                  'low>prev_high, close>open），结合涨停基因（近期有涨停证明有大资金参与）'
                  '和强势收盘（收盘在振幅上方70%），三重共振筛选出买盘极强的突破点',
    '缩量突破': '缩量回调(成交量≤70%均量) → 放量突破10日高(成交量≥1.5倍) + '
              '收盘位≥80% + CCI>50 + 非60日新高（仅主板）—— 最强PF策略。'
              '缩量阶段主力悄悄吸筹清理浮筹，放量突破新高确认建仓完成、拉升开始。'
              '收盘位≥80%说明尾盘无抛压，CCI>50确认动量，'
              '非60日新高过滤避免冲高回落（数据挖掘验证的关键增强条件）',
    '群龙夺宝': '横盘平台内≥2涨停 + 阳成团阴分散 + 红肥绿瘦 + 放量突破颈线(≥2倍均量) + '
              '日涨≥5%（仅主板）—— 多个涨停板聚集在同一平台说明多路资金争夺同一标的，'
              '阳日放量阴日缩量证明控盘能力强，放量突破颈线启动主升浪',
}

TRADING_METHODS = {
    'OBV涨停梦': '信号日收盘买入 → 次日高点达5%目标价卖出，否则次日收盘卖出',
    '形态识别': '第3天回收确认信号日收盘买入，持有5天等待涨停或大涨。'
               '适合中线持有（5日涨停率27%），非次日短线策略',
    '涨停接力': '信号日收盘买入 → 次日高点达5%目标价卖出，否则次日收盘卖出。'
               '适合辅助人工判断（追涨策略，机械PF<1），需结合盘口和板块热度',
    '跳空涨停基因': '信号日收盘买入 → 次日高点达7%目标价卖出，否则次日收盘卖出。'
                  '7%目标优于5%（测试集PF 1.49 vs 1.33），因跳空突破后上行空间大。'
                  '回测验证：训练PF=1.31，测试PF=1.49，涨停率17.4%，'
                  '是所有策略中唯一在训练集和测试集均PF>1的策略',
    '缩量突破': '信号日收盘买入 → 次日高点达7%目标价卖出，否则次日收盘卖出。'
              '回测验证：训练PF=1.45，测试PF=2.39，是所有策略中PF最高的。'
              '7%目标最优（突破后上行空间大）。信号稀少但精准',
    '群龙夺宝': '信号日收盘买入 → 次日高点达7%目标价卖出，否则次日收盘卖出。'
              '回测验证：测试PF@7%=1.11，5日涨停率30%。'
              '适合持有数天等待主升浪启动（5日均max涨幅9.4%）',
}

# Per-strategy target override (strategies not listed use config.DEFAULT_TARGET_PCT)
STRATEGY_TARGET_PCT = {
    '跳空涨停基因': config.GAP_LU_TARGET_PCT,  # 7% optimized target
    '缩量突破': config.BREAKOUT_TARGET_PCT,     # 7% optimized target
    '群龙夺宝': config.DRAGON_TARGET_PCT,       # 7% optimized target
}

# Financial explanations for each strategy
STRATEGY_EXPLANATIONS = {
    'OBV涨停梦': {
        'logic': 'OBV（能量潮）通过累计成交量变化追踪资金流向。当OBV突破30日整理平台的高点，'
                 '同时成交量相比近期放大，意味着大资金正在积极介入。',
        'why_works': '量在价先是技术分析核心原理。OBV突破新高代表买方力量持续积累到临界点，'
                     '配合成交量激增确认资金加速流入。CCI>50确保动量向上，'
                     '日涨幅>2%过滤弱势品种，筛选出资金积极介入且动量充足的标的。',
        'risk': '放量突破可能是主力出货伪装。仅限主板（10%涨停板）以降低波动风险。'
                '排除当日涨停股（封板无法买入）。',
    },
    '形态识别': {
        'logic': '投资笔记核心形态：横盘≥22天（主力建仓期）→ 缩量涨停突破平台高点'
                 '（成交量非平台最大量，说明抛压已耗尽）→ 倍量阴线洗盘'
                 '（成交量≥涨停日1.5倍，震出跟风盘）→ 第3天收盘回到阴线上方 + 无下影线'
                 '（确认洗盘结束，买盘坚决无犹豫）。',
        'why_works': '横盘≥22天意味着主力已完成充分建仓，散户失去耐心离场。'
                     '缩量涨停（成交量不是平台期最大量）说明场内浮筹已被充分消化。'
                     '倍量阴线洗盘是主力故意制造恐慌，震出短线跟风盘。'
                     '无下影线的回收是最关键信号 —— 说明洗盘完毕后买盘坚决，没有犹豫试探。'
                     '数据挖掘验证：66个信号中27%在5日内涨停，远高于1.5%基准率。',
        'risk': '信号稀少（约每7天出现1次），需要耐心等待。'
                '大盘系统性下跌时形态失效。',
    },
    '涨停接力': {
        'logic': '寻找近期已有涨停表现、当日大涨但未封板、且收盘价在振幅最高10%的股票。'
                 '这类股票具有"涨停基因"——近期已证明有大资金运作的能力。',
        'why_works': '数据挖掘10,000+涨停事件发现的关键规律：'
                     '收盘位≥90%（close_position）是最强区分因子。'
                     '当日大涨5%+但未封板时，如果收盘价仍在最高价附近（上影线极短），'
                     '说明尾盘买盘极强、无人获利了结，次日继续冲击涨停的概率大幅提升。'
                     '近5日有涨停确认该股具有大资金运作的"基因"，CCI>50确认动量。',
        'risk': '典型追涨策略，PF<1（机械交易亏损）。涨停率高（15-20%）但非涨停时'
                '跌幅也大。适合结合板块热度、盘口特征做人工筛选，不建议完全机械执行。',
    },
    '跳空涨停基因': {
        'logic': '三重共振策略：①跳空高开不回补（开盘价>昨日最高价，最低价>昨日最高价，'
                 '收盘价>开盘价）②收盘位≥70%（收盘价在当日振幅上方70%）'
                 '③近10个交易日内有涨停记录。仅限主板（10%涨停板），排除当日涨停股。',
        'why_works': '这是通过PF（盈亏比）优化挖掘出的首个盈利策略，其金融逻辑：\n'
                     '①跳空不回补 = 极强买压信号。开盘直接跳过昨日最高价且全天不回落，'
                     '说明隔夜有重大利好或大资金集中挂单买入，买盘力度压倒性超过卖盘。\n'
                     '②收盘位≥70% = 无抛压确认。如果跳空后出现长上影线（收盘位低），'
                     '说明高位有大量卖单等待出货；反之收盘位高意味着买盘持续到尾盘，'
                     '没有人愿意在高位卖出。\n'
                     '③涨停基因 = 大资金参与证明。近期有涨停记录说明该股已被游资或机构关注，'
                     '有能力推动涨停的大资金仍在运作中。跳空突破是这些资金的延续动作。\n'
                     '三重共振将随机噪声过滤掉，只留下买盘极强+无抛压+有大资金背景的标的。'
                     '7%目标价优于5%是因为这类强势突破后通常有更大的上行空间。',
        'risk': '信号稀少（730天仅276笔），单日可能多只股票同时触发导致集中风险。'
                '强依赖前一日有涨停记录，本质仍是追涨策略。'
                '训练PF=1.31，测试PF=1.49，是所有策略中唯一可机械交易的，'
                '但仍需注意大盘极端行情下的系统性风险。',
    },
    '缩量突破': {
        'logic': '两阶段量价策略：①缩量回调阶段——成交量连续萎缩至5日均量的70%以下，'
                 '同时股价小幅阴跌或横盘。②放量突破阶段——成交量骤增至5日均量1.5倍以上，'
                 '收盘价突破近10日最高收盘价，收盘位≥80%，CCI>50确认动量。',
        'why_works': '缩量回调→放量突破是经典的主力操盘手法：\n'
                     '①缩量阶段 = 主力清理浮筹。成交量萎缩说明散户卖出意愿耗尽，'
                     '轻微阴跌是主力刻意压价吸收最后的卖单。量能越缩，说明浮筹越干净。\n'
                     '②放量突破 = 主力拉升启动。缩量到极致后突然放量1.5倍+，'
                     '是大资金集中买入的标志。突破10日高确认这不是普通反弹而是新趋势的开始。\n'
                     '③收盘位≥80% = "关门"确认。突破日收盘在高位说明全天持续有买盘，'
                     '没有获利回吐，买方完全控盘。\n'
                     '数据挖掘发现：单独的缩量天数越多反而PF越低（过度缩量=无人关注而非主力吸筹），'
                     '放量倍数越大也越差（暴量=分歧大而非一致看多）。'
                     '真正有效的是"适度缩量→适度放量+突破新高+强势收盘"的组合。\n'
                     '⑤非60日新高 = 避免追高。数据挖掘发现做新高的股票PF显著下降（冲高回落效应），'
                     '而"突破10日高但低于60日高"选出的是从回调中恢复的突破，成功率更高。',
        'risk': '信号较稀少（平均每天0.7个），需要耐心等待。'
                '策略本质仍是追涨突破，大盘弱势时突破容易失败。'
                '建议结合板块趋势和大盘环境使用。',
    },
    '群龙夺宝': {
        'logic': '在一个横盘平台（≥15天）内，多个涨停板聚集（≥2个），形成H型或N型反包结构。'
                 '平台期量价关系呈现"阳成团阴分散"（阳日均量>阴日均量）和'
                 '"红肥绿瘦"（阳线实体>阴线实体），说明主力控盘能力强。'
                 '当放量（≥2倍平台均量）突破平台颈线（收盘>平台最高价）时，'
                 '主升浪行情开启。',
        'why_works': '群龙夺宝的金融逻辑：\n'
                     '①多涨停聚集 = 多路资金争夺。一个平台内出现多次涨停，'
                     '说明不止一路资金在运作该股，共识度极高。涨停板是资金实力的证明。\n'
                     '②阳成团阴分散 = 控盘强势。阳日放量说明主力在涨时积极吸筹，'
                     '阴日缩量说明调整时无人卖出，主力锁仓筹码不松手。\n'
                     '③红肥绿瘦 = 攻守兼备。阳线实体大（涨幅猛）阴线实体小（跌幅浅），'
                     '体现出多方占绝对优势的力量对比。\n'
                     '④倍量突破颈线 = 主升启动信号。前期横盘积累的能量在突破时集中释放，'
                     '成交量≥2倍确认这是真突破而非假突破。\n'
                     '本质上，群龙夺宝 = OBV涨停梦的底层原因（阳日放量阴日缩量→OBV持续上升→突破）。'
                     '数据验证：208信号，测试集5日涨停率30%，大肉案例+47.7%、+46.4%。',
        'risk': '信号量适中（486天208笔，约0.4/天）。'
                '横盘平台判定依赖回看窗口选择，边界情况可能有偏差。'
                '大盘系统性下跌时平台突破容易失败。'
                '建议关注标的是否为行业/题材龙头（符合原战法"珍宝"要求）。',
    },
}

# Maps strategy name -> list of signal function names used in that strategy
STRATEGY_SIGNALS = {
    'OBV涨停梦': [
        'signal_obv_breakout',
        'signal_volume_surge',
        'signal_cci_momentum_floor',
        'signal_daily_gain_gt2',
    ],
}


def compute_deep_stats(stock_data, all_dates):
    """Compute detailed multi-day performance stats for each strategy.

    Scans all strategies across ALL dates, tracks D+1..D+5 returns and
    limit-ups. Returns a dict of detailed stats per strategy.
    """
    import screener as _screener
    from universe import get_limit_ratio as _glr

    def _is_lu(code, close, prev_close):
        lr = _glr(code)
        lp = round(prev_close * (1 + lr), 2)
        return close >= lp * (1 - config.LIMIT_TOLERANCE)

    date_strs = [d if isinstance(d, str) else d.strftime('%Y-%m-%d') for d in all_dates]
    results = {name: [] for name in _screener.ALL_SCREENS}

    for di, date_str in enumerate(date_strs[:-5]):
        if di < 30:
            continue
        for sname, sfunc in _screener.ALL_SCREENS.items():
            try:
                codes = sfunc(stock_data, date_str)
            except Exception:
                continue
            ts = pd.Timestamp(date_str)
            for code in codes:
                df = stock_data.get(code)
                if df is None or ts not in df.index:
                    continue
                idx = df.index.get_loc(ts)
                c = df['Close'].values
                h = df['High'].values
                buy = float(c[idx])
                if buy <= 0:
                    continue
                sig = {}
                any_lu = False
                max_g = 0.0
                for d in range(1, 6):
                    fi = idx + d
                    if fi >= len(c):
                        sig[f'd{d}_ret'] = np.nan
                        sig[f'd{d}_lu'] = False
                        sig[f'd{d}_high'] = np.nan
                        continue
                    ret = float(c[fi]) / buy - 1
                    hret = float(h[fi]) / buy - 1
                    prev = float(c[fi - 1])
                    lu = _is_lu(code, c[fi], prev) if prev > 0 else False
                    sig[f'd{d}_ret'] = ret
                    sig[f'd{d}_lu'] = lu
                    sig[f'd{d}_high'] = hret
                    if lu:
                        any_lu = True
                    max_g = max(max_g, hret)
                sig['any_lu_5d'] = any_lu
                sig['max_gain_5d'] = max_g
                if idx + 1 < len(c):
                    sig['next_ret'] = float(c[idx + 1]) / buy - 1
                    sig['next_high'] = float(h[idx + 1]) / buy - 1
                results[sname].append(sig)

    # Build stats dict for each strategy
    target_map = dict(STRATEGY_TARGET_PCT)
    stats = {}
    for sname, sigs in results.items():
        if not sigs:
            stats[sname] = None
            continue
        df = pd.DataFrame(sigs)
        n = len(df)
        target = target_map.get(sname, config.DEFAULT_TARGET_PCT)
        s = {'total_signals': n, 'daily_avg': round(n / max(len(date_strs), 1), 1)}

        # D+1 return stats
        d1 = df['d1_ret'].dropna() * 100
        s['d1_avg'] = round(float(d1.mean()), 2)
        s['d1_median'] = round(float(d1.median()), 2)
        s['d1_win_rate'] = round(float((d1 > 0).mean() * 100), 1)
        for t in [3, 5, 7, 10]:
            s[f'd1_ge{t}'] = round(float((d1 >= t).mean() * 100), 1)
        s['d1_le_neg5'] = round(float((d1 <= -5).mean() * 100), 1)

        # Limit-up rates
        s['d1_lu_rate'] = round(float(df['d1_lu'].mean() * 100), 1)
        s['lu5_rate'] = round(float(df['any_lu_5d'].mean() * 100), 1)
        lu_by_day = []
        for d in range(1, 6):
            col = f'd{d}_lu'
            rate = round(float(df[col].mean() * 100), 1) if col in df.columns else 0
            lu_by_day.append(rate)
        s['lu_by_day'] = lu_by_day

        # Max gain 5d distribution
        mg = df['max_gain_5d'].dropna() * 100
        s['max5d_avg'] = round(float(mg.mean()), 1)
        s['max5d_median'] = round(float(mg.median()), 1)
        for t in [5, 10, 15, 20]:
            s[f'max5d_ge{t}'] = round(float((mg >= t).mean() * 100), 1)

        # Multi-day cumulative returns D+1..D+5
        multiday = []
        for d in range(1, 6):
            col = f'd{d}_ret'
            if col not in df.columns:
                continue
            vals = df[col].dropna() * 100
            if len(vals) == 0:
                continue
            row = {
                'day': d,
                'avg': round(float(vals.mean()), 1),
                'median': round(float(vals.median()), 1),
                'win_rate': round(float((vals > 0).mean() * 100), 0),
                'gt5_pct': round(float((vals > 5).mean() * 100), 1),
                'gt10_pct': round(float((vals > 10).mean() * 100), 1),
            }
            multiday.append(row)
        s['multiday'] = multiday

        # PF at various targets
        if 'next_high' in df.columns and 'next_ret' in df.columns:
            nh = df['next_high'].dropna()
            nr = df['next_ret'].dropna()
            common = nh.index.intersection(nr.index)
            nh, nr = nh.loc[common], nr.loc[common]
            for tp in [0.05, 0.07]:
                hit = nh >= tp
                pnl = np.where(hit, tp, nr)
                g = float(pnl[pnl > 0].sum())
                l = abs(float(pnl[pnl < 0].sum()))
                pf = round(g / l, 2) if l > 0 else 0
                wr = round(float(hit.mean() * 100), 1)
                s[f'pf{int(tp*100)}'] = pf
                s[f'wr{int(tp*100)}'] = wr

        stats[sname] = s

    return stats


def load_stock_data():
    """Load processed parquet and compute indicators.

    Returns:
        dict[str, DataFrame] with indicators computed, or None if no data.
    """
    stock_data = downloader.load_processed()
    if not stock_data:
        return None

    print("Computing indicators for web...")
    for code in list(stock_data.keys()):
        stock_data[code] = indicators.compute_all(stock_data[code])

    return stock_data


def run_screen_for_date(stock_data, date):
    """Run all screens for a given date.

    Returns:
        dict[str, list[str]] mapping screen name -> list of stock codes.
    """
    results = {}
    for name, func in screener.ALL_SCREENS.items():
        codes = func(stock_data, date)
        results[name] = codes
    return results


def get_pattern_signals_for_date(stock_data, date):
    """Get detailed pattern signal info for a given date.

    Returns:
        list of dicts with pattern match details (code, explanation, confidence, etc.)
    """
    from pattern_screen import get_latest_pattern_signals
    signals = get_latest_pattern_signals()
    # Filter to signals matching this date
    date_ts = date.replace('-', '')
    return [s for s in signals if s.get('signal_date') == date_ts]


def list_available_dates(stock_data):
    """Get sorted list of all trading dates from the data.

    Returns:
        list[str] of date strings.
    """
    sample = next(iter(stock_data.values()))
    dates = sorted(sample.index.strftime('%Y-%m-%d').tolist())
    return dates


def get_stock_indicators(stock_data, code, date):
    """Extract key indicator values for a stock on a given date.

    Returns:
        dict with indicator values, or None if data unavailable.
    """
    if code not in stock_data:
        return None
    df = stock_data[code]
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return None

    row = df.loc[ts]
    prev_close = df['Close'].shift(1).loc[ts] if ts in df.index else None
    change_pct = ((row['Close'] - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0

    return {
        'code': code,
        'close': round(float(row['Close']), 2),
        'change_pct': round(float(change_pct), 2),
        'volume': int(row['Volume']),
        'rsi6': round(float(row.get('Rsi', 0)), 1) if pd.notna(row.get('Rsi')) else '-',
        'rsi12': round(float(row.get('Rsi12', 0)), 1) if pd.notna(row.get('Rsi12')) else '-',
        'crsi': round(float(row.get('CRSI', 0)), 1) if pd.notna(row.get('CRSI')) else '-',
        'cci': round(float(row.get('CCI', 0)), 1) if pd.notna(row.get('CCI')) else '-',
        'obv': int(row.get('OBV', 0)) if pd.notna(row.get('OBV')) else '-',
        'adx': round(float(row.get('ADX', 0)), 1) if pd.notna(row.get('ADX')) else '-',
        'adxr': round(float(row.get('ADXR', 0)), 1) if pd.notna(row.get('ADXR')) else '-',
        'vol_7d': round(float(row.get('Volume percentage 7d', 0)), 1) if pd.notna(row.get('Volume percentage 7d')) else '-',
    }


def prepare_candlestick_data(stock_data, code, tail=60):
    """Convert stock data to ECharts K-line format.

    Returns:
        dict with dates, ohlc, volume, and indicator arrays for ECharts.
    """
    if code not in stock_data:
        return None
    df = stock_data[code].tail(tail).copy()

    dates = df.index.strftime('%Y-%m-%d').tolist()
    ohlc = df[['Open', 'Close', 'Low', 'High']].values.tolist()
    ohlc = [[round(v, 2) for v in row] for row in ohlc]
    volume = df['Volume'].fillna(0).astype(int).tolist()

    def safe_list(col):
        if col in df.columns:
            return [round(float(v), 2) if pd.notna(v) else None for v in df[col]]
        return [None] * len(dates)

    return {
        'dates': dates,
        'ohlc': ohlc,
        'volume': volume,
        'adx': safe_list('ADX'),
        'adxr': safe_list('ADXR'),
        'plus_dm': safe_list('+DM'),
        'minus_dm': safe_list('-DM'),
        'crsi': safe_list('CRSI'),
        'rsi6': safe_list('Rsi'),
        'rsi12': safe_list('Rsi12'),
        'obv': [int(v) if pd.notna(v) else None for v in df.get('OBV', pd.Series([0]*len(df)))],
    }


def get_signal_diagnostics(stock_data, code, date, strategy_name):
    """Extract actual indicator values vs thresholds for each signal in a strategy.

    Returns:
        dict[signal_name -> {label, items: [{name, value, threshold, op}]}]
        or empty dict if strategy not in STRATEGY_SIGNALS.
    """
    if strategy_name not in STRATEGY_SIGNALS:
        return {}
    if code not in stock_data:
        return {}

    df = stock_data[code]
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return {}

    idx = df.index.get_loc(ts)
    result = {}

    for sig_name in STRATEGY_SIGNALS[strategy_name]:
        items = []

        if sig_name == 'signal_cci_cross_neg100':
            cci_today = float(df['CCI'].iloc[idx]) if pd.notna(df['CCI'].iloc[idx]) else None
            cci_yest = float(df['CCI'].iloc[idx - 1]) if idx > 0 and pd.notna(df['CCI'].iloc[idx - 1]) else None
            items = [
                {'name': 'CCI(今日)', 'value': round(cci_today, 1) if cci_today is not None else '-',
                 'threshold': str(config.CCI_CROSS_LEVEL), 'op': '>'},
                {'name': 'CCI(昨日)', 'value': round(cci_yest, 1) if cci_yest is not None else '-',
                 'threshold': str(config.CCI_CROSS_LEVEL), 'op': '<'},
            ]
            result[sig_name] = {'label': 'CCI穿越-100', 'items': items}

        elif sig_name == 'signal_escape_bottom':
            if idx >= config.ESCAPE_BOTTOM_SHORT_WINDOW and idx >= config.ESCAPE_BOTTOM_LONG_WINDOW:
                short_pct = abs(float(df['Close'].iloc[idx]) / float(df['Close'].iloc[idx - config.ESCAPE_BOTTOM_SHORT_WINDOW]) - 1)
                long_pct = float(df['Close'].iloc[idx]) / float(df['Close'].iloc[idx - config.ESCAPE_BOTTOM_LONG_WINDOW]) - 1
            else:
                short_pct = None
                long_pct = None
            items = [
                {'name': '15日距离%', 'value': f'{short_pct * 100:.1f}%' if short_pct is not None else '-',
                 'threshold': f'{config.ESCAPE_BOTTOM_SHORT_PCT * 100:.0f}%', 'op': '<='},
                {'name': '55日深度%', 'value': f'{long_pct * 100:.1f}%' if long_pct is not None else '-',
                 'threshold': f'{config.ESCAPE_BOTTOM_LONG_PCT * 100:.0f}%', 'op': '<='},
            ]
            result[sig_name] = {'label': '底部逃逸', 'items': items}

        elif sig_name == 'signal_cci_deep_oversold':
            def _cci(offset):
                i = idx - offset
                if 0 <= i < len(df) and pd.notna(df['CCI'].iloc[i]):
                    return round(float(df['CCI'].iloc[i]), 1)
                return '-'
            items = [
                {'name': 'CCI(t-1)', 'value': _cci(1), 'threshold': str(config.CCI_CROSS_LEVEL), 'op': '<'},
                {'name': 'CCI(t-2)', 'value': _cci(2), 'threshold': str(config.CCI_DEEP_LEVEL), 'op': '<'},
                {'name': 'CCI(t-3)', 'value': _cci(3), 'threshold': str(config.CCI_DEEP_LEVEL), 'op': '<'},
                {'name': 'CCI(t-7)', 'value': _cci(6), 'threshold': f'{config.CCI_DEEP_BEFORE7D_LOWER}~{config.CCI_DEEP_BEFORE7D_UPPER}', 'op': '区间'},
            ]
            result[sig_name] = {'label': 'CCI深度超卖', 'items': items}

        elif sig_name == 'signal_daily_gain_lt3':
            prev_close = float(df['Close'].iloc[idx - 1]) if idx > 0 else None
            cur_close = float(df['Close'].iloc[idx])
            gain = (cur_close / prev_close - 1) * 100 if prev_close and prev_close > 0 else None
            items = [
                {'name': '日涨幅%', 'value': f'{gain:.2f}%' if gain is not None else '-',
                 'threshold': f'{config.DAILY_GAIN_LT3_PCT * 100:.0f}%', 'op': '<'},
            ]
            result[sig_name] = {'label': '日涨幅<3%', 'items': items}

        elif sig_name == 'signal_ema5_proximity':
            close = float(df['Close'].iloc[idx])
            ema5 = float(df['EMA'].iloc[idx]) if pd.notna(df['EMA'].iloc[idx]) else None
            prox = abs(close / ema5 - 1) * 100 if ema5 and ema5 > 0 else None
            items = [
                {'name': '|Close/EMA5-1|%', 'value': f'{prox:.2f}%' if prox is not None else '-',
                 'threshold': f'{config.EMA5_PROXIMITY_PCT * 100:.0f}%', 'op': '<'},
                {'name': 'EMA5', 'value': round(ema5, 2) if ema5 is not None else '-'},
            ]
            result[sig_name] = {'label': '价格贴近EMA5', 'items': items}

        elif sig_name == 'signal_obv_breakout':
            obv = float(df['OBV'].iloc[idx]) if pd.notna(df['OBV'].iloc[idx]) else None
            window = config.OBV_CONSOLIDATION_WINDOW
            if idx >= window:
                obv_max_30 = float(df['OBV'].iloc[idx - window:idx].max())
            else:
                obv_max_30 = None
            items = [
                {'name': 'OBV', 'value': int(obv) if obv is not None else '-'},
                {'name': f'{window}日最高OBV', 'value': int(obv_max_30) if obv_max_30 is not None else '-'},
                {'name': '突破量', 'value': int(obv - obv_max_30) if obv is not None and obv_max_30 is not None else '-',
                 'threshold': '> 0', 'op': '>'},
            ]
            result[sig_name] = {'label': 'OBV突破', 'items': items}

        elif sig_name == 'signal_volume_surge':
            vol_7d = float(df['Volume percentage 7d'].iloc[idx]) if pd.notna(df['Volume percentage 7d'].iloc[idx]) else None
            vol_3d = float(df['Volume percentage 3d'].iloc[idx]) if pd.notna(df['Volume percentage 3d'].iloc[idx]) else None
            items = [
                {'name': '量比7d%', 'value': f'{vol_7d:.1f}' if vol_7d is not None else '-',
                 'threshold': f'{config.VOLUME_7D_LOWER}~{config.VOLUME_7D_UPPER}', 'op': '区间'},
                {'name': '量比3d%', 'value': f'{vol_3d:.1f}' if vol_3d is not None else '-',
                 'threshold': str(config.VOLUME_3D_UPPER), 'op': '<'},
            ]
            result[sig_name] = {'label': '成交量激增', 'items': items}

        elif sig_name == 'signal_cci_momentum_floor':
            cci = float(df['CCI'].iloc[idx]) if pd.notna(df['CCI'].iloc[idx]) else None
            items = [
                {'name': 'CCI', 'value': round(cci, 1) if cci is not None else '-',
                 'threshold': str(config.CCI_MOMENTUM_FLOOR), 'op': '>'},
            ]
            result[sig_name] = {'label': 'CCI动量底线', 'items': items}

        elif sig_name == 'signal_daily_gain_gt2':
            prev_close = float(df['Close'].iloc[idx - 1]) if idx > 0 else None
            cur_close = float(df['Close'].iloc[idx])
            gain = (cur_close / prev_close - 1) * 100 if prev_close and prev_close > 0 else None
            items = [
                {'name': '日涨幅%', 'value': f'{gain:.2f}%' if gain is not None else '-',
                 'threshold': f'{config.DAILY_GAIN_GT2_PCT * 100:.0f}%', 'op': '>'},
            ]
            result[sig_name] = {'label': '日涨幅>2%', 'items': items}

        elif sig_name == 'signal_adx_below_max':
            adx = float(df['ADX'].iloc[idx]) if pd.notna(df['ADX'].iloc[idx]) else None
            items = [
                {'name': 'ADX', 'value': round(adx, 1) if adx is not None else '-',
                 'threshold': str(config.ADX_BUY_MAX), 'op': '<'},
            ]
            result[sig_name] = {'label': 'ADX趋势过滤', 'items': items}

    return result


def compute_motive_label(stock_data, code, date):
    """Classify OBV strategy motive based on price-volume characteristics.

    Returns:
        str: one of '出货', '试盘', '吸筹', '待确认'
    """
    if code not in stock_data:
        return '待确认'

    df = stock_data[code]
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return '待确认'

    idx = df.index.get_loc(ts)
    row = df.iloc[idx]

    # Position within 55-day range (0% = low, 100% = high)
    if idx >= 55:
        high_55 = df['Close'].iloc[idx - 55:idx + 1].max()
        low_55 = df['Close'].iloc[idx - 55:idx + 1].min()
        if high_55 > low_55:
            position_pct = (float(row['Close']) - low_55) / (high_55 - low_55) * 100
        else:
            position_pct = 50
    else:
        position_pct = 50

    # Volume ratio vs 20-day average
    if idx >= 20:
        avg_vol_20 = df['Volume'].iloc[idx - 20:idx].mean()
        vol_ratio = float(row['Volume']) / avg_vol_20 if avg_vol_20 > 0 else 1
    else:
        vol_ratio = 1

    # Upper shadow ratio
    high = float(row['High'])
    close = float(row['Close'])
    open_ = float(row['Open'])
    low = float(row['Low'])
    body_top = max(close, open_)
    full_range = high - low
    upper_shadow_ratio = (high - body_top) / full_range if full_range > 0 else 0

    # Daily amplitude
    amplitude = (high - low) / float(row['Close']) * 100 if close > 0 else 0

    # OBV 5-day trend
    if idx >= 5:
        obv_5d_rising = float(df['OBV'].iloc[idx]) > float(df['OBV'].iloc[idx - 5])
    else:
        obv_5d_rising = False

    # Classification rules
    if position_pct >= 80 and vol_ratio >= 3 and upper_shadow_ratio > 0.3:
        return '出货'
    if vol_ratio < 2 and upper_shadow_ratio > 0.5:
        return '试盘'
    if position_pct <= 30 and amplitude < 3 and obv_5d_rising:
        return '吸筹'
    return '待确认'


def compute_failure_analysis(stock_data, trade_details, n=10):
    """Analyze the most recent losing trades for common failure patterns.

    Args:
        stock_data: dict[str, DataFrame]
        trade_details: list of trade detail dicts from run_all_backtests
        n: number of recent losses to analyze

    Returns:
        dict with failure pattern statistics, or None if no losses.
    """
    # Filter to losing trades
    losses = [t for t in trade_details if t['pnl_pct'] < 0]
    if not losses:
        return None

    # Take last n losses (most recent)
    recent = losses[-n:]

    long_upper_shadow = 0
    next_day_gap_down = 0
    high_volume_retreat = 0
    bearish_close = 0
    detail_trades = []

    for t in recent:
        code = t['code']
        buy_date = t['buy_date']
        sell_date = t['sell_date']

        if code not in stock_data:
            continue
        df = stock_data[code]
        buy_ts = pd.Timestamp(buy_date)
        sell_ts = pd.Timestamp(sell_date)

        if buy_ts not in df.index or sell_ts not in df.index:
            continue

        buy_row = df.loc[buy_ts]
        sell_row = df.loc[sell_ts]

        # Upper shadow ratio on buy day
        high = float(buy_row['High'])
        close = float(buy_row['Close'])
        open_ = float(buy_row['Open'])
        low = float(buy_row['Low'])
        body_top = max(close, open_)
        full_range = high - low
        upper_shadow = (high - body_top) / full_range if full_range > 0 else 0

        # Next day gap = sell_day open vs buy_day close
        gap_pct = (float(sell_row['Open']) / close - 1) * 100 if close > 0 else 0

        # Volume ratio on sell day
        sell_idx = df.index.get_loc(sell_ts)
        if sell_idx >= 7:
            avg_vol = df['Volume'].iloc[sell_idx - 7:sell_idx].mean()
            sell_vol_ratio = float(sell_row['Volume']) / avg_vol if avg_vol > 0 else 1
        else:
            sell_vol_ratio = 1
        sell_bearish = float(sell_row['Close']) < float(sell_row['Open'])

        flags = []
        if upper_shadow > 0.5:
            long_upper_shadow += 1
            flags.append('上影线过长')
        if gap_pct < -1:
            next_day_gap_down += 1
            flags.append('次日低开')
        if sell_vol_ratio > 1.5 and sell_bearish:
            high_volume_retreat += 1
            flags.append('放量回落')
        if sell_bearish:
            bearish_close += 1
            flags.append('收阴')

        detail_trades.append({
            'buy_date': buy_date,
            'sell_date': sell_date,
            'code': code,
            'pnl_pct': t['pnl_pct'],
            'upper_shadow': round(upper_shadow, 2),
            'gap_pct': round(gap_pct, 2),
            'sell_vol_ratio': round(sell_vol_ratio, 2),
            'flags': flags,
        })

    total = len(recent)
    parts = []
    if long_upper_shadow:
        parts.append(f'{long_upper_shadow}次上影线过长')
    if next_day_gap_down:
        parts.append(f'{next_day_gap_down}次次日低开')
    if high_volume_retreat:
        parts.append(f'{high_volume_retreat}次放量回落')
    if bearish_close:
        parts.append(f'{bearish_close}次收阴')
    summary = f'最近{total}次失败中，' + '，'.join(parts) if parts else f'最近{total}次失败无明显共同特征'

    return {
        'total_losses': total,
        'long_upper_shadow': long_upper_shadow,
        'next_day_gap_down': next_day_gap_down,
        'high_volume_retreat': high_volume_retreat,
        'bearish_close': bearish_close,
        'summary': summary,
        'trades': detail_trades,
    }


def prepare_equity_curve(daily_details):
    """Convert backtest daily details to ECharts equity curve format.

    Args:
        daily_details: DataFrame from BacktestEngine.generate_report()['daily_details']

    Returns:
        dict with dates, equity, drawdown arrays.
    """
    if daily_details is None or daily_details.empty:
        return {'dates': [], 'equity': [], 'drawdown': []}

    dates = daily_details.index.tolist()
    returns = daily_details['return']
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max)

    return {
        'dates': dates,
        'equity': [round(float(v), 4) for v in equity],
        'drawdown': [round(float(v), 4) for v in drawdown],
    }


def run_all_backtests(stock_data, days):
    """Run backtests for all strategies.

    Args:
        stock_data: dict[str, DataFrame] with indicators.
        days: list[str] of trading day strings.

    Returns:
        dict[str, dict] mapping screen name -> {report, results, equity_data}.
    """
    engine = BacktestEngine()
    all_reports = {}

    for name, screen_func in screener.ALL_SCREENS.items():
        print(f"  Backtesting: {name}...")
        target_pct = STRATEGY_TARGET_PCT.get(name, config.DEFAULT_TARGET_PCT)
        results = engine.run(screen_func, stock_data, days, target_pct=target_pct)
        report = engine.generate_report(results)
        equity_data = prepare_equity_curve(report.get('daily_details'))

        # Build trade detail list for the template
        trade_details = []
        if not results.empty:
            for _, row in results.iterrows():
                # Compute actual close-to-close return (not backtest simulation)
                code = row['code']
                buy_date = row['buy_date']
                sell_date = row['sell_date']
                actual_return = 0.0
                buy_close = 0.0
                sell_close = 0.0
                if code in stock_data:
                    df = stock_data[code]
                    buy_ts = pd.Timestamp(buy_date)
                    sell_ts = pd.Timestamp(sell_date)
                    if buy_ts in df.index and sell_ts in df.index:
                        buy_close = float(df.loc[buy_ts, 'Close'])
                        sell_close = float(df.loc[sell_ts, 'Close'])
                        if buy_close > 0:
                            actual_return = (sell_close / buy_close - 1) * 100

                trade_details.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'code': code,
                    'buy_price': round(buy_close, 2),
                    'sell_price': round(sell_close, 2),
                    'pnl_pct': round(actual_return, 2),
                    'hit_target': bool(row['hit_target']),
                    'hit_limit_up': bool(row['hit_limit_up']),
                })

        failure_analysis = compute_failure_analysis(stock_data, trade_details)

        all_reports[name] = {
            'report': {
                'total_trades': report['total_trades'],
                'win_rate': round(report['win_rate'] * 100, 1),
                'limit_up_rate': round(report['limit_up_rate'] * 100, 1),
                'profit_factor': round(report['profit_factor'], 2),
                'cumulative_return': round(report['cumulative_return'] * 100, 1),
                'annualized_return': round(report['annualized_return'] * 100, 1),
                'max_drawdown': round(report['max_drawdown'] * 100, 1),
                'sharpe_ratio': round(report['sharpe_ratio'], 2),
                'sortino_ratio': round(report['sortino_ratio'], 2),
                'total_pnl': round(report['total_pnl'], 2),
            },
            'equity_data': equity_data,
            'trade_details': trade_details,
            'failure_analysis': failure_analysis,
        }

    return all_reports
