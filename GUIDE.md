# SZ 量化交易系统使用指南

深交所 A 股短线量化交易系统，基于技术指标组合筛选 + 模拟交易 Agent。

---

## 环境准备

### 依赖安装

```bash
pip install yfinance pandas numpy matplotlib mplfinance
```

TA-Lib 需要单独安装（需要 C 库）：

```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux
# 先安装 ta-lib C 库，再 pip install TA-Lib
```

### 项目结构

```
sz/
├── run.py              # CLI 入口（所有命令从这里进入）
├── config.py           # 全局配置（路径、指标参数、策略阈值）
├── universe.py         # 股票池（深交所代码段、板块判断）
├── downloader.py       # 数据下载（yfinance）与本地缓存
├── indicators.py       # 技术指标计算
├── labels.py           # 涨停标签构建
├── strategies.py       # 信号函数（26个）
├── screener.py         # 组合筛选策略（8个）
├── backtest.py         # 回测引擎
├── agent.py            # 模拟交易 Agent
├── diagnose.py         # 特征诊断工具
├── optimizer.py        # 信号组合优化器
├── visualize.py        # 可视化（K线、资金曲线、回撤）
├── pipeline.py         # 数据流水线
├── web/                # 静态网站生成器
│   ├── generator.py
│   ├── data_prep.py
│   ├── templates/
│   └── static/
├── data/               # 数据目录（自动创建）
│   ├── raw/            # 原始下载数据
│   ├── processed/      # 处理后数据
│   ├── features/       # 特征数据
│   └── predictions/    # 预测结果
└── site/               # 生成的静态网站
```

---

## 快速开始

### 1. 初始化数据（首次运行）

```bash
cd /path/to/sz
python run.py init
```

下载深交所全部股票（~1800只）近一年日线数据，计算技术指标，保存到 `data/` 目录。

可指定起始日期：

```bash
python run.py init --start 2024-01-01
```

### 2. 每日更新

```bash
python run.py daily
```

增量拉取最近 60 天数据，更新指标和筛选结果。可指定日期：

```bash
python run.py daily --date 2025-10-27
```

### 3. 运行筛选

```bash
python run.py screen
```

对当日所有 8 个筛选策略输出命中的股票代码。

---

## 核心命令

### `agent` — 模拟交易

最核心的命令。启动模拟交易 Agent 对历史数据进行仿真交易。

```bash
python run.py agent [--days N] [--capital AMOUNT] [--screen NAME]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--days` | 60 | 回测最近 N 个交易日 |
| `--capital` | 100000 | 初始资金（元） |
| `--screen` | OBV涨停梦 | 使用的筛选策略名称 |
| `--days-list` | 无 | 指定具体日期列表（逗号分隔） |

示例：

```bash
# 默认策略，60天
python run.py agent --days 60

# 指定策略和周期
python run.py agent --days 120 --screen 当日金叉买入

# 更大资金
python run.py agent --days 60 --capital 500000
```

**Agent 交易逻辑**：

1. **买入决策**：
   - 筛选策略输出候选股
   - 排除当日涨停股（封板无法买入）
   - 资金充足检查

2. **卖出决策（三段式）**：
   - **硬止损**：Low ≤ 买入价 × 0.97（-3%）
   - **涨停模式**：持仓触及涨停后启用追踪止损 + 续板概率判断
   - **普通模式**：8% 止盈 / RSI>80 / CCI 连降 / 最多持仓 5 天

3. **涨停续板学习**：Agent 从历史数据学习涨停续板概率（按连板天数、RSI/CCI/ADX/成交量分桶），用于涨停模式下的持仓决策。

**输出内容**：每日决策日志、交易明细、涨停交易统计、绩效报告（胜率、总收益、夏普比率、最大回撤）、净值曲线。

### `backtest` — 回测

对所有注册的筛选策略执行固定目标收益回测。

```bash
python run.py backtest [--days DATES] [--target PCT] [--plot]
```

| 参数 | 说明 |
|------|------|
| `--days` | 逗号分隔的日期列表（默认最近60天） |
| `--target` | 目标收益率（默认 0.03 即 3%） |
| `--plot` | 显示资金曲线和回撤图 |

### `diagnose` — 特征诊断

分析指定策略筛选出的交易，对比涨停/非涨停组的特征分布差异。

```bash
python run.py diagnose [--screen NAME] [--days DATES] [--target PCT]
```

### `optimize` — 信号组合优化

穷举信号组合，寻找最优策略参数。

```bash
python run.py optimize [--focus SIGNALS] [--min-signals N] [--max-signals N]
                        [--train-window N] [--test-window N] [--top-n N]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--focus` | 无 | 必须包含的信号（逗号分隔） |
| `--min-signals` | 2 | 最少信号数 |
| `--max-signals` | 4 | 最多信号数 |
| `--train-window` | 120 | 训练窗口天数 |
| `--test-window` | 20 | 测试窗口天数 |
| `--top-n` | 5 | Phase 2 候选组合数 |

### `plot` — K线图

绘制单只股票的多面板 K 线图。

```bash
python run.py plot 002906 [--tail 40]
```

面板包含：价格 K 线、ADX/ADXR/±DM、CRSI/RSI-6/RSI-12、OBV。

### `web` — 静态网站

生成包含每日筛选结果的静态 HTML 网站。

```bash
python run.py web [--output site] [--date DATE]
```

---

## 筛选策略

系统包含 8 个组合筛选策略，每个策略由多个信号函数 AND 组合而成：

| 策略名称 | 信号组合 | 思路 |
|----------|---------|------|
| **超卖** | CCI穿越-100 + 底部逃离 + CCI深度超卖 + 日涨<3% | 深度超卖反弹 |
| **当日金叉买入** | UOS穿越65 + 非涨停 + RSI金叉 | 多指标共振看多 |
| **底部异动** | 量能堆积 + 底部逃离 | 底部放量异动 |
| **震荡指标** | ADX穿越ADXR + 底部逃离 | 底部趋势启动 |
| **近日上涨** | ADX穿越ADXR + DM多头 + 非涨停 | 确认多头趋势 |
| **CCI快进出** | CCI穿越-100 + 底部逃离 + CCI深度超卖 + 日涨<3% | 快速CCI反弹 |
| **OBV涨停梦** | OBV突破 + 量能堆积 + CCI>50 + 日涨>2% | T0买入博T+1涨停 |
| **OBV波段** | OBV涨停梦 + ADX<40 | 过滤趋势过强，做2-8%波段 |
| **OBV底部突破** | OBV突破 + 量能堆积 + 底部逃离 | OBV底部放量突破 |

使用 `--screen` 参数指定策略（仅 `agent` 和 `diagnose` 命令支持）。

---

## 信号函数

系统定义了 27 个信号函数（`strategies.py`），分为 5 类：

**CRSI 类**：`crsi_cross50`、`crsi_above50`、`crsi_below25`、`crsi_decrease`、`crsi_cross20`

**RSI 类**：`rsi_golden_cross`、`rsi_strengthening`、`rsi_declining`

**量能类**：`volume_surge`、`obv_breakout`、`uos_cross65`

**趋势类**：`adx_cross_adxr`、`adxr_above25`、`dm_positive`、`adx_below_max`

**其他**：`escape_bottom`、`cci_cross_neg100`、`cci_deep_oversold`、`cci_momentum_floor`、`ema5_proximity`、`sideways`、`consecutive_decline`、`pct_rank_spike`、`no_limit_up`、`avoid_high`、`daily_gain_gt5`、`daily_gain_lt3`、`daily_gain_gt2`

---

## 配置参数

所有可调参数集中在 `config.py`，主要分类：

### 路径配置
- `DATA_DIR` / `RAW_DIR` / `PROCESSED_DIR` / `FEATURES_DIR` / `PREDICTIONS_DIR`

### 下载参数
- `LOOKBACK_DAYS = 365`：初始化时回溯天数
- `INCREMENTAL_DAYS = 60`：每日增量拉取天数

### 交易成本
- `COMMISSION_RATE = 0.0003`：佣金万三
- `SLIPPAGE_RATE = 0.001`：滑点千一
- `STAMP_TAX_RATE = 0.001`：印花税千一（卖出收取）

### 策略阈值
参见 `config.py` 中各信号对应的常量。关键参数：
- `ADX_BUY_MAX = 40`：Agent 买入时 ADX 上限（过滤趋势过强的股票）
- `CCI_MOMENTUM_FLOOR = 50`：OBV涨停梦的 CCI 下限
- `DAILY_GAIN_GT2_PCT = 0.02`：日涨幅>2%

### 指标参数
- `RSI_SHORT_PERIOD = 6` / `RSI_LONG_PERIOD = 12`
- `EMA_PERIOD = 5`
- `ADX_PERIOD = 8`
- `CCI_PERIOD = 14`
- `UOS_PERIODS = (7, 14, 28)`

---

## 典型工作流

### 日常选股

```bash
# 1. 更新数据
python run.py daily

# 2. 查看筛选结果
python run.py screen

# 3. 对感兴趣的股票查看K线
python run.py plot 002906
```

### 策略评估

```bash
# 1. 用 Agent 模拟交易（推荐）
python run.py agent --days 120 --screen OBV涨停梦

# 2. 诊断策略特征
python run.py diagnose --screen OBV涨停梦

# 3. 探索新信号组合
python run.py optimize --focus obv_breakout --max-signals 4
```

### 生成报告网站

```bash
python run.py web --output site
# 生成的 HTML 在 site/ 目录
```

---

## 注意事项

1. **数据依赖**：首次使用必须先执行 `python run.py init` 下载数据，下载约需 20-30 分钟（~1800只股票）。
2. **网络要求**：数据通过 yfinance 从 Yahoo Finance 获取，需要稳定的网络连接。
3. **仅限深交所**：本系统仅覆盖深交所股票（代码后缀 `.SZ`），不包含上交所。
4. **不含 ST 股票**：股票池已排除 ST 类股票。
5. **风险提示**：本系统仅供学习和研究用途，不构成投资建议。量化策略的历史回测表现不代表未来收益。
