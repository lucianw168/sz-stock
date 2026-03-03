# SZ 量化选股系统使用指南

深交所 A 股短线量化选股系统，基于技术指标组合筛选 + 回测验证 + 实时盘中选股 + 静态网站展示。

数据源：Tushare Pro（主力）/ yfinance（备用）。

---

## 环境准备

### 依赖安装

```bash
pip install tushare pandas numpy jinja2 yfinance
```

TA-Lib 需要单独安装（需要 C 库）：

```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux
# 先安装 ta-lib C 库，再 pip install TA-Lib
```

### Tushare Token 配置

注册 [Tushare Pro](https://tushare.pro/) 获取 token，设置环境变量：

```bash
export TUSHARE_TOKEN='你的token'
```

实时选股功能需要 `rt_k` 接口权限（需单独申请）。若无权限，系统会自动切换 yfinance（约 15 分钟延迟）。

### 项目结构

```
sz/
├── run.py              # CLI 入口（所有命令从这里进入）
├── config.py           # 全局配置（路径、指标参数、策略阈值）
├── universe.py         # 股票池定义（深交所代码段、板块判断）
├── downloader.py       # 数据下载（Tushare + yfinance fallback）与本地缓存
├── indicators.py       # 技术指标计算（RSI、CRSI、ADX、CCI、OBV 等）
├── labels.py           # 涨停标签构建（用于 ML）
├── strategies.py       # 信号函数定义（统一接口 signal_xxx(df) -> Series[bool]）
├── screener.py         # 组合筛选策略（多信号 AND 组合）
├── pattern_screen.py   # AI 形态识别（横盘缩量涨停延迟回收）
├── backtest.py         # 回测引擎
├── agent.py            # 模拟交易 Agent（三段式卖出、涨停续板学习）
├── diagnose.py         # 特征诊断（涨停 vs 非涨停对比分析）
├── optimizer.py        # 信号组合优化器
├── visualize.py        # K线图、资金曲线、回撤图
├── pipeline.py         # 数据流水线（初始化 / 每日更新）
├── deploy.sh           # GitHub Pages 部署脚本
├── web/                # 静态网站生成器
│   ├── generator.py    # 页面生成主逻辑
│   ├── data_prep.py    # 数据准备（回测、指标、诊断）
│   ├── templates/      # Jinja2 模板（深色金融终端风格）
│   │   ├── base.html
│   │   ├── index.html      # Dashboard 首页
│   │   ├── daily.html      # 每日选股详情
│   │   ├── strategy.html   # 策略回测详情
│   │   ├── stock.html      # 个股 K 线图
│   │   └── history.html    # 历史记录
│   └── static/
│       └── style.css       # 深色主题样式
├── data/               # 数据目录（自动创建）
│   ├── raw/            # Tushare 原始缓存（parquet）
│   ├── processed/      # 处理后合并数据
│   ├── features/       # 特征数据
│   ├── models/         # 形态模板库
│   └── predictions/    # 预测结果
└── site/               # 生成的静态网站（gh-pages 部署）
```

---

## 快速开始

### 1. 初始化数据（首次运行）

```bash
cd /path/to/sz
python run.py init
```

下载深交所全部股票（~2700只）近两年日线数据，计算技术指标，保存到 `data/` 目录。

可指定起始日期：

```bash
python run.py init --start 2024-01-01
```

如已有 `limit_up` 项目的缓存数据，可直接导入：

```bash
python run.py init --from-cache
```

### 2. 每日更新

```bash
python run.py daily
```

增量拉取最近 60 天数据，更新指标和筛选结果。可指定日期：

```bash
python run.py daily --date 2026-03-02
```

### 3. 运行筛选

```bash
python run.py screen
```

对当日所有筛选策略输出命中的股票代码。

---

## 核心命令

### `live` — 实时盘中选股（重点功能）

盘中使用实时行情数据运行选股，在收盘前发现候选标的。

```bash
python run.py live [--loop] [--interval N] [--web] [--output DIR] [--force]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--loop` | 否 | 持续循环刷新（否则只运行一次） |
| `--interval` | 15 | 刷新间隔（分钟） |
| `--web` | 否 | 每次刷新后重新生成网站 |
| `--output` | site | 网站输出目录 |
| `--force` | 否 | 非交易时段也执行（测试用） |

**工作流程**：

```
1. 加载历史日线数据 + 预计算指标（只做一次）
2. 循环（每 N 分钟）：
   ├── 检查交易时间（9:15 - 15:05）
   ├── 获取实时行情
   │   ├── 优先：Tushare pro.rt_k()（实时）
   │   └── 备用：yfinance（~15 分钟延迟）
   ├── 构造「今日临时日线」→ 注入每只股票的 DataFrame
   ├── 重算全部技术指标
   ├── 运行选股 → 输出候选列表
   └── [可选] 重新生成网站
```

**典型使用方式**：

```bash
# 开盘后启动，每 15 分钟自动刷新 + 生成网站
nohup python run.py live --loop --interval 15 --web >> live.log 2>&1 &

# 午后密集观察，5 分钟间隔
python run.py live --loop --interval 5

# 手动触发一次看结果
python run.py live

# 非交易时段测试
python run.py live --force
```

**数据源 fallback**：

当 Tushare `rt_k` 接口不可用（token 未设、权限不足、API 异常）时，系统自动切换到 yfinance 批量下载。yfinance 有约 15 分钟延迟，但仍可在盘中提供有效参考。

### `agent` — 模拟交易

启动模拟交易 Agent 对历史数据进行仿真交易。

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
python run.py agent --days 60
python run.py agent --days 120 --screen CCI快进出
python run.py agent --days 60 --capital 500000
```

### `backtest` — 回测

对所有注册的筛选策略执行固定目标收益回测。

```bash
python run.py backtest [--days DATES] [--target PCT] [--plot]
```

| 参数 | 说明 |
|------|------|
| `--days` | 逗号分隔的日期列表（默认最近60天） |
| `--target` | 目标收益率（默认 0.05 即 5%） |
| `--plot` | 显示资金曲线和回撤图 |

### `web` — 生成静态网站

生成深色金融终端风格的 Dashboard 网站。

```bash
python run.py web [--output site] [--date DATE]
```

生成内容：
- **首页**：KPI 卡片（候选数、最佳胜率、累计收益、最大回撤、策略数、总交易数） + 今日选股 + 策略排行榜 + 30 日日历
- **每日详情**：sticky 表头、数值右对齐、涨跌箭头、信号诊断展开、动机徽章
- **策略页**：回测指标、收益曲线（ECharts 深色主题）、回撤图、交易明细、失败特征分析
- **个股页**：60 日 K 线（价格 + 成交量 + ADX + RSI + OBV 五面板）
- **历史记录**：每日各策略选股数量汇总

### `diagnose` — 特征诊断

分析指定策略筛选出的交易，对比涨停/非涨停组的特征分布差异。

```bash
python run.py diagnose [--screen NAME] [--days DATES] [--target PCT]
```

### `optimize` — 信号组合优化

穷举信号组合，寻找最优策略参数。

```bash
python run.py optimize [--focus SIGNALS] [--min-signals N] [--max-signals N]
```

### `plot` — K线图

绘制单只股票的多面板 K 线图。

```bash
python run.py plot 002906 [--tail 40]
```

### `scan` — 形态扫描

运行 AI 形态识别模式扫描。

```bash
python run.py scan [--date 2026-03-02]
```

---

## 部署到 GitHub Pages

```bash
# 一键部署（生成网站 + 推送到 gh-pages）
./deploy.sh
```

部署脚本会：
1. 运行 `python run.py web` 生成静态网站
2. 在临时目录创建 gh-pages 分支
3. 强制推送到远程 gh-pages 分支

**定时自动部署**（cron）：

```bash
# 工作日 15:15 自动部署（A股 15:00 收盘，数据约 15:10 就绪）
15 15 * * 1-5 cd /path/to/sz && ./deploy.sh >> deploy.log 2>&1
```

---

## 筛选策略

系统包含 4 个组合筛选策略：

| 策略名称 | 信号组合 | 思路 |
|----------|---------|------|
| **CCI快进出** | CCI穿越-100 + 底部逃逸 + CCI深度超卖 + 日涨<3% + EMA5贴近 | CCI 深度超卖反弹 |
| **OBV涨停梦** | OBV突破 + 量能堆积 + CCI>50 + 日涨>2%（仅主板） | T0 买入博 T+1 涨停 |
| **OBV波段** | OBV涨停梦 + ADX<40 + 底部逃逸 | 过滤趋势过强，做 2-8% 波段 |
| **形态识别** | AI 横盘缩量涨停延迟回收模式 | 从历史涨停事件学习的多日形态 |

**交易方针**：
- CCI快进出 / OBV涨停梦 / OBV波段：信号日收盘买入 → 次日高点达 5% 目标价卖出，否则次日收盘卖出
- 形态识别：信号确认后择机买入，持有至目标价或止损

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

### Tushare
- `TUSHARE_TOKEN`：通过环境变量 `TUSHARE_TOKEN` 设置
- `API_CALL_INTERVAL = 0.12`：API 调用间隔（秒），对应 500 次/分钟限额

### 下载参数
- `LOOKBACK_DAYS = 730`：初始化时回溯天数（约 2 年）
- `INCREMENTAL_DAYS = 60`：每日增量拉取天数

### 交易成本
- `COMMISSION_RATE = 0.0003`：佣金万三
- `SLIPPAGE_RATE = 0.001`：滑点千一
- `STAMP_TAX_RATE = 0.001`：印花税千一（卖出收取）
- `DEFAULT_TARGET_PCT = 0.05`：默认目标收益 5%

### 指标参数
- `RSI_SHORT_PERIOD = 6` / `RSI_LONG_PERIOD = 12`
- `EMA_PERIOD = 5`
- `ADX_PERIOD = 8`
- `CCI_PERIOD = 14`
- `OBV_CONSOLIDATION_WINDOW = 30`
- `UOS_PERIODS = (7, 14, 28)`

---

## 典型工作流

### 盘中实时选股（推荐）

```bash
# 1. 确保历史数据已更新
python run.py daily

# 2. 开盘后启动实时选股（每 15 分钟刷新 + 生成网站）
python run.py live --loop --interval 15 --web

# 3. 浏览器打开 site/index.html 查看实时候选
# 4. 午后 13:00-14:50 密集观察候选列表
# 5. 14:55 对候选股下单买入
```

### 盘后分析

```bash
# 1. 更新当日数据
python run.py daily

# 2. 查看筛选结果
python run.py screen

# 3. 生成网站 + 部署
python run.py web && ./deploy.sh

# 4. 查看个股 K 线
python run.py plot 002906
```

### 策略评估

```bash
# Agent 模拟交易
python run.py agent --days 120 --screen OBV涨停梦

# 特征诊断
python run.py diagnose --screen OBV涨停梦

# 信号组合优化
python run.py optimize --focus obv_breakout --max-signals 4
```

---

## 网站功能

网站采用深色金融终端风格（`data-theme="dark"`），使用 Pico CSS + 自定义深色配色：

| 颜色 | 用途 |
|------|------|
| `#0B1220` | 页面背景 |
| `#101A2E` | 卡片/表格背景 |
| `#4C8DFF` | 强调蓝（链接、品牌） |
| `#24D18A` | 盈利绿 |
| `#FF4D4F` | 亏损红 |
| `#9FB0C6` | 次要文字 |

字体：Inter（英文/数字）+ Noto Sans SC（中文），数字列使用 `tabular-nums` 等宽数字。

---

## 注意事项

1. **数据依赖**：首次使用必须先执行 `python run.py init` 下载数据，下载约需 20-30 分钟（~2700只股票）。
2. **Tushare 权限**：基础日线数据无需额外权限，实时行情 `rt_k` 需单独申请。
3. **yfinance 备用**：当 Tushare 不可用时自动切换 yfinance，有约 15 分钟延迟。
4. **仅限深交所**：本系统仅覆盖深交所股票（代码后缀 `.SZ`），不包含上交所。
5. **交易时间**：`live` 命令默认仅在 9:15-15:05 执行，非交易时段跳过（`--force` 可覆盖）。
6. **风险提示**：本系统仅供学习和研究用途，不构成投资建议。量化策略的历史回测表现不代表未来收益。
