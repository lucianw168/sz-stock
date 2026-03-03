# 变更日志 (CHANGELOG)

本文档记录从原始 notebook 版本迁移到模块化项目后的主要改动，以及后续优化迭代。

---

## v2.0 — 金融终端 UI + 实时选股（2026-03）

### 深色金融终端 UI

完全重写前端视觉系统，从 Pico CSS 浅色主题升级为深色金融终端风格。

**`web/static/style.css`** — 完全重写：
- 深色配色系统：`--bg-primary: #0B1220`、`--bg-card: #101A2E`、`--accent: #4C8DFF`、`--green: #24D18A`、`--red: #FF4D4F`
- 字体系统：Inter（英文/数字）+ Noto Sans SC（中文），数字列 `tabular-nums`
- 表格终端风：sticky 表头、数值右对齐、hover 淡蓝高亮、涨跌箭头（↑↓）
- KPI 卡片组件：大字号数值 + 小字标签 + 左边框颜色标识
- Pico CSS 深色变量覆盖

**`web/templates/base.html`**：
- `data-theme="dark"` 启用 Pico 深色模式
- Google Fonts 加载 Inter + Noto Sans SC
- 导航栏：左侧品牌 `SZ α`，右侧导航链接
- 页脚：`SZ 量化选股系统 v2.0 | 数据来源: 深交所 | 仅供研究参考`

**`web/templates/index.html`** — 三段式 Dashboard：
- 标题区：左 `每日选股`，右 `Data as of: {date}`
- KPI 卡片行（6 个）：今日候选数、最佳策略胜率、30日累计收益、最大回撤、活跃策略数、总回测交易
- 底部两栏：左栏（今日选股结果卡片） + 右栏（策略排行榜 + 30日日历）

**`web/templates/daily.html`**：
- sticky 表头、数值列 `.num` 右对齐
- 涨幅列加 ↑↓ 箭头（`.arrow-up` / `.arrow-down`）

**`web/templates/strategy.html`** / **`stock.html`** — ECharts 深色配色：
- 透明背景、`#101A2E` tooltip、`#1E2A3A` 网格线、`#9FB0C6` 轴标签
- 收益曲线：蓝色渐变填充；回撤：红色半透明

**`web/templates/history.html`**：
- sticky 表头 + 数值列右对齐

**`web/generator.py`**：
- `_build_index()` 新增 KPI 聚合计算：`total_candidates`、`best_strategy`、`total_bt_trades`
- `build()` 支持 `stock_data` 参数跳过磁盘加载（用于 live 模式）

### 实时盘中选股

新增 `python run.py live` 命令，支持盘中使用实时数据运行选股。

**`downloader.py`** — 新增实时数据层：
- `_fetch_rt_tushare()`：调用 `pro.rt_k(ts_code='0*.SZ,3*.SZ')` 获取全市场实时日K
- `_fetch_rt_yfinance(stock_codes)`：yfinance 备用方案，分批下载（每批 500 只）
- `fetch_realtime(stock_codes)`：自动 fallback（Tushare → yfinance）
- `inject_realtime(stock_data, realtime)`：将实时数据注入为今日临时行，自动推导 `pre_close`
- Volume 单位自动对齐：rt_k 和 yfinance 的股 → 手（÷100）

**`run.py`** — 新增 `live` 子命令：
- `--loop`：持续循环刷新
- `--interval N`：刷新间隔（分钟，默认 15）
- `--web`：每次刷新后重新生成网站
- `--deploy`：每次刷新后自动部署到 gh-pages（需配合 `--web`）
- `--force`：非交易时段也执行
- 历史数据只加载一次，每次循环 deepcopy + 注入实时数据 + 重算指标
- 新增 `_deploy_to_ghpages()` 辅助函数：临时目录 → git init → force push gh-pages

**`deploy.sh`** — 修复数据更新流程：
- 部署前先运行 `python run.py daily` 从 Tushare 增量下载最新日线数据
- 修复了之前只运行 `web` 导致网站数据不更新的问题

**`web/generator.py`** — 修复历史页面：
- `_determine_dates(date=)` 传入特定日期时，不再只生成单日页面
- 改为包含最近 30 个历史交易日 + 当日，确保历史记录完整

---

## v1.1 — Tushare 迁移 + 形态识别

### 数据源迁移：yfinance → Tushare

**`downloader.py`** — 完全重写：
- 替换 yfinance 为 Tushare Pro API（`pro.daily()`、`pro.stock_basic()`、`pro.trade_cal()`、`pro.limit_list_d()`）
- Parquet 本地缓存 + 增量更新
- 前复权处理（通过 pre_close 检测除权日，计算调整因子）
- 限速控制（0.12s/call = 500次/分钟）
- 支持从 `limit_up` 项目导入缓存数据

**`config.py`**：
- 新增 `TUSHARE_TOKEN`（环境变量）
- `LOOKBACK_DAYS` 从 365 调整为 730（约 2 年）
- `DEFAULT_TARGET_PCT` 从 0.03 调整为 0.05（5%）

### 形态识别策略

新增 AI 驱动的形态识别策略（`pattern_screen.py`）：

- 从历史涨停事件挖掘多日 K 线形态模板
- 当前发现模式：横盘缩量涨停延迟回收（横盘→缩量涨停→阴线→第3天回收确认）
- 形态模板持久化到 `data/models/pattern_library.pkl`
- 新增 `run.py scan` 命令

### 策略精简

从 8 个策略精简为 4 个：

| 保留 | 移除（理由） |
|------|-------------|
| CCI快进出 | 超卖（与 CCI快进出 重复） |
| OBV涨停梦 | 当日金叉买入（表现不佳） |
| OBV波段 | 近日上涨（表现不佳） |
| 形态识别（新增） | 底部异动（被 OBV 策略覆盖） |
| | OBV底部突破（被 OBV波段 覆盖） |

### 网站生成器

**`web/`** — 新增静态网站生成系统：
- Jinja2 模板 + Pico CSS 框架
- ECharts 5 交互式图表（K 线、收益曲线、回撤图）
- 信号诊断面板（每个信号的实际值 vs 阈值）
- 动机分析徽章（出货/试盘/吸筹/待确认）
- 失败特征分析（上影线、次日低开、放量回落、收阴）
- `deploy.sh` 一键部署到 GitHub Pages

---

## v1.0 — 原始版本 → 模块化重构

原始系统是一个 Jupyter notebook (`sz_codes.ipynb`)，包含数据下载、指标计算、信号筛选、回测的全部逻辑。重构后拆分为以下独立模块：

| 模块 | 职责 |
|------|------|
| `config.py` | 全局配置常量 |
| `universe.py` | 股票池定义（深交所代码段、板块判断） |
| `downloader.py` | 数据下载与本地缓存 |
| `indicators.py` | 技术指标计算（RSI、CRSI、ADX、CCI、OBV 等） |
| `labels.py` | 涨停标签构建（用于 ML） |
| `strategies.py` | 信号函数定义（统一接口 `signal_xxx(df) -> Series[bool]`） |
| `screener.py` | 组合筛选策略（多信号 AND 组合） |
| `backtest.py` | 回测引擎 |
| `agent.py` | 模拟交易 Agent（三段式卖出、涨停续板学习） |
| `diagnose.py` | 特征诊断（涨停 vs 非涨停对比分析） |
| `optimizer.py` | 信号组合优化器 |
| `visualize.py` | K线图、资金曲线、回撤图 |
| `pipeline.py` | 数据流水线（初始化 / 每日更新） |
| `run.py` | CLI 入口 |

---

## 关键优化记录

### 1. OBV涨停梦信号优化

**问题**：原始 OBV涨停梦只有 OBV 突破 + 量能堆积 2 个信号，筛选出的股票太多，涨停命中率极低（549 笔交易仅 16 笔涨停，2.9%）。

**改动**：通过 `diagnose.py` 对涨停/非涨停交易的特征做统计对比（Cohen's d 排序），新增 2 个过滤条件：

| 过滤 | 依据 | 效果 |
|------|------|------|
| CCI > 50 | 动量确认，与OBV/量能正交（d=1.41） | 保留87.5%涨停 |
| 日涨幅 > 2% | 价格验证OBV突破（涨停组均值4.5% vs 非涨停1.3%） | 进一步筛除弱信号 |

**涉及文件**：
- `config.py`：新增 `CCI_MOMENTUM_FLOOR = 50`、`DAILY_GAIN_GT2_PCT = 0.02`
- `strategies.py`：新增 `signal_cci_momentum_floor`、`signal_daily_gain_gt2`
- `screener.py`：`screen_obv_momentum` 新增两个信号

### 2. 交易真实性约束

#### 2a. 涨停股无法买入

当日收盘价达到涨停价的股票处于封板状态，实际无法买入。

**改动** (`screener.py` / `agent.py`)：涨停股自动排除。

**影响**：某些激进策略的"利润"从 +14% 变为 -8%，证明其收益完全来自买入涨停股的虚假交易。

#### 2b. 印花税

A 股卖出时收取千分之一印花税。

**改动**：`config.py` 新增 `STAMP_TAX_RATE = 0.001`。

### 3. ADX 过滤器（基于亏损分析）

**发现**：ADX ≥ 40 的交易 67% 大亏损率（趋势过强 = 末期反转风险高）。

**改动**：新增 `ADX_BUY_MAX = 40`，独立为 OBV波段 策略。

**效果**（60天回测）：

| 指标 | 无ADX过滤 | 有ADX过滤 |
|------|-----------|-----------|
| 交易数 | 42 | 21 |
| 胜率 | 47.6% | 66.7% |
| 总收益 | +0.49% | +1.14% |
| 最大回撤 | -1.77% | -1.14% |

---

## 当前交易成本模型

| 费用项 | 费率 | 收取时机 |
|--------|------|----------|
| 佣金 | 万三 (0.03%) | 买入和卖出 |
| 滑点 | 千一 (0.1%) | 买入和卖出 |
| 印花税 | 千一 (0.1%) | 仅卖出 |

## 核心教训

1. **不能买涨停股**：这是 A 股最基本的约束。任何忽略此规则的回测都会高估收益。
2. **预测赢家不如过滤输家**：涨停预测样本太少容易过拟合；反向分析大亏损交易的特征更可靠。
3. **简单规则胜过复杂模型**：单一 ADX<40 过滤器的效果优于 6 维指标分桶评分系统。
4. **多时间窗口验证**：单一回测窗口容易误导，应在不同周期确认策略稳定性。
