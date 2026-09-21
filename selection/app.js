/* Public research display only. Never fabricates new signals or calls a broker. */
(function () {
  'use strict';
  const app = document.getElementById('cn-selection');
  if (!app) return;
  const data = window.CN_SELECTION_DATA;
  if (!data || !window.CN_SELECTION_VIEW || data.schema_version !== 1 || data.cards.length !== 5) {
    app.textContent = '研究记录暂不可用，请稍后刷新。没有发布新的交易信号。';
    app.setAttribute('role', 'alert');
    return;
  }
  const root = app.dataset.siteRoot || '../';
  const asset = app.dataset.assetRoot || (root + 'selection/');
  const esc = x => String(x == null ? '' : x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = (x, n=2) => x == null || !Number.isFinite(Number(x)) ? '—' : Number(x).toFixed(n);
  const pct = x => num(x) + (x == null ? '' : '%');
  const signed = x => (x > 0 ? '+' : '') + pct(x);
  const tone = x => x == null ? 'cn-muted' : x > 0 ? 'cn-good' : x < 0 ? 'cn-bad' : 'cn-muted';
  const icon = name => `<i data-lucide="${name}" aria-hidden="true"></i>`;
  const pf = m => m.no_losses ? '无亏损样本' : num(m.profit_factor);
  const cardMap = Object.fromEntries(data.cards.map(c => [c.key, c]));
  const view = window.CN_SELECTION_VIEW;
  const stockLink = (r,product,back) => root+r.market+'/stock/'+encodeURIComponent(r.ts_code.slice(0,6))+'.html?'+new URLSearchParams({signal:r.date,product:product||data.cards.find(c=>c.candidates.includes(r))?.key||active,return:back||location.hash}).toString();
  const evidenceButton = r => window.CN_CONTEXT_DATA?.stocks?.[r.ts_code] ? `<button class="cn-icon cn-evidence-open" data-evidence="${esc(r.ts_code)}" title="${esc(r.name||r.ts_code)}公司与市场证据" aria-label="${esc(r.ts_code)}公司与市场证据">${icon('notebook-text')}</button>` : '';
  let currentView = view.route(location.hash, Object.keys(cardMap));
  let active = cardMap[currentView] ? currentView : 'second';
  let monitor = currentView === 'monitor';
  let market = app.dataset.market || 'all';
  let resizeTimer;
  const periods = [['2y','2年'],['1y','1年'],['6m','半年'],['3m','三个月']];

  function status(c) {
    if(c.daily_run) {
      const r=c.daily_run;
      const label=r.state==='blocked'?'数据待补齐':r.date!==data.expected_session?'等待最新行情':r.timely?'日更正常':'补生成记录';
      return `<span class="cn-status ${r.state==='ready'&&r.timely&&r.date===data.expected_session?'cn-good':'cn-warn'}" title="每日筛选状态；历史回测截至日另列">${label}</span>`;
    }
    return `<span class="cn-status cn-${esc(c.health.tone)}">${esc(c.health.status)}</span>`;
  }
  function kpis(m) {
    return `<div class="cn-kpis">${[
      ['账户累计收益',signed(m.return_pct),tone(m.return_pct)],
      ['净盈利胜率',pct(m.win_rate),''],['PF',pf(m),''],
      ['账户最大回撤',pct(m.drawdown_pct),'cn-bad'],['已结算交易',String(m.trades),'']
    ].map(([label,value,cls]) => `<div class="cn-kpi"><label>${label}</label><strong class="${cls}">${value}</strong></div>`).join('')}</div>`;
  }
  function periodRows(c) {
    return periods.map(([key,label]) => {
      const m=c.windows[key];
      if (!m) return `<tr><td>${label}</td><td colspan="5">无覆盖数据</td></tr>`;
      return `<tr><td>${label}${!m.coverage_complete?'<small class="cn-warn">历史不足，以下为已覆盖期</small>':''}</td><td>${esc(m.start)}<small>至 ${esc(m.end)} · ${m.sessions} 日</small></td><td class="cn-num ${tone(m.return_pct)}">${signed(m.return_pct)}</td><td class="cn-num">${pct(m.win_rate)}<small>${m.trades} 笔退出</small></td><td class="cn-num">${pf(m)}</td><td class="cn-num cn-bad">${pct(m.drawdown_pct)}</td></tr>`;
    }).join('');
  }
  function renderShell() {
    const overview = currentView === 'overview';
    const records=currentView==='history', product=Boolean(cardMap[currentView]);
    app.innerHTML = `${!overview?`<a class="cn-back" href="#overview" id="cn-back">${icon('arrow-left')}全部产品线</a>`:''}<div class="cn-page-head"><div><h1>${overview?'每日选股':records?'选股历史':monitor?'运行监测':'选股研究'}</h1><p class="cn-muted">${esc(data.expected_session)} 收盘口径 · 研究观察</p></div><div class="cn-header-actions"><button id="cn-monitor" aria-pressed="${monitor}">运行监测</button><button class="cn-icon" id="cn-refresh" title="刷新发布记录" aria-label="刷新发布记录">${icon('refresh-cw')}</button></div></div>
      <nav class="cn-workspace-nav" aria-label="研究工作台"><a href="#overview" data-nav="overview" ${overview?'aria-current="page"':''}>${icon('layout-grid')}今日选股</a><a href="#history" data-nav="history" ${records?'aria-current="page"':''}>${icon('calendar-days')}选股历史</a><a href="#monitor" data-nav="monitor" ${monitor?'aria-current="page"':''}>${icon('activity')}运行监测</a></nav>
      ${product?`<div class="cn-product-tabs" role="tablist" aria-label="选股产品线">${data.cards.map(c=>`<button role="tab" id="cn-tab-${c.key}" data-product="${c.key}" aria-controls="cn-panel" aria-selected="${c.key===active}" tabindex="${c.key===active?0:-1}" style="--cn-line:${c.color}">${esc(c.name)}</button>`).join('')}</div>`:''}
      <div id="cn-panel" ${product?`role="tabpanel" aria-labelledby="cn-tab-${active}"`:''}></div>
      <footer class="cn-footer"><p class="cn-muted">研究参考，非投资建议。历史模拟不保证未来收益，页面不发送交易订单。</p><small>页面生成 ${esc(data.built_at.replace('T',' ').slice(0,19))}（北京时间） · 各版本数据截至日单列 · <a href="${esc(asset)}data.json">研究数据</a> · <a href="${esc(asset)}build.json">发布版本</a> · <a href="${esc(root)}methodology.html">统计口径</a></small></footer>`;
    document.getElementById('cn-monitor').onclick=()=>navigate('monitor');
    app.querySelectorAll('[data-nav]').forEach(a=>a.onclick=e=>{e.preventDefault();navigate(a.dataset.nav);});
    const back=document.getElementById('cn-back');
    if(back) back.onclick=e=>{e.preventDefault();navigate('overview');};
    document.getElementById('cn-refresh').onclick=()=>location.reload();
    app.querySelectorAll('[data-product]').forEach(button=>{
      button.onclick=()=>navigate(button.dataset.product);
      button.onkeydown=e=>{
        const keys=data.cards.map(c=>c.key), i=keys.indexOf(button.dataset.product);
        let target;
        if(e.key==='ArrowRight') target=keys[(i+1)%keys.length];
        if(e.key==='ArrowLeft') target=keys[(i+keys.length-1)%keys.length];
        if(e.key==='Home') target=keys[0];
        if(e.key==='End') target=keys[keys.length-1];
        if(target){e.preventDefault();navigate(target);document.getElementById('cn-tab-'+target).focus();}
      };
    });
    if(overview) renderOverview(); else if(monitor) renderMonitor(); else if(records)renderRecords();else renderProduct();
    if(window.lucide) window.lucide.createIcons();
  }
  function navigate(key, writeHistory=true) {
    currentView=view.route('#'+key,Object.keys(cardMap));
    monitor=currentView==='monitor';
    if(cardMap[currentView]) active=currentView;
    if(writeHistory && location.hash!=='#'+key) history.pushState(null,'','#'+key);
    renderShell();
  }
  function renderOverview() {
    const panel=document.getElementById('cn-panel');
    panel.innerHTML=`<div class="cn-overview-tools"><label>市场 <select id="cn-overview-market"><option value="all">全市场</option><option value="sz">深圳主板</option><option value="sh">上海主板</option><option value="cy">创业板</option><option value="kc">科创板</option></select></label><span class="cn-muted">回测每只50%仓位 · 历史与当日名单分开标注</span></div>
      <div class="cn-card-grid">${data.cards.map(c=>{
        const s=view.snapshot(c,market,data.expected_session);
        return `<article class="cn-product-card" style="--cn-line:${c.color}" data-card="${c.key}">
          <header><h2><a href="#${c.key}" data-detail="${c.key}">${esc(c.name)}</a></h2><a class="cn-card-arrow" href="#${c.key}" data-detail="${c.key}" aria-label="${esc(c.name)}详情" title="${esc(c.name)}详情">${icon('arrow-up-right')}</a></header>
          <p class="cn-card-date">${s.archived?'历史归档':s.state==='blocked'?'待数据更新':s.timely?'盘后观察':'补生成记录'} · ${esc(s.date||'暂无记录')}${s.stale?' <span class="cn-warn">非今日名单</span>':''}</p>
          <div class="cn-stock-list">${s.rows.length?s.rows.slice(0,8).map(r=>`<span class="cn-stock-entry"><a class="cn-stock" href="${esc(stockLink(r))}"><strong>${esc(r.ts_code.slice(0,6))}</strong><span>${esc(r.name||r.ts_code.slice(-2))}</span></a>${evidenceButton(r)}</span>`).join(''):`<p class="cn-muted">${esc(s.state==='blocked'?'数据未齐，本次暂停选股':s.state==='ready'?'本次此市场无精选股票':'此市场暂无该版本记录')}</p>`}</div>
          ${s.rows.length>8?`<small>另有 ${s.rows.length-8} 只，详情中查看</small>`:''}
          <footer><span class="cn-muted">${c.key==='manual'?`开发期胜率 ${pct(c.full.win_rate)} · ${c.full.trades} 笔`:`历史胜率 ${pct(c.full.win_rate)} · PF ${pf(c.full)}`}</span><a href="#${c.key}" data-detail="${c.key}">逻辑与回测</a></footer>
        </article>`;
      }).join('')}</div>
      <section class="cn-section"><div class="cn-chart-head"><h2>近期选股记录</h2><a href="#history" data-records>全部日期 ${icon('arrow-right')}</a></div>${recentRecords()}</section>
      <section class="cn-section"><div class="cn-chart-head"><h2>运行监测</h2><a href="#monitor" id="cn-overview-monitor">全部窗口 ${icon('arrow-right')}</a></div><div class="cn-table-wrap"><table><thead><tr><th>产品线</th><th>数据状态</th><th class="cn-num">近三个月收益</th><th class="cn-num">盈利胜率</th><th class="cn-num">退出笔数</th></tr></thead><tbody>${data.cards.map(c=>{const m=c.windows['3m'];return `<tr><td><a href="#${c.key}" data-detail="${c.key}">${esc(c.name)}</a>${c.key==='manual'?'<small>精选开发测试</small>':''}</td><td>${status(c)}<small>账户至 ${esc(c.account_asof)}</small></td><td class="cn-num ${tone(m?.return_pct)}">${signed(m?.return_pct)}</td><td class="cn-num">${pct(m?.win_rate)}</td><td class="cn-num">${m?.trades??'无记录'}</td></tr>`;}).join('')}</tbody></table></div></section>`;
    document.getElementById('cn-overview-market').value=market;
    document.getElementById('cn-overview-market').onchange=e=>{market=e.target.value;renderOverview();if(window.lucide)window.lucide.createIcons();};
    const open=key=>{navigate(key);window.scrollTo({top:0});};
    panel.querySelectorAll('[data-detail]').forEach(a=>a.onclick=e=>{e.preventDefault();open(a.dataset.detail);});
    panel.querySelectorAll('[data-card]').forEach(card=>card.onclick=e=>{if(!e.target.closest('a,button'))open(card.dataset.card);});
    document.getElementById('cn-overview-monitor').onclick=e=>{e.preventDefault();navigate('monitor');};
    panel.querySelectorAll('[data-records]').forEach(a=>a.onclick=e=>{e.preventDefault();navigate(a.hash.slice(1));});
  }
  function recentRecords(){
    const page=view.historyPage(data.cards,data.history_sessions||[],{expected:data.expected_session,market,range:20,pageSize:5});
    return `<div class="cn-table-wrap"><table class="cn-record-matrix"><thead><tr><th>信号日</th>${data.cards.map(c=>`<th class="cn-num">${esc(c.name)}</th>`).join('')}</tr></thead><tbody>${page.days.map(day=>`<tr><td><a href="#history?date=${day.date}&market=${market}&range=20" data-records>${day.date}</a></td>${day.cells.map(cell=>`<td class="cn-num">${cell.state==='unknown'?'<span class="cn-muted" title="该日没有留存记录">—</span>':`<a href="#history?date=${day.date}&focus=${cell.key}&market=${market}&range=20" data-records>${cell.state==='blocked'?'待数据':cell.count+' 只'}</a>`}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function mountRecords(product){
    if(!window.CN_SELECTION_HISTORY){document.getElementById('cn-records').textContent='历史记录暂不可用，请刷新页面。';return;}
    window.CN_SELECTION_HISTORY.mount(document.getElementById('cn-records'),{cards:data.cards,
      sessions:data.history_sessions||[],expected:data.expected_session,market,product,
      defaultDate:product?view.snapshot(cardMap[product],market,data.expected_session).date:'',
      stockLink,evidenceButton,root,onMarket:value=>{market=value;}});
  }
  function renderRecords(){
    document.getElementById('cn-panel').innerHTML=`<div class="cn-chart-head"><h2>每日产品线记录</h2><details class="cn-legacy-links"><summary>旧市场档案</summary><nav aria-label="旧市场策略历史">${[['sz','深圳主板'],['sh','上海主板'],['cy','创业板'],['kc','科创板']].map(([m,n])=>`<a href="${esc(root+m+'/history.html')}">${n}</a>`).join('')}</nav></details></div><div id="cn-records"></div>`;
    mountRecords();
  }
  function renderProduct() {
    const c=cardMap[active], panel=document.getElementById('cn-panel');
    panel.innerHTML=`<div class="cn-title-line"><h2>${esc(c.name)}</h2>${status(c)}</div><p class="cn-muted cn-model">${esc(c.model)}<small>${esc(c.version)}</small></p>
      <p>${esc(c.thesis)}</p>
      <div class="cn-notice">${esc(c.signal_message)}${c.key==='manual'?`<br>${esc(c.reference_label)}`:''}</div>
      ${c.daily_run?`<div class="cn-toolbar"><span>最近筛选 ${esc(c.daily_run.date)}</span><span>候选 ${c.daily_run.pool_rows??'未完成'} 只 · 精选 ${c.daily_run.selected_rows} 只</span><span class="cn-muted">记录于 ${esc(c.daily_run.captured_at.replace('T',' ').slice(0,19))}</span></div>`:''}
      <nav class="cn-section-links" aria-label="产品详情章节"><a href="#${active}" data-section="cn-records">${icon('list')}观察记录</a><a href="#${active}" data-section="cn-performance">${icon('chart-no-axes-combined')}回测表现</a><a href="#${active}" data-section="cn-logic">${icon('book-open')}逻辑与执行</a></nav>
      <section class="cn-section"><div class="cn-chart-head"><h2>观察名单与历史记录</h2><a href="#history?product=${active}" data-all-history>五线历史总览 ${icon('arrow-right')}</a></div><div id="cn-records"></div></section>
      <section id="cn-performance" class="cn-section">
      <div class="cn-chart-head"><h2>${c.key==='manual'?'精选确认 · 开发期模拟':'历史模拟表现'}</h2><span class="cn-muted">信号截至 ${esc(c.signal_asof)} · 账户截至 ${esc(c.account_asof)}</span></div>
      ${kpis(c.full)}<small>每只股票 ${num(c.allocation_pct,0)}% 仓位 · ${c.key==='bottom'?'同批最多两只，不加杠杆':'同一时间一批'} · 历史模拟，非实盘收益</small>
      <canvas id="cn-equity" class="cn-chart" role="img" aria-label="${esc(c.name)}历史模拟账户净值曲线"></canvas>
      <p class="cn-muted">${esc(c.reference_note)}</p>
      ${c.version_comparison?releaseComparison(c):''}
      ${c.upgrade_study?manualComparison(c):''}
      <section class="cn-section"><div class="cn-chart-head"><h2>滚动表现</h2><button id="cn-open-monitor">查看五线监测</button></div><div class="cn-table-wrap"><table><thead><tr><th>窗口</th><th>实际覆盖</th><th class="cn-num">账户收益</th><th class="cn-num">盈利胜率</th><th class="cn-num">PF</th><th class="cn-num">最大回撤</th></tr></thead><tbody>${periodRows(c)}</tbody></table></div><p class="cn-muted">截至各账户最近数据日回看，不以今天冒充最新回测。胜率按窗口内退出计，收益包含跨窗持仓的净值变化。</p></section>
      </section><section id="cn-logic" class="cn-section cn-financial-grid"><div><h2>金融逻辑与筛选过程</h2><ol>${c.process.map(x=>`<li>${esc(x)}</li>`).join('')}</ol></div><div><h2>操作思路与执行边界</h2><ol>${c.operation.map(x=>`<li>${esc(x)}</li>`).join('')}</ol></div></section><p class="cn-notice">${esc(c.limitation)}</p>
      ${c.key==='manual'?rulesSection(c):''}<details><summary>统计定义与风险说明</summary>${Object.entries(data.methodology).map(([,v])=>`<p class="cn-muted">${esc(v)}</p>`).join('')}</details>`;
    document.getElementById('cn-open-monitor').onclick=()=>navigate('monitor');
    panel.querySelector('[data-all-history]').onclick=e=>{e.preventDefault();navigate('history?product='+active);};
    panel.querySelectorAll('[data-section]').forEach(a=>a.onclick=e=>{e.preventDefault();document.getElementById(a.dataset.section).scrollIntoView({block:'start'});});
    mountRecords(active);
    requestAnimationFrame(()=>drawChart(c));
  }
  function releaseComparison(c) {
    const study=c.version_comparison;
    return `<section class="cn-section"><h2>本次升级对比</h2><p class="cn-muted">同一历史区间、相同候选资格与交易规则，每次买入50%仓位；仅升级联合排序输入。以下是开发期对照，不是独立留出测试。</p><div class="cn-table-wrap"><table class="cn-release-table"><thead><tr><th>版本</th><th class="cn-num">已结算</th><th class="cn-num">盈利胜率</th><th class="cn-num">PF</th><th class="cn-num">账户收益</th><th class="cn-num">最大回撤</th></tr></thead><tbody>${[['升级前',study.previous],['趋势结构与板块联动',study.current]].map(([label,m])=>`<tr><td>${label}</td><td class="cn-num">${m.trades}</td><td class="cn-num">${pct(m.win_rate)}</td><td class="cn-num">${pf(m)}</td><td class="cn-num">${signed(m.return_pct)}</td><td class="cn-num">${pct(m.drawdown_pct)}</td></tr>`).join('')}</tbody></table></div><p class="cn-muted">${study.planned_days} 次盘后观察，${study.confirmed_plans} 次通过开盘确认。忽略资金占用的全部确认机会胜率为 ${pct(study.confirmed_win_rate)}，不能与实际入账胜率混用。</p></section>`;
  }
  function manualComparison(c) {
    const rows=c.upgrade_study.rows.filter(r=>r.period==='full');
    return `<section class="cn-section"><h2>精选确认方案对比</h2><p class="cn-muted">2024-08-01 至 2026-04-29 的信号，账户结算至 2026-05-11；两种方案均按每次买入50%仓位、相同交易费用逐笔重算。以下为开发期比较，不是独立留出集或实盘。</p><div class="cn-table-wrap"><table><thead><tr><th>方案</th><th class="cn-num">笔数</th><th class="cn-num">盈利胜率</th><th class="cn-num">PF</th><th class="cn-num">账户收益</th><th class="cn-num">最大回撤</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.arm==='consensus'?'规则共识参照':'共识 + 承接确认'}</td><td class="cn-num">${r.trades}</td><td class="cn-num">${pct(r.win_pct)}</td><td class="cn-num">${num(r.pf)}</td><td class="cn-num">${signed(r.return_pct)}</td><td class="cn-num">${pct(r.dd_pct)}</td></tr>`).join('')}</tbody></table></div><p class="cn-muted">确认方案的精度和回撤有所改善，但交易减少，累计收益也降低。当前规则代码的前向验证仍在建立，不能将29笔开发样本外推为稳定收益。</p></section>`;
  }
  function rulesSection(c) {
    return `<section class="cn-section"><h2>规则来源与独立表现</h2><p class="cn-muted">单一来源不再单独发出精选信号；它们作为不同机制的组合证据，接受统一承接确认。以下为各来源独立运行的历史审计，不是当前组合绩效。</p><div class="cn-table-wrap"><table><thead><tr><th>市场 / 规则</th><th>机制</th><th>处理</th><th class="cn-num">样本</th><th class="cn-num">胜率</th><th class="cn-num">PF / 压力PF</th></tr></thead><tbody>${c.rules.map(r=>`<tr><td><a href="${esc(root+r.market+'/strategy/'+encodeURIComponent(r.strategy)+'.html')}">${esc(r.source)}</a></td><td>${esc(r.family)}</td><td>${esc(r.stage)}</td><td class="cn-num">${r.trades}</td><td class="cn-num">${pct(r.win_rate)}</td><td class="cn-num">${num(r.profit_factor)} / ${num(r.stress_pf)}</td></tr>`).join('')}</tbody></table></div></section>`;
  }
  function renderMonitor() {
    const panel=document.getElementById('cn-panel');
    panel.removeAttribute('aria-labelledby');
    panel.setAttribute('aria-label','五条产品线运行监测');
    panel.innerHTML=`<h2>运行监测</h2><p class="cn-muted">每日筛选状态与历史账户表现分别展示</p><div class="cn-notice">数据中断、样本不足与策略承压分开处理。目前没有任何一条线被认证为可直接实盘。每日筛选已有独立记录，但历史模拟账户尚未接续为前向账户；以下窗口指标以各账户截止日计算，不冒充当前实时收益。</div><div class="cn-table-wrap"><table class="cn-health-table"><thead><tr><th>产品线 / 版本</th><th>监测状态</th>${periods.map(([,label])=>`<th class="cn-num">${label}<small>收益 / 胜率</small></th>`).join('')}</tr></thead><tbody>${data.cards.map(c=>`<tr><td><a href="#${c.key}" data-open="${c.key}">${esc(c.name)}</a><small>账户至 ${esc(c.account_asof)}</small>${c.key==='manual'?'<small>精选开发测试</small>':''}</td><td>${status(c)}<small>截止当时：${esc(c.health.performance)}</small><small>历史回测信号距今 ${c.health.lag_sessions} 个交易日</small></td>${periods.map(([key])=>{const m=c.windows[key];return m?`<td class="cn-num"><strong class="${tone(m.return_pct)}">${signed(m.return_pct)}</strong><br>${pct(m.win_rate)}<small>${m.trades} 笔${m.trades<20?' · 小样本':''}</small>${!m.coverage_complete?'<small class="cn-warn">窗口未满</small>':''}</td>`:'<td>无记录</td>';}).join('')}</tr>`).join('')}</tbody></table></div>
      <section class="cn-section"><h2>近三个月窗口的历史变化</h2><div class="cn-toolbar"><label>产品线 <select id="cn-monitor-product">${data.cards.map(c=>`<option value="${c.key}" ${active===c.key?'selected':''}>${esc(c.name)}</option>`).join('')}</select></label></div><p class="cn-muted">沿同一连续账户，每5个交易日观察一次当时近三个月表现；这是历史回看，不是前向影子盘。</p><div class="cn-table-wrap"><table class="cn-history"><thead><tr><th>统计截至</th><th>实际窗口起点</th><th class="cn-num">收益</th><th class="cn-num">胜率</th><th class="cn-num">退出数</th><th class="cn-num">PF</th><th class="cn-num">回撤</th></tr></thead><tbody id="cn-health-history"></tbody></table></div></section><details open><summary>怎样解读状态</summary>${['status','windows','returns','wins','forward'].map(k=>`<p class="cn-muted">${esc(data.methodology[k])}</p>`).join('')}</details>`;
    panel.querySelectorAll('[data-open]').forEach(a=>a.onclick=e=>{e.preventDefault();navigate(a.dataset.open);});
    document.getElementById('cn-monitor-product').onchange=e=>{active=e.target.value;renderHistory();};
    renderHistory();
  }
  function renderHistory() {
    document.getElementById('cn-health-history').innerHTML=[...cardMap[active].history].reverse().map(m=>`<tr><td>${esc(m.date)}</td><td>${esc(m.start)}</td><td class="cn-num ${tone(m.return_pct)}">${signed(m.return_pct)}</td><td class="cn-num">${pct(m.win_rate)}</td><td class="cn-num">${m.trades}</td><td class="cn-num">${pf(m)}</td><td class="cn-num">${pct(m.drawdown_pct)}</td></tr>`).join('');
  }
  function drawChart(c) {
    const canvas=document.getElementById('cn-equity');
    if(!canvas) return;
    const box=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    if(box.width<=0) return;
    canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const w=box.width,h=box.height,left=52,right=w-10,top=20,bottom=h-30;
    const values=c.equity.map(p=>p.equity), lo=Math.min(1,...values),hi=Math.max(1,...values), pad=Math.max((hi-lo)*.12,.01);
    const y=v=>bottom-(v-lo+pad)/(hi-lo+pad*2)*(bottom-top),x=i=>left+i/Math.max(1,values.length-1)*(right-left);
    ctx.font='12px system-ui';ctx.lineWidth=1;ctx.fillStyle='#9FB0C6';
    for(let i=0;i<4;i++){const v=lo-pad+(hi-lo+2*pad)*i/3, yy=y(v);ctx.strokeStyle='#253346';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(right,yy);ctx.stroke();ctx.fillText(v.toFixed(2),0,yy+4);}
    ctx.setLineDash([4,4]);ctx.strokeStyle='#63728A';ctx.beginPath();ctx.moveTo(left,y(1));ctx.lineTo(right,y(1));ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=c.color;ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.stroke();
    ctx.fillText(c.equity[0].date,left,h-6);ctx.textAlign='right';ctx.fillText(c.equity.at(-1).date,right,h-6);
  }
  window.addEventListener('hashchange',()=>navigate(location.hash.slice(1),false));
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(!monitor)drawChart(cardMap[active]);},100);});
  renderShell();
})();
