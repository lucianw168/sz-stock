/* Date-first observation journal. No inference, pricing, or broker calls. */
(function(scope){
  'use strict';
  const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icon=name=>`<i data-lucide="${name}" aria-hidden="true"></i>`;
  const markets=[['all','全市场'],['sz','深圳主板'],['sh','上海主板'],['cy','创业板'],['kc','科创板']];
  const validDate=x=>/^\d{4}-\d{2}-\d{2}$/.test(x||'');
  function mount(host,config){
    const view=scope.CN_SELECTION_VIEW;
    const params=new URLSearchParams(location.hash.split('?')[1]||'');
    const key=config.product||'all';
    const defaults=config.defaultDate||'';
    const state={product:key==='all'?(params.get('product')||'all'):key,
      market:params.get('market')||config.market||'all',range:params.get('range')||'20',
      kind:params.get('kind')||'all',query:params.get('q')||'',settled:params.get('settled')==='1',
      end:params.get('end')||config.expected,page:Number(params.get('page'))||0,
      date:params.get('date')||defaults,focus:params.get('focus')||'all'};
    if(!config.cards.some(c=>c.key===state.product))state.product=key;
    if(!config.cards.some(c=>c.key===state.focus))state.focus='all';
    if(!markets.some(([m])=>m===state.market))state.market='all';
    if(!['20','60','120','all'].includes(state.range))state.range='20';
    if(!['all','daily','historical'].includes(state.kind))state.kind='all';
    if(!validDate(state.end)||state.end>config.expected)state.end=config.expected;
    if(!validDate(state.date)||state.date>config.expected)state.date='';
    // A deep-linked signal must remain reachable even outside the default range.
    if(params.has('date')&&!params.has('range'))state.range='all';
    let result;
    function options(){return {...state,expected:config.expected,pageSize:10};}
    function hash(){
      const p=new URLSearchParams();
      if(key==='all'&&state.product!=='all')p.set('product',state.product);
      if(state.market!=='all')p.set('market',state.market);
      p.set('range',state.range);
      if(state.kind!=='all')p.set('kind',state.kind);
      if(state.query)p.set('q',state.query);
      if(state.settled)p.set('settled','1');
      if(state.end!==config.expected)p.set('end',state.end);
      if(state.page)p.set('page',state.page);
      if(state.date)p.set('date',state.date);
      if(state.focus!=='all')p.set('focus',state.focus);
      return '#'+(key==='all'?'history':key)+'?'+p.toString();
    }
    function commit(replace=false){
      history[replace?'replaceState':'pushState'](null,'',hash());
      if(config.onMarket)config.onMarket(state.market);
    }
    function label(cell){
      if(cell.state==='blocked')return '待数据';
      if(cell.state==='unknown')return '—';
      return String(cell.count);
    }
    function provenance(cell){
      if(cell.run)return cell.run.timely?'盘后留存':'补生成';
      return cell.historical?'历史回测':'观察留存';
    }
    function stockRows(day){
      return day.cells.filter(c=>state.focus==='all'||state.focus===c.key).map(cell=>{
        const card=config.cards.find(c=>c.key===cell.key);
        if(!cell.rows.length){
          let text=cell.state==='blocked'?cell.run.message:cell.state==='unknown'?'该日没有可核验记录':
            cell.total?'当前筛选无匹配记录':'已完成筛选，未产生精选名单';
          return `<tr><td>${esc(card.name)}</td><td colspan="4" class="cn-muted">${esc(text)}</td></tr>`;
        }
        return cell.rows.map(row=>{
          const score=view.scoreDisplay(row,card.key);
          const kind=row.kind==='daily_observation'?'盘后观察':row.kind==='late_reconstruction'?'开盘后补生成':
            row.kind==='manual_observation'?'规则观察':'历史回测';
          const pnl=row.pnl_pct==null?'未结算':`${row.pnl_pct>0?'+':''}${Number(row.pnl_pct).toFixed(2)}%`;
          return `<tr><td><a href="#${esc(card.key)}?date=${esc(day.date)}&range=all">${esc(card.name)}</a></td>
            <td class="cn-code"><a href="${esc(config.stockLink(row,card.key,hash()))}"><strong>${esc(row.ts_code.slice(0,6))}</strong> ${esc(row.name)}</a><small>${esc(row.ts_code)} · <a href="${esc(config.root+'analyzer/search.html?q='+encodeURIComponent(row.ts_code.slice(0,6)))}">公司分析</a></small>${config.evidenceButton?.(row)||''}</td>
            <td title="${esc(score.tooltip)}">${esc(score.main)}<small>${esc(score.detail)}</small>${row.sources?`<small>${esc(row.sources)}</small>`:''}</td>
            <td>${esc(kind)}<small>${esc(row.account_status)}</small></td>
            <td class="cn-num ${row.pnl_pct>0?'cn-good':row.pnl_pct<0?'cn-bad':'cn-muted'}">${pnl}<small>${row.entry_date?esc(row.entry_date)+' 买入':''}</small><small>${row.exit_date?esc(row.exit_date)+' 退出':''}</small></td></tr>`;
        }).join('');
      }).join('');
    }
    function details(day){
      return `<tr class="cn-record-expanded"><td colspan="${result.cards.length+2}"><div class="cn-record-detail-head"><strong>${esc(day.date)} · ${state.focus==='all'?'全部产品线':esc(config.cards.find(c=>c.key===state.focus)?.name||'选股记录')}</strong><span class="cn-muted">选股记录，非交易指令</span></div><div class="cn-table-wrap"><table class="cn-record-stocks"><thead><tr><th>产品线</th><th>股票 / K线</th><th>池内排名 / 来源</th><th>记录性质 / 参与</th><th class="cn-num">已兑现净收益</th></tr></thead><tbody>${stockRows(day)}</tbody></table></div></td></tr>`;
    }
    function render(){
      result=view.historyPage(config.cards,config.sessions||[],options());state.page=result.page;
      host.innerHTML=`<div class="cn-record-toolbar cn-toolbar">
        ${key==='all'?`<label>产品线<select data-filter="product"><option value="all">全部产品线</option>${config.cards.map(c=>`<option value="${c.key}">${esc(c.name)}</option>`).join('')}</select></label>`:''}
        <label>市场<select data-filter="market">${markets.map(([m,n])=>`<option value="${m}">${n}</option>`).join('')}</select></label>
        <label>区间<select data-filter="range"><option value="20">近20个交易日</option><option value="60">近60个交易日</option><option value="120">近120个交易日</option><option value="all">全部记录</option></select></label>
        <label>截至<input type="date" data-filter="end" max="${esc(config.expected)}"></label>
        <label>记录<select data-filter="kind"><option value="all">全部来源</option><option value="daily">每日观察</option><option value="historical">历史回测</option></select></label>
        <label class="cn-record-search">股票<input type="search" data-filter="query" placeholder="代码 / 名称" aria-label="搜索股票代码或名称"></label>
        <label><input type="checkbox" data-filter="settled">仅已结算</label>
        <button class="cn-icon" data-action="reset" title="恢复最近记录" aria-label="恢复最近记录">${icon('rotate-ccw')}</button>
        <button class="cn-icon" data-action="export" title="导出筛选范围内全部记录" aria-label="导出筛选范围内全部记录" ${result.totalRecords?'':'disabled'}>${icon('download')}</button></div>
        <div class="cn-record-summary"><span role="status">${result.totalDays} 个交易日 · ${result.totalRecords} 条匹配记录</span><small>0 已运行无匹配 · — 无留存 · 待数据 运行受阻</small></div>
        <div class="cn-table-wrap cn-record-scroll" tabindex="0" role="region" aria-label="按日期排列的产品线选股数量">
        <table class="cn-record-matrix ${result.cards.length===1?'cn-record-single':''}"><thead><tr><th>信号日</th>${result.cards.map(c=>`<th class="cn-num"><span style="border-color:${c.color}" class="cn-record-line">${esc(c.name)}</span></th>`).join('')}<th class="cn-num">合计条目</th></tr></thead><tbody>${result.days.length?result.days.map(day=>{
          const open=state.date===day.date;
          return `<tr class="cn-record-day ${open?'cn-record-active':''}"><td><button class="cn-record-date" data-day="${day.date}" data-focus="all" aria-expanded="${open&&state.focus==='all'}" aria-controls="cn-record-details" title="${day.date}全部选股">${icon(open?'chevron-down':'chevron-right')}<time>${day.date}</time></button></td>${day.cells.map(cell=>`<td class="cn-num"><button class="cn-record-count ${cell.state==='blocked'?'cn-warn':''}" data-day="${day.date}" data-focus="${cell.key}" aria-label="${day.date} ${esc(cell.name)} ${label(cell)}${cell.state==='ready'||cell.state==='recorded'?'条':''}" aria-expanded="${open&&state.focus===cell.key}" aria-controls="cn-record-details" ${cell.state==='unknown'?'disabled':''}>${label(cell)}</button>${cell.state!=='unknown'?`<small>${provenance(cell)}</small>`:''}</td>`).join('')}<td class="cn-num"><strong>${day.count}</strong><small>${day.unique} 只股票${day.recorded<day.cells.length?' · 部分缺记录':''}</small></td></tr>${open?details(day):''}`;
        }).join(''):`<tr><td colspan="${result.cards.length+2}" class="cn-empty">该筛选范围没有匹配记录。</td></tr>`}</tbody></table></div>
        <div class="cn-record-pagination"><span class="cn-muted">第 ${result.page+1} / ${result.pages} 页</span><button class="cn-icon" data-action="previous" title="上一页" aria-label="上一页" ${result.page?'':'disabled'}>${icon('chevron-left')}</button><button class="cn-icon" data-action="next" title="下一页" aria-label="下一页" ${result.page+1<result.pages?'':'disabled'}>${icon('chevron-right')}</button></div>`;
      const expanded=host.querySelector('.cn-record-expanded');
      if(expanded)expanded.id='cn-record-details';
      else host.querySelectorAll('[aria-controls]').forEach(el=>el.removeAttribute('aria-controls'));
      host.querySelectorAll('[data-filter]').forEach(el=>{
        const name=el.dataset.filter;
        if(name==='settled')el.checked=state.settled;else el.value=state[name];
        el.onchange=()=>{state[name]=name==='settled'?el.checked:el.value;state.page=0;state.date='';commit();render();};
        if(name==='query')el.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();el.blur();}};
      });
      host.querySelectorAll('[data-day]').forEach(el=>el.onclick=()=>{
        const closing=state.date===el.dataset.day&&state.focus===el.dataset.focus;
        state.date=closing?'':el.dataset.day;state.focus=el.dataset.focus;commit();render();
        const focused=host.querySelector(`[data-day="${el.dataset.day}"][data-focus="${el.dataset.focus}"]`);
        focused?.focus({preventScroll:true});
      });
      host.querySelector('[data-action="previous"]').onclick=()=>page(-1);
      host.querySelector('[data-action="next"]').onclick=()=>page(1);
      host.querySelector('[data-action="reset"]').onclick=()=>{
        Object.assign(state,{product:key,market:config.market||'all',range:'20',kind:'all',query:'',settled:false,end:config.expected,page:0,date:'',focus:'all'});commit();render();
      };
      host.querySelector('[data-action="export"]').onclick=exportRows;
      if(scope.lucide)scope.lucide.createIcons();
    }
    function page(delta){state.page+=delta;state.date='';commit();render();host.scrollIntoView({block:'start'});}
    function exportRows(){
      const cell=v=>{let s=String(v??'');if(/^[=+\-@\t\r]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};
      const rows=[['信号日','产品线','股票代码','名称','记录类型','账户状态','已兑现净收益%','模型版本']];
      for(const day of result.allDays)for(const group of day.cells)for(const r of group.rows){
        rows.push([day.date,group.name,r.ts_code,r.name,r.kind,r.account_status,r.pnl_pct,config.cards.find(c=>c.key===group.key).version]);
      }
      const url=URL.createObjectURL(new Blob(['\uFEFF'+rows.map(r=>r.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));
      const a=document.createElement('a');a.href=url;a.download=`CN_history_${state.product}_${state.end}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    if(state.date){
      const all=view.historyPage(config.cards,config.sessions||[],options());
      const index=all.allDays.findIndex(d=>d.date===state.date);
      if(index>=0)state.page=Math.floor(index/10);
    }
    render();
  }
  scope.CN_SELECTION_HISTORY={mount};
})(window);
