/* Pure display selection: an archived list must never become a live signal. */
(function (scope) {
  'use strict';
  function snapshot(card, market, expected) {
    if(card.daily_run && card.daily_run.date<=expected) {
      const run=card.daily_run;
      const rows=card.candidates.filter(r=>r.date===run.date &&
        ['daily_observation','late_reconstruction'].includes(r.kind) &&
        (market==='all'||r.market===market));
      return {date:run.date,rows,archived:false,stale:run.date!==expected,
        state:run.state,timely:run.timely,message:run.message};
    }
    const rows = card.candidates.filter(r => (market === 'all' || r.market === market) &&
      (card.key !== 'manual' || r.kind === 'manual_observation'));
    const dates = rows.map(r => r.date).filter(d => d <= expected).sort();
    const date = dates.at(-1) || null;
    const selected = rows.filter(r => r.date === date);
    return {date, rows:selected, archived:card.key !== 'manual', stale:date !== expected};
  }
  function route(hash, keys) {
    const value = hash.replace(/^#/, '').split('?')[0];
    return keys.includes(value) || ['monitor','history'].includes(value) ? value : 'overview';
  }
  const dailyKinds = ['daily_observation','late_reconstruction','manual_observation'];
  function historyCell(card, date, options={}) {
    const market=options.market||'all', kind=options.kind||'all';
    const query=String(options.query||'').trim().toLowerCase();
    const accepts=r=>kind==='all'||(kind==='daily')===dailyKinds.includes(r.kind);
    const records=card.candidates.filter(r=>r.date===date&&accepts(r));
    const run=kind==='historical'?null:[...(card.daily_runs||[]),card.daily_run].filter(Boolean).find(r=>r.date===date);
    const rows=records.filter(r=>(market==='all'||r.market===market)&&
      (!query||[r.ts_code,r.name].some(v=>String(v||'').toLowerCase().includes(query)))&&
      (!options.settled||r.pnl_pct!=null));
    const state=run?.state==='blocked'?'blocked':run?.state==='ready'?'ready':records.length?'recorded':'unknown';
    return {key:card.key,name:card.name,rows,count:rows.length,total:records.length,state,run,
      filtered:rows.length!==records.length,
      historical:records.some(r=>!dailyKinds.includes(r.kind)),
      late:records.some(r=>r.kind==='late_reconstruction')||Boolean(run&&!run.timely)};
  }
  function historyPage(cards, sessions, options={}) {
    const expected=options.expected||'9999-12-31', end=options.end&&options.end<expected?options.end:expected;
    const selected=cards.filter(c=>!options.product||options.product==='all'||c.key===options.product);
    const keys=new Set([...sessions,...selected.flatMap(c=>[
      ...c.candidates.map(r=>r.date),...(c.daily_runs||[]).map(r=>r.date),c.daily_run?.date].filter(Boolean))]);
    let dates=[...keys].filter(d=>d<=end).sort().reverse();
    const range=Number(options.range);
    if([20,60,120].includes(range))dates=dates.slice(0,range);
    let days=dates.map(date=>{
      const cells=selected.map(c=>historyCell(c,date,options));
      const rows=cells.flatMap(c=>c.rows);
      return {date,cells,count:rows.length,unique:new Set(rows.map(r=>r.ts_code)).size,
        recorded:cells.filter(c=>c.state!=='unknown').length};
    });
    // Searching and settled-only are record filters, not claims of zero picks.
    if(options.query?.trim()||options.settled)days=days.filter(d=>d.count>0);
    const pageSize=Math.max(1,Math.min(50,Number(options.pageSize)||10));
    const pages=Math.max(1,Math.ceil(days.length/pageSize));
    const page=Math.min(pages-1,Math.max(0,Math.floor(Number(options.page)||0)));
    return {cards:selected,days:days.slice(page*pageSize,(page+1)*pageSize),allDays:days,
      totalDays:days.length,totalRecords:days.reduce((n,d)=>n+d.count,0),page,pages,pageSize};
  }
  function scoreDisplay(row, key) {
    const names={abc:'形态偏离原分',bottom:'相对收益原分',second:'收益排序原分',first_open:'收益排序原分',manual:'规则原分'};
    const raw=row.score==null?'':Number(row.score).toFixed(4);
    const tooltip=`${names[key]||'模型原分'} ${raw}；不是盈利概率，不跨产品比较。`;
    if(Number.isInteger(row.pool_size)&&row.pool_size>0&&Number.isInteger(row.pool_rank)&&row.pool_rank>0&&row.pool_rank<=row.pool_size) {
      const main=`${row.score_tie_count>1?'并列':''}第 ${row.pool_rank} / ${row.pool_size} 名`;
      if(row.pool_size===1)return {main,detail:'唯一候选，未做横向比较',tooltip};
      const value=row.score_percentile;
      const detail=Number.isFinite(value)&&value>=0&&value<=100?`池内分位 ${value.toFixed(1)} / 100 · 非胜率`:'完整池分位未记录';
      return {main,detail,tooltip};
    }
    return {main:raw||row.sources||'未记录',detail:raw?(names[key]||'模型原分')+' · 非胜率':'',tooltip};
  }
  const api = {snapshot, route, scoreDisplay, historyCell, historyPage};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else scope.CN_SELECTION_VIEW = api;
})(typeof window === 'undefined' ? globalThis : window);
