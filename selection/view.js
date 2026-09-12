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
    const value = hash.replace(/^#/, '');
    return keys.includes(value) || value === 'monitor' ? value : 'overview';
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
  const api = {snapshot, route, scoreDisplay};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else scope.CN_SELECTION_VIEW = api;
})(typeof window === 'undefined' ? globalThis : window);
