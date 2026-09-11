/* Pure display selection: an archived list must never become a live signal. */
(function (scope) {
  'use strict';
  function snapshot(card, market, expected) {
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
  const api = {snapshot, route};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else scope.CN_SELECTION_VIEW = api;
})(typeof window === 'undefined' ? globalThis : window);
