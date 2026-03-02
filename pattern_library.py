"""Pattern definition framework: enums, parameter containers, match results.

Pure data structures with no external dependencies beyond stdlib + pickle.
All numerical thresholds are learned by PatternMiner, not hardcoded here.
"""

import pickle
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class PatternPhase(Enum):
    """Phases a stock can be in within a pattern lifecycle."""
    IDLE = 'idle'
    CONSOLIDATION = 'consolidation'
    BREAKOUT_LIMIT_UP = 'breakout_limit_up'
    PULLBACK = 'pullback'
    CONFIRMATION = 'confirmation'
    SIGNAL = 'signal'
    FAILED = 'failed'


@dataclass
class PatternParams:
    """All numerical thresholds learned from data by PatternMiner."""
    # Consolidation detection
    consol_min_days: int = 15            # min consolidation days (expected 15-25)
    consol_max_range_pct: float = 0.12   # max price range during consolidation (8%-15%)

    # Breakout characteristics
    breakout_vol_ratio_min: float = 0.3  # limit-up vol / consol avg vol lower bound
    breakout_vol_ratio_max: float = 1.0  # upper bound (shrinking < 1.0)
    breakout_seal_quality_min: float = 0.3  # min seal quality score

    # Pullback characteristics
    pullback_vol_ratio_min: float = 1.5  # pullback vol / breakout vol (double volume >= 1.5)
    pullback_max_drop_pct: float = -0.08 # max allowed drop from breakout close
    pullback_max_days: int = 3           # max days to wait for pullback

    # Confirmation characteristics
    confirm_recovery_pct: float = 0.0    # close must be above pullback open by this pct
    confirm_max_days: int = 3            # max days to wait for confirmation

    # Statistics learned from data
    sample_count: int = 0
    success_rate: float = 0.0
    avg_return_5d: float = 0.0
    avg_return_10d: float = 0.0
    median_return_5d: float = 0.0


@dataclass
class PatternTemplate:
    """A named pattern with learned parameters and phase sequence."""
    name: str                          # e.g. 'consol_shrink_lu_double_vol_recovery'
    display_name: str                  # e.g. '横盘缩量涨停倍量回收'
    params: PatternParams              # learned from data
    phases: list                       # list of PatternPhase for this pattern
    min_confidence: float = 0.45       # min success rate threshold
    description: str = ''              # human-readable description


@dataclass
class PatternMatch:
    """A detected pattern match for a stock on a specific date."""
    ts_code: str
    pattern_name: str
    current_phase: PatternPhase
    phase_dates: dict = field(default_factory=dict)   # {phase_name: date_str}
    confidence: float = 0.0            # historical success rate
    explanation: str = ''              # human-readable explanation

    # Key metrics for output
    consol_days: int = 0
    consol_range_pct: float = 0.0
    breakout_vol_ratio: float = 0.0
    breakout_seal_quality: float = 0.0
    pullback_vol_ratio: float = 0.0
    pullback_drop_pct: float = 0.0

    # Signal date info
    signal_date: str = ''
    signal_price: float = 0.0


@dataclass
class PatternLibrary:
    """Collection of learned pattern templates. Serializable via pickle."""
    templates: dict = field(default_factory=dict)  # name -> PatternTemplate
    total_events_analyzed: int = 0
    total_valid_windows: int = 0
    mining_date: str = ''

    def add_template(self, template: PatternTemplate):
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PatternTemplate]:
        return self.templates.get(name)

    def list_templates(self) -> list:
        return list(self.templates.values())

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'PatternLibrary':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def summary(self) -> str:
        lines = [
            f'PatternLibrary: {len(self.templates)} templates, '
            f'{self.total_events_analyzed} events analyzed',
            f'Mining date: {self.mining_date}',
            '',
        ]
        for t in self.templates.values():
            p = t.params
            lines.append(f'  {t.display_name} ({t.name})')
            lines.append(f'    samples: {p.sample_count} | '
                         f'success: {p.success_rate:.1%} | '
                         f'avg 5d return: {p.avg_return_5d:+.1%}')
            lines.append(f'    consol >= {p.consol_min_days}d, '
                         f'range < {p.consol_max_range_pct:.0%}, '
                         f'vol ratio {p.breakout_vol_ratio_min:.1f}-{p.breakout_vol_ratio_max:.1f}, '
                         f'pullback vol >= {p.pullback_vol_ratio_min:.1f}x')
            lines.append('')
        return '\n'.join(lines)
