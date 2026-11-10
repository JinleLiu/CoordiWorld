"""Exact deployed selector AST extracted from sealed C1 lines 89-99.
No world inference, head fitting, or official scoring. C0=before; C1=choice.
Requires original head/base-eligibility inputs and bank receipt.
"""
import numpy as np
from types import SimpleNamespace

def select(a, h, ec, bankrec, unsafe):
    args = SimpleNamespace(policy='static_guard')
    eligible = h['eligible'].copy()
    score = h['scores'].copy() if ec is None else (14 * h['scores'] + 2 * h['projected'][:, 0] * ec) / 16
    extra = np.arange(len(a['valid'])) >= 12
    if bankrec['variant_steps']:
        assert ec is not None
        eligible |= extra & a['valid'] & (ec > 0) & (h['raw_physical_nc'] <= 0.5) & (h['raw_dac_event'] <= 0.5)
        p0 = extra & (a['x674'][:, 668] > 0)
        score[p0] += 5 / 16 * (h['projected'][p0, 0] * a['x674'][p0, 662] - h['projected'][p0, 1])
    if args.policy == 'static_guard_absDAC':
        add = a['valid'] & ~unsafe & (h['raw_physical_nc'] <= 0.5) & (h['raw_dac_event'] <= 0.5)
        add &= ~extra if ec is None else ~extra | (ec > 0)
        eligible |= add
    choice = int(np.where(eligible, score, -np.inf).argmax())
    before = choice
    clear = eligible & ~unsafe
    if unsafe[choice] and clear.any():
        choice = int(np.where(clear, score, -np.inf).argmax())
    return (before, choice, eligible, score)
