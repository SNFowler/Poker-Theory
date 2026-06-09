"""Construction of parameterized range shapes over ranks.

A *range* here is a weight vector over the game's ranks, used as ``deal_weights``
to bias how a player's private card is dealt.  We build a one-parameter family
that interpolates between a **condensed** range (mass concentrated on a middle
rank) and a **polarized** range (mass split on the extreme ranks), so we can
sweep a scalar *polarization* parameter and watch equilibrium EV respond.
"""

from __future__ import annotations

import math
from typing import Tuple


def _gaussian_bump(n: int, center: float, sigma: float) -> list:
    return [math.exp(-((i - center) ** 2) / (2.0 * sigma * sigma)) for i in range(n)]


def _normalize(w: list) -> Tuple[float, ...]:
    s = sum(w)
    return tuple(x / s for x in w)


def condensed_range(n: int, sigma: float = 0.9) -> Tuple[float, ...]:
    """Mass concentrated on the middle rank(s)."""
    center = (n - 1) / 2.0
    return _normalize(_gaussian_bump(n, center, sigma))


def polarized_range(n: int, sigma: float = 0.9) -> Tuple[float, ...]:
    """Mass split on the two extreme ranks (strongest + weakest)."""
    lo = _gaussian_bump(n, 0.0, sigma)
    hi = _gaussian_bump(n, n - 1, sigma)
    return _normalize([a + b for a, b in zip(lo, hi)])


def uniform_range(n: int) -> Tuple[float, ...]:
    return tuple(1.0 / n for _ in range(n))


def polarization_family(n: int, p: float, sigma: float = 0.9) -> Tuple[float, ...]:
    """Interpolate condensed (``p=0``) -> polarized (``p=1``)."""
    cond = condensed_range(n, sigma)
    pol = polarized_range(n, sigma)
    return _normalize([(1 - p) * c + p * q for c, q in zip(cond, pol)])


def mean_strength(weights: Tuple[float, ...]) -> float:
    """Mean rank *strength* of a range (strength = n-index; index 0 strongest).

    Returns a value in roughly ``[1, n]``; useful for summarizing a range.
    """
    n = len(weights)
    s = sum(weights)
    return sum(w * (n - i) for i, w in enumerate(weights)) / s
