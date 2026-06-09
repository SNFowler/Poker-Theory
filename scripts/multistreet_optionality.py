#!/usr/bin/env python3
"""Multi-street test of the optionality thesis: does narrowing a GTO range pay?

The thesis (Channel B): a street-1 bet narrows the opponent's range, destroying
its optionality, and that loss is cashed on a later street.  Two links are clean
and a third one fails -- and the failure is the real lesson.

Setup: a 2-street game, passive defender (call/fold), per-street bet menus so the
street-1 size s1 is isolated and street 2 is a fixed pot bet.  For each s1 we
measure, conditional on the defender continuing:

  * condensation of the continuing range  (1 - H/Hmax)
  * mean strength of the continuing range
  * the defender's street-2 card value-of-information, per street-2 chip
  * the bettor's street-2 extraction  =  V(can bet street 2) - V(street 2 checked
    down)  -- a confound-free difference, isolating the street-2 betting value.

Findings:
  1. Narrowing is real: bigger s1 -> more condensed continuing range.
  2. Optionality drops: the defender's street-2 card-VoI/chip falls sharply.
  3. BUT the bettor's street-2 extraction does NOT rise -- it is flat / slightly
     falling, and VoI and extraction are *positively* correlated.

Why: a passive defender narrows from BELOW -- it folds its weak hands and keeps
its strong ones, so the continuing range gets *stronger* (mean strength rises),
not weaker.  A GTO opponent controls which hands continue and keeps the
continuing range defensible.  You can measure its optionality collapse, but you
cannot bank it: betting cannot force a GTO range into an exploitable shape.

The penalty for a condensed range (the clairvoyance result) is real only when the
shape is IMPOSED -- the opponent is dealt / constrained to it.  When the opponent
chooses its own continuing range, it chooses a good one.  Which is the project's
recurring lesson once more: value comes from the opponent being *constrained*,
not from information per se.

Writes figures/fig10_multistreet_optionality.png.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poker_theory.game import Game, GameConfig, AKQJT9_RANKS  # noqa: E402
from poker_theory import sequence_form as sf, analysis as an, ranges as rg  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RANKS = AKQJT9_RANKS
S2 = 1.0
BETTOR = rg.uniform_range(6)
DEFENDER = rg.uniform_range(6)


def _game(s1, bet_street2: bool):
    by_round = (((s1,) if s1 is not None else ()), ((S2,) if bet_street2 else ()))
    return Game(GameConfig(
        ranks=RANKS, suits=2, ante=1, num_rounds=2, bet_mode="no-limit", stack=50.0,
        raise_fractions=(), allow_allin=False, max_raises=1, aggressors=(0,),
        bet_fractions_by_round=by_round, bet_fractions=(S2,),
        deal_weights=(BETTOR, DEFENDER)))


def measure(s1):
    g = _game(s1, True)
    sol = sf.solve(g)
    reach = an.ProfileReach(g, sol.strategy)
    vals = an.node_values(g, sol.strategy)
    extraction = sol.value - sf.solve(_game(s1, False)).value

    cond = strength = float("nan")
    s1lab = f"b{s1:g}" if s1 is not None else None
    if s1lab:
        for d in an.decision_points(g):
            if d.player == 1 and d.community == "-" and d.history == s1lab:
                prior = an.rank_prior(d, reach)
                cont = {r: prior[r] * sol.strategy[d.infoset_by_rank[r]].get("c", 0.0)
                        for r in prior}
                tot = sum(cont.values())
                if tot > 0:
                    cont = {r: v / tot for r, v in cont.items()}
                    cond = 1 - an._entropy(cont) / np.log2(6)
                    strength = sum(cont[r] * (6 - RANKS.index(r)) for r in cont)

    nvoi = den = 0.0
    for d in an.decision_points(g):
        if d.player == 1 and d.community != "-" and d.history.endswith(f"b{S2:g}"):
            try:
                qv = an.question_value(g, sol.strategy, d, reach, vals)
            except ValueError:
                continue
            w = sum(reach.infoset_reach.get(h, 0.0) for h in d.infoset_by_rank.values())
            if w <= 0:
                continue
            bet2 = S2 * (2 + 4 * (s1 if s1 else 0.0))
            nvoi += w * qv.voi / bet2
            den += w
    voi2 = nvoi / den if den else float("nan")
    return cond, strength, voi2, extraction


def main() -> None:
    print("=" * 74)
    print("MULTI-STREET OPTIONALITY  ::  does narrowing a GTO range pay later?")
    print("=" * 74)
    s1s = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    print(f"\n  {'s1':>5} {'cond(R)':>9} {'strength(R)':>11} {'VoI2/chip':>10} {'extract2':>9}")
    rows = []
    for s1 in s1s:
        rows.append((s1,) + measure(s1))
        print(f"  {s1:>5.2f} {rows[-1][1]:>9.4f} {rows[-1][2]:>11.3f}"
              f" {rows[-1][3]:>10.4f} {rows[-1][4]:>9.4f}")
    cond = np.array([r[1] for r in rows])
    strength = np.array([r[2] for r in rows])
    voi2 = np.array([r[3] for r in rows])
    extr = np.array([r[4] for r in rows])
    print(f"\n  corr(VoI2/chip, extraction2)        = {np.corrcoef(voi2, extr)[0,1]:+.3f}")
    print(f"  corr(condensation, mean strength)   = {np.corrcoef(cond, strength)[0,1]:+.3f}")
    print("\n  => Links 1-2 hold (narrowing real, VoI collapses) but extraction does NOT")
    print("     rise: the passive defender caps from BELOW, keeping a STRONG range.")
    print("     A GTO opponent narrows itself optimally; the optionality drop is not")
    print("     bankable.  The condensed-range penalty needs an IMPOSED shape.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    axA.plot(s1s, cond, "-o", color="tab:purple", label="condensation 1-H/Hmax")
    ax2 = axA.twinx()
    ax2.plot(s1s, strength, "-s", color="tab:orange", label="mean strength")
    axA.set_xlabel("street-1 bet size s1 (fraction of pot)")
    axA.set_ylabel("condensation of continuing range", color="tab:purple")
    ax2.set_ylabel("mean strength of continuing range", color="tab:orange")
    axA.set_title("(A) A bet narrows the continuing range\nbut it caps from BELOW (strength rises)")
    axA.grid(alpha=0.3)

    axB.plot(s1s, voi2, "-D", color="tab:blue", label="defender street-2 VoI / chip")
    ax3 = axB.twinx()
    ax3.plot(s1s, extr, "-^", color="tab:green", label="bettor street-2 extraction")
    axB.set_xlabel("street-1 bet size s1 (fraction of pot)")
    axB.set_ylabel("defender VoI per street-2 chip", color="tab:blue")
    ax3.set_ylabel("bettor street-2 extraction (chips)", color="tab:green")
    ax3.set_ylim(0, max(extr) * 1.5)
    axB.set_title("(B) Optionality collapses, but extraction does NOT rise\n"
                  "(a GTO range can't be bet into an exploitable shape)")
    axB.grid(alpha=0.3)

    path = os.path.join(FIG_DIR, "fig10_multistreet_optionality.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
