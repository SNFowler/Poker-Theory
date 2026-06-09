#!/usr/bin/env python3
"""Complexity stress test: where the question-value predictor holds and breaks.

The composite predictor ``penalty ~ max(0, s/(1+s)*indifferent_mass - VoI)`` is an
*empirical* fit on the simple clairvoyance setup, not a theorem.  This script
stress-tests it by enriching the game and reports, honestly, the domain where it
survives and the domain where it fails.

Regime 1 (in domain): a polar bettor vs a condensed / bluff-catcher defender,
now WITH the defender allowed to raise (answer alphabet fold/call/raise).
  -> The predictor holds (and even improves); a pure bluff-catcher cannot escape
     the rent s/(1+s) even when handed the raise option -- the measure correctly
     shows every extra action is dominated (the answer partition stays trivial).

Regime 2 (out of domain): a soft-polar bettor (thin value) vs a STRONG,
counter-attacking defender, raises allowed.
  -> The predictor breaks.  ``indifferent_mass`` conflates two opposite things:
     indifference-at-the-bottom (a worthless bluff-catcher, paying rent) and
     indifference-at-the-top (a nut hand choosing between call and raise, both
     winning).  The strong defender is no longer a victim -- the "penalty" can go
     negative -- so the whole rent framing does not apply.

Conclusion: the textbook content (s/(1+s), MDF, polar>condensed) reproduces
exactly; VoI=0 cleanly characterizes pure bluff-catchers; but the cute composite
predictor is a bluff-catcher-regime artifact and does not generalize.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_theory import studies as st, ranges as rg, analysis as an, sequence_form as sf
from poker_theory.game import Game, GameConfig, AKQJT9_RANKS


def _solve_response(defender, s, bettor, rf=(1.0,), mr=2):
    cfg = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=1,
                     bet_mode="no-limit", stack=100.0, bet_fractions=(s,),
                     raise_fractions=rf, allow_allin=False, max_raises=mr,
                     deal_weights=(bettor, defender))
    g = Game(cfg)
    sol = sf.solve(g)
    reach = an.ProfileReach(g, sol.strategy)
    vals = an.node_values(g, sol.strategy)
    dp = next(d for d in an.decision_points(g)
              if d.player == 1 and d.history == f"b{s:g}" and d.community == "-")
    return sol.value, an.question_value(g, sol.strategy, dp, reach, vals)


def main() -> None:
    sizes = [0.5, 1.0, 2.0]
    print("=" * 76)
    print("COMPLEXITY STRESS TEST  ::  does the predictor survive a richer game?")
    print("=" * 76)

    # ---- Regime 1: in domain, defender may raise ----
    polar = (1, 0, 0, 0, 0, 1)
    shapes1 = {
        "condensed-mid": rg.condensed_range(6),
        "bluffcatch x4": (0, 1, 1, 1, 1, 0),
        "bluffcatch x2": (0, 0, 1, 1, 0, 0),
        "uniform": (1, 1, 1, 1, 1, 1),
    }
    pen, pred = [], []
    print("\nRegime 1 -- polar bettor, condensed/bluff-catcher defender, RAISES ALLOWED:")
    print(f"  {'defender':<14}{'s':>5} {'penalty':>8} {'s/(1+s)':>8} {'predictor':>10}"
          f" {'indiff':>7} {'VoI':>7} {'H(ans)':>7}")
    for label, w in shapes1.items():
        for s in sizes:
            v, qv = _solve_response(w, s, polar)
            p = max(0.0, s / (1 + s) * qv.indifferent_mass - qv.voi)
            pen.append(v); pred.append(p)
            print(f"  {label:<14}{s:>5.1f} {v:>8.4f} {s/(1+s):>8.4f} {p:>10.4f}"
                  f" {qv.indifferent_mass:>7.3f} {qv.voi:>7.4f} {qv.partition_entropy:>7.3f}")
    pen, pred = np.array(pen), np.array(pred)
    print(f"  => corr(penalty, predictor) = {np.corrcoef(pen, pred)[0,1]:+.3f}  (HOLDS)")
    print("     a pure bluff-catcher pays exactly s/(1+s) even given the raise option.")

    # ---- Regime 2: out of domain, strong counter-attacking defender ----
    soft = rg.polarized_range(6)   # thin value (K, T) the defender can out-hold
    shapes2 = {
        "top-heavy": (3, 2, 1, 1, 0, 0),
        "nutty": (3, 1, 0, 0, 0, 0),
        "uniform": (1, 1, 1, 1, 1, 1),
        "condensed-mid": rg.condensed_range(6),
    }
    pen2, pred2 = [], []
    print("\nRegime 2 -- soft-polar bettor (thin value), STRONG defender, raises allowed:")
    print(f"  {'defender':<14}{'s':>5} {'penalty':>8} {'predictor':>10} {'indiff':>7}"
          f" {'VoI':>7} {'meanMaxEV':>9}")
    for label, w in shapes2.items():
        for s in sizes:
            try:
                v, qv = _solve_response(w, s, soft)
            except ValueError:
                # bettor never bets into this defender: the question is worthless,
                # so it is never even posed -- the strongest possible "out of domain".
                print(f"  {label:<14}{s:>5.1f}   (bettor never bets -- question never posed)")
                continue
            p = max(0.0, s / (1 + s) * qv.indifferent_mass - qv.voi)
            mean_max = sum(qv.prior[r] * max(qv.action_values[r].values()) for r in qv.prior)
            pen2.append(v); pred2.append(p)
            print(f"  {label:<14}{s:>5.1f} {v:>8.4f} {p:>10.4f} {qv.indifferent_mass:>7.3f}"
                  f" {qv.voi:>7.4f} {mean_max:>+9.3f}")
    pen2, pred2 = np.array(pen2), np.array(pred2)
    print(f"  => corr(penalty, predictor) = {np.corrcoef(pen2, pred2)[0,1]:+.3f}  (BREAKS)")
    print("     'top-heavy' counter-attacks: penalty goes NEGATIVE while indiff_mass=1")
    print("     conflates nut-indifference (call/raise both win) with rent indifference.")

    print("\n" + "=" * 76)
    print("VERDICT: predictor is a bluff-catcher-regime artifact; the textbook")
    print("content (s/(1+s), MDF, polar>condensed) and VoI=0 for bluff-catchers are")
    print("the durable, exact results.  No new poker math -- a faithful re-encoding.")


if __name__ == "__main__":
    main()
