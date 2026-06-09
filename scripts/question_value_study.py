#!/usr/bin/env python3
"""The corrected measure: a bet is a question -- measure the question, not the range.

The original brief asked for *an information measure that captures how exploitable
the shape of a range is*.  The earlier measures (entropy/KL of the rank
distribution) cannot express "shape": they are permutation-invariant, blind to
where mass sits in the strength order relative to the opponent.  This study
instruments the structure the brief actually named -- **the partition a bet's
question induces on the responder's range** -- via three quantities computed at
the response decision point (see ``analysis.question_value``):

  * VoI        -- decision-value of the responder's card for answering;
  * H(answer)  -- entropy of the induced answer partition;
  * indifferent (bluff-catcher) mass -- the mass held at indifference.

Findings (exact, from the LP):

  1. THE COUNTEREXAMPLE.  A defender uniform over four middle ranks has 2 bits
     of range entropy yet VoI = 0, answer entropy = 0, and pays exactly the
     clairvoyance rent s/(1+s) -- strategically identical to a single card.
     Range entropy cannot see this; the question measures see it exactly.

  2. THE PREDICTOR.  penalty ~~ rent x indifferent_mass - VoI
     ("what the question would extract from a blind range, minus what the
     defender's card buys back").  corr ~ 0.93 across all shapes, sizes, and
     the condensation surface; range entropy is incoherent across the same data.

  3. rent x indifferent_mass alone is the clairvoyance BOUND: exact for a pure
     polar question, an overestimate when the bettor has thin-value hands.

Writes figures/fig7_question_value.png.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poker_theory import studies as st  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main() -> None:
    print("=" * 76)
    print("THE QUESTION-VALUE MEASURE  ::  measuring the partition, not the range")
    print("=" * 76)

    # ---- gallery: shapes at fixed mean strength, pure polar bettor ----
    sizes = [0.5, 1.0, 2.0]
    pts = st.question_value_gallery(sizes)
    print("\nShape gallery (pure nuts/air bettor; defender mean strength fixed):")
    print(f"  {'shape':<22}{'s':>5} {'H(range)':>9} {'H(answer)':>9} {'VoI':>8}"
          f" {'indiff':>7} {'penalty':>8} {'rent*ind':>9} {'-VoI':>8}")
    for q in pts:
        print(f"  {q.label:<22}{q.bet_fraction:>5.2f} {q.range_entropy:>9.3f}"
              f" {q.partition_entropy:>9.3f} {q.voi:>8.4f} {q.indifferent_mass:>7.3f}"
              f" {q.penalty:>8.4f} {q.predictor:>9.4f} {q.predictor_net:>8.4f}")

    # ---- surface: condensation x size, soft-polar bettor ----
    conds = list(np.linspace(0.0, 1.0, 9))
    fracs = list(np.linspace(0.25, 3.0, 12))
    g = st.question_value_surface(conds, fracs)

    def corrs(pen, pred, pred_net, hr, name):
        pen, pred, pred_net, hr = map(np.asarray, (pen, pred, pred_net, hr))
        m = ~np.isnan(pred)
        c_bound = np.corrcoef(pen[m], pred[m])[0, 1]
        c_net = np.corrcoef(pen[m], pred_net[m])[0, 1]
        c_h = np.corrcoef(pen, hr)[0, 1]
        rmse = float(np.sqrt(np.mean((pen[m] - pred_net[m]) ** 2)))
        print(f"\n{name}")
        print(f"  corr(penalty, rent*indiff)        = {c_bound:+.3f}   (clairvoyance bound)")
        print(f"  corr(penalty, rent*indiff - VoI)  = {c_net:+.3f}   RMSE = {rmse:.4f}")
        print(f"  corr(penalty, range entropy)      = {c_h:+.3f}   (the old measure)")
        return c_net

    corrs([q.penalty for q in pts], [q.predictor for q in pts],
          [q.predictor_net for q in pts], [q.range_entropy for q in pts],
          "GALLERY (pure polar question):")
    corrs(g["penalty"].ravel(), g["predictor"].ravel(),
          g["predictor_net"].ravel(), g["range_entropy"].ravel(),
          "SURFACE (soft-polar question, condensation x size):")
    allpen = np.concatenate([[q.penalty for q in pts], g["penalty"].ravel()])
    allpred = np.concatenate([[q.predictor for q in pts], g["predictor"].ravel()])
    allnet = np.concatenate([[q.predictor_net for q in pts], g["predictor_net"].ravel()])
    allhr = np.concatenate([[q.range_entropy for q in pts], g["range_entropy"].ravel()])
    c_net = corrs(allpen, allpred, allnet, allhr, "COMBINED:")

    bc = next(q for q in pts if q.label == "mid-4 bluffcatchers" and q.bet_fraction == 1.0)
    print(f"\nThe counterexample, explicitly (s=1):")
    print(f"  'mid-4 bluffcatchers': H(range) = {bc.range_entropy:.1f} bits, yet")
    print(f"  VoI = {bc.voi:.4f}, H(answer) = {bc.partition_entropy:.4f}, "
          f"indifferent mass = {bc.indifferent_mass:.3f}")
    print(f"  penalty = {bc.penalty:.4f} = s/(1+s) -- identical to a single card.")

    # ---------------- figure ----------------
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.2))

    # (A) penalty vs range entropy: the old measure has no content across shapes
    s_show = 1.0
    gal1 = [q for q in pts if q.bet_fraction == s_show]
    axA.scatter([q.range_entropy for q in gal1], [q.penalty for q in gal1],
                s=70, color="tab:red", zorder=3)
    for q in gal1:
        axA.annotate(q.label, (q.range_entropy, q.penalty), fontsize=8,
                     xytext=(5, 4), textcoords="offset points")
    axA.set_xlabel("range entropy H(defender range)  (bits)")
    axA.set_ylabel("penalty (value to polar bettor, s = 1)")
    axA.set_title("(A) The old measure fails:\nrange entropy carries no shape information")
    axA.grid(alpha=0.3)

    # (B) penalty vs composite predictor, everything
    mg = ~np.isnan(allpred)
    axB.scatter(g["predictor_net"].ravel(), g["penalty"].ravel(), s=22,
                color="tab:blue", alpha=0.6, label="condensation surface")
    axB.scatter([q.predictor_net for q in pts], [q.penalty for q in pts], s=55,
                color="tab:red", marker="D", label="shape gallery")
    lim = [min(np.nanmin(allnet), 0) - 0.05, np.nanmax(allnet) + 0.05]
    axB.plot(lim, lim, "k--", lw=1, label="y = x")
    axB.set_xlabel("rent x indifferent mass  -  VoI")
    axB.set_ylabel("penalty (exact equilibrium value)")
    axB.set_title(f"(B) The question measure predicts:\ncorr = {c_net:+.2f} across all shapes and sizes")
    axB.legend(loc="upper left", fontsize=9)
    axB.grid(alpha=0.3)

    # (C) per-size tracking for three instructive shapes
    fine = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    shapes = st.shape_gallery()
    for label, color in (("mid-4 bluffcatchers", "tab:red"),
                         ("uniform", "tab:blue"),
                         ("semi-polar", "tab:green")):
        qs = [st.question_value_point(shapes[label], s, label=label) for s in fine]
        axC.plot(fine, [q.penalty for q in qs], "-o", color=color, label=f"{label}: penalty")
        axC.plot(fine, [q.predictor_net for q in qs], "--", color=color, alpha=0.7)
    axC.plot([], [], "k--", alpha=0.7, label="(dashed = predictor)")
    axC.set_xlabel("bet size s (fraction of pot)")
    axC.set_ylabel("penalty / predictor (chips)")
    axC.set_title("(C) Tracking across sizes:\nbluff-catcher ranges pay full rent at every size")
    axC.legend(fontsize=8)
    axC.grid(alpha=0.3)

    path = os.path.join(FIG_DIR, "fig7_question_value.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
