#!/usr/bin/env python3
"""Bet-sizing abstraction: how few sizes recover the GTO continuum?

The practical question behind the whole project: a solver with a *continuum* of
bet sizes achieves value V*, but you can only afford k discrete sizes.  How much
does the best k-set recover, and is there a cheap heuristic that does nearly as
well as the (expensive) greedy-optimal set?

Setup: a one-sided single-street game -- the bettor holds a range and may check
or bet any size in its menu; the defender is passive (call/fold only), so the
only sizing dimension is the bettor's menu and there is no out-of-position
confound.  The "continuum" is a fine geometric grid; capture% normalises value
between check-only (0%) and the full grid (100%).

Findings:
  * Diminishing returns are steep: 1 good size ~ 85-90%, TWO sizes ~ 98-100%,
    three ~ 100%.  Greedy-optimal matches exhaustive optimal at k<=2.
  * The optimal small menu is {small, ~pot} -- a polarising big size plus a thin
    small size; over-bets are essentially never needed against these ranges.
  * A single FIXED canonical menu {0.33, 1.0} captures >=~97% across every bettor
    shape: you do not need to know the range to pick near-optimal sizes -- a
    small + pot pair is robust.  Geometric spacing across the whole size axis is
    a poor heuristic because it squanders sizes on unused over-bets.

Writes figures/fig8_sizing_abstraction.png.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poker_theory import studies as st, ranges as rg  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GRID = [round(float(x), 3) for x in np.geomspace(0.1, 6.0, 20)]
SHAPES = {
    "uniform": rg.uniform_range(6),
    "polarized": rg.polarized_range(6),
    "condensed": rg.condensed_range(6),
    "linear/top": (3, 2.4, 1.8, 1.2, 0.6, 0.2),
}
CANONICAL = {
    "{1.0}": (1.0,),
    "{0.5,1.0}": (0.5, 1.0),
    "{0.33,1.0}": (0.33, 1.0),
    "{0.33,0.75,1.5}": (0.33, 0.75, 1.5),
}


def main() -> None:
    defender = rg.uniform_range(6)
    print("=" * 74)
    print("BET-SIZING ABSTRACTION  ::  how few sizes recover the continuum?")
    print("=" * 74)

    curves = {}
    for name, bettor in SHAPES.items():
        curves[name] = st.sizing_abstraction_curve(bettor, defender, grid=GRID, k_max=5)

    print("\nGreedy-optimal capture %% of the continuum vs number of sizes k:")
    print(f"  {'bettor':<12}" + "".join(f"k={k:<5}" for k in range(1, 6)))
    for name in SHAPES:
        row = curves[name].chosen["greedy"]
        print(f"  {name:<12}" + "".join(f"{100*row[k][2]:<6.1f}" for k in range(5)))

    # greedy-vs-exhaustive validation at k=2
    print("\nValidation -- greedy vs exhaustive optimal (k=2):")
    for name, bettor in SHAPES.items():
        c = curves[name]
        best_v = max(st.sizing_value(combo, bettor, defender)
                     for combo in itertools.combinations(GRID, 2))
        best_cap = st._capture(best_v, c.v_check, c.v_full)
        greedy_cap = c.chosen["greedy"][1][2]
        print(f"  {name:<12} greedy={100*greedy_cap:5.1f}%   exhaustive={100*best_cap:5.1f}%"
              f"   gap={100*(best_cap-greedy_cap):+.2f}pp")

    # robustness of a fixed canonical menu across shapes
    print("\nRobustness of FIXED canonical menus (capture %, no range knowledge):")
    print(f"  {'menu':<16}" + "".join(f"{n:<11}" for n in SHAPES))
    canon_caps = {}
    for mname, menu in CANONICAL.items():
        caps = []
        for sname, bettor in SHAPES.items():
            c = curves[sname]
            v = st.sizing_value(menu, bettor, defender)
            caps.append(st._capture(v, c.v_check, c.v_full))
        canon_caps[mname] = caps
        print(f"  {mname:<16}" + "".join(f"{100*x:<11.1f}" for x in caps))
    best_menu = max(canon_caps, key=lambda m: min(canon_caps[m]))
    print(f"  => most robust fixed menu: {best_menu} (worst-case "
          f"{100*min(canon_caps[best_menu]):.1f}% across shapes)")

    # ---------------- figure ----------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    ks = list(range(1, 6))
    for name in SHAPES:
        caps = [100 * curves[name].chosen["greedy"][k - 1][2] for k in ks]
        axA.plot(ks, caps, "-o", label=name)
    axA.axhline(99, color="k", ls=":", lw=0.8)
    axA.set_xlabel("number of bet sizes k (greedy-optimal)")
    axA.set_ylabel("% of the GTO continuum recovered")
    axA.set_title("(A) Diminishing returns:\ntwo well-chosen sizes ~ 98-100% of the continuum")
    axA.set_xticks(ks); axA.set_ylim(0, 103)
    axA.legend(fontsize=9); axA.grid(alpha=0.3)

    x = np.arange(len(SHAPES))
    w = 0.8 / len(CANONICAL)
    for i, (mname, caps) in enumerate(canon_caps.items()):
        axB.bar(x + i * w, [100 * c for c in caps], w, label=mname)
    axB.axhline(100, color="k", lw=0.6)
    axB.set_xticks(x + 0.4 - w / 2); axB.set_xticklabels(list(SHAPES), rotation=15)
    axB.set_ylabel("% of continuum recovered")
    axB.set_title("(B) A fixed {0.33, 1.0} menu is robust across ranges\n"
                  "(no range knowledge needed)")
    axB.legend(fontsize=8, ncol=2); axB.grid(alpha=0.3, axis="y")

    path = os.path.join(FIG_DIR, "fig8_sizing_abstraction.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
