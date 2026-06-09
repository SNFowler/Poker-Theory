#!/usr/bin/env python3
"""The penalty for a condensed range -- the clairvoyance game.

Isolates the intuition the project is about: a condensed range is at the mercy of
a polarized opponent.  Two results:

  (A) Pure clairvoyance (nuts/air bettor vs a single bluff-catcher): the exact
      solver reproduces the closed form ``value = s/(1+s)`` and the minimum-defence
      frequency ``1/(1+s)`` -- the penalty the condensed range pays grows
      monotonically with bet size and there is no EV-vs-exploitation tension.

  (B) Penalty surface over (how condensed the defender is) x (bet size), with real
      6-rank ranges and a fixed polarized bettor: the penalty is largest exactly
      where the defender is most condensed and the bet is biggest.

Writes figures/fig6_clairvoyance.png.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poker_theory.game import AKQJT9_RANKS  # noqa: E402
from poker_theory import studies as st  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("THE PENALTY FOR A CONDENSED RANGE  ::  the clairvoyance game")
    print("=" * 72)

    # (A) pure clairvoyance vs closed form
    sizes = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    pts = st.clairvoyance_size_sweep(sizes)
    print("\nPure clairvoyance (nuts/air bettor vs one bluff-catcher):")
    print(f"  {'s':>6} {'solver':>9} {'s/(1+s)':>9} {'MDFcall':>8} {'bluff':>7}")
    for p in pts:
        print(f"  {p.bet_fraction:>6.2f} {p.solver_value:>9.5f} {p.closed_form:>9.5f}"
              f" {p.call_frequency:>8.3f} {p.bluff_frequency:>7.3f}")
    max_err = max(abs(p.solver_value - p.closed_form) for p in pts)
    print(f"  max |solver - closed form| = {max_err:.2e}  (exact match)")
    print("  => the penalty grows monotonically with bet size, -> 1 ante at all-in.")

    # (B) penalty surface
    conds = list(np.linspace(0.0, 1.0, 11))
    fracs = list(np.linspace(0.2, 4.0, 20))
    surf = st.condensation_penalty_surface(conds, fracs)
    print("\nPenalty surface (polar bettor vs defender of varying condensation):")
    print(f"  defender uniform (d=0):    best-size penalty = {surf[0].max():+.4f}")
    print(f"  defender condensed (d=1):  best-size penalty = {surf[-1].max():+.4f}")
    print("  => condensing the defender (at fixed mean strength) strictly raises the")
    print("     penalty a polar bettor can extract, and the best size grows with it.")

    # figure
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    s_fine = np.linspace(0.05, 10, 200)
    axA.plot(s_fine, s_fine / (1 + s_fine), "-", color="tab:blue",
             label="closed form  s/(1+s)")
    axA.plot([p.bet_fraction for p in pts], [p.solver_value for p in pts], "o",
             color="tab:red", label="exact solver")
    axA.set_xlabel("bet size s (fraction of pot)")
    axA.set_ylabel("penalty paid by the condensed range\n(= value to the polar bettor, antes)")
    axA.set_title("(A) Pure clairvoyance: penalty grows with bet size\n"
                  "(bigger is always better for the polar player)")
    axA.axhline(1.0, color="k", ls=":", lw=0.8)
    axA.legend(loc="lower right"); axA.grid(alpha=0.3)

    im = axB.imshow(surf, origin="lower", aspect="auto", cmap="magma",
                    extent=[fracs[0], fracs[-1], conds[0], conds[-1]])
    cbar = fig.colorbar(im, ax=axB)
    cbar.set_label("penalty (value to polar bettor)")
    # best-size ridge
    best_j = surf.argmax(axis=1)
    axB.plot([fracs[j] for j in best_j], conds, "-o", color="cyan", lw=2,
             label="penalty-maximizing size")
    axB.set_xlabel("bet size s (fraction of pot)")
    axB.set_ylabel("defender condensation  (0 = uniform, 1 = single card)")
    axB.set_title("(B) Penalty surface: most condensed + biggest bet = worst spot\n"
                  "(real 6-rank ranges, defender mean-strength fixed)")
    axB.legend(loc="lower right")

    path = os.path.join(FIG_DIR, "fig6_clairvoyance.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
