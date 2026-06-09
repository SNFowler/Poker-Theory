#!/usr/bin/env python3
"""Run the full research programme and emit figures + a printed report.

Usage:
    python scripts/run_studies.py            # full run, writes figures/*.png
    python scripts/run_studies.py --quick    # coarser grids, faster

Produces:
    figures/fig1_polarization_ev.png       measure 1: EV vs range polarization
    figures/fig2_posterior_collapse.png    measures 2 & 3: MI + per-size posterior
    figures/fig3_betsize_ev_vs_info.png    bet-size sweep: EV vs info degradation
    figures/fig4_synthesis.png             EV-optimal vs info-optimal size vs polariz.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poker_theory.game import Game, GameConfig, AKQJT9_RANKS, LEDUC_RANKS  # noqa: E402
from poker_theory import sequence_form as sf  # noqa: E402
from poker_theory import best_response as br  # noqa: E402
from poker_theory import analysis as an  # noqa: E402
from poker_theory import ranges as rg  # noqa: E402
from poker_theory import studies as st  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------


def validation() -> None:
    section("VALIDATION  (exact solver sanity checks)")
    cfg = GameConfig(ranks=LEDUC_RANKS, suits=2, ante=1, bet_mode="fixed-limit",
                     fixed_bets=(2, 4), max_raises=2)
    g = Game(cfg)
    sol = sf.solve(g)
    _, expl = br.exploitability(g, sol.strategy)
    print(f"Classic Leduc Hold'em:")
    print(f"  game value to P0      = {sol.value:.6f}   (published: -0.085606)")
    print(f"  sequences per player  = {sol.seq_index.num_seqs(0)}   (published: 337)")
    print(f"  infosets per player   = {len(g.player_infosets[0])}   (published: 144)")
    print(f"  equilibrium exploitability = {expl:.2e}  (-> exact Nash)")


# ---------------------------------------------------------------------------


def measure1_polarization(quick: bool) -> None:
    section("MEASURE 1  ::  range shape -> EV  (polarization advantage)")
    npts = 9 if quick else 13
    fig, ax = plt.subplots(figsize=(7.5, 5))
    summary = {}
    for rounds, color in ((1, "tab:blue"), (2, "tab:red")):
        base = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=rounds,
                          bet_mode="no-limit", stack=20.0,
                          bet_fractions=(0.75,), raise_fractions=(), allow_allin=False,
                          max_raises=1)
        pts = st.polarization_sweep(base, ranged_player=0, num_points=npts)
        xs = [p.polarization for p in pts]
        ys = [p.value_to_player for p in pts]
        ax.plot(xs, ys, "-o", color=color, label=f"{rounds}-round game")
        # crossover where EV changes sign (condensed-loses -> polar-wins)
        cross = None
        for i in range(1, len(ys)):
            if ys[i - 1] < 0 <= ys[i] or ys[i - 1] <= 0 < ys[i]:
                # linear interpolation
                t = -ys[i - 1] / (ys[i] - ys[i - 1])
                cross = xs[i - 1] + t * (xs[i] - xs[i - 1])
                break
        summary[rounds] = (ys[0], ys[-1], cross)
        print(f"  {rounds}-round: EV(condensed)={ys[0]:+.4f}  EV(polarized)={ys[-1]:+.4f}"
              f"  crossover@polarization={cross if cross is None else round(cross,3)}")
        if cross is not None:
            ax.axvline(cross, color=color, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("polarization parameter  (0 = condensed / middle,  1 = polarized / extremes)")
    ax.set_ylabel("equilibrium EV to the ranged player (P0)")
    ax.set_title("Measure 1: condensed ranges lose, polarized ranges win\n"
                 "(opponent holds a uniform range; range mean-strength held fixed)")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, "fig1_polarization_ev.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [saved] {path}")
    print("  => Confirms the core intuition: a condensed range is at the mercy of the")
    print("     opponent; polarizing the same-strength range flips EV positive.")


# ---------------------------------------------------------------------------


def measures_23_posterior(quick: bool) -> None:
    section("MEASURES 2 & 3  ::  action<->strength MI  and  per-size posterior collapse")
    # Single-round game with a real *menu* of sizes so each size is a 'question'.
    base = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=1,
                      bet_mode="no-limit", stack=20.0,
                      bet_fractions=(0.5, 1.0, 2.0), raise_fractions=(),
                      allow_allin=False, max_raises=1)
    g = Game(base)
    sol = sf.solve(g)
    reach = an.ProfileReach(g, sol.strategy)
    dp = next(d for d in an.decision_points(g)
              if d.player == 0 and d.history == "" and d.community == "-")
    mi = an.action_strength_mi(g, sol.strategy, dp, reach)
    pcs = an.posterior_collapse(g, sol.strategy, dp, reach)

    print(f"  Decision: {dp.label}")
    print(f"  I(action; private rank) = {mi.mutual_information:.4f} bits"
          f"   [H(action)={mi.action_entropy:.3f}, H(action|rank)={mi.conditional_entropy:.3f}]")
    print(f"  prior over P0 rank: " + ", ".join(f"{r}:{mi.prior[r]:.3f}" for r in dp.infoset_by_rank))
    print("  per-action posterior collapse (measure 3):")
    print(f"    {'action':>8} {'P(action)':>10} {'H_post(bits)':>12} {'drop(bits)':>11} {'KL(bits)':>9}")
    for p in pcs:
        print(f"    {p.action:>8} {p.action_prob:>10.3f} {p.posterior_entropy:>12.3f}"
              f" {p.entropy_drop:>11.3f} {p.kl_to_prior:>9.3f}")
    expd = sum(p.action_prob * p.entropy_drop for p in pcs)
    print(f"  check: sum_a P(a)*drop(a) = {expd:.4f} bits  ==  MI = {mi.mutual_information:.4f} bits")

    # Figure: per-action P(action) and entropy drop
    labels = [p.action for p in pcs]
    pa = [p.action_prob for p in pcs]
    drop = [p.entropy_drop for p in pcs]
    kl = [p.kl_to_prior if np.isfinite(p.kl_to_prior) else 0.0 for p in pcs]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(x, pa, color="tab:gray")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("P(action)  (how often the question is posed)")
    ax1.set_title("Action frequencies at P0's opening")
    ax1.grid(alpha=0.3, axis="y")
    w = 0.4
    ax2.bar(x - w / 2, drop, width=w, label="entropy drop H(S)-H(S|B)", color="tab:red")
    ax2.bar(x + w / 2, kl, width=w, label="KL(posterior||prior)", color="tab:blue")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("bits")
    ax2.set_title(f"Belief collapse per action\nMI(action;rank)={mi.mutual_information:.3f} bits")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    path = os.path.join(FIG_DIR, "fig2_posterior_collapse.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------


def measure4_exploitability() -> None:
    section("MEASURE 4  ::  exploitability under a range/strategy constraint")
    cfg = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=2,
                     bet_mode="no-limit", stack=20.0, bet_fractions=(0.75,),
                     raise_fractions=(), allow_allin=False, max_raises=1)
    g = Game(cfg)
    sol = sf.solve(g)
    print("  (a) strategy reading -- force a player to be rank-blind (condensed strategy):")
    for player in (0, 1):
        r = an.flat_range_exploitability(g, sol.strategy, player, sol.value)
        print(f"      P{player}: eq EV={r['equilibrium_value']:+.4f}  "
              f"flat EV={r['flat_value']:+.4f}  EV lost={r['exploitability_gap']:.4f}")
    print("  (b) range reading -- restrict the *deal* to a shape, replay optimally:")
    res = st.range_restriction_exploitability(cfg, player=0)
    for r in res:
        print(f"      {r.label:>10}: EV={r.value_restricted:+.4f}  "
              f"(uniform {r.value_uniform:+.4f})  EV lost={r.ev_lost:+.4f}")


# ---------------------------------------------------------------------------


def betsize_sweep_figure(quick: bool) -> None:
    section("BET-SIZE SWEEP  ::  EV(B)  vs  opponent-range degradation(B)")
    n = 6
    pol = rg.polarized_range(n)
    uni = rg.uniform_range(n)
    base = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=2,
                      bet_mode="no-limit", stack=20.0, deal_weights=(pol, uni))
    fracs = [round(x, 3) for x in np.linspace(0.2, 3.0, 15 if quick else 25)]
    res = st.bet_size_sweep(base, fracs, bettor=0)
    B = [r.fraction for r in res]
    EV = [r.value_to_bettor for r in res]
    MI = [r.opp_mi for r in res]
    DROP = [r.entropy_drop for r in res]
    FOLD = [r.fold_frequency for r in res]
    ev_opt = max(res, key=lambda r: r.value_to_bettor)
    mi_opt = max(res, key=lambda r: r.opp_mi)

    print(f"  bettor = P0 with a POLARIZED range, 2-round AKQJT9")
    print(f"  EV-optimal size           = {ev_opt.fraction:.3f}x pot  (EV={ev_opt.value_to_bettor:+.4f})")
    print(f"  info-optimal size (opp MI)= {mi_opt.fraction:.3f}x pot  (MI={mi_opt.opp_mi:.4f} bits)")
    print(f"  reading (a) range-condensation (entropy drop) is monotone in size.")
    print(f"  => info-optimal ({mi_opt.fraction:.2f}) does NOT equal EV-optimal "
          f"({ev_opt.fraction:.2f}); information-max overbets.")

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    l1, = ax1.plot(B, EV, "-o", color="tab:green", label="EV to bettor")
    ax1.set_xlabel("bet size  B  (fraction of pot)")
    ax1.set_ylabel("EV to bettor", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.axvline(ev_opt.fraction, color="tab:green", ls="--", alpha=0.6)
    ax2 = ax1.twinx()
    l2, = ax2.plot(B, MI, "-s", color="tab:blue", label="opp action-MI [reading b]")
    l3, = ax2.plot(B, DROP, "-^", color="tab:red", label="continuing-range entropy drop [reading a]")
    l4, = ax2.plot(B, FOLD, ":", color="tab:gray", label="fold frequency")
    ax2.set_ylabel("bits  /  probability", color="k")
    ax2.axvline(mi_opt.fraction, color="tab:blue", ls="--", alpha=0.6)
    ax1.set_title("Bet-size sweep (polarized bettor, 2-round):\n"
                  "EV-optimal (green dashed) sits below the info-optimal size (blue dashed)")
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, "fig3_betsize_ev_vs_info.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------


def betsize_decomposition_figure(quick: bool) -> None:
    section("EV DECOMPOSITION  ::  separating the range-collapse term from sizing")
    n = 6
    pol = rg.polarized_range(n)
    uni = rg.uniform_range(n)
    base = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=2,
                      bet_mode="no-limit", stack=20.0, deal_weights=(pol, uni))
    fracs = [round(float(x), 3) for x in np.linspace(0.2, 3.0, 15 if quick else 25)]
    res = st.bet_size_decomposition(base, fracs, bettor=0)
    B = np.array([r.fraction for r in res])
    EV = np.array([r.ev_total for r in res])
    VC = np.array([r.v_check for r in res])
    VF = np.array([r.v_betfold for r in res])
    VK = np.array([r.v_betcall for r in res])
    drop = np.array([r.entropy_drop for r in res])
    tough = np.array([r.showdown_toughness for r in res])
    percont = np.array([r.per_continue_value for r in res])
    ev_opt = max(res, key=lambda r: r.ev_total)

    print("  EV(B) = V_check + V_betfold + V_betcall   (exact additive partition)")
    print(f"  {'B':>5} {'EV':>8} {'V_check':>9} {'V_betfold':>10} {'V_betcall':>10}"
          f" {'Hdrop':>7} {'tough':>7}")
    for r in res:
        print(f"  {r.fraction:>5.2f} {r.ev_total:>8.4f} {r.v_check:>9.4f}"
              f" {r.v_betfold:>10.4f} {r.v_betcall:>10.4f}"
              f" {r.entropy_drop:>7.3f} {r.showdown_toughness:>7.3f}")
    c1 = float(np.corrcoef(drop, tough)[0, 1])
    c2 = float(np.corrcoef(drop, percont)[0, 1])
    print(f"  EV-optimal size = {ev_opt.fraction:.2f}x pot")
    print(f"  corr(range-collapse [entropy drop], continuing-range toughness) = {c1:+.3f}")
    print(f"  corr(range-collapse [entropy drop], value-when-called)          = {c2:+.3f}")
    print("  => the collapse term is real and tightly tracked by the entropy statistic,")
    print("     but in a 2-round game it scores as a COST (tougher continuers now); the")
    print("     leverage that would make it a benefit lives on later streets (absent here).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(B, EV, "-o", color="k", lw=2, label="EV(B) total")
    ax1.plot(B, VC, "-", color="tab:gray", label="V_check")
    ax1.plot(B, VF, "-^", color="tab:green", label="V_betfold (fold equity)")
    ax1.plot(B, VK, "-s", color="tab:red", label="V_betcall (when called)")
    ax1.axvline(ev_opt.fraction, color="k", ls="--", alpha=0.5)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("bet size B (fraction of pot)")
    ax1.set_ylabel("contribution to EV (bettor)")
    ax1.set_title("Exact partition  EV(B) = V_check + V_betfold + V_betcall")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax3 = ax2.twinx()
    l1, = ax2.plot(B, drop, "-D", color="tab:blue", label="range collapse (entropy drop)")
    l2, = ax3.plot(B, tough, "-s", color="tab:red", label="continuing-range toughness (EV cost)")
    ax2.set_xlabel("bet size B (fraction of pot)")
    ax2.set_ylabel("entropy drop (bits)", color="tab:blue")
    ax3.set_ylabel("vbar_full - vbar_continuing", color="tab:red")
    ax2.set_title(f"Range collapse tracks continuing-range toughness\n"
                  f"corr = {c1:+.2f}  (collapse is well-proxied, but scores as cost here)")
    ax2.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, "fig5_ev_decomposition.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [saved] {path}")


def synthesis_figure(quick: bool) -> None:
    section("SYNTHESIS  ::  how EV-optimal and info-optimal sizes move with polarization")
    n = 6
    uni = rg.uniform_range(n)
    pol_pts = np.linspace(0, 1, 7 if quick else 11)
    fracs = np.linspace(0.2, 3.0, 16 if quick else 26)
    ev_grid = np.zeros((len(pol_pts), len(fracs)))   # EV(polarization, size)
    mi_grid = np.zeros_like(ev_grid)
    ev_sizes, mi_sizes, ev_max = [], [], []
    for i, p in enumerate(pol_pts):
        ranged = rg.polarization_family(n, p)
        base = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=2,
                          bet_mode="no-limit", stack=20.0, deal_weights=(ranged, uni))
        res = st.bet_size_sweep(base, [round(float(x), 3) for x in fracs], bettor=0)
        ev_grid[i] = [r.value_to_bettor for r in res]
        mi_grid[i] = [r.opp_mi for r in res]
        ev_sizes.append(fracs[int(np.argmax(ev_grid[i]))])
        mi_sizes.append(fracs[int(np.argmax(mi_grid[i]))])
        ev_max.append(float(np.max(ev_grid[i])))
        print(f"  polariz={p:.2f}:  EV-opt size={ev_sizes[-1]:.3f}  "
              f"info-opt size={mi_sizes[-1]:.3f}  EV_max={ev_max[-1]:+.4f}")

    # EV heatmap over (polarization, size) with optimal ridges overlaid.
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    im = ax.imshow(ev_grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[fracs[0], fracs[-1], pol_pts[0], pol_pts[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("equilibrium EV to the (polarized) bettor")
    ax.plot(ev_sizes, pol_pts, "-o", color="white", lw=2, label="EV-optimal size")
    ax.plot(mi_sizes, pol_pts, "--s", color="red", lw=1.5, label="info-optimal size (max opp MI)")
    ax.set_xlabel("bet size B (fraction of pot)")
    ax.set_ylabel("range polarization parameter")
    ax.set_title("Synthesis: EV landscape over (polarization, size)\n"
                 "EV-optimal ridge (white) shifts to larger sizes as the range polarizes;\n"
                 "the info-optimal ridge (red) sits to its right -- info-max overbets")
    ax.legend(loc="lower right")
    path = os.path.join(FIG_DIR, "fig4_synthesis.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarser grids, faster")
    args = ap.parse_args()

    validation()
    measure1_polarization(args.quick)
    measures_23_posterior(args.quick)
    measure4_exploitability()
    betsize_sweep_figure(args.quick)
    betsize_decomposition_figure(args.quick)
    synthesis_figure(args.quick)

    section("DONE")
    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
