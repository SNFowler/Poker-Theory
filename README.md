# Poker-Theory — information-theoretic study of range exploitability vs bet sizing

A small, **exactly solvable** toy poker game (a Leduc-style generalization to the
ranks **A K Q J T 9**) plus the machinery to ask one research question precisely:

> Can we define an **information measure** that captures *how exploitable the
> shape of a range is*, and use it to find the **bet size that most degrades the
> opponent's range** — and does that information-optimal size coincide with the
> EV-optimal size?

Everything runs on CPU in seconds. The exact solver is a **sequence-form linear
program** (Nash equilibrium + exact game value); an independent **CFR+** solver
cross-checks it and scales to finer sizing grids.

---

## The headline result: the penalty for a condensed range

The clean piece of poker theory this project was really after lives in the
**clairvoyance game** (`poker_theory/clairvoyance.py`, `scripts/clairvoyance_study.py`),
not in the full symmetric solve (which averages the effect away). Pit a
**polarized** bettor (nuts or air, 50/50) against a maximally **condensed**
defender ("a single card", a pure bluff-catcher):

- the exact solver reproduces the textbook closed form to machine precision:
  **value to the bettor = `s/(1+s)`**, defender minimum-defence frequency
  `1/(1+s)`, bluff frequency `s/(1+s)` — where `s` is the bet as a fraction of pot;
- that value **is the penalty the condensed range pays**, and it **grows
  monotonically with bet size**, approaching a full ante at all-in. Against a
  purely condensed range the polar player simply wants to bet as large as
  possible — **EV-optimal and "maximally exploit their range" sizing coincide**;
- with real 6-rank ranges, **condensing the defender (at fixed mean strength)
  strictly raises the extractable penalty** (`+0.10 → +0.54` from uniform to a
  single card) and pushes the penalty-maximizing size larger.

This also explains the earlier "information-max overbets relative to EV" puzzle:
that tension only appears because a *symmetric* bettor also holds value hands that
want to be called. Strip the range to pure polar-vs-condensed and the tension
vanishes — the penalty and the EV point the same way. → `figures/fig6_clairvoyance.png`

## TL;DR findings

Using the AKQJT9 game (see below), with the bettor's range mean-strength held
fixed so we isolate *shape*:

1. **Condensed ranges lose, polarized ranges win** (measure 1). Sweeping a range
   from condensed (mass on a middle rank) to polarized (mass on the extremes)
   takes the equilibrium EV from **−0.52 → +0.16** (2-round game), crossing zero
   at polarization ≈ **0.72**. A condensed range really is "at the mercy" of the
   opponent. → `figures/fig1_polarization_ev.png`

2. **A bet is a question; bigger questions collapse beliefs harder** (measures
   2 & 3). At a single decision the betting line leaks `I(action; rank) ≈ 0.99`
   bits about the hand. Decomposed per action, a pot-sized bet collapses the
   opponent's belief by **1.67 bits** vs **0.82 bits** for a check.
   → `figures/fig2_posterior_collapse.png`

3. **Being unable to use your hand is catastrophic** (measure 4). Forcing a
   player onto a rank-blind ("condensed") strategy costs ~**1.0–1.3** per hand
   against a best responder. Restricting the *deal* to a condensed range costs
   **0.42**; to a polarized range *gains* **0.26**.

4. **EV(B) decomposes exactly, and the range-collapse term separates out
   cleanly** (the synthesizing result). The value of betting size B partitions
   *exactly* into `EV(B) = V_check + V_betfold + V_betcall`. A bigger bet **buys
   fold equity** (`V_betfold` rises) but the **continuers get tougher**
   (`V_betcall` falls, going negative past ~1× pot); EV peaks where these
   balance (~**0.6× pot**). The information statistic (entropy collapse of the
   continuing range) is a *very good proxy* for the continuing-range term —
   `corr(entropy_drop, continuing-range toughness) = +0.93` — so range-collapse
   **is** an identifiable, well-measured term in the sizing balance. But in a
   2-round game it enters with a **cost** sign (condensing their range leaves the
   survivors *strong*, which hurts you *when called now*).
   → `figures/fig5_ev_decomposition.png`

5. **The "collapse pays off" benefit is a later-street term — and the two
   criteria co-move with polarization.** The romantic effect (cap them now,
   leverage it by barrelling later) is a *separate* term that is ≈0 here because
   there is only one shallow street to spend the cap on; this is why a literal
   information *maximum* overbets relative to EV (info-optimal ~1.7× vs
   EV-optimal ~0.65×, `fig3`). As the range polarizes, *both* the EV-optimal and
   the information-optimal size grow and the EV ridge migrates toward larger
   sizes (`fig4`). Net rule: **more polarized ⇒ bet bigger**; the information
   measures predict the *direction* of EV-optimal sizing and *proxy* the
   collapse term well, but whether that term is a net benefit is
   **depth-dependent**. → `figures/fig3_betsize_ev_vs_info.png`,
   `figures/fig4_synthesis.png`

---

## The game: Leduc-AKQJT9

A faithful generalization of Leduc Hold'em (`poker_theory/game.py`):

- **Deck:** 6 ranks `{A,K,Q,J,T,9}`, 2 suits → 12 cards (configurable).
- **Deal:** each player one private card; round-1 betting; one community card is
  revealed; round-2 betting; showdown.
- **Hand ranking:** a **pair** (private == community) beats any unpaired hand;
  otherwise the higher private rank wins; equal ranks split.
- **Antes:** each player antes 1 (pot starts at 2).
- **Two-player, zero-sum, imperfect information.**

The tree is built combinatorially over **ranks**; suits are integrated out
exactly via card-removal probabilities. This is a *lossless* abstraction here
because payoffs and card-removal probabilities depend only on ranks — it keeps
the game small enough for an exact LP while remaining the true game.

### Number of streets (`num_rounds`)

`num_rounds` is configurable. `num_rounds` rounds reveal `num_rounds-1` community
cards (one between each pair of rounds) and the showdown uses the whole board:

- `1` — single-round (deal, one betting round, showdown on private ranks) — the
  simplest spot for the sizing study;
- `2` — classic Leduc (validated to the published value);
- `3+` — **multi-street depth**, needed for range-leverage to compound (so the
  "cap them now, barrel later" benefit can actually materialize — see the
  decomposition finding above).

**Multi-board hand ranking** (a clean generalization of Leduc's rule): a hand is
a *pair* if its private rank appears anywhere on the board; a pair beats a
non-pair, and among equal pair-status the higher private rank wins (equal ranks
split). The classic single-card Leduc rule is the `num_rounds=2` special case.

**Computational note.** The fully-expanded tree grows multiplicatively per street,
so the exact LP is the right tool only for small configs (it solves a 3-round,
3-rank, single-raise game in ~0.1 s). For deeper/wider multi-street games CFR+ is
the intended solver — it agrees with the LP value where both run (e.g. 3-round
fixed-limit, `|LP − CFR| ≈ 4e-4`). The next scaling step (not yet implemented) is
a vectorized / public-tree CFR that carries reach vectors over private ranks and
computes showdown counterfactual values with an `O(n log n)` sort instead of
enumerating every private-card pair — that makes the number of ranks nearly free
and leaves street depth as the only real cost.

### Betting model (the lever)

The bet-size **menu is a parameter** — the whole point is to sweep it.

- **Fixed-limit** mode reproduces classic Leduc (`fixed_bets=(2,4)`, 2 raises/round).
- **No-limit** mode offers bet/raise sizes as **fractions of the pot** plus an
  optional all-in, bounded by a finite stack and a raise cap.

```python
from poker_theory import Game, GameConfig, AKQJT9_RANKS
cfg = GameConfig(ranks=AKQJT9_RANKS, num_rounds=2, bet_mode="no-limit",
                 stack=20.0, bet_fractions=(0.5, 1.0, 2.0))
g = Game(cfg)
```

---

## Solvers

### Sequence-form LP (exact) — `poker_theory/sequence_form.py`
The Koller–Megiddo–von Stengel sequence-form LP. Solving it yields an **exact**
Nash equilibrium (a pair of realization plans) and the **exact** game value. We
solve both players' LPs and check `v0 == −v1`. The realization plans are the raw
material for reach probabilities and posteriors.

### CFR+ (cross-check) — `poker_theory/cfr.py`
Counterfactual Regret Minimisation with the CFR+ enhancements (regret-matching⁺,
alternating updates, linear averaging). Independent of the LP; used to confirm
the value and to scale to fine sizing grids.

### Best response / exploitability — `poker_theory/best_response.py`
Exact tabular best response for imperfect-information games. Drives the
convergence check (the LP equilibrium has exploitability ≈ `5e-16`) and
measure 4.

### Validation (all automated in `tests/`)
| quantity | computed | published Leduc |
|---|---|---|
| game value to P0 | **−0.085606** | −0.085606 |
| sequences / player | **337** | 337 |
| infosets / player | **144** | 144 |
| equilibrium exploitability | **5e-16** | 0 (exact) |
| CFR+ avg-profile value | matches LP to ~1e-3 | — |

```
python -m pytest -q          # 11 invariants, ~2s
```

---

## The four information measures — `poker_theory/analysis.py`

All operate on the equilibrium realization plan; entropies are in **bits**.

1. **Range-shape → EV (polarization advantage).** A player's range is a weight
   vector over ranks (`poker_theory/ranges.py`) imposed via `deal_weights`. We
   sweep a scalar polarization parameter (condensed ↔ polarized, *mean strength
   held fixed*), re-solve the exact equilibrium each time, and plot EV. →
   `studies.polarization_sweep`

2. **Action↔strength mutual information.** At a decision point,
   `I(action; private rank) = H(action) − H(action | rank)` — how much a betting
   line leaks about the hand. → `analysis.action_strength_mi`

3. **Posterior entropy collapse from a bet.** For each action `B`, the opponent's
   posterior `P(rank | B) ∝ P(rank) P(B | rank)`; we report the entropy drop
   `H(S) − H(S | B)` and `KL(posterior ‖ prior)` per size. The
   action-probability-weighted mean of the drops **equals** the mutual
   information of measure 2 (verified to machine precision). → `analysis.posterior_collapse`
   *(Note: when the prior over ranks is uniform, `KL(posterior‖prior)` equals the
   entropy drop exactly, since `KL(p‖uniform) = log₂n − H(p)`.)*

4. **Exploitability gap under a range constraint.** Two readings, both provided:
   - *strategy reading* — force a player to be **rank-blind** (it must play the
     same action distribution regardless of its card), then let the opponent best
     respond; report EV lost. → `analysis.flat_range_exploitability`
   - *range reading* — restrict the **deal** to a shape but keep optimal play and
     re-solve; report EV lost. → `studies.range_restriction_exploitability`

---

## The synthesizing question

"A bet size that maximally worsens the opponent's range" has **two readings**,
both instrumented in `studies.bet_size_sweep`:

- **(a) continuing-range degradation** — the fold/continue split leaves the
  opponent's *continuing* range condensed/capped for the rest of the hand
  (entropy drop, KL from prior, fold equity, mean-strength shift);
- **(b) information extraction** — the bet makes the opponent's *reply* leak about
  their hand (`I(opp action; opp rank)` facing the bet).

For a chosen spot we build a **separate game per single bet size** so that `EV(B)`
is a clean function of the size, solve each exactly, and compare the EV-optimal
size to the degradation-optimal sizes. Reading (a)'s condensation is *monotone* in
size (no interior optimum); reading (b)'s MI is single-peaked but peaks larger
than EV — so a literal "maximally degrade" rule overbets.

**The cleaner framing — separate the collapse term, don't argmax it.**
Range-collapse is not *the* sizing rule; it is *one term* in the sizing balance.
`studies.bet_size_decomposition` makes that exact:

```
EV(B)  =  V_check(B)  +  V_betfold(B)  +  V_betcall(B)
                         └ fold equity   └ value when called (where the collapse lives)
```

A bigger bet trades rising fold equity against a falling when-called term, and
the EV optimum is where they balance. We then probe the when-called term with the
information statistic and find the collapse is **real and tightly proxied**
(`corr(entropy_drop, continuing-range toughness) = +0.93`) — but in this shallow
game it scores as a *cost* (tougher continuers now), with the *leverage benefit*
that would flip its sign living on later streets that the 2-round game does not
have. So: **the term separates out and is well-measured; its EV payoff is
depth-dependent.** → `figures/fig5_ev_decomposition.png`

---

## Reproducing everything

```bash
pip install -r requirements.txt
python -m pytest -q                  # validation suite
python scripts/run_studies.py        # full report + figures/  (~20s)
python scripts/run_studies.py --quick   # coarser grids        (~10s)
python scripts/clairvoyance_study.py    # the penalty-for-a-condensed-range result
```

Outputs (committed under `figures/`):

| file | content |
|---|---|
| `fig1_polarization_ev.png` | measure 1 — EV vs range polarization (1- & 2-round) |
| `fig2_posterior_collapse.png` | measures 2 & 3 — MI + per-size belief collapse |
| `fig3_betsize_ev_vs_info.png` | bet-size sweep — EV(B) vs degradation(B) |
| `fig5_ev_decomposition.png` | exact `EV(B)=V_check+V_betfold+V_betcall`; collapse term vs toughness |
| `fig4_synthesis.png` | EV landscape over (polarization, size) with optimal ridges |
| `fig6_clairvoyance.png` | **the penalty for a condensed range**: `s/(1+s)` + penalty surface |

---

## Layout

```
poker_theory/
  game.py            extensive-form Leduc-AKQJT9 (configurable bet menu, 1 or 2 rounds)
  sequence_form.py   exact Nash via sequence-form LP
  cfr.py             CFR+ cross-check solver
  best_response.py   best response + exploitability
  analysis.py        reach/posteriors + the four measures
  ranges.py          condensed/polarized range families
  studies.py         polarization sweep + bet-size degradation sweep
scripts/run_studies.py   runs the programme, writes figures + a report
tests/test_core.py       11 invariants (Leduc value, exploitability, MI identity, ...)
figures/                 generated plots
```

## Notes, caveats, and next steps

- **Position.** P0 always acts first (out of position). With a *symmetric* range
  this positional disadvantage dominates EV(B), so EV-optimal is "bet small";
  the polarization-advantage and sizing effects are studied by giving the bettor
  a range edge, which is where the information story becomes interesting.
- **Reading (a) is monotone.** Condensation of the continuing range grows with
  size, so "maximally degrade the continuing range" has no interior optimum — its
  value only matters via the *rest of the hand*, which is why a 2-round game is
  the right vehicle for it and a 1-round game is not.
- **Scaling to fine grids.** The single-size games are tiny; the LP handles the
  sweep directly. For large multi-size menus the LP grows and CFR+ is the
  intended fallback (the sparse LP already accepts thousands of sequences).
- **Provenance.** Built fresh in this repo. If you have prior AKQ-game code to
  fold in or cross-check against, point me at it.
