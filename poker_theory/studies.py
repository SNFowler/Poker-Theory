"""Research studies: polarization sweep and the bet-size degradation sweep.

This module ties the solver and the measures together into the two experiments
the brief asks for:

* :func:`polarization_sweep` (measure 1) -- impose a one-parameter range shape on
  a player, re-solve the exact equilibrium at each setting, and record how the
  game value moves as the range goes from *condensed* to *polarized*.

* :func:`bet_size_sweep` (the synthesizing question) -- for a fixed spot, build a
  separate game for each single bet size on a menu, solve it exactly, and record
  both the **EV** of that size and how much it **degrades the opponent's range**,
  under both readings from the brief:

    (a) continuing-range degradation -- the fold/continue split leaves the
        opponent's continuing range condensed/capped (entropy drop, KL, fold
        equity);
    (b) information extraction -- the bet makes the opponent's response leak
        about their hand (``I(action; rank)`` of the opponent's reply).

  The headline question is whether the **information-optimal** size coincides
  with the **EV-optimal** size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import analysis as an
from . import ranges as rg
from . import sequence_form as sf
from .game import Game, GameConfig


# ---------------------------------------------------------------------------
# Measure 1: polarization sweep
# ---------------------------------------------------------------------------


@dataclass
class PolarizationPoint:
    polarization: float
    value_to_player: float       # equilibrium EV to the ranged player
    mean_strength: float          # mean strength of the imposed range
    range_weights: Tuple[float, ...]


def polarization_sweep(
    base_config: GameConfig, ranged_player: int = 0, num_points: int = 11,
    sigma: float = 0.9,
) -> List[PolarizationPoint]:
    """Sweep ``ranged_player``'s range from condensed to polarized.

    The opponent keeps a uniform range.  At each polarization setting we re-solve
    the *exact* equilibrium and record the value to the ranged player.
    """
    n = base_config.num_ranks
    out: List[PolarizationPoint] = []
    for k in range(num_points):
        p = k / (num_points - 1)
        ranged = rg.polarization_family(n, p, sigma)
        uniform = rg.uniform_range(n)
        if ranged_player == 0:
            weights = (ranged, uniform)
        else:
            weights = (uniform, ranged)
        cfg = _replace(base_config, deal_weights=weights)
        sol = sf.solve(Game(cfg))
        value_to_player = sol.value if ranged_player == 0 else -sol.value
        out.append(PolarizationPoint(
            polarization=p, value_to_player=value_to_player,
            mean_strength=rg.mean_strength(ranged), range_weights=ranged,
        ))
    return out


# ---------------------------------------------------------------------------
# The bet-size sweep (synthesizing question)
# ---------------------------------------------------------------------------


@dataclass
class BetSizeResult:
    fraction: float                  # bet size as a fraction of the pot
    label: str                       # action label in the game
    value_to_bettor: float           # equilibrium EV to the betting player
    bet_frequency: float             # how often the bettor bets (any rank)
    bettor_mi: float                 # I(bettor action; bettor rank), bits (leak)
    # opponent reaction to the bet:
    fold_frequency: float            # P(opponent folds | facing the bet)
    opp_mi: float                    # I(opp action; opp rank) facing the bet -- reading (b)
    prior_entropy: float             # H(opp range) before the bet, bits
    continuing_entropy: float        # H(opp continuing range), bits -- reading (a)
    entropy_drop: float              # prior - continuing (condensation)
    continuing_kl: float             # KL(continuing range || prior), bits
    mean_strength_prior: float
    mean_strength_continuing: float
    reached: bool = True


def _opponent_response_dp(game: Game, bettor: int, bet_label: str) -> Optional[an.DecisionPoint]:
    """The opponent's decision point immediately facing the bettor's first bet."""
    opp = 1 - bettor
    for dp in an.decision_points(game):
        if dp.player == opp and dp.community == "-" and dp.history == bet_label:
            return dp
    return None


def _bettor_open_dp(game: Game, bettor: int) -> Optional[an.DecisionPoint]:
    for dp in an.decision_points(game):
        if dp.player == bettor and dp.community == "-" and dp.history == "":
            return dp
    return None


def _continuing_range(
    dp: an.DecisionPoint, strategy, reach: an.ProfileReach,
    continue_actions: Tuple[str, ...],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Return (prior over opp rank, continuing range, fold frequency)."""
    prior = an.rank_prior(dp, reach)
    cont_weight = {}
    for r, infoset in dp.infoset_by_rank.items():
        pcont = sum(strategy[infoset].get(a, 0.0) for a in continue_actions)
        cont_weight[r] = prior[r] * pcont
    cont_total = sum(cont_weight.values())
    fold_freq = 1.0 - cont_total
    if cont_total > 0:
        continuing = {r: w / cont_total for r, w in cont_weight.items()}
    else:
        continuing = {r: 0.0 for r in prior}
    return prior, continuing, fold_freq


def _strength_dist_stats(dist: Dict[str, float], ranks: Tuple[str, ...]) -> float:
    """Mean strength (n..1, strongest=n) of a distribution over rank labels."""
    n = len(ranks)
    idx = {r: i for i, r in enumerate(ranks)}
    s = sum(dist.values())
    if s <= 0:
        return 0.0
    return sum(p * (n - idx[r]) for r, p in dist.items()) / s


def bet_size_sweep(
    base_config: GameConfig, fractions: List[float], bettor: int = 0,
) -> List[BetSizeResult]:
    """For each bet size, solve a single-size game and measure EV + degradation.

    ``base_config`` should describe the spot (ranks, rounds, stack, ante).  The
    bet menu is replaced by a single fraction for each game so that EV(B) is a
    clean function of the size B.
    """
    results: List[BetSizeResult] = []
    ranks = base_config.ranks
    for frac in fractions:
        cfg = _replace(
            base_config, bet_mode="no-limit", bet_fractions=(frac,),
            raise_fractions=(), allow_allin=False, max_raises=1,
        )
        game = Game(cfg)
        sol = sf.solve(game)
        reach = an.ProfileReach(game, sol.strategy)
        value_to_bettor = sol.value if bettor == 0 else -sol.value

        # Identify the single bet label (e.g. "b1" for pot-sized).
        open_dp = _bettor_open_dp(game, bettor)
        bet_labels = [a for a in open_dp.actions if a not in ("x", "c", "f")]
        bet_label = bet_labels[0] if bet_labels else None

        # bettor leak + frequency at the opening decision
        mi_open = an.action_strength_mi(game, sol.strategy, open_dp, reach)
        bet_freq = sum(mi_open.action_marginal.get(b, 0.0) for b in bet_labels)

        resp_dp = _opponent_response_dp(game, bettor, bet_label) if bet_label else None
        if resp_dp is None:
            results.append(BetSizeResult(
                fraction=frac, label=bet_label or "", value_to_bettor=value_to_bettor,
                bet_frequency=bet_freq, bettor_mi=mi_open.mutual_information,
                fold_frequency=float("nan"), opp_mi=float("nan"),
                prior_entropy=float("nan"), continuing_entropy=float("nan"),
                entropy_drop=float("nan"), continuing_kl=float("nan"),
                mean_strength_prior=float("nan"), mean_strength_continuing=float("nan"),
                reached=False,
            ))
            continue

        opp_mi = an.action_strength_mi(game, sol.strategy, resp_dp, reach)
        continue_actions = tuple(a for a in resp_dp.actions if a != "f")
        prior, continuing, fold_freq = _continuing_range(
            resp_dp, sol.strategy, reach, continue_actions
        )
        h_prior = an._entropy(prior)
        h_cont = an._entropy(continuing)
        kl = an._kl(continuing, prior) if fold_freq < 1.0 else float("inf")
        results.append(BetSizeResult(
            fraction=frac, label=bet_label, value_to_bettor=value_to_bettor,
            bet_frequency=bet_freq, bettor_mi=mi_open.mutual_information,
            fold_frequency=fold_freq, opp_mi=opp_mi.mutual_information,
            prior_entropy=h_prior, continuing_entropy=h_cont,
            entropy_drop=h_prior - h_cont, continuing_kl=kl,
            mean_strength_prior=_strength_dist_stats(prior, ranks),
            mean_strength_continuing=_strength_dist_stats(continuing, ranks),
        ))
    return results


# ---------------------------------------------------------------------------
# Measure 4 (range reading): exploitability of a restricted/condensed range
# ---------------------------------------------------------------------------


@dataclass
class RangeExploitResult:
    label: str
    value_uniform: float             # value to player with a uniform range
    value_restricted: float          # value with the restricted range
    ev_lost: float                   # uniform - restricted (>= 0 if shape hurts)
    mean_strength: float


def range_restriction_exploitability(
    base_config: GameConfig, player: int = 0,
    shapes: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> List[RangeExploitResult]:
    """EV of *optimally playing* a restricted range vs a uniform range.

    Unlike the strategy-flattening reading (in :mod:`analysis`), here the player
    keeps optimal play but their *deal* is constrained to a given shape; we
    re-solve exactly.  This isolates the value of the range *shape* itself.
    """
    n = base_config.num_ranks
    if shapes is None:
        shapes = {
            "condensed": rg.condensed_range(n),
            "polarized": rg.polarized_range(n),
        }
    uniform = rg.uniform_range(n)

    def solve_value(ranged: Tuple[float, ...]) -> float:
        if player == 0:
            weights = (ranged, uniform)
        else:
            weights = (uniform, ranged)
        sol = sf.solve(Game(_replace(base_config, deal_weights=weights)))
        return sol.value if player == 0 else -sol.value

    v_uniform = solve_value(uniform)
    out = []
    for label, shape in shapes.items():
        v = solve_value(shape)
        out.append(RangeExploitResult(
            label=label, value_uniform=v_uniform, value_restricted=v,
            ev_lost=v_uniform - v, mean_strength=rg.mean_strength(shape),
        ))
    return out


# ---------------------------------------------------------------------------
# small helper: dataclasses.replace for a frozen GameConfig
# ---------------------------------------------------------------------------


def _replace(cfg: GameConfig, **changes) -> GameConfig:
    import dataclasses
    return dataclasses.replace(cfg, **changes)
