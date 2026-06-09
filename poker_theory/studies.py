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
# Bet-size EV decomposition: separating the range-collapse term
# ---------------------------------------------------------------------------


@dataclass
class DecompResult:
    """Exact additive decomposition of EV(B) for a single bet size B.

    ``ev_total == v_check + v_betfold + v_betcall`` (reach-weighted contributions
    to the game value, so they sum exactly).  The fold/continue split and the
    per-continue continuation value isolate where the bet's value comes from.
    """

    fraction: float
    label: str
    ev_total: float
    v_check: float                 # value contributed by the check line
    v_betfold: float               # fold equity: bet -> opponent folds
    v_betcall: float               # value when the bet is called (round-2 outcomes)
    p_bet: float                   # P(bettor bets)
    p_fold_given_bet: float
    p_call_given_bet: float
    per_continue_value: float      # v_betcall / P(bet & call): EV per called hand
    vbar_continuing: float         # per-continue value vs the *continuing* range
    vbar_full: float               # per-continue value vs the *full* range (frozen play)
    showdown_toughness: float      # vbar_full - vbar_continuing (>0: continuers tougher)
    prior_entropy: float
    continuing_entropy: float
    entropy_drop: float            # how condensed the continuing range is
    prior_strength: float
    continuing_strength: float


def _partition_ev(game: Game, strategy, bettor: int, bet_label: str):
    """Reach-weighted EV contributions of the {check, bet-fold, bet-call} lines.

    Returns (values, probs) dicts keyed by the three line tags.
    """
    from .game import ChanceNode, PlayerNode, TerminalNode
    vals = {"check": 0.0, "betfold": 0.0, "betcall": 0.0}
    probs = {"check": 0.0, "betfold": 0.0, "betcall": 0.0}

    def child_tag(node: PlayerNode, action: str, tag):
        if tag is None and node.player == bettor:          # bettor's opening
            return "check" if action != bet_label else "bet_pending"
        if tag == "bet_pending" and node.player != bettor:  # opponent's reply
            return "betfold" if action == "f" else "betcall"
        return tag

    def rec(idx: int, reach: float, tag) -> None:
        node = game.nodes[idx]
        if isinstance(node, TerminalNode):
            if tag in vals:
                vals[tag] += reach * node.payoff
                probs[tag] += reach
            return
        if isinstance(node, ChanceNode):
            for _l, prob, child in node.branches:
                rec(child, reach * prob, tag)
            return
        sp = strategy[node.infoset]
        for action, child in node.actions:
            p = sp.get(action, 0.0)
            if p > 0:
                rec(child, reach * p, child_tag(node, action, tag))

    rec(game.root, 1.0, None)
    return vals, probs


def bet_size_decomposition(
    base_config: GameConfig, fractions: List[float], bettor: int = 0,
) -> List[DecompResult]:
    """Decompose EV(B) into fold equity vs continuation value, and probe whether
    the bet *degrades* the opponent's continuing range (range-collapse term)."""
    results: List[DecompResult] = []
    ranks = base_config.ranks
    n = len(ranks)
    idx_of = {r: i for i, r in enumerate(ranks)}
    opp = 1 - bettor

    for frac in fractions:
        cfg = _replace(
            base_config, bet_mode="no-limit", bet_fractions=(frac,),
            raise_fractions=(), allow_allin=False, max_raises=1,
        )
        game = Game(cfg)
        sol = sf.solve(game)
        reach = an.ProfileReach(game, sol.strategy)
        values = an.node_values(game, sol.strategy)
        ev_total = sol.value if bettor == 0 else -sol.value
        sgn = 1.0 if bettor == 0 else -1.0  # convert P0-values to bettor-values

        open_dp = _bettor_open_dp(game, bettor)
        bet_labels = [a for a in open_dp.actions if a not in ("x", "c", "f")]
        bet_label = bet_labels[0] if bet_labels else None
        resp_dp = _opponent_response_dp(game, bettor, bet_label) if bet_label else None
        if resp_dp is None:
            continue

        vals, probs = _partition_ev(game, sol.strategy, bettor, bet_label)
        v_check = sgn * vals["check"]
        v_betfold = sgn * vals["betfold"]
        v_betcall = sgn * vals["betcall"]
        p_bet = probs["betfold"] + probs["betcall"]
        p_fold = probs["betfold"] / p_bet if p_bet > 0 else float("nan")
        p_call = probs["betcall"] / p_bet if p_bet > 0 else float("nan")
        per_continue = (v_betcall / probs["betcall"]) if probs["betcall"] > 0 else float("nan")

        # Per-rank continuation value to the bettor, and continuing vs full range.
        continue_actions = [a for a in resp_dp.actions if a != "f"]
        v2: Dict[str, float] = {}
        W: Dict[str, float] = {}
        callprob: Dict[str, float] = {}
        for r, infoset in resp_dp.infoset_by_rank.items():
            num = 0.0
            wtot = 0.0
            for ndx in game.infoset_nodes[infoset]:
                node = game.nodes[ndx]
                child = dict(node.actions).get("c")
                if child is None:
                    continue
                wn = reach.node_reach.get(ndx, 0.0)
                num += wn * values[child]
                wtot += wn
            v2[r] = (num / wtot) if wtot > 0 else 0.0
            W[r] = wtot
            callprob[r] = sum(sol.strategy[infoset].get(a, 0.0) for a in continue_actions)

        cont_den = sum(callprob[r] * W[r] for r in v2)
        full_den = sum(W[r] for r in v2)
        vbar_cont = (sgn * sum(callprob[r] * W[r] * v2[r] for r in v2) / cont_den) if cont_den > 0 else float("nan")
        vbar_full = (sgn * sum(W[r] * v2[r] for r in v2) / full_den) if full_den > 0 else float("nan")

        prior, continuing, _fold = _continuing_range(
            resp_dp, sol.strategy, reach, tuple(continue_actions)
        )
        h_prior = an._entropy(prior)
        h_cont = an._entropy(continuing)
        results.append(DecompResult(
            fraction=frac, label=bet_label, ev_total=ev_total,
            v_check=v_check, v_betfold=v_betfold, v_betcall=v_betcall,
            p_bet=p_bet, p_fold_given_bet=p_fold, p_call_given_bet=p_call,
            per_continue_value=per_continue, vbar_continuing=vbar_cont, vbar_full=vbar_full,
            showdown_toughness=vbar_full - vbar_cont,
            prior_entropy=h_prior, continuing_entropy=h_cont, entropy_drop=h_prior - h_cont,
            prior_strength=_strength_dist_stats(prior, ranks),
            continuing_strength=_strength_dist_stats(continuing, ranks),
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
