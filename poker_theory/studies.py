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
from .best_response import best_response_value
from .game import Game, GameConfig, AKQJT9_RANKS


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
# The clairvoyance game: the penalty for a condensed range
# ---------------------------------------------------------------------------


def _bluff_catcher_ranges(ranks: Tuple[str, ...]):
    """(bettor, defender) deal weights for the pure clairvoyance game.

    Bettor is polarized onto the strongest (nuts) and weakest (air) rank; the
    defender is a single mid bluff-catcher that beats air and loses to the nuts.
    """
    n = len(ranks)
    bettor = [0.0] * n
    bettor[0] = 1.0           # nuts (strongest)
    bettor[-1] = 1.0          # air (weakest)
    defender = [0.0] * n
    defender[n // 2] = 1.0    # a single middle card
    return tuple(bettor), tuple(defender)


@dataclass
class ClairvoyancePoint:
    bet_fraction: float
    solver_value: float        # value to the polar bettor = penalty to defender
    closed_form: float         # s/(1+s)
    call_frequency: float      # defender's MDF at equilibrium (numeric)
    bluff_frequency: float     # bettor's air-bet frequency (numeric)


def clairvoyance_size_sweep(
    fractions: List[float], ranks: Tuple[str, ...] = AKQJT9_RANKS,
    ante: int = 1, stack: float = 100.0,
) -> List[ClairvoyancePoint]:
    """Solve the pure clairvoyance game across bet sizes; compare to closed form."""
    from . import clairvoyance as cl
    bettor_w, defender_w = _bluff_catcher_ranges(ranks)
    nuts, air, catcher = ranks[0], ranks[-1], ranks[len(ranks) // 2]
    out: List[ClairvoyancePoint] = []
    for s in fractions:
        cfg = GameConfig(ranks=ranks, suits=2, ante=ante, num_rounds=1,
                         bet_mode="no-limit", stack=stack, bet_fractions=(s,),
                         raise_fractions=(), allow_allin=False, max_raises=1,
                         deal_weights=(bettor_w, defender_w))
        game = Game(cfg)
        sol = sf.solve(game)
        bet_label = f"b{s:g}"
        call = bluff = float("nan")
        for h, probs in sol.strategy.items():
            p, own, comm, hist = an.parse_infoset(h)
            if p == 1 and own == catcher and hist == bet_label:
                call = probs.get("c", 0.0)
            if p == 0 and own == air and hist == "":
                bluff = probs.get(bet_label, 0.0)
        eq = cl.clairvoyant_equilibrium(s)
        out.append(ClairvoyancePoint(
            bet_fraction=s, solver_value=sol.value, closed_form=eq.value_to_bettor,
            call_frequency=call, bluff_frequency=bluff,
        ))
    return out


def condensation_penalty_surface(
    condensations: List[float], fractions: List[float],
    ranks: Tuple[str, ...] = AKQJT9_RANKS, ante: int = 1, stack: float = 100.0,
):
    """Penalty (value to a polar bettor) over (defender condensation, bet size).

    The bettor holds a fixed *polarized* range; the defender's range is swept from
    uniform (``d=0``) to fully condensed/middle (``d=1``) at fixed mean strength.
    Returns a 2-D array ``penalty[i, j]`` for condensation ``i`` and size ``j``.
    """
    import numpy as np
    n = len(ranks)
    bettor_w = rg.polarized_range(n)
    penalty = np.zeros((len(condensations), len(fractions)))
    for i, d in enumerate(condensations):
        defender_w = rg.condensation_family(n, d)
        for j, s in enumerate(fractions):
            cfg = GameConfig(ranks=ranks, suits=2, ante=ante, num_rounds=1,
                             bet_mode="no-limit", stack=stack, bet_fractions=(s,),
                             raise_fractions=(), allow_allin=False, max_raises=1,
                             deal_weights=(bettor_w, defender_w))
            penalty[i, j] = sf.solve(Game(cfg)).value
    return penalty


# ---------------------------------------------------------------------------
# The corrected measure: question value (VoI / answer partition) vs penalty
# ---------------------------------------------------------------------------


@dataclass
class QuestionPoint:
    """One (defender shape, bet size) cell of the question-value study."""

    label: str
    bet_fraction: float
    penalty: float            # equilibrium value to the polar bettor
    range_entropy: float      # H(defender deal weights), bits -- the OLD measure
    partition_entropy: float  # H(answer partition), bits
    voi: float                # value of the defender's card for the answer, chips
    indifferent_mass: float   # bluff-catcher mass held at indifference
    rent: float               # clairvoyance rent scale s/(1+s)
    predictor: float          # rent * indifferent_mass: the clairvoyance bound
    # The composite measure: max(0, rent * indifferent_mass - voi).  "What the
    # question would extract from a blind range, minus what the defender's card
    # buys back" -- clamped at zero because the question is optional (the bettor
    # can always check rather than pose a value-losing question).
    predictor_net: float


def question_value_point(
    defender_weights: Tuple[float, ...], s: float,
    bettor_weights: Optional[Tuple[float, ...]] = None,
    ranks: Tuple[str, ...] = AKQJT9_RANKS, ante: int = 1, stack: float = 100.0,
    label: str = "",
    raise_fractions: Tuple[float, ...] = (), max_raises: int = 1,
    num_rounds: int = 1,
) -> QuestionPoint:
    """Solve one game and measure the question the bet poses.

    Default bettor is pure nuts-or-air (mass on the strongest and weakest rank).
    ``raise_fractions``/``max_raises`` enrich the responder's answer alphabet
    (e.g. allowing a raise -> fold/call/raise); ``num_rounds`` adds streets.
    """
    n = len(ranks)
    if bettor_weights is None:
        bw = [0.0] * n
        bw[0] = bw[-1] = 1.0
        bettor_weights = tuple(bw)
    cfg = GameConfig(ranks=ranks, suits=2, ante=ante, num_rounds=num_rounds,
                     bet_mode="no-limit", stack=stack, bet_fractions=(s,),
                     raise_fractions=raise_fractions, allow_allin=False,
                     max_raises=max_raises,
                     deal_weights=(bettor_weights, tuple(defender_weights)))
    game = Game(cfg)
    sol = sf.solve(game)
    reach = an.ProfileReach(game, sol.strategy)
    values = an.node_values(game, sol.strategy)
    dp = _opponent_response_dp(game, bettor=0, bet_label=f"b{s:g}")
    if dp is None:
        raise RuntimeError(f"no response decision point for size {s}")
    rent = s / (1.0 + s)
    h_range = an._entropy({ranks[i]: w for i, w in enumerate(defender_weights)})
    try:
        qv = an.question_value(game, sol.strategy, dp, reach, values)
    except ValueError:
        # The bettor never bets at equilibrium: the question is worthless against
        # this defender and is never posed.  Measures are undefined; the rent
        # extracted (and hence the predictor) is zero.
        return QuestionPoint(
            label=label, bet_fraction=s, penalty=sol.value, range_entropy=h_range,
            partition_entropy=float("nan"), voi=float("nan"),
            indifferent_mass=float("nan"), rent=rent, predictor=0.0,
            predictor_net=0.0,
        )
    return QuestionPoint(
        label=label, bet_fraction=s, penalty=sol.value, range_entropy=h_range,
        partition_entropy=qv.partition_entropy, voi=qv.voi,
        indifferent_mass=qv.indifferent_mass, rent=rent,
        predictor=rent * qv.indifferent_mass,
        predictor_net=max(0.0, rent * qv.indifferent_mass - qv.voi),
    )


def shape_gallery(n: int = 6) -> Dict[str, Tuple[float, ...]]:
    """Defender range shapes, all with mean rank-strength fixed at the centre.

    Includes the counterexample to entropy-as-shape: ``mid-4 bluffcatchers`` has
    2 bits of range entropy yet is strategically identical to a single card.
    """
    assert n == 6, "gallery is written for the 6-rank game"
    return {
        "uniform": (1, 1, 1, 1, 1, 1),
        "condensed-mid": rg.condensed_range(n),
        "mid-4 bluffcatchers": (0, 1, 1, 1, 1, 0),
        "mid-2 bluffcatchers": (0, 0, 1, 1, 0, 0),
        "two-point K/T": (0, 1, 0, 0, 1, 0),
        "semi-polar": (0.35, 0.15, 0, 0, 0.15, 0.35),
        "polar A/9": (1, 0, 0, 0, 0, 1),
    }


def question_value_gallery(
    fractions: List[float], ranks: Tuple[str, ...] = AKQJT9_RANKS,
) -> List[QuestionPoint]:
    """Evaluate every gallery shape at every bet size (pure polar bettor)."""
    out: List[QuestionPoint] = []
    for label, w in shape_gallery(len(ranks)).items():
        for s in fractions:
            out.append(question_value_point(w, s, ranks=ranks, label=label))
    return out


def question_value_surface(
    condensations: List[float], fractions: List[float],
    ranks: Tuple[str, ...] = AKQJT9_RANKS,
):
    """Penalty + question measures over (defender condensation, bet size).

    Same setup as :func:`condensation_penalty_surface` (soft-polarized bettor),
    but also recording the new measures so the predictor can be tested on the
    whole surface.  Returns a dict of 2-D arrays keyed by measure name.
    """
    import numpy as np
    n = len(ranks)
    bettor = rg.polarized_range(n)
    shape = (len(condensations), len(fractions))
    grids = {k: np.zeros(shape) for k in
             ("penalty", "predictor", "predictor_net", "voi",
              "indifferent_mass", "range_entropy")}
    for i, d in enumerate(condensations):
        defender = rg.condensation_family(n, d)
        for j, s in enumerate(fractions):
            qp = question_value_point(defender, s, bettor_weights=bettor,
                                      ranks=ranks, label=f"d={d:g}")
            grids["penalty"][i, j] = qp.penalty
            grids["predictor"][i, j] = qp.predictor
            grids["predictor_net"][i, j] = qp.predictor_net
            grids["voi"][i, j] = qp.voi
            grids["indifferent_mass"][i, j] = qp.indifferent_mass
            grids["range_entropy"][i, j] = qp.range_entropy
    return grids


# ---------------------------------------------------------------------------
# Bet-sizing abstraction: how few sizes recover the continuum?
# ---------------------------------------------------------------------------


def sizing_value(
    sizes: Tuple[float, ...], bettor_weights: Tuple[float, ...],
    defender_weights: Tuple[float, ...], ranks: Tuple[str, ...] = AKQJT9_RANKS,
    ante: int = 1, stack: float = 50.0,
) -> float:
    """Value to the bettor (P0) when it may check or bet any size in ``sizes``.

    Single street; the defender may only call or fold (so the only sizing
    dimension in the game is the bettor's menu).  An empty menu means check-only.
    """
    common = dict(ranks=ranks, suits=2, ante=ante, num_rounds=1, bet_mode="no-limit",
                  stack=stack, raise_fractions=(), allow_allin=False,
                  deal_weights=(bettor_weights, defender_weights),
                  aggressors=(0,))  # defender is passive: check/call/fold only
    if not sizes:
        cfg = GameConfig(bet_fractions=(1.0,), max_raises=0, **common)
    else:
        cfg = GameConfig(bet_fractions=tuple(sorted(set(sizes))), max_raises=1, **common)
    return sf.solve(Game(cfg)).value


@dataclass
class AbstractionCurve:
    grid: List[float]
    v_check: float
    v_full: float
    # method -> list over k of (chosen sizes, value, capture fraction)
    chosen: Dict[str, List[Tuple[List[float], float, float]]]


def _capture(v: float, v_check: float, v_full: float) -> float:
    denom = v_full - v_check
    return 0.0 if abs(denom) < 1e-12 else (v - v_check) / denom


def greedy_sizes(
    grid: List[float], bettor_weights: Tuple[float, ...],
    defender_weights: Tuple[float, ...], k_max: int = 6, **kw,
) -> List[Tuple[List[float], float]]:
    """Forward-selection benchmark: repeatedly add the size that helps EV most."""
    chosen: List[float] = []
    out: List[Tuple[List[float], float]] = []
    remaining = list(grid)
    for _ in range(k_max):
        best_size, best_val = None, -1e18
        for s in remaining:
            v = sizing_value(tuple(chosen + [s]), bettor_weights, defender_weights, **kw)
            if v > best_val:
                best_val, best_size = v, s
        chosen = sorted(chosen + [best_size])
        remaining.remove(best_size)
        out.append((list(chosen), best_val))
    return out


def geometric_sizes(grid: List[float], k: int) -> List[float]:
    """k sizes geometrically spaced across the grid range (the folklore rule)."""
    import numpy as np
    lo, hi = grid[0], grid[-1]
    targets = np.geomspace(lo, hi, k)
    return sorted({min(grid, key=lambda g: abs(g - t)) for t in targets})


def linear_sizes(grid: List[float], k: int) -> List[float]:
    import numpy as np
    targets = np.linspace(grid[0], grid[-1], k)
    return sorted({min(grid, key=lambda g: abs(g - t)) for t in targets})


def sizing_abstraction_curve(
    bettor_weights: Tuple[float, ...], defender_weights: Tuple[float, ...],
    grid: Optional[List[float]] = None, k_max: int = 6, **kw,
) -> AbstractionCurve:
    """Compare greedy-optimal vs geometric vs linear size menus against the grid."""
    import numpy as np
    if grid is None:
        grid = [round(float(x), 3) for x in np.geomspace(0.1, 15.0, 24)]
    v_check = sizing_value((), bettor_weights, defender_weights, **kw)
    v_full = sizing_value(tuple(grid), bettor_weights, defender_weights, **kw)

    chosen: Dict[str, List[Tuple[List[float], float, float]]] = {}
    greedy = greedy_sizes(grid, bettor_weights, defender_weights, k_max, **kw)
    chosen["greedy"] = [(s, v, _capture(v, v_check, v_full)) for s, v in greedy]
    for name, fn in (("geometric", geometric_sizes), ("linear", linear_sizes)):
        rows = []
        for k in range(1, k_max + 1):
            sizes = fn(grid, k)
            v = sizing_value(tuple(sizes), bettor_weights, defender_weights, **kw)
            rows.append((sizes, v, _capture(v, v_check, v_full)))
        chosen[name] = rows
    return AbstractionCurve(grid=grid, v_check=v_check, v_full=v_full, chosen=chosen)


# ---------------------------------------------------------------------------
# Two-sided abstraction: exploitability of a translated bucket strategy
# ---------------------------------------------------------------------------


def translate(b: float, menu: List[float], rule: str = "nearest") -> Dict[float, float]:
    """Map an off-menu bet fraction ``b`` to a distribution over menu sizes.

    ``nearest``: round to the closest menu size.
    ``pseudoharmonic``: the Ganzfried-Sandholm randomized mapping, which makes a
    rational bettor closer to indifferent about choosing an intermediate size.
    """
    m = sorted(menu)
    if b <= m[0]:
        return {m[0]: 1.0}
    if b >= m[-1]:
        return {m[-1]: 1.0}
    # bracketing menu sizes A < b < B
    A = max(s for s in m if s <= b)
    B = min(s for s in m if s >= b)
    if A == B:
        return {A: 1.0}
    if rule == "nearest":
        return {A: 1.0} if (b - A) <= (B - b) else {B: 1.0}
    if rule == "pseudoharmonic":
        wA = ((B - b) * (1.0 + A)) / ((B - A) * (1.0 + b))
        return {A: wA, B: 1.0 - wA}
    raise ValueError(f"unknown translation rule {rule!r}")


def _defender_call_freqs(
    menu: List[float], bettor_w, defender_w, ranks, ante, stack,
) -> Dict[Tuple[str, float], float]:
    """GTO call frequency c*(hand, size) for the passive defender on menu ``menu``."""
    cfg = GameConfig(ranks=ranks, suits=2, ante=ante, num_rounds=1,
                     bet_mode="no-limit", stack=stack,
                     bet_fractions=tuple(sorted(menu)), raise_fractions=(),
                     allow_allin=False, max_raises=1, aggressors=(0,),
                     deal_weights=(bettor_w, defender_w))
    sol = sf.solve(Game(cfg))
    out: Dict[Tuple[str, float], float] = {}
    for h in ranks:
        for s in menu:
            key = f"P1|{h}|-|b{s:g}"
            out[(h, s)] = sol.strategy.get(key, {}).get("c", 0.0)
    return out


@dataclass
class AbstractionExploit:
    menu: List[float]
    rule: str
    value_full_gto: float     # bettor value vs a perfectly-defending opponent
    value_exploited: float    # bettor best-response value vs the bucket defender
    exploitability: float     # >= 0; extra chips the gap concedes


def abstraction_exploitability(
    menu: List[float], bettor_w, defender_w, grid: List[float],
    rule: str = "nearest", ranks: Tuple[str, ...] = AKQJT9_RANKS,
    ante: int = 1, stack: float = 50.0,
) -> AbstractionExploit:
    """How exploitable is a defender that only prepared responses for ``menu``?

    The defender computes its GTO defence on ``menu``, then faces a best-responder
    that may bet any size on the fine ``grid``; off-menu bets are handled by the
    translation ``rule``.  Exploitability is the extra value the best-responder
    extracts versus a defender that defends the full grid perfectly.
    """
    full_grid = sorted(set(grid) | set(menu))
    cfg_full = GameConfig(ranks=ranks, suits=2, ante=ante, num_rounds=1,
                          bet_mode="no-limit", stack=stack,
                          bet_fractions=tuple(full_grid), raise_fractions=(),
                          allow_allin=False, max_raises=1, aggressors=(0,),
                          deal_weights=(bettor_w, defender_w))
    game = Game(cfg_full)
    v_full = sf.solve(game).value

    cf = _defender_call_freqs(menu, bettor_w, defender_w, ranks, ante, stack)

    # build the bucket defender's behavioural strategy on the full-grid tree
    strat: Dict[str, Dict[str, float]] = {}
    for infoset in game.player_infosets[1]:
        _p, h, _comm, hist = an.parse_infoset(infoset)
        acts = game.infoset_actions[infoset]
        if hist.startswith("b") and "c" in acts and "f" in acts:
            b = float(hist[1:])
            tw = translate(b, menu, rule)
            cp = sum(w * cf[(h, s)] for s, w in tw.items())
            strat[infoset] = {"c": cp, "f": 1.0 - cp}
        else:
            strat[infoset] = {a: 1.0 / len(acts) for a in acts}

    v_exploited = best_response_value(game, strat, br_player=0)
    return AbstractionExploit(
        menu=sorted(menu), rule=rule, value_full_gto=v_full,
        value_exploited=v_exploited, exploitability=v_exploited - v_full,
    )


# ---------------------------------------------------------------------------
# small helper: dataclasses.replace for a frozen GameConfig
# ---------------------------------------------------------------------------


def _replace(cfg: GameConfig, **changes) -> GameConfig:
    import dataclasses
    return dataclasses.replace(cfg, **changes)
