"""Reach / posterior extraction and the four information measures.

Everything here operates on a *behavioural strategy profile* (the
``infoset -> {action: prob}`` dict produced by either the LP or CFR solver) and
the game tree.  It provides:

* node and infoset **reach probabilities** under the profile;
* **decision points** -- the set of infosets that share a public state but differ
  in the acting player's private rank -- and the prior over the actor's rank
  there;
* **posterior** beliefs over a player's private rank after a particular action;
* the four measures from the brief:

  1. :func:`range_shape_ev` -- equilibrium EV as a function of a range-shape
     (polarization) parameter (re-solves the game per range; see
     :mod:`poker_theory.studies`).
  2. :func:`action_strength_mi` -- ``I(action; private rank)`` at a decision
     point: how much a betting line leaks about the hand.
  3. :func:`posterior_collapse` -- per-action posterior entropy drop / KL: how
     sharply each bet size collapses the opponent's belief.
  4. :func:`flat_range_exploitability` -- EV lost when a player is forced onto a
     rank-blind ("condensed") strategy, a single "how exploitable is this shape"
     number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .best_response import best_response_value
from .game import ChanceNode, Game, PlayerNode, TerminalNode

Strategy = Dict[str, Dict[str, float]]

LOG2 = math.log(2.0)


def _entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy in bits of a (sub)distribution; ignores zero mass."""
    total = sum(dist.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for p in dist.values():
        if p > 0:
            q = p / total
            h -= q * math.log(q) / LOG2
    return h


def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(p || q) in bits.  p, q are distributions over the same keys."""
    kl = 0.0
    for k, pk in p.items():
        if pk > 0:
            qk = q.get(k, 0.0)
            if qk <= 0:
                return math.inf
            kl += pk * math.log(pk / qk) / LOG2
    return kl


def parse_infoset(key: str) -> Tuple[int, str, str, str]:
    """Return ``(player, own_rank, community, history)`` from an infoset key."""
    player_s, own, comm, hist = key.split("|")
    return int(player_s[1:]), own, comm, hist


def public_key(key: str) -> Tuple[int, str, str]:
    """Public part of an infoset: ``(player, community, history)`` (rank-agnostic)."""
    player, _own, comm, hist = parse_infoset(key)
    return (player, comm, hist)


# ---------------------------------------------------------------------------
# Reach probabilities under a profile
# ---------------------------------------------------------------------------


class ProfileReach:
    """Reach probabilities of every node / infoset under a strategy profile."""

    def __init__(self, game: Game, strategy: Strategy):
        self.game = game
        self.strategy = strategy
        self.node_reach: Dict[int, float] = {}
        self._compute(game.root, 1.0)
        self.infoset_reach: Dict[str, float] = {}
        for infoset, nodes in game.infoset_nodes.items():
            self.infoset_reach[infoset] = sum(self.node_reach.get(n, 0.0) for n in nodes)

    def _compute(self, idx: int, reach: float) -> None:
        self.node_reach[idx] = self.node_reach.get(idx, 0.0) + reach
        node = self.game.nodes[idx]
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for _label, prob, child in node.branches:
                self._compute(child, reach * prob)
            return
        probs = self.strategy[node.infoset]
        for action, child in node.actions:
            p = probs.get(action, 0.0)
            if p > 0:
                self._compute(child, reach * p)


# ---------------------------------------------------------------------------
# Decision points
# ---------------------------------------------------------------------------


@dataclass
class DecisionPoint:
    """A public spot where ``player`` acts, grouped over their private rank."""

    player: int
    community: str
    history: str
    # rank -> infoset key
    infoset_by_rank: Dict[str, str]
    actions: List[str]

    @property
    def label(self) -> str:
        comm = self.community if self.community != "-" else "preflop"
        hist = self.history if self.history else "(open)"
        return f"P{self.player} @ comm={comm} hist={hist}"


def decision_points(game: Game) -> List[DecisionPoint]:
    """Group infosets sharing a public state into decision points."""
    groups: Dict[Tuple[int, str, str], Dict[str, str]] = {}
    actions: Dict[Tuple[int, str, str], List[str]] = {}
    for player in (0, 1):
        for infoset in game.player_infosets[player]:
            p, own, comm, hist = parse_infoset(infoset)
            key = (p, comm, hist)
            groups.setdefault(key, {})[own] = infoset
            actions[key] = game.infoset_actions[infoset]
    out = []
    for (p, comm, hist), by_rank in groups.items():
        out.append(DecisionPoint(player=p, community=comm, history=hist,
                                 infoset_by_rank=by_rank, actions=actions[(p, comm, hist)]))
    return out


def rank_prior(dp: DecisionPoint, reach: ProfileReach) -> Dict[str, float]:
    """Prior P(actor's private rank | this public state is reached)."""
    weights = {r: reach.infoset_reach.get(h, 0.0) for r, h in dp.infoset_by_rank.items()}
    total = sum(weights.values())
    if total <= 0:
        return {r: 0.0 for r in weights}
    return {r: w / total for r, w in weights.items()}


# ---------------------------------------------------------------------------
# Measure 2: action <-> strength mutual information
# ---------------------------------------------------------------------------


@dataclass
class MIResult:
    decision: str
    mutual_information: float    # bits
    action_entropy: float        # H(action), bits
    conditional_entropy: float   # H(action | rank), bits
    prior: Dict[str, float]
    action_marginal: Dict[str, float]
    reach: float                 # probability this decision point is reached


def action_strength_mi(
    game: Game, strategy: Strategy, dp: DecisionPoint,
    reach: Optional[ProfileReach] = None,
) -> MIResult:
    """``I(action ; private rank)`` at a decision point (in bits)."""
    if reach is None:
        reach = ProfileReach(game, strategy)
    prior = rank_prior(dp, reach)
    actions = dp.actions

    # conditional action distributions q(a | rank)
    cond: Dict[str, Dict[str, float]] = {}
    for r, infoset in dp.infoset_by_rank.items():
        cond[r] = {a: strategy[infoset].get(a, 0.0) for a in actions}

    marginal = {a: sum(prior[r] * cond[r][a] for r in prior) for a in actions}

    h_action = _entropy(marginal)
    h_cond = sum(prior[r] * _entropy(cond[r]) for r in prior)
    mi = h_action - h_cond
    dp_reach = sum(reach.infoset_reach.get(h, 0.0) for h in dp.infoset_by_rank.values())
    return MIResult(
        decision=dp.label, mutual_information=mi, action_entropy=h_action,
        conditional_entropy=h_cond, prior=prior, action_marginal=marginal,
        reach=dp_reach,
    )


# ---------------------------------------------------------------------------
# Measure 3: posterior collapse per action (bet size)
# ---------------------------------------------------------------------------


@dataclass
class PosteriorResult:
    action: str
    action_prob: float              # P(action) -- how often this "question" is posed
    posterior: Dict[str, float]      # P(rank | action)
    prior_entropy: float             # H(S), bits
    posterior_entropy: float         # H(S | action=B), bits
    entropy_drop: float              # H(S) - H(S | action=B), bits
    kl_to_prior: float               # KL(posterior || prior), bits


def posterior_collapse(
    game: Game, strategy: Strategy, dp: DecisionPoint,
    reach: Optional[ProfileReach] = None,
) -> List[PosteriorResult]:
    """Per-action posterior over the actor's rank and its entropy collapse.

    For each action ``B`` available at ``dp`` we form the opponent's posterior
    ``P(rank | action=B) ∝ P(rank) P(B | rank)`` and report the entropy drop and
    KL divergence from the prior.  The action-probability-weighted mean of the
    entropy drops equals the mutual information of measure 2.
    """
    if reach is None:
        reach = ProfileReach(game, strategy)
    prior = rank_prior(dp, reach)
    h_prior = _entropy(prior)
    cond = {r: {a: strategy[dp.infoset_by_rank[r]].get(a, 0.0) for a in dp.actions}
            for r in prior}
    marginal = {a: sum(prior[r] * cond[r][a] for r in prior) for a in dp.actions}

    results = []
    for a in dp.actions:
        pa = marginal[a]
        if pa <= 0:
            posterior = {r: 0.0 for r in prior}
            h_post = 0.0
            drop = 0.0
            kl = 0.0
        else:
            posterior = {r: prior[r] * cond[r][a] / pa for r in prior}
            h_post = _entropy(posterior)
            drop = h_prior - h_post
            kl = _kl(posterior, prior)
        results.append(PosteriorResult(
            action=a, action_prob=pa, posterior=posterior, prior_entropy=h_prior,
            posterior_entropy=h_post, entropy_drop=drop, kl_to_prior=kl,
        ))
    return results


# ---------------------------------------------------------------------------
# Measure 4: exploitability of a flat / condensed (rank-blind) strategy
# ---------------------------------------------------------------------------


def flatten_strategy(
    game: Game, strategy: Strategy, player: int,
    reach: Optional[ProfileReach] = None,
) -> Strategy:
    """Return ``strategy`` with ``player`` forced to be *rank-blind*.

    At every public decision the player is made to play the reach-weighted
    average of its equilibrium action distribution across ranks -- i.e. it can no
    longer condition on its own hand strength.  This is the maximally
    "condensed" version of the player's strategy: a range that cannot
    differentiate.
    """
    if reach is None:
        reach = ProfileReach(game, strategy)
    flat = dict(strategy)
    for dp in decision_points(game):
        if dp.player != player:
            continue
        prior = rank_prior(dp, reach)
        actions = dp.actions
        if sum(prior.values()) <= 0:
            blended = {a: 1.0 / len(actions) for a in actions}
        else:
            blended = {a: sum(prior[r] * strategy[dp.infoset_by_rank[r]].get(a, 0.0)
                              for r in prior) for a in actions}
            s = sum(blended.values())
            blended = {a: (blended[a] / s if s > 0 else 1.0 / len(actions)) for a in actions}
        for r, infoset in dp.infoset_by_rank.items():
            flat[infoset] = dict(blended)
    return flat


def flat_range_exploitability(
    game: Game, strategy: Strategy, player: int,
    equilibrium_value_p0: float,
) -> Dict[str, float]:
    """EV lost when ``player`` is forced onto a rank-blind (condensed) strategy.

    Returns the player's equilibrium value, the value of the flattened strategy
    against a best-responding opponent, and the gap (>= 0).
    """
    reach = ProfileReach(game, strategy)
    flat = flatten_strategy(game, strategy, player, reach)
    opp = 1 - player
    # opponent best-responds to the flattened player
    opp_value = best_response_value(game, flat, br_player=opp)
    # value to `player` of the flattened strategy = -opp_value (zero sum)
    flat_value_player = -opp_value
    eq_value_player = equilibrium_value_p0 if player == 0 else -equilibrium_value_p0
    gap = eq_value_player - flat_value_player
    return {
        "equilibrium_value": eq_value_player,
        "flat_value": flat_value_player,
        "exploitability_gap": gap,
    }
