"""Best response and exploitability for the extensive-form game.

Given a fixed behavioural strategy profile, we compute each player's exact best
response value.  The gap between the best-response value and the game value
measures how far a profile is from equilibrium; it is the primary convergence
check for CFR and the engine behind measure D (exploitability under a range
constraint).

The best-response algorithm is the standard tabular one for imperfect-information
games: an opponent/chance *reach* is computed top-down, and the best response
action at each infoset is the argmax over actions of the reach-weighted
continuation value, with continuation values memoised so each infoset's choice
depends only on (strictly deeper) infosets.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .game import ChanceNode, Game, PlayerNode, TerminalNode

Strategy = Dict[str, Dict[str, float]]


class _BestResponder:
    def __init__(self, game: Game, opp_strategy: Strategy, br_player: int):
        self.game = game
        self.opp = opp_strategy
        self.br_player = br_player
        # reach contributed by chance + opponent (NOT the BR player)
        self.reach_opp: Dict[int, float] = {}
        # nodes belonging to each BR infoset, with their opp reach
        self.infoset_nodes: Dict[str, list] = {}
        self._best_action: Dict[str, str] = {}
        self._value_cache: Dict[int, float] = {}

        self._compute_reach(game.root, 1.0)

    def _sign(self) -> float:
        # payoffs are stored to player 0; convert to BR player's perspective
        return 1.0 if self.br_player == 0 else -1.0

    def _compute_reach(self, idx: int, reach: float) -> None:
        node = self.game.nodes[idx]
        self.reach_opp[idx] = self.reach_opp.get(idx, 0.0) + reach
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for _label, prob, child in node.branches:
                self._compute_reach(child, reach * prob)
            return
        assert isinstance(node, PlayerNode)
        if node.player == self.br_player:
            self.infoset_nodes.setdefault(node.infoset, []).append(idx)
            for _action, child in node.actions:
                self._compute_reach(child, reach)  # BR action prob is counterfactual
        else:
            probs = self.opp[node.infoset]
            for action, child in node.actions:
                self._compute_reach(child, reach * probs.get(action, 0.0))

    def _value(self, idx: int) -> float:
        """Continuation value to BR player under opp strategy + BR best play."""
        if idx in self._value_cache:
            return self._value_cache[idx]
        node = self.game.nodes[idx]
        if isinstance(node, TerminalNode):
            val = node.payoff * self._sign()
        elif isinstance(node, ChanceNode):
            val = sum(prob * self._value(child) for _l, prob, child in node.branches)
        elif node.player == self.br_player:
            action = self._best_response_action(node.infoset)
            child = dict(node.actions)[action]
            val = self._value(child)
        else:
            probs = self.opp[node.infoset]
            val = sum(probs.get(a, 0.0) * self._value(child) for a, child in node.actions)
        self._value_cache[idx] = val
        return val

    def _best_response_action(self, infoset: str) -> str:
        if infoset in self._best_action:
            return self._best_action[infoset]
        nodes = self.infoset_nodes[infoset]
        action_labels = self.game.infoset_actions[infoset]
        best_label, best_val = None, float("-inf")
        for action in action_labels:
            total = 0.0
            for n in nodes:
                child = dict(self.game.nodes[n].actions)[action]
                total += self.reach_opp[n] * self._value(child)
            if total > best_val:
                best_val, best_label = total, action
        self._best_action[infoset] = best_label
        return best_label

    def value(self) -> float:
        return self._value(self.game.root)

    def policy(self) -> Strategy:
        pol: Strategy = {}
        for infoset in self.infoset_nodes:
            a = self._best_response_action(infoset)
            pol[infoset] = {
                act: (1.0 if act == a else 0.0)
                for act in self.game.infoset_actions[infoset]
            }
        return pol


def best_response_value(game: Game, opp_strategy: Strategy, br_player: int) -> float:
    """Value to ``br_player`` of best-responding to a fixed opponent strategy."""
    return _BestResponder(game, opp_strategy, br_player).value()


def best_response_policy(game: Game, opp_strategy: Strategy, br_player: int) -> Strategy:
    return _BestResponder(game, opp_strategy, br_player).policy()


def exploitability(game: Game, strategy: Strategy) -> Tuple[float, float]:
    """Return ``(nash_conv, exploitability)`` for a full strategy profile.

    ``nash_conv = u0_BR + u1_BR`` (>= 0, zero iff Nash);
    ``exploitability = nash_conv / 2`` is the average per-player gain from
    unilaterally best-responding.
    """
    u0 = _BestResponder(game, strategy, br_player=0).value()
    u1 = _BestResponder(game, strategy, br_player=1).value()
    nash_conv = u0 + u1
    return nash_conv, nash_conv / 2.0
