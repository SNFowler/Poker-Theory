"""CFR+ solver as an independent cross-check (and for finer sizing grids).

A compact implementation of Counterfactual Regret Minimisation with the CFR+
modifications (regret matching with non-negative regret flooring and linear
strategy averaging).  CFR+ converges to a Nash equilibrium of the two-player
zero-sum game and serves two purposes here:

* an *independent* check that the sequence-form LP found the right value /
  strategy (they should agree to within CFR's convergence tolerance), and
* a fallback solver for fine sizing grids where the LP becomes large.

The public :func:`train` returns the average strategy in the same
``infoset -> {action: prob}`` format produced by the LP solver, so it is a drop-in
for the best-response and measure code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .game import ChanceNode, Game, PlayerNode, TerminalNode

Strategy = Dict[str, Dict[str, float]]


@dataclass
class CFRState:
    regret_sum: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strategy_sum: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _ensure(state: CFRState, infoset: str, actions: List[str]) -> None:
    if infoset not in state.regret_sum:
        state.regret_sum[infoset] = {a: 0.0 for a in actions}
        state.strategy_sum[infoset] = {a: 0.0 for a in actions}


def _current_strategy(state: CFRState, infoset: str, actions: List[str]) -> Dict[str, float]:
    regrets = state.regret_sum[infoset]
    pos = {a: max(regrets[a], 0.0) for a in actions}
    total = sum(pos.values())
    if total > 0:
        return {a: pos[a] / total for a in actions}
    return {a: 1.0 / len(actions) for a in actions}


def _cfr(game: Game, state: CFRState, idx: int,
         pi0: float, pi1: float, pc: float, weight: float, updating: int) -> float:
    """Recursive CFR pass (alternating updates); returns payoff to player 0.

    Only the regrets and strategy average of ``updating`` are touched on this
    pass -- this is the canonical CFR+ alternating-update scheme, which
    converges markedly faster than simultaneous updates.
    """
    node = game.nodes[idx]
    if isinstance(node, TerminalNode):
        return node.payoff
    if isinstance(node, ChanceNode):
        return sum(prob * _cfr(game, state, child, pi0, pi1, pc * prob, weight, updating)
                   for _l, prob, child in node.branches)

    assert isinstance(node, PlayerNode)
    infoset = node.infoset
    actions = game.infoset_actions[infoset]
    _ensure(state, infoset, actions)
    strat = _current_strategy(state, infoset, actions)

    util_action: Dict[str, float] = {}
    node_util = 0.0
    for action, child in node.actions:
        if node.player == 0:
            u = _cfr(game, state, child, pi0 * strat[action], pi1, pc, weight, updating)
        else:
            u = _cfr(game, state, child, pi0, pi1 * strat[action], pc, weight, updating)
        util_action[action] = u
        node_util += strat[action] * u

    if node.player == updating:
        cf_reach = (pi1 * pc) if node.player == 0 else (pi0 * pc)
        sign = 1.0 if node.player == 0 else -1.0
        regrets = state.regret_sum[infoset]
        strat_sum = state.strategy_sum[infoset]
        own_reach = pi0 if node.player == 0 else pi1
        for action in actions:
            regret = sign * (util_action[action] - node_util) * cf_reach
            # CFR+: floor accumulated regret at zero.
            regrets[action] = max(0.0, regrets[action] + regret)
            strat_sum[action] += weight * own_reach * strat[action]
    return node_util


def average_strategy(state: CFRState, game: Game) -> Strategy:
    strat: Strategy = {}
    for player in (0, 1):
        for infoset in game.player_infosets[player]:
            actions = game.infoset_actions[infoset]
            if infoset not in state.strategy_sum:
                strat[infoset] = {a: 1.0 / len(actions) for a in actions}
                continue
            ssum = state.strategy_sum[infoset]
            total = sum(ssum.values())
            if total > 0:
                strat[infoset] = {a: ssum[a] / total for a in actions}
            else:
                strat[infoset] = {a: 1.0 / len(actions) for a in actions}
    return strat


def train(game: Game, iterations: int = 1000, verbose: bool = False) -> Strategy:
    """Run CFR+ for ``iterations`` and return the average strategy."""
    state = CFRState()
    for t in range(1, iterations + 1):
        # CFR+ linear averaging: weight by iteration index; alternate updates.
        for updating in (0, 1):
            _cfr(game, state, game.root, 1.0, 1.0, 1.0, weight=float(t), updating=updating)
        if verbose and (t % max(1, iterations // 10) == 0):
            from .best_response import exploitability
            _, expl = exploitability(game, average_strategy(state, game))
            print(f"  iter {t:5d}  exploitability = {expl:.6f}")
    return average_strategy(state, game)
