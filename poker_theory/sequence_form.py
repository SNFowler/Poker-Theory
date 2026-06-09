"""Exact Nash equilibrium via sequence-form linear programming.

Implements the Koller--Megiddo--von Stengel sequence-form LP for two-player
zero-sum extensive-form games with perfect recall.  Solving the LP yields an
*exact* Nash equilibrium (a pair of realization plans) and the exact game value.

The realization plans are the raw material for everything downstream: behavioural
strategies, infoset reach probabilities, and the posterior beliefs that feed the
information measures.

Notation (von Stengel 1996)
---------------------------
Player 0 (the *row* / maximising player) chooses a realization plan ``x`` over
its sequences with ``E0 x = e0``, ``x >= 0``.  Player 1 (the *column* /
minimising player) chooses ``y`` with ``E1 y = e1``, ``y >= 0``.  The payoff to
player 0 is ``x^T A y``.  The value is obtained from the LP::

    max_{x,q}  e1^T q
    s.t.       A^T x - E1^T q >= 0
               E0 x = e0,  x >= 0

and ``y`` is recovered by solving the symmetric LP for player 1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .game import ChanceNode, Game, PlayerNode, TerminalNode

sys.setrecursionlimit(1_000_000)


@dataclass
class SequenceFormSolution:
    value: float                                   # game value to player 0
    realization: Dict[int, Dict[int, float]]       # player -> {seq_id: prob}
    strategy: Dict[str, Dict[str, float]]          # infoset -> {action: prob}
    seq_index: "SequenceIndex"


class SequenceIndex:
    """Enumerates player sequences and builds the constraint matrices."""

    def __init__(self, game: Game):
        self.game = game
        # seq id 0 is the empty sequence for each player.
        self.seq_label: Dict[int, List[Tuple[str, str]]] = {0: [("", "")], 1: [("", "")]}
        # (infoset, action) -> seq id, per player
        self.action_seq: Dict[int, Dict[Tuple[str, str], int]] = {0: {}, 1: {}}
        # infoset -> {action: seq id}
        self.infoset_seqs: Dict[str, Dict[str, int]] = {}
        # infoset -> parent sequence id (the seq leading into the infoset)
        self.infoset_parent: Dict[str, int] = {}
        # payoff contributions A[seq0, seq1] += reach * payoff
        self._A_entries: Dict[Tuple[int, int], float] = {}

        self._enumerate()
        self.A = self._build_payoff_matrix()
        self.E0, self.e0 = self._build_constraints(0)
        self.E1, self.e1 = self._build_constraints(1)

    # -- enumeration -------------------------------------------------------

    def _seq_for(self, player: int, infoset: str, action: str) -> int:
        key = (infoset, action)
        table = self.action_seq[player]
        if key not in table:
            sid = len(self.seq_label[player])
            self.seq_label[player].append(key)
            table[key] = sid
            self.infoset_seqs.setdefault(infoset, {})[action] = sid
        return table[key]

    def _enumerate(self) -> None:
        game = self.game

        def walk(idx: int, seq: Tuple[int, int], reach: float) -> None:
            node = game.nodes[idx]
            if isinstance(node, TerminalNode):
                if reach != 0.0 and node.payoff != 0.0:
                    key = seq
                    self._A_entries[key] = self._A_entries.get(key, 0.0) + reach * node.payoff
                return
            if isinstance(node, ChanceNode):
                for _label, prob, child in node.branches:
                    walk(child, seq, reach * prob)
                return
            assert isinstance(node, PlayerNode)
            p = node.player
            parent_seq = seq[p]
            # All nodes in an infoset share the same parent sequence (perfect recall).
            if node.infoset in self.infoset_parent:
                assert self.infoset_parent[node.infoset] == parent_seq, (
                    f"perfect-recall violation at {node.infoset}"
                )
            else:
                self.infoset_parent[node.infoset] = parent_seq
            for action, child in node.actions:
                sid = self._seq_for(p, node.infoset, action)
                new_seq = (sid, seq[1]) if p == 0 else (seq[0], sid)
                walk(child, new_seq, reach)

        walk(game.root, (0, 0), 1.0)

    # -- matrices ----------------------------------------------------------

    def num_seqs(self, player: int) -> int:
        return len(self.seq_label[player])

    def _build_payoff_matrix(self) -> sp.csr_matrix:
        n0, n1 = self.num_seqs(0), self.num_seqs(1)
        if self._A_entries:
            rows, cols, data = zip(*[(i, j, v) for (i, j), v in self._A_entries.items()])
            A = sp.coo_matrix((data, (rows, cols)), shape=(n0, n1)).tocsr()
        else:
            A = sp.csr_matrix((n0, n1))
        return A

    def _build_constraints(self, player: int) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Build E (rows: root + one per infoset) and e for ``player``."""
        infosets = self.game.player_infosets[player]
        n_seq = self.num_seqs(player)
        row_of_infoset = {h: i + 1 for i, h in enumerate(infosets)}
        m = 1 + len(infosets)

        rows: List[int] = [0]
        cols: List[int] = [0]
        data: List[float] = [1.0]  # root: x[empty] = 1

        for h in infosets:
            r = row_of_infoset[h]
            parent = self.infoset_parent[h]
            rows.append(r)
            cols.append(parent)
            data.append(-1.0)
            for _action, sid in self.infoset_seqs[h].items():
                rows.append(r)
                cols.append(sid)
                data.append(1.0)

        E = sp.coo_matrix((data, (rows, cols)), shape=(m, n_seq)).tocsr()
        e = np.zeros(m)
        e[0] = 1.0
        return E, e


def _solve_oneside(
    A: sp.csr_matrix,
    E_row: sp.csr_matrix,
    e_row: np.ndarray,
    E_col: sp.csr_matrix,
    e_col: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Solve the LP for the *row* (maximising) player.

    Returns ``(value_to_row_player, row_realization)``.
    """
    n_row = A.shape[0]
    m_col = E_col.shape[0]

    # variables z = [x (n_row); q (m_col)]
    c = np.concatenate([np.zeros(n_row), -e_col])  # maximise e_col^T q

    # inequality: -A^T x + E_col^T q <= 0
    A_ub = sp.hstack([-A.T, E_col.T]).tocsr()
    b_ub = np.zeros(A.shape[1])

    # equality: E_row x = e_row  (q columns zero)
    A_eq = sp.hstack([E_row, sp.csr_matrix((E_row.shape[0], m_col))]).tocsr()
    b_eq = e_row

    bounds = [(0, None)] * n_row + [(None, None)] * m_col

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    value = -res.fun  # we minimised -e_col^T q
    x = res.x[:n_row]
    return value, x


def solve(game: Game) -> SequenceFormSolution:
    """Solve ``game`` exactly and return realization plans + behavioural strategy."""
    idx = SequenceIndex(game)
    A = idx.A

    v0, x = _solve_oneside(A, idx.E0, idx.e0, idx.E1, idx.e1)
    # Player 1 maximises payoff matrix M = -A^T (rows indexed by player-1 seqs).
    M = (-A.T).tocsr()
    v1, y = _solve_oneside(M, idx.E1, idx.e1, idx.E0, idx.e0)

    if abs(v0 + v1) > 1e-6:
        raise RuntimeError(
            f"value mismatch between primal/dual solves: v0={v0}, v1={v1}"
        )

    realization = {
        0: {i: float(x[i]) for i in range(len(x))},
        1: {j: float(y[j]) for j in range(len(y))},
    }
    strategy = _behavioural_strategy(idx, realization)
    return SequenceFormSolution(value=v0, realization=realization,
                                strategy=strategy, seq_index=idx)


def _behavioural_strategy(
    idx: SequenceIndex, realization: Dict[int, Dict[int, float]]
) -> Dict[str, Dict[str, float]]:
    """Convert realization plans to behavioural probabilities at each infoset."""
    strat: Dict[str, Dict[str, float]] = {}
    for player in (0, 1):
        real = realization[player]
        for h in idx.game.player_infosets[player]:
            parent = idx.infoset_parent[h]
            denom = real.get(parent, 0.0)
            actions = idx.infoset_seqs[h]
            probs: Dict[str, float] = {}
            if denom > 1e-12:
                for action, sid in actions.items():
                    probs[action] = real.get(sid, 0.0) / denom
            else:
                # unreached infoset: default to uniform for a fully-specified strategy
                n = len(actions)
                for action in actions:
                    probs[action] = 1.0 / n
            strat[h] = probs
    return strat
