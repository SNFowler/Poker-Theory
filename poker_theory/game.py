"""Extensive-form game model for Leduc-style AKQJT9 poker.

This module builds a fully-expanded extensive-form game (EFG) tree for a
two-player, zero-sum, imperfect-information poker variant that generalises
Leduc Hold'em to an arbitrary set of ranks and a configurable betting model.

Key design goals
----------------
* **Exactly solvable on CPU.**  The tree is built combinatorially over *ranks*
  (suits are strategically irrelevant here and are integrated out exactly via
  card-removal probabilities), keeping the tree small enough for an exact
  sequence-form LP while remaining a faithful model of the game.
* **Pluggable betting model.**  The bet-size *menu* is a parameter of
  :class:`GameConfig` -- the whole point of the research programme is to sweep
  it.  We support both classic *fixed-limit* Leduc (for sanity-checking against
  known results) and *no-limit* multi-size betting (pot fractions + all-in).
* **Information sets keyed by observables.**  A player's information set is the
  pair *(own private rank, public state)* where the public state is the
  community rank (once revealed) plus the full public betting history.  Because
  payoffs and card-removal probabilities depend only on ranks, keying infosets
  by rank is a *lossless* abstraction.

The resulting :class:`Game` object is consumed by the sequence-form LP solver,
the CFR solver, the best-response routine, and the information-measure analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Canonical rank orderings.  Index 0 is the *strongest* rank.
LEDUC_RANKS = ("K", "Q", "J")          # classic Leduc Hold'em
AKQJT9_RANKS = ("A", "K", "Q", "J", "T", "9")


@dataclass(frozen=True)
class GameConfig:
    """Static description of a Leduc-style game.

    Parameters
    ----------
    ranks:
        Tuple of rank labels, strongest first.  ``("K","Q","J")`` reproduces
        classic Leduc; ``("A","K","Q","J","T","9")`` is the AKQJT9 game.
    suits:
        Number of suits (copies) of each rank.  Standard Leduc uses 2.
    ante:
        Forced ante per player (pot starts at ``2*ante``).
    bet_mode:
        ``"fixed-limit"`` or ``"no-limit"``.
    fixed_bets:
        For fixed-limit: the bet/raise increment in each round, e.g. ``(2, 4)``
        for classic Leduc.
    max_raises:
        Maximum number of aggressive actions (bet counts as the first raise)
        per round.
    stack:
        Total chips each player has available *including* the ante.  Bounds
        no-limit bet sizes.  Ignored (treated as effectively infinite) in
        fixed-limit mode.
    bet_fractions / raise_fractions:
        For no-limit: candidate bet/raise sizes as fractions of the pot.
    allow_allin:
        Whether an explicit all-in shove is always offered in no-limit.
    """

    ranks: Tuple[str, ...] = AKQJT9_RANKS
    suits: int = 2
    ante: int = 1
    bet_mode: str = "no-limit"

    # Number of betting rounds.  2 = full Leduc (private + community card).
    # 1 = single-round game (deal, one betting round, showdown on private ranks,
    # no community card) -- the simplest case for the bet-sizing study.
    num_rounds: int = 2

    # fixed-limit parameters
    fixed_bets: Tuple[int, ...] = (2, 4)
    max_raises: int = 2

    # no-limit parameters
    stack: float = 20.0
    bet_fractions: Tuple[float, ...] = (0.5, 1.0)
    raise_fractions: Tuple[float, ...] = (1.0,)
    allow_allin: bool = True

    # Optional per-player *range* weighting over ranks (aligned to ``ranks``).
    # ``None`` means a uniform deal.  Used to study how the *shape* of a player's
    # range (condensed vs polarized) affects equilibrium EV.
    deal_weights: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None

    def __post_init__(self) -> None:
        if self.bet_mode not in ("fixed-limit", "no-limit"):
            raise ValueError(f"unknown bet_mode {self.bet_mode!r}")
        if self.bet_mode == "fixed-limit" and len(self.fixed_bets) < self.num_rounds:
            raise ValueError("fixed-limit needs a bet size for each round")

    @property
    def num_ranks(self) -> int:
        return len(self.ranks)

    def rank_strength(self, rank: str) -> int:
        """Higher is stronger.  Index 0 in ``ranks`` is strongest."""
        return self.num_ranks - self.ranks.index(rank)


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------


@dataclass
class ChanceNode:
    kind: str = field(default="chance", init=False)
    # list of (label, probability, child_index)
    branches: List[Tuple[str, float, int]] = field(default_factory=list)


@dataclass
class PlayerNode:
    kind: str = field(default="player", init=False)
    player: int = 0
    infoset: str = ""
    # list of (action_label, child_index)
    actions: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class TerminalNode:
    kind: str = field(default="terminal", init=False)
    # payoff to player 0 (player 1 receives the negation)
    payoff: float = 0.0


Node = object  # ChanceNode | PlayerNode | TerminalNode


# ---------------------------------------------------------------------------
# Betting-state bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetState:
    """Immutable snapshot of the chips committed by each player."""

    committed: Tuple[float, float]

    @property
    def pot(self) -> float:
        return self.committed[0] + self.committed[1]


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------


class Game:
    """A fully-expanded extensive-form game built from a :class:`GameConfig`."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.nodes: List[Node] = []
        self.root: int = -1

        # infoset key -> list of node indices in that set
        self.infoset_nodes: Dict[str, List[int]] = {}
        # infoset key -> canonical action label list
        self.infoset_actions: Dict[str, List[str]] = {}
        # infoset key -> player
        self.infoset_player: Dict[str, int] = {}
        # ordered list of infoset keys per player
        self.player_infosets: Dict[int, List[str]] = {0: [], 1: []}

        self._build()

    # -- node creation helpers ---------------------------------------------

    def _add(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _register_player_node(
        self, player: int, infoset: str, idx: int, action_labels: List[str]
    ) -> None:
        if infoset not in self.infoset_nodes:
            self.infoset_nodes[infoset] = []
            self.infoset_actions[infoset] = action_labels
            self.infoset_player[infoset] = player
            self.player_infosets[player].append(infoset)
        else:
            # All nodes in an infoset must share the same legal actions.
            assert self.infoset_actions[infoset] == action_labels, (
                f"inconsistent actions in infoset {infoset!r}: "
                f"{self.infoset_actions[infoset]} vs {action_labels}"
            )
        self.infoset_nodes[infoset].append(idx)

    # -- combinatorial chance ----------------------------------------------

    def _deal_branches(self) -> List[Tuple[str, str, float]]:
        """Enumerate ordered (rank_p0, rank_p1) deals with card-removal probs.

        If ``config.deal_weights`` is set, each player's private rank is drawn in
        proportion to ``weight(rank) * available_copies(rank)`` (with physical
        card removal between the two draws).  This lets us impose a *range* shape
        on a player without otherwise changing the game.
        """
        cfg = self.config
        ranks = cfg.ranks
        if cfg.deal_weights is None:
            w0 = {r: 1.0 for r in ranks}
            w1 = {r: 1.0 for r in ranks}
        else:
            w0 = {r: cfg.deal_weights[0][i] for i, r in enumerate(ranks)}
            w1 = {r: cfg.deal_weights[1][i] for i, r in enumerate(ranks)}

        out: List[Tuple[str, str, float]] = []
        counts0 = {r: cfg.suits for r in ranks}
        z0 = sum(w0[r] * counts0[r] for r in ranks)
        for a in ranks:
            if counts0[a] == 0 or w0[a] == 0:
                continue
            pa = w0[a] * counts0[a] / z0
            counts1 = dict(counts0)
            counts1[a] -= 1
            z1 = sum(w1[r] * counts1[r] for r in ranks)
            if z1 <= 0:
                continue
            for b in ranks:
                if counts1[b] == 0 or w1[b] == 0:
                    continue
                pb = w1[b] * counts1[b] / z1
                out.append((a, b, pa * pb))
        return out

    def _community_branches(self, a: str, b: str) -> List[Tuple[str, float]]:
        """Enumerate community ranks given the two private ranks already dealt."""
        cfg = self.config
        counts = {r: cfg.suits for r in cfg.ranks}
        counts[a] -= 1
        counts[b] -= 1
        remaining = sum(counts.values())
        return [(c, counts[c] / remaining) for c in cfg.ranks if counts[c] > 0]

    # -- showdown ----------------------------------------------------------

    def _showdown_payoff(
        self, a: str, b: str, community: Optional[str], committed: Tuple[float, float]
    ) -> float:
        """Net payoff to player 0 at showdown given both ranks and community.

        ``community is None`` (single-round game) compares private ranks only.
        """
        cfg = self.config
        p0_pair = community is not None and a == community
        p1_pair = community is not None and b == community
        if p0_pair and not p1_pair:
            result = 1
        elif p1_pair and not p0_pair:
            result = -1
        elif p0_pair and p1_pair:
            # both paired the (single remaining) community rank is impossible
            # because only one copy of `community` remains; kept for safety.
            result = 0
        else:
            sa, sb = cfg.rank_strength(a), cfg.rank_strength(b)
            result = (sa > sb) - (sa < sb)
        c0, c1 = committed
        if result > 0:
            return c1            # win: gain opponent's contribution
        if result < 0:
            return -c0           # lose: forfeit own contribution
        return (c1 - c0) / 2.0   # split

    # -- legal bet/raise enumeration ---------------------------------------

    def _legal_aggressive(
        self, committed: Tuple[float, float], to_act: int, round_idx: int, is_raise: bool
    ) -> List[Tuple[str, float]]:
        """Return list of (label, new_committed_for_actor) aggressive actions."""
        cfg = self.config
        opp = 1 - to_act
        out: List[Tuple[str, float]] = []
        if cfg.bet_mode == "fixed-limit":
            inc = cfg.fixed_bets[round_idx]
            target = committed[opp] + inc  # match opponent then add increment
            label = "raise" if is_raise else "bet"
            out.append((label, target))
            return out

        # no-limit
        stack = cfg.stack
        call_to = committed[opp]                 # amount needed to match
        pot_if_called = committed[0] + committed[1] + (call_to - committed[to_act])
        fractions = cfg.raise_fractions if is_raise else cfg.bet_fractions
        candidates: List[Tuple[str, float]] = []
        for frac in fractions:
            extra = frac * pot_if_called
            target = call_to + extra
            if target > stack:
                continue
            tag = "r" if is_raise else "b"
            candidates.append((f"{tag}{frac:g}", target))
        if cfg.allow_allin:
            candidates.append(("allin", stack))
        # filter: must be a strict raise over current commitment, within stack,
        # and de-duplicate identical target amounts (e.g. frac shove == all-in).
        seen: set = set()
        for label, target in candidates:
            target = round(target, 9)
            if target <= committed[opp] + 1e-12:   # not a genuine raise/bet
                continue
            if target > stack + 1e-9:
                continue
            if target in seen:
                continue
            seen.add(target)
            out.append((label, target))
        return out

    # -- recursive betting builder -----------------------------------------

    def _build_betting(
        self,
        a: str,
        b: str,
        community: Optional[str],
        committed: Tuple[float, float],
        to_act: int,
        round_idx: int,
        num_raises: int,
        opp_checked: bool,
        history: Tuple[str, ...],
        close_round: Callable[[Tuple[float, float], Tuple[str, ...]], int],
    ) -> int:
        """Build the subtree for one betting decision.  Returns node index.

        ``close_round`` is invoked with ``(committed, history)`` when the round
        ends without a fold; it constructs the next stage (community reveal +
        round 2, or showdown) and returns its node index.
        """
        cfg = self.config
        opp = 1 - to_act
        to_call = committed[opp] - committed[to_act]
        action_specs: List[Tuple[str, int]] = []
        labels: List[str] = []

        if to_call <= 1e-12:
            # ---- no outstanding bet: check or bet ----
            if opp_checked:
                child = close_round(committed, history + ("x",))
            else:
                child = self._build_betting(
                    a, b, community, committed, opp, round_idx, num_raises,
                    opp_checked=True, history=history + ("x",),
                    close_round=close_round,
                )
            action_specs.append(("x", child))
            labels.append("x")

            if num_raises < cfg.max_raises:
                for label, target in self._legal_aggressive(
                    committed, to_act, round_idx, is_raise=False
                ):
                    new_committed = _with(committed, to_act, target)
                    child = self._build_betting(
                        a, b, community, new_committed, opp, round_idx,
                        num_raises + 1, opp_checked=False,
                        history=history + (label,), close_round=close_round,
                    )
                    action_specs.append((label, child))
                    labels.append(label)
        else:
            # ---- facing a bet: fold / call / raise ----
            # Net to player 0 when `to_act` folds: if P0 folds -> -c0; else +c1.
            fold_payoff = -committed[0] if to_act == 0 else committed[1]
            fold_child = self._add(TerminalNode(payoff=fold_payoff))
            action_specs.append(("f", fold_child))
            labels.append("f")

            call_committed = _with(committed, to_act, committed[opp])
            child = close_round(call_committed, history + ("c",))
            action_specs.append(("c", child))
            labels.append("c")

            if num_raises < cfg.max_raises:
                for label, target in self._legal_aggressive(
                    committed, to_act, round_idx, is_raise=True
                ):
                    new_committed = _with(committed, to_act, target)
                    child = self._build_betting(
                        a, b, community, new_committed, opp, round_idx,
                        num_raises + 1, opp_checked=False,
                        history=history + (label,), close_round=close_round,
                    )
                    action_specs.append((label, child))
                    labels.append(label)

        node = PlayerNode(player=to_act)
        idx = self._add(node)
        # Infoset key: acting player's private rank + public state.
        infoset = self._infoset_key(to_act, a, b, community, history)
        node.infoset = infoset
        node.actions = action_specs
        self._register_player_node(to_act, infoset, idx, labels)
        return idx

    def _infoset_key(
        self,
        player: int,
        a: str,
        b: str,
        community: Optional[str],
        history: Tuple[str, ...],
    ) -> str:
        own = a if player == 0 else b
        comm = community if community is not None else "-"
        hist = "".join(history)
        return f"P{player}|{own}|{comm}|{hist}"

    # -- stage assembly ----------------------------------------------------

    def _build(self) -> None:
        cfg = self.config
        ante = float(cfg.ante)

        deal = ChanceNode()
        self.root = self._add(deal)

        for a, b, prob in self._deal_branches():

            # Stage transition after round 1 closes.
            def after_round1(committed, history, a=a, b=b):
                if cfg.num_rounds == 1:
                    # Single-round game: showdown immediately on private ranks.
                    term = TerminalNode(
                        payoff=self._showdown_payoff(a, b, None, committed)
                    )
                    return self._add(term)
                # Otherwise: reveal community card, then run round 2.
                comm_chance = ChanceNode()
                comm_idx = self._add(comm_chance)
                for c, cprob in self._community_branches(a, b):
                    def close_round2(committed2, history2, a=a, b=b, c=c):
                        term = TerminalNode(
                            payoff=self._showdown_payoff(a, b, c, committed2)
                        )
                        return self._add(term)

                    r2_history = history + ("/",)
                    r2_root = self._build_betting(
                        a, b, c, committed, to_act=0, round_idx=1,
                        num_raises=0, opp_checked=False, history=r2_history,
                        close_round=close_round2,
                    )
                    comm_chance.branches.append((c, cprob, r2_root))
                return comm_idx

            committed0 = (ante, ante)
            r1_root = self._build_betting(
                a, b, None, committed0, to_act=0, round_idx=0,
                num_raises=0, opp_checked=False, history=(),
                close_round=after_round1,
            )
            deal.branches.append((f"{a}{b}", prob, r1_root))

    # -- public summary ----------------------------------------------------

    def summary(self) -> Dict[str, int]:
        n_term = sum(1 for n in self.nodes if isinstance(n, TerminalNode))
        n_chance = sum(1 for n in self.nodes if isinstance(n, ChanceNode))
        n_player = sum(1 for n in self.nodes if isinstance(n, PlayerNode))
        return {
            "nodes": len(self.nodes),
            "terminals": n_term,
            "chance": n_chance,
            "player": n_player,
            "infosets_p0": len(self.player_infosets[0]),
            "infosets_p1": len(self.player_infosets[1]),
        }


def _with(committed: Tuple[float, float], player: int, value: float) -> Tuple[float, float]:
    if player == 0:
        return (value, committed[1])
    return (committed[0], value)
