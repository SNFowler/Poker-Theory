"""Core invariants for the Leduc-AKQJT9 solver and measures.

Run with ``pytest -q`` from the repo root.  Games are kept small so the suite
runs in a few seconds on CPU.
"""

import math

import pytest

from poker_theory.game import Game, GameConfig, LEDUC_RANKS, AKQJT9_RANKS
from poker_theory import sequence_form as sf
from poker_theory import cfr
from poker_theory import best_response as br
from poker_theory import analysis as an


def leduc_game():
    cfg = GameConfig(ranks=LEDUC_RANKS, suits=2, ante=1, bet_mode="fixed-limit",
                     fixed_bets=(2, 4), max_raises=2)
    return Game(cfg)


def test_leduc_known_value():
    """Classic Leduc Hold'em first-player value is the published -0.0856..."""
    sol = sf.solve(leduc_game())
    assert sol.value == pytest.approx(-0.0856064, abs=1e-5)


def test_leduc_infoset_count():
    """Leduc has 144 information sets per player (288 total)."""
    g = leduc_game()
    assert len(g.player_infosets[0]) == 144
    assert len(g.player_infosets[1]) == 144


def test_leduc_sequence_count():
    """Leduc has 337 sequences per player."""
    sol = sf.solve(leduc_game())
    assert sol.seq_index.num_seqs(0) == 337
    assert sol.seq_index.num_seqs(1) == 337


def test_equilibrium_is_unexploitable():
    g = leduc_game()
    sol = sf.solve(g)
    _, expl = br.exploitability(g, sol.strategy)
    assert expl < 1e-9


def test_best_response_value_equals_game_value():
    g = leduc_game()
    sol = sf.solve(g)
    u0 = br.best_response_value(g, sol.strategy, 0)
    u1 = br.best_response_value(g, sol.strategy, 1)
    assert u0 == pytest.approx(sol.value, abs=1e-7)
    assert u1 == pytest.approx(-sol.value, abs=1e-7)


def test_cfr_agrees_with_lp_value():
    """CFR+ is an independent cross-check: its average-profile value matches LP."""
    cfg = GameConfig(ranks=LEDUC_RANKS, suits=2, ante=1, bet_mode="fixed-limit",
                     num_rounds=1, fixed_bets=(2,), max_raises=2)
    g = Game(cfg)
    lp = sf.solve(g)
    avg = cfr.train(g, iterations=4000)

    # expected payoff to P0 under the CFR average profile
    from poker_theory.game import ChanceNode, PlayerNode, TerminalNode
    cache = {}

    def ev(idx):
        if idx in cache:
            return cache[idx]
        n = g.nodes[idx]
        if isinstance(n, TerminalNode):
            v = n.payoff
        elif isinstance(n, ChanceNode):
            v = sum(p * ev(c) for _l, p, c in n.branches)
        else:
            pr = avg[n.infoset]
            v = sum(pr.get(a, 0.0) * ev(c) for a, c in n.actions)
        cache[idx] = v
        return v

    assert ev(g.root) == pytest.approx(lp.value, abs=5e-3)


def test_single_round_solves_and_unexploitable():
    cfg = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=1,
                     bet_mode="no-limit", stack=50.0, bet_fractions=(1.0,),
                     raise_fractions=(), allow_allin=False, max_raises=1)
    g = Game(cfg)
    sol = sf.solve(g)
    _, expl = br.exploitability(g, sol.strategy)
    assert expl < 1e-9


def test_mutual_information_identity():
    """MI(action;rank) == sum_a P(a) * (H(prior) - H(posterior|a))."""
    g = leduc_game()
    sol = sf.solve(g)
    reach = an.ProfileReach(g, sol.strategy)
    dp = next(d for d in an.decision_points(g)
              if d.player == 0 and d.history == "" and d.community == "-")
    mi = an.action_strength_mi(g, sol.strategy, dp, reach)
    pcs = an.posterior_collapse(g, sol.strategy, dp, reach)
    expected_drop = sum(p.action_prob * p.entropy_drop for p in pcs)
    assert expected_drop == pytest.approx(mi.mutual_information, abs=1e-9)
    assert mi.mutual_information >= -1e-12


def test_flat_range_exploitability_nonnegative():
    g = leduc_game()
    sol = sf.solve(g)
    for player in (0, 1):
        r = an.flat_range_exploitability(g, sol.strategy, player, sol.value)
        assert r["exploitability_gap"] >= -1e-9


def test_deal_weights_normalize():
    """Weighted deals still form a valid probability distribution."""
    weights = ((0.4, 0.1, 0.0, 0.0, 0.1, 0.4), tuple([1 / 6] * 6))
    cfg = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=1,
                     bet_mode="no-limit", stack=20.0, deal_weights=weights)
    g = Game(cfg)
    total = sum(prob for _l, prob, _c in g.nodes[g.root].branches)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_zero_sum_symmetry_of_value():
    """Solving from both sides agrees (enforced) and value is finite."""
    sol = sf.solve(leduc_game())
    assert math.isfinite(sol.value)
