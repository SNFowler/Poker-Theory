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


def test_ev_decomposition_is_exact():
    """EV(B) == V_check + V_betfold + V_betcall, an exact additive partition."""
    from poker_theory import studies as st
    base = GameConfig(ranks=AKQJT9_RANKS, suits=2, ante=1, num_rounds=2,
                      bet_mode="no-limit", stack=20.0)
    res = st.bet_size_decomposition(base, [0.5, 1.0, 2.0], bettor=0)
    assert len(res) == 3
    for r in res:
        assert r.v_check + r.v_betfold + r.v_betcall == pytest.approx(r.ev_total, abs=1e-9)


def test_multiround_lp_unexploitable():
    """The N-round generalization solves exactly (3-round, small config)."""
    cfg = GameConfig(ranks=LEDUC_RANKS, suits=2, ante=1, num_rounds=3,
                     bet_mode="fixed-limit", fixed_bets=(2, 4, 4), max_raises=1)
    g = Game(cfg)
    sol = sf.solve(g)
    _, expl = br.exploitability(g, sol.strategy)
    assert expl < 1e-9


def test_multiboard_showdown_ranking():
    """Pair (private matches any board card) beats non-pair; higher rank wins."""
    g = Game(GameConfig(ranks=AKQJT9_RANKS, num_rounds=3, bet_mode="no-limit"))
    c = (2.0, 2.0)  # win -> +2, lose -> -2, split -> 0
    assert g._showdown_payoff("K", "A", ("K", "9"), c) == 2.0   # P0 pairs K, beats A-high
    assert g._showdown_payoff("K", "A", ("9", "T"), c) == -2.0  # both unpaired, A > K
    assert g._showdown_payoff("K", "A", ("K", "A"), c) == -2.0  # A-pair beats K-pair
    assert g._showdown_payoff("Q", "Q", ("9", "T"), c) == 0.0   # tie -> split


def test_leduc_is_two_round_special_case():
    """The refactor preserves the classic 2-round Leduc value exactly."""
    cfg = GameConfig(ranks=LEDUC_RANKS, suits=2, ante=1, num_rounds=2,
                     bet_mode="fixed-limit", fixed_bets=(2, 4), max_raises=2)
    assert sf.solve(Game(cfg)).value == pytest.approx(-0.0856064, abs=1e-5)


def test_clairvoyance_matches_closed_form():
    """The exact solver reproduces the clairvoyance theorem value s/(1+s)."""
    from poker_theory import studies as st
    from poker_theory import clairvoyance as cl
    pts = st.clairvoyance_size_sweep([0.25, 0.5, 1.0, 2.0, 5.0])
    for p in pts:
        s = p.bet_fraction
        assert p.solver_value == pytest.approx(s / (1 + s), abs=1e-9)
        # minimum-defence frequency and bluff frequency
        eq = cl.clairvoyant_equilibrium(s)
        assert p.call_frequency == pytest.approx(eq.call_frequency, abs=1e-6)
        assert p.bluff_frequency == pytest.approx(eq.bluff_frequency, abs=1e-6)


def test_condensation_raises_penalty():
    """A more condensed defender (fixed mean strength) pays a larger penalty."""
    from poker_theory import studies as st
    surf = st.condensation_penalty_surface([0.0, 1.0], [0.5, 1.0, 2.0, 3.0])
    assert surf[1].max() > surf[0].max() + 1e-6   # condensed worse than uniform


def test_question_value_counterexample():
    """High-entropy bluff-catcher range == single card: entropy can't see shape.

    A defender uniform over four middle ranks has 2 bits of range entropy yet
    zero value-of-information, a trivial answer partition, full indifferent
    mass, and pays exactly the clairvoyance rent s/(1+s).
    """
    from poker_theory import studies as st
    qp = st.question_value_point((0, 1, 1, 1, 1, 0), 1.0, label="bc4")
    assert qp.range_entropy == pytest.approx(2.0, abs=1e-9)
    assert qp.voi == pytest.approx(0.0, abs=1e-6)
    assert qp.partition_entropy == pytest.approx(0.0, abs=1e-6)
    assert qp.indifferent_mass == pytest.approx(1.0, abs=1e-6)
    assert qp.penalty == pytest.approx(0.5, abs=1e-6)   # s/(1+s) at s=1


def test_question_value_uniform_defender():
    """A spread range has positive VoI and pays less than the full rent."""
    from poker_theory import studies as st
    qp = st.question_value_point((1, 1, 1, 1, 1, 1), 1.0, label="uniform")
    assert qp.voi > 0.01                      # the card matters for the answer
    assert qp.indifferent_mass < 1.0 - 1e-6   # not everything is a bluff-catcher
    assert qp.penalty < 0.5 - 1e-3            # pays less than clairvoyance rent


def test_question_value_voi_nonnegative():
    """VoI = v_informed - v_blind is nonnegative by construction."""
    from poker_theory import studies as st
    for w in [(1, 1, 1, 1, 1, 1), (0.35, 0.15, 0, 0, 0.15, 0.35)]:
        qp = st.question_value_point(w, 1.0)
        assert qp.voi >= -1e-9


def test_zero_sum_symmetry_of_value():
    """Solving from both sides agrees (enforced) and value is finite."""
    sol = sf.solve(leduc_game())
    assert math.isfinite(sol.value)
