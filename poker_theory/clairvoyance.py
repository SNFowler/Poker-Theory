"""The clairvoyance game: the clean "penalty for a condensed range".

This is the toy that actually isolates the intuition behind the whole project --
*a condensed range is at the mercy of a polarized opponent* -- which a full
symmetric game averages away.

Setup (Chen & Ankenman, *The Mathematics of Poker*):

* a pot of ``P`` sits in the middle (here ``P = 2*ante``);
* the **bettor** is *polarized* -- holds the nuts or pure air, 50/50;
* the **defender** is maximally *condensed* -- a single bluff-catcher that beats
  air and loses to the nuts ("a single card");
* the bettor may bet ``b`` (a fraction ``s = b/P`` of the pot) or check; the
  defender may only call or fold.

Exact equilibrium (derived below), with ``s = b/P``:

* the bettor **value-bets the nuts** and **bluffs air** with frequency
  ``beta = s/(1+s)`` (so bluffs are a fraction ``s/(1+2s)`` of the betting range);
* the defender is held at **indifference**, calling with the minimum-defence
  frequency ``c = 1/(1+s)`` and folding ``s/(1+s)``;
* the **value to the bettor is ``s/(1+s)``** (in ante units, over a check-down
  baseline of 0) -- i.e. the *penalty the condensed range pays*.

The penalty is **monotone increasing in bet size** and approaches one full ante
as ``s -> infinity``: against a purely condensed range the polar player wants to
bet as large as possible.  There is no fold-equity-vs-value tension here (the
nuts always want the biggest bet and the bluffs ride along), so the EV-optimal
size and the "maximally exploit their range" size *coincide* -- the divergence
seen with symmetric ranges is an artefact of the bettor also holding value hands
that want to be called.

Derivation of the value
-----------------------
Take ``P = 2`` (antes of 1) and bet ``b = sP = 2s``; payoffs are net chips to the
bettor (each player has already posted its ante).

*Defender indifference* (call vs fold facing a bet) fixes the bluff frequency.
With ``P(air | bet) = beta/(1+beta)`` and ``P(nuts | bet) = 1/(1+beta)``,
``EV_call = [beta(P+b) - b]/(1+beta) = 0`` gives ``beta = b/(P+b) = s/(1+s)``.

*Bettor indifference* (bluff vs check, holding air) fixes the call frequency.
Bluffing air wins ``P`` when the defender folds and loses ``b`` when called;
checking air loses at showdown (EV 0 relative to the dead pot).  ``(1-c)P - cb = 0``
gives ``c = P/(P+b) = 1/(1+s)``.

*Value to the bettor*: with the nuts (prob 1/2) the bettor wins ``c*b`` more than
the pot it already had; with air (prob 1/2) the bettor is held to its check value
of 0.  Hence ``value = (1/2) c b = (1/2)(1/(1+s))(2s) = s/(1+s)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClairvoyantEquilibrium:
    bet_fraction: float       # s = b / P
    value_to_bettor: float    # s/(1+s) -- the penalty the condensed range pays
    bluff_frequency: float    # beta = s/(1+s): how often air bets
    call_frequency: float     # c = 1/(1+s): defender minimum-defence frequency
    fold_frequency: float     # s/(1+s)
    bluff_fraction_of_bets: float  # bluffs / (value+bluffs) = s/(1+2s)


def clairvoyant_equilibrium(s: float) -> ClairvoyantEquilibrium:
    """Closed-form GTO of the clairvoyance game for pot-fraction bet size ``s``."""
    if s < 0:
        raise ValueError("bet fraction must be non-negative")
    return ClairvoyantEquilibrium(
        bet_fraction=s,
        value_to_bettor=s / (1.0 + s),
        bluff_frequency=s / (1.0 + s),
        call_frequency=1.0 / (1.0 + s),
        fold_frequency=s / (1.0 + s),
        bluff_fraction_of_bets=s / (1.0 + 2.0 * s),
    )
