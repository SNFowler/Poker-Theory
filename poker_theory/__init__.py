"""Poker-Theory: information-theoretic study of range exploitability vs sizing.

A small, exactly-solvable Leduc-style toy game (AKQJT9) plus an exact
sequence-form LP solver, a CFR+ cross-check, best-response/exploitability
routines, and the four information measures described in the research brief.
"""

from .game import Game, GameConfig, LEDUC_RANKS, AKQJT9_RANKS  # noqa: F401

__all__ = ["Game", "GameConfig", "LEDUC_RANKS", "AKQJT9_RANKS"]
