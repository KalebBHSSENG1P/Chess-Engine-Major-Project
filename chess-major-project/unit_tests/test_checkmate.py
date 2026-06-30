import pytest
from ChessEngine import GameState, Move

def make(gs, start, end):
    move = Move(start, end, gs.board)
    gs.makeMove(move)

def test_checkmate_detected():
    gs = GameState()

    # Fool's Mate (shortest real checkmate)
    make(gs, (6, 5), (5, 5))  # f3
    make(gs, (1, 4), (3, 4))  # e5
    make(gs, (6, 6), (4, 6))  # g4
    make(gs, (0, 3), (4, 7))  # Qh4#

    assert gs.inCheckmate is True
