import pytest
from ChessEngine import GameState

def test_pawn_has_legal_forward_move():
    gs = GameState()
    moves = gs.getValidMoves()

    # White pawn at row 6 should be able to move to row 5
    assert any(m.startRow == 6 and m.endRow == 5 for m in moves)
