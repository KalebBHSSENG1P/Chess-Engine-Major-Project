import pytest
from ChessEngine import GameState, Move, King, Rook

def test_white_can_castle_kingside():
    gs = GameState()

    # Clear board
    gs.board = [[None for _ in range(8)] for _ in range(8)]

    # Place king and rook in castling positions
    gs.board[7][4] = King("w")
    gs.whiteKingLocation = (7, 4)

    gs.board[7][7] = Rook("w")

    # Enable castling rights
    gs.currentCastlingRights.wks = True

    moves = gs.getValidMoves()

    # King-side castle: e1 → g1
    assert any(
        m.startRow == 7 and m.startCol == 4 and m.endRow == 7 and m.endCol == 6
        for m in moves
    )
