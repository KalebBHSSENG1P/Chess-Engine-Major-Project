import pytest
from ChessEngine import GameState, Move

def test_pawn_promotion():
    gs = GameState()

    # Clear board
    gs.board = [[None for _ in range(8)] for _ in range(8)]

    from ChessEngine import Pawn, Queen

    # White pawn on the 7th rank ready to promote
    gs.board[1][0] = Pawn("w")

    # White to move
    gs.whiteToMove = True

    # Move pawn from a7 → a8 and promote to a queen
    move = Move((1, 0), (0, 0), gs.board, promotionChoice="Q")
    gs.makeMove(move)

    # Pawn should now be a queen
    assert isinstance(gs.board[0][0], Queen)