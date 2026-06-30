import pytest
from ChessEngine import GameState, Move

def make(gs, start, end):
    move = Move(start, end, gs.board)
    gs.makeMove(move)

def test_stalemate_detected():
    gs = GameState()

    # Clear board
    gs.board = [[None for _ in range(8)] for _ in range(8)]

    from ChessEngine import King, Queen

    # Black king trapped in the corner (a8)
    gs.board[0][0] = King("b")
    gs.blackKingLocation = (0, 0)

    # White king far enough away (c6)
    gs.board[2][2] = King("w")
    gs.whiteKingLocation = (2, 2)

    # White queen delivering stalemate pressure (b6)
    gs.board[2][1] = Queen("w")

    gs.whiteToMove = False  # Black to move

    moves = gs.getValidMoves()

    # Black has no legal moves but is NOT in check → stalemate
    assert len(moves) == 0
    assert gs.inStalemate is True