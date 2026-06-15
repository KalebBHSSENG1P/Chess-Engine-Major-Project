import random
import numpy as np

class ChessAI:
    pieceScore = {"K": 0, "Q": 10, "R": 5, "B": 3, "N": 3, "p": 1}

    knightScores = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 3, 3, 3, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 3, 3, 3, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=int)

    bishopScores = np.array([
        [4, 3, 2, 1, 1, 2, 3, 4],
        [3, 4, 3, 2, 2, 3, 4, 3],
        [2, 3, 4, 3, 3, 4, 3, 2],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [2, 3, 4, 3, 3, 4, 3, 2],
        [3, 4, 3, 2, 2, 3, 4, 3],
        [4, 3, 2, 1, 1, 2, 3, 4],
    ], dtype=int)

    queenScores = np.array([
        [1, 1, 1, 3, 1, 1, 1, 1],
        [1, 2, 3, 3, 3, 1, 1, 1],
        [1, 4, 3, 3, 3, 4, 2, 1],
        [1, 2, 3, 3, 3, 2, 2, 1],
        [1, 2, 3, 3, 3, 2, 2, 1],
        [1, 4, 3, 3, 3, 4, 2, 1],
        [1, 1, 2, 3, 3, 1, 1, 1],
        [1, 1, 1, 3, 1, 1, 1, 1],
    ], dtype=int)

    rookScores = np.array([
        [4, 3, 4, 4, 4, 4, 3, 4],
        [4, 4, 4, 4, 4, 4, 4, 4],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [4, 4, 4, 4, 4, 4, 4, 4],
        [4, 3, 4, 4, 4, 4, 3, 4],
    ], dtype=int)

    whitePawnScores = np.array([
        [8, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [5, 6, 6, 7, 7, 6, 6, 5],
        [2, 3, 3, 5, 5, 3, 3, 2],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    blackPawnScores = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [2, 3, 3, 5, 5, 3, 3, 2],
        [5, 6, 6, 7, 7, 6, 6, 5],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8],
    ], dtype=int)

    piecePositionScores = {
        "N": knightScores,
        "B": bishopScores,
        "Q": queenScores,
        "R": rookScores,
        "bp": blackPawnScores,
        "wp": whitePawnScores,
    }

    CHECKMATE = 1000
    STALEMATE = 0
    MAX_DEPTH = 3

    @staticmethod
    def find_random_move(valid_moves):
        return random.choice(valid_moves)

    @classmethod
    def find_best_move_minmax(cls, gs, valid_moves, return_queue):
        print(f"[ChessAI] Starting search depth={cls.MAX_DEPTH} color={'white' if gs.whiteToMove else 'black'} valid_moves={len(valid_moves)}")
        cls.next_move = None
        ordered_moves = sorted(valid_moves, key=cls._move_order_key, reverse=True)
        cls._find_move_negamax_alphabeta(
            gs, ordered_moves, cls.MAX_DEPTH, -cls.CHECKMATE, cls.CHECKMATE, 1 if gs.whiteToMove else -1
        )
        print(f"[ChessAI] Best move: {cls.next_move}")
        return_queue.put(cls.next_move)

    @classmethod
    def _move_order_key(cls, move):
        victim_value = cls.pieceScore.get(move.pieceCaptured.kind, 0) if move.pieceCaptured is not None else 0
        attacker_value = cls.pieceScore.get(move.pieceMoved.kind, 0)
        promotion_bonus = 10 if move.isPawnPromotion else 0
        capture_bonus = victim_value * 10 - attacker_value
        return capture_bonus + promotion_bonus

    @classmethod
    def _find_move_negamax_alphabeta(cls, gs, valid_moves, depth, alpha, beta, turn_multiplier):
        if depth == 0:
            return turn_multiplier * cls.score_board(gs)

        max_score = -cls.CHECKMATE
        ordered_moves = sorted(valid_moves, key=cls._move_order_key, reverse=True)
        for move in ordered_moves:
            gs.makeMove(move)
            next_moves = gs.getValidMoves()
            score = -cls._find_move_negamax_alphabeta(
                gs, next_moves, depth - 1, -beta, -alpha, -turn_multiplier
            )
            if score > max_score:
                max_score = score
                if depth == cls.MAX_DEPTH:
                    cls.next_move = move
            gs.undoMove()
            if max_score > alpha:
                alpha = max_score
            if alpha >= beta:
                break
        return max_score

    @classmethod
    def score_board(cls, gs):
        if gs.checkmate:
            return -cls.CHECKMATE if gs.whiteToMove else cls.CHECKMATE
        if gs.stalemate:
            return cls.STALEMATE

        score = 0
        for row in range(len(gs.board)):
            for col in range(len(gs.board[row])):
                square = gs.board[row, col]
                if square is None:
                    continue
                position_key = square.code if square.kind == "p" else square.kind
                pps = cls.piecePositionScores[position_key][row][col] * 0.1 if square.kind != "K" else 0
                if square.color == "w":
                    score += cls.pieceScore[square.kind] + pps
                else:
                    score -= cls.pieceScore[square.kind] + pps
        return score
