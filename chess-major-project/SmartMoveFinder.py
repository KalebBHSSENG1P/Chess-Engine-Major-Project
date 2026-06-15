import random
import time
import numpy as np

class ChessAI:
    pieceScore = {"K": 1000, "Q": 10, "R": 5, "B": 3.25, "N": 3, "p": 1}

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
    MAX_DEPTH = 4
    TIME_LIMIT = 0.6
    killer_moves = {}
    transposition_table = {}
    start_time = 0.0
    stop_search = False

    @staticmethod
    def find_random_move(valid_moves):
        return random.choice(valid_moves)

    @classmethod
    def find_best_move_minmax(cls, gs, valid_moves):
        cls.transposition_table = {}
        cls.killer_moves = {}
        cls.stop_search = False
        cls.start_time = time.perf_counter()

        best_move = None
        turn_multiplier = 1 if gs.whiteToMove else -1
        for depth in range(1, cls.MAX_DEPTH + 1):
            if cls.stop_search:
                break
            alpha = -cls.CHECKMATE
            beta = cls.CHECKMATE
            ordered_moves = sorted(valid_moves, key=cls._move_order_key, reverse=True)
            best_move_at_depth = None
            best_score = -cls.CHECKMATE
            for move in ordered_moves:
                if cls.stop_search:
                    break
                gs.makeMove(move)
                next_moves = gs.getValidMoves()
                score = -cls._find_move_negamax_alphabeta(
                    gs,
                    next_moves,
                    depth - 1,
                    -beta,
                    -alpha,
                    -turn_multiplier,
                )
                gs.undoMove()
                if cls.stop_search:
                    break
                if score > best_score:
                    best_score = score
                    best_move_at_depth = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            if cls.stop_search:
                break
            if best_move_at_depth is not None:
                best_move = best_move_at_depth
        return best_move if best_move is not None else (valid_moves[0] if valid_moves else None)

    @classmethod
    def _move_order_key(cls, move):
        victim_value = cls.pieceScore.get(move.pieceCaptured.kind, 0) if move.pieceCaptured is not None else 0
        attacker_value = cls.pieceScore.get(move.pieceMoved.kind, 0)
        promotion_bonus = 100 if move.isPawnPromotion else 0
        capture_bonus = victim_value * 10 - attacker_value
        return capture_bonus + promotion_bonus

    @classmethod
    def _position_key(cls, gs):
        flat_codes = tuple(
            square.code if square is not None else "--"
            for row in gs.board
            for square in row
        )
        return (
            flat_codes,
            gs.whiteToMove,
            gs.currentCastlingRights.wks,
            gs.currentCastlingRights.wqs,
            gs.currentCastlingRights.bks,
            gs.currentCastlingRights.bqs,
            gs.enpassantPossible,
        )

    @classmethod
    def _find_move_negamax_alphabeta(cls, gs, valid_moves, depth, alpha, beta, turn_multiplier):
        if cls.stop_search:
            return 0
        if time.perf_counter() - cls.start_time > cls.TIME_LIMIT:
            cls.stop_search = True
            return 0

        position_key = cls._position_key(gs)
        cached = cls.transposition_table.get(position_key)
        if cached is not None and cached[0] >= depth:
            return cached[1]

        if depth == 0:
            score = cls._quiescence_search(gs, valid_moves, alpha, beta, turn_multiplier)
            cls.transposition_table[position_key] = (depth, score)
            return score

        max_score = -cls.CHECKMATE
        ordered_moves = sorted(valid_moves, key=cls._move_order_key, reverse=True)
        killer_id = cls.killer_moves.get(depth)
        if killer_id is not None:
            for i, move in enumerate(ordered_moves):
                if move.moveID == killer_id:
                    ordered_moves.insert(0, ordered_moves.pop(i))
                    break

        for move in ordered_moves:
            gs.makeMove(move)
            next_moves = gs.getValidMoves()
            score = -cls._find_move_negamax_alphabeta(
                gs, next_moves, depth - 1, -beta, -alpha, -turn_multiplier
            )
            gs.undoMove()
            if score > max_score:
                max_score = score
            if max_score > alpha:
                alpha = max_score
            if alpha >= beta:
                if not move.isCapture:
                    cls.killer_moves[depth] = move.moveID
                break

        cls.transposition_table[position_key] = (depth, max_score)
        return max_score

    @classmethod
    def _quiescence_search(cls, gs, valid_moves, alpha, beta, turn_multiplier):
        if cls.stop_search:
            return 0
        if time.perf_counter() - cls.start_time > cls.TIME_LIMIT:
            cls.stop_search = True
            return 0

        stand_pat = turn_multiplier * cls.score_board(gs)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        captures = [move for move in valid_moves if move.isCapture or move.isPawnPromotion]
        ordered_captures = sorted(captures, key=cls._move_order_key, reverse=True)
        for move in ordered_captures:
            if cls.stop_search:
                break
            gs.makeMove(move)
            next_moves = gs.getValidMoves()
            score = -cls._quiescence_search(
                gs,
                next_moves,
                -beta,
                -alpha,
                -turn_multiplier,
            )
            gs.undoMove()
            if cls.stop_search:
                break
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    @classmethod
    def score_board(cls, gs):
        if gs.checkmate:
            return -cls.CHECKMATE if gs.whiteToMove else cls.CHECKMATE
        if gs.stalemate:
            return cls.STALEMATE

        score = 0.0
        for row in range(len(gs.board)):
            for col in range(len(gs.board[row])):
                square = gs.board[row, col]
                if square is None:
                    continue
                position_key = square.code if square.kind == "p" else square.kind
                piece_value = cls.pieceScore[square.kind]
                position_value = cls.piecePositionScores[position_key][row][col] * 0.1 if square.kind != "K" else 0
                piece_score = piece_value + position_value
                if square.color == "w":
                    score += piece_score
                else:
                    score -= piece_score
        return score
