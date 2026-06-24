"""
Chess AI engine implementing negamax with alpha-beta pruning.
Features: iterative deepening, transposition table, killer moves, quiescence search.
SmartMoveFinder.py originally coded by Eddie Sharick (2021), code copied and modified by Kaleb Vong (2026)
"""
import random
import time
from Debug import debug_print, prof_start, prof_end, prof_report

class ChessAI:
    # Piece values (in pawns): used for material counting and move evaluation
    pieceScore = {"K": 1000, "Q": 10, "R": 5, "B": 3.25, "N": 3, "p": 1}

    # Debug node counter for performance analysis
    node_count = 0

    # Piece-square tables: position bonus/penalty for each piece type on each square
    # Higher values = more desirable positions (e.g., knights in center > knights on edges)
    knightScores = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 3, 3, 3, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 3, 3, 3, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # Bishop position bonuses: favors long diagonals and center
    bishopScores = [
        [4, 3, 2, 1, 1, 2, 3, 4],
        [3, 4, 3, 2, 2, 3, 4, 3],
        [2, 3, 4, 3, 3, 4, 3, 2],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [2, 3, 4, 3, 3, 4, 3, 2],
        [3, 4, 3, 2, 2, 3, 4, 3],
        [4, 3, 2, 1, 1, 2, 3, 4],
    ]

    # Queen position bonuses: prefers center with slight bias toward back rank
    queenScores = [
        [1, 1, 1, 3, 1, 1, 1, 1],
        [1, 2, 3, 3, 3, 1, 1, 1],
        [1, 4, 3, 3, 3, 4, 2, 1],
        [1, 2, 3, 3, 3, 2, 2, 1],
        [1, 2, 3, 3, 3, 2, 2, 1],
        [1, 4, 3, 3, 3, 4, 2, 1],
        [1, 1, 2, 3, 3, 1, 1, 1],
        [1, 1, 1, 3, 1, 1, 1, 1],
    ]

    # Rook position bonuses: favors open files and back rank
    rookScores = [
        [4, 3, 4, 4, 4, 4, 3, 4],
        [4, 4, 4, 4, 4, 4, 4, 4],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [4, 4, 4, 4, 4, 4, 4, 4],
        [4, 3, 4, 4, 4, 4, 3, 4],
    ]

    # White pawn position bonuses: encourages advancing toward promotion
    whitePawnScores = [
        [8, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [5, 6, 6, 7, 7, 6, 6, 5],
        [2, 3, 3, 5, 5, 3, 3, 2],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # Black pawn position bonuses: mirror of white (favors advancement toward rank 0)
    blackPawnScores = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 2, 3, 4, 4, 3, 2, 1],
        [2, 3, 3, 5, 5, 3, 3, 2],
        [5, 6, 6, 7, 7, 6, 6, 5],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8],
    ]

    # Maps piece codes to their position score tables
    piecePositionScores = {
        "N": knightScores,
        "B": bishopScores,
        "Q": queenScores,
        "R": rookScores,
        "bp": blackPawnScores,
        "wp": whitePawnScores,
    }

    # Opening book: common opening moves and responses for early game variation
    # Maps move sequences (as algebraic notation) to list of recommended response moves
    opening_book = {
        # White first moves: popular main openings
        "": ["e4", "d4", "c4", "Nf3"],  # 1.e4, 1.d4, 1.c4, 1.Nf3
        
        # White 1.e4 responses: Italian Game, French Defense, Sicilian
        "e4": ["c5", "e5", "c6"],  # Sicilian, Open game, Caro-Kann
        "e4|c5": ["Nf3", "c4"],  # Sicilian main lines
        "e4|e5": ["Nf3", "f4"],  # Open game, King's Gambit
        "e4|c6": ["d4", "Nf3"],  # Caro-Kann main lines
        
        # White 1.d4 responses: Queen's Gambit, Semi-Slav, Nf6
        "d4": ["Nf6", "d5", "c6"],  # Nf6, QGD, Semi-Slav
        "d4|Nf6": ["c4", "Bg5"],  # Reti, London
        "d4|d5": ["c4", "Nf3"],  # QGD, London
        "d4|c6": ["Nf3", "c4"],  # Semi-Slav, Slav
        
        # White 1.c4 responses: English Opening
        "c4": ["e5", "c6", "Nf6"],  # English variations
        "c4|e5": ["Nc3", "g3"],  # Symmetrical English
        
        # White 1.Nf3 responses: Reti Opening
        "Nf3": ["d5", "c5", "Nf6"],  # Reti lines
    }

    # Search parameters and engine constants
    CHECKMATE = 1000  # Maximum score (one side has won)
    STALEMATE = 0  # No advantage (draw)
    MAX_DEPTH = 4  # Maximum search depth
    TIME_LIMIT = 4 # Time limit for move search (seconds)
    
    # Optimization caches: transposition table stores evaluated positions
    killer_moves = {}  # Killer move heuristic for move ordering
    eval_cache = {}  # Cache for static board evaluations
    transposition_table = {}  # Cache of (depth, score) for positions
    start_time = 0.0  # Search start timestamp
    stop_search = False  # Flag to terminate search on timeout

    @staticmethod
    def find_random_move(valid_moves):
        """Return a random legal move (for testing or fallback)."""
        return random.choice(valid_moves)

    @classmethod
    def _get_opening_book_move(cls, gs, valid_moves):
        """
        Check opening book for current position and return random book move if available.
        Uses move log to track opening sequence in algebraic notation.
        """
        # Only use book in opening phase (first 8 moves total = 4 for each side)
        if len(gs.moveLog) >= 8:
            return None
        
        # Build position key from move log in algebraic notation
        move_sequence = "|".join(str(move) for move in gs.moveLog)
        
        # Look up position in opening book
        if move_sequence in cls.opening_book:
            book_moves = cls.opening_book[move_sequence]
            # Filter book moves to only those that are legal in current position
            legal_book_moves = []
            for move in valid_moves:
                if str(move) in book_moves:
                    legal_book_moves.append(move)
            # Return random legal book move if any exist
            if legal_book_moves:
                debug_print(f"Book move found for position '{move_sequence}': {[str(m) for m in legal_book_moves]}")
                return random.choice(legal_book_moves)
        
        return None

    @classmethod
    def find_best_move_minmax(cls, gs, valid_moves):
        """
        Find best move using iterative deepening with alpha-beta pruning.
        Returns deepest completed move; retains previous if time limit exceeded.
        """
        # Check opening book for first 8 moves (early game variation)
        book_move = cls._get_opening_book_move(gs, valid_moves)
        if book_move is not None:
            return book_move
        
        # Optimization caches
        cls.transposition_table = {}
        cls.killer_moves = {}
        cls.eval_cache = {}

        # reset number of nodes searched (for debugging only)
        cls.node_count = 0

        # Iterative deepening setup
        cls.stop_search = False
        cls.start_time = time.perf_counter()
        best_move = None
        best_move_fallback = None
        best_score_fallback = None
        turn_multiplier = 1 if gs.whiteToMove else -1

        # Iterative deepening: search progressively deeper until time limit
        for depth in range(1, cls.MAX_DEPTH + 1):
            if cls.stop_search:
                break
            alpha = -cls.CHECKMATE
            beta = cls.CHECKMATE
            ordered_moves = sorted(valid_moves, key=cls._move_order_key, reverse=True)
            best_moves_at_depth = []
            best_score = -cls.CHECKMATE
            
            # Debug message
            debug_print(f"Starting search at depth {depth}, no. of valid moves: {len(valid_moves)}")

            # Evaluate each root move at current depth
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
                
                # Track all moves with best score (for tiebreaker selection)
                if score > best_score:
                    best_score = score
                    best_moves_at_depth = [move]
                elif score == best_score:
                    best_moves_at_depth.append(move)
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            
            if cls.stop_search:
                break
            
            # Select best move at this depth using move-quality heuristic as tiebreaker
            if best_moves_at_depth:
                best_move = max(
                    best_moves_at_depth,
                    key=lambda move: (cls._move_quality(move, gs), cls._move_order_key(move)),
                )
                best_move_fallback = best_move
                best_score_fallback = best_score
        
        # get move quality for debugging output (only for final selected move, not all moves at depth)
        move_quality = cls._move_quality(best_move, gs)
        # print debug message for best move found at end of search, including score and qualities
        if best_move is not None:
            move_quality = cls._move_quality(best_move, gs)
            debug_print(f"Finished search. Best move: {best_move} with score: {best_score_fallback} and tiebreaker quality: {move_quality}")
        else:
            debug_print("Finished search. No best move found.")
        # print other debug information at the end of each move search
        debug_print(f"Nodes searched: {cls.node_count}, time taken: {time.perf_counter() - cls.start_time:.2f} seconds")
        prof_report()
        if best_move_fallback is not None:
            return best_move_fallback
        else:
            return valid_moves[0]  # fallback only if nothing was searched

    @classmethod
    def _move_order_key(cls, move):
        """
        Heuristic for move ordering: prioritizes captures and promotions.
        Used for alpha-beta pruning efficiency (examine promising moves first).
        """
        victim_value = cls.pieceScore.get(move.pieceCaptured.kind, 0) if move.pieceCaptured is not None else 0
        attacker_value = cls.pieceScore.get(move.pieceMoved.kind, 0)
        promotion_bonus = 100 if move.isPawnPromotion else 0
        capture_bonus = victim_value * 10 - attacker_value
        return capture_bonus + promotion_bonus

    @classmethod
    def _move_quality(cls, move, gs):
        """
        Tiebreaker heuristic when multiple moves evaluate equally.
        Prioritizes: capturing undefended pieces > check blocking > piece safety > avoiding undefended moves > special moves > piece development.
        """
        quality = 0.0
        
        # Highest priority: captures (tactical moves that win material)
        if move.isCapture:
            captured_value = cls.pieceScore.get(move.pieceCaptured.kind, 0)
            attacker_value = cls.pieceScore.get(move.pieceMoved.kind, 0)
            
            # Check if captured piece is defended (protected by opponent)
            gs.makeMove(move)
            captured_piece_defended = gs.squareUnderAttack(move.endRow, move.endCol)
            gs.undoMove()
            
            if not captured_piece_defended:
                # Undefended piece capture: always good (winning material)
                if move.pieceCaptured.kind == "Q":
                    quality += 10.0
                elif move.pieceCaptured.kind == "R":
                    quality += 5.0
                elif move.pieceCaptured.kind in ("B", "N"):
                    quality += 3.2
                elif move.pieceCaptured.kind == "p":
                    quality += 1.0
                quality += 0.5  # Bonus for capturing undefended piece
            else:
                # Defended piece capture: evaluate the trade
                # Only apply bonuses if we're winning material in the exchange
                if captured_value > attacker_value:
                    # We're winning material
                    if move.pieceCaptured.kind == "Q":
                        quality += 10.0
                    elif move.pieceCaptured.kind == "R":
                        quality += 5.0
                    elif move.pieceCaptured.kind in ("B", "N"):
                        quality += 3.2
                    elif move.pieceCaptured.kind == "p":
                        quality += 1.0
                    quality += 0.5  # Bonus for winning trade
                else:
                    # We're losing material or even trade: heavy penalty
                    material_loss = attacker_value - captured_value
                    quality -= 10.0 + material_loss * 0.3
            
            # Extra bonus for capturing a piece that was attacking our pieces
            # Find which of our pieces are currently under attack
            our_color = "w" if gs.whiteToMove else "b"
            attacked_our_pieces = 0
            for r in range(8):
                for c in range(8):
                    piece = gs.board[r, c]
                    if piece is not None and piece.color == our_color:
                        if gs.squareUnderAttack(r, c):
                            attacked_our_pieces += 1
            
            # Make the move and see how many of our pieces are still under attack
            gs.makeMove(move)
            still_attacked = 0
            for r in range(8):
                for c in range(8):
                    piece = gs.board[r, c]
                    if piece is not None and piece.color == our_color:
                        if gs.squareUnderAttack(r, c):
                            still_attacked += 1
            gs.undoMove()
            
            # Bonus for removing threats (pieces that are no longer under attack)
            threats_removed = attacked_our_pieces - still_attacked
            if threats_removed > 0:
                quality += threats_removed * 0.8
        
        # Check blocking preference: prefer blocking check with non-king pieces over moving king
        if move.pieceMoved.kind != "K":
            king_was_in_check = gs.inCheck()
            if king_was_in_check:
                gs.makeMove(move)
                king_still_in_check = gs.inCheck()
                gs.undoMove()
                # Strong bonus for blocking check without moving king
                if not king_still_in_check:
                    quality += 3.5
        
        # Piece safety: dynamic bonus for moving any piece under attack to safety
        # Bonus scales with piece value: queen gets biggest bonus, pawns smallest
        if not move.isCapture:
            piece_under_attack = gs.squareUnderAttack(move.startRow, move.startCol)
            if piece_under_attack:
                # Check if piece is still under attack after the move
                gs.makeMove(move)
                piece_still_under_attack = gs.squareUnderAttack(move.endRow, move.endCol)
                gs.undoMove()
                # If piece evades capture, give dynamic bonus based on piece value
                if not piece_still_under_attack:
                    if move.pieceMoved.kind != "K":
                        piece_value = cls.pieceScore.get(move.pieceMoved.kind, 0)
                        # Queen: 10 -> +1.0, Rook: 5 -> +0.5, Bishop/Knight: 3-3.25 -> +0.3-0.32, Pawn: 1 -> +0.1
                        quality += piece_value * 0.15
                    else:
                        quality += 0.1
        
        # CRITICAL: Heavy penalty for moving piece to a square under attack
        # Extra severe penalty for pieces captured by pawns (bad trades)
        if not move.isCapture and move.pieceMoved.kind != "K":
            gs.makeMove(move)
            dest_under_attack = gs.squareUnderAttack(move.endRow, move.endCol)
            gs.undoMove()
            if dest_under_attack:
                # Check specifically if a pawn can capture this piece
                enemy_color = "b" if gs.whiteToMove else "w"
                piece_value = cls.pieceScore.get(move.pieceMoved.kind, 0)
                pawn_can_capture = False
                
                # Check diagonal squares for enemy pawns that can capture
                pawn_attack_direction = -1 if enemy_color == "b" else 1
                for delta_col in (-1, 1):
                    pawn_row = move.endRow - pawn_attack_direction
                    pawn_col = move.endCol + delta_col
                    if 0 <= pawn_row < 8 and 0 <= pawn_col < 8:
                        pawn = gs.board[pawn_row, pawn_col]
                        if pawn is not None and pawn.kind == "p" and pawn.color == enemy_color:
                            pawn_can_capture = True
                            break
                
                # Much heavier penalty for pieces captured by pawns (bad trades)
                if pawn_can_capture:
                    quality -= 3.0 + piece_value
                else:
                    # Very heavy penalty for moving to any attacked square
                    quality -= 2.0 + piece_value * 0.4
        
        # Second priority: move pieces under attack to safety
        if not move.isCapture and gs.squareUnderAttack(move.startRow, move.startCol):
            # Higher bonus for moving queen to safety
            if move.pieceMoved.kind != "Q":
                quality += 1.5
        
        # Special moves: promotion and castling
        if move.isPawnPromotion:
            quality += 2.0
        if move.isCastleMove:
            quality += 0.75
        
        # Slight penalty for checking opponent's king (very minor disincentive)
        gs.makeMove(move)
        if gs.inCheck():
            quality -= 0.03
        gs.undoMove()
        
        # Special moves: promotion and castling
        if move.isPawnPromotion:
            quality += 2.0
        if move.isCastleMove:
            quality += 0.75
        
        # Piece development: only evaluate if no tactical moves available
        if quality == 0.0:
            # Check if square is under attack
            gs.makeMove(move)
            dest_under_attack = gs.squareUnderAttack(move.endRow, move.endCol)
            gs.undoMove()
            if not dest_under_attack: # If square is not under attack, only then evaluate development bonuses/penalties
                # Knight development from starting position
                if move.pieceMoved.kind == "N":
                    if move.startRow in (7, 0):
                        quality += 0.10
                    if move.endCol in (0, 7) or move.endRow in (0, 7):
                        quality -= 0.15
                    else:
                        quality += 0.05
                # Bishop development
                elif move.pieceMoved.kind == "B":
                    if move.startRow in (7, 0):
                        quality += 0.08
                    if move.endRow in (2, 5):
                        quality -= 0.15
                    else:
                        quality += 0.01
                # King: heavy penalty for moving (preserves castling rights)
                elif move.pieceMoved.kind == "K":
                    quality -= 10.0
                # Rook: slight penalty for moving from corner (conserve for castling)
                elif move.pieceMoved.kind == "R":
                    if move.startRow in (7, 0) and move.startCol in (0, 7):
                        quality -= 1.0
                # Queen: penalty for early development
                elif move.pieceMoved.kind == "Q":
                    quality -= 0.08
                # Pawn strategy: advance central pawns more than flank pawns
                elif move.pieceMoved.kind == "p":
                    is_white = move.pieceMoved.color == "w"
                    forward_row = move.endRow if is_white else (7 - move.endRow)
                    is_center = move.endCol in (3, 4)
                    is_center_adjacent = move.endCol in (2, 5)
                    
                    if forward_row <= 2:
                        quality -= 0.25
                    elif forward_row == 3:
                        if is_center:
                            quality += 0.22
                        elif is_center_adjacent:
                            quality += 0.14
                        else:
                            quality -= 0.05
                    elif forward_row == 4:
                        if is_center:
                            quality += 0.18
                        elif is_center_adjacent:
                            quality += 0.10
                        else:
                            quality -= 0.02
                    else:
                        if is_center or is_center_adjacent:
                            quality += 0.05
                        else:
                            quality -= 0.10
            else:
                pass # No development bonus if moving to attacked square

            # Bonus for moving toward center (except pawns) even when square is under attack, hopefully balancing out lost piece bonuses/penalties
            center_distance = abs(move.endRow - 3.5) + abs(move.endCol - 3.5)
            if move.pieceMoved.kind != "p":
                quality += max(0.0, 2.0 - center_distance) * 0.01
        
        return quality

    @classmethod
    def _position_key(cls, gs):
        """
        Create hashable position key for transposition table.
        Includes board state, whose turn it is, castling rights, en passant.
        """
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
        """
        Negamax search with alpha-beta pruning.
        Maintains transposition table and killer move heuristic.
        Returns 0 if time limit exceeded or search stopped.
        """
        # Profling start for negamax search
        t0 = prof_start("negamax")

        # Debug message printing negamax depth, prints each node visited so off by default even though debug mode is on. 
        # debug_print(f"Negamax depth={depth}")
        # increment node count (for debugging purposes only)
        cls.node_count += 1

        # Check for timeout or stop signal
        if cls.stop_search:
            return 0
        if time.perf_counter() - cls.start_time > cls.TIME_LIMIT:
            debug_print("Time limit exceeded, stopping search")
            cls.stop_search = True
            return 0
        
        # Check transposition table for cached evaluation at this depth
        position_key = cls._position_key(gs)
        cached = cls.transposition_table.get(position_key)
        if cached is not None and cached[0] >= depth:
            return cached[1]

        # Quiescence search at depth 0 to resolve captures
        if depth == 0:
            score = cls._quiescence_search(gs, valid_moves, alpha, beta, turn_multiplier)
            cls.transposition_table[position_key] = (depth, score)
            return score

        # Negamax loop: evaluate child positions
        max_score = -cls.CHECKMATE
        ordered_moves = sorted(valid_moves, key=cls._move_order_key, reverse=True)
        
        # Apply killer move heuristic: prioritize moves that caused cutoffs at this depth
        killer_id = cls.killer_moves.get(depth)
        if killer_id is not None:
            for i, move in enumerate(ordered_moves):
                if move.moveID == killer_id:
                    ordered_moves.insert(0, ordered_moves.pop(i))
                    break

        # Evaluate moves with alpha-beta pruning
        for move in ordered_moves:
            gs.makeMove(move)
            next_moves = gs.getValidMoves()
            score = -cls._find_move_negamax_alphabeta(
                gs, next_moves, depth - 1, -beta, -alpha, -turn_multiplier
            )
            gs.undoMove()
            
            # Update alpha and check for beta cutoff
            if score > max_score:
                max_score = score
            if max_score > alpha:
                alpha = max_score
            if alpha >= beta:
                # Store killer move if it's a quiet move (non-capture)
                if not move.isCapture:
                    cls.killer_moves[depth] = move.moveID
                break

        cls.transposition_table[position_key] = (depth, max_score)
        prof_end("negamax", t0)
        return max_score

    @classmethod
    def _quiescence_search(cls, gs, valid_move, alpha, beta, turn_multiplier, qdepth = 0):
        """
        Quiescence search: extends depth 0 by analyzing all captures/promotions.
        Prevents horizon effect where AI misses critical tactics.
        """
        # Profiling start for quiescence search
        t0 = prof_start("quiescence")

        # Debug message printing quiescence search depth, prints each node visited so off by default even though debug mode is on. 
        # debug_print(f"Quiescence search depth={qdepth}")
        # Check for timeout
        if cls.stop_search:
            return 0

        # increment node count (for debugging purposes only)
        cls.node_count += 1

        # Stand-pat: current position evaluation (baseline)
        stand_pat = turn_multiplier * cls.score_board(gs)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Only consider forcing moves (captures and promotions)
        captures = [move for move in valid_move if move.isCapture or move.isPawnPromotion]
        # debug_print(f"Quiescence captures: {len(captures)}")

        # Order captures by most promising first
        ordered_captures = sorted(captures, key=cls._move_order_key, reverse=True)
        
        for move in ordered_captures:
            if cls.stop_search:
                break
            gs.makeMove(move)
            valid_move = gs.getValidMoves()
            score = -cls._quiescence_search(
                gs,
                valid_move,
                -beta,
                -alpha,
                -turn_multiplier,
                qdepth + 1
            )
            gs.undoMove()
            if cls.stop_search:
                break
            
            # Alpha-beta pruning in quiescence search
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        prof_end("quiescence", t0)
        return alpha

    @classmethod
    def score_board(cls, gs, valid_moves=None):
        """
        Evaluate position: material + position tables + center control + mobility.
        Returns positive for white advantage, negative for black advantage.
        """
        # Profiling start for board evaluation
        t0 = prof_start("score_board")

        # eval cache check: if position has been evaluated before, return cached score
        key = cls._position_key(gs)
        if key in cls.eval_cache:
            return cls.eval_cache[key]

        # Checkmate/stalemate are terminal states
        if gs.checkmate:
            debug_print("Checkmate detected")
            return -cls.CHECKMATE if gs.whiteToMove else cls.CHECKMATE
        if gs.stalemate:
            debug_print("Stalemate detected")
            return cls.STALEMATE

        score = 0.0
        
        # Material evaluation: piece values + position bonuses
        for row in range(len(gs.board)):
            for col in range(len(gs.board[row])):
                square = gs.board[row, col]
                if square is None:
                    continue
                position_key = square.code if square.kind == "p" else square.kind
                piece_value = cls.pieceScore[square.kind]
                position_value = cls.piecePositionScores[position_key][row][col] * 0.1 if square.kind != "K" else 0
                piece_score = piece_value + position_value
                
                # Add to white or subtract from white's perspective
                if square.color == "w":
                    score += piece_score
                else:
                    score -= piece_score

        # Center control bonus: controlling center squares (d4, e4, d5, e5)
        white_center_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}
        for row, col in white_center_squares:
            square = gs.board[row, col]
            if square is not None:
                score += 0.1 if square.color == "w" else -0.1

        # Mobility bonus: more legal moves = better position
        if valid_moves is None:
            valid_moves = gs.getValidMoves()
        mobility_score = 0.05 * len(valid_moves)
        if not gs.whiteToMove:
            mobility_score = -mobility_score
        score += mobility_score

        cls.eval_cache[key] = score
        prof_end("score_board", t0)
        return score
