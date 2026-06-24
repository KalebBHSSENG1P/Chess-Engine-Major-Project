"""
Chess game state management and move generation engine.
Handles board representation, legal move generation, special moves (castling, en passant, promotion).
Maintains move history and game state (checkmate, stalemate, check).
ChessEngine.py originally coded by Eddie Sharick (2021), code copied and modified by Kaleb Vong (2026)
"""

import numpy as np
from Debug import debug_print, prof_start, prof_end, prof_report

class Piece:
    """Base class for all chess pieces. Stores color ('w'/'b') and kind ('p','N','B','R','Q','K')."""
    def __init__(self, color, kind):
        self.color = color  # 'w' for white, 'b' for black
        self.kind = kind  # Piece type: 'p', 'N', 'B', 'R', 'Q', 'K'

    @property
    def code(self):
        # Returns two-character code like 'wp' (white pawn) or 'bR' (black rook)
        return f"{self.color}{self.kind}"

    def __str__(self):
        return self.code

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.color}')"


class Pawn(Piece):
    # Pawn has kind 'p' (lowercase to distinguish from other pieces)
    def __init__(self, color):
        super().__init__(color, "p")


class Knight(Piece):
    # Knight has kind 'N' (uppercase to distinguish from black pawn)
    def __init__(self, color):
        super().__init__(color, "N")


class Bishop(Piece):
    # Bishop has kind 'B' (uppercase)
    def __init__(self, color):
        super().__init__(color, "B")


class Rook(Piece):
    # Rook has kind 'R' (uppercase)
    def __init__(self, color):
        super().__init__(color, "R")


class Queen(Piece):
    # Queen has kind 'Q' (uppercase)
    def __init__(self, color):
        super().__init__(color, "Q")


class King(Piece):
    # King has kind 'K' (uppercase)
    def __init__(self, color):
        super().__init__(color, "K")


def create_starting_board():
    # Initialize 8x8 board with numpy array (object dtype for piece storage)
    board = np.full((8, 8), None, dtype=object)
    # Place black pieces on rows 0-1 (top of board)
    board[0] = [Rook("b"), Knight("b"), Bishop("b"), Queen("b"), King("b"), Bishop("b"), Knight("b"), Rook("b")]
    board[1] = [Pawn("b") for _ in range(8)]
    # Place white pieces on rows 6-7 (bottom of board)
    board[6] = [Pawn("w") for _ in range(8)]
    board[7] = [Rook("w"), Knight("w"), Bishop("w"), Queen("w"), King("w"), Bishop("w"), Knight("w"), Rook("w")]
    return board


class GameState:
    """
    Maintains complete game state: board, turn, move history, castling rights, check/checkmate status.
    Generates all legal moves and handles special moves (castling, en passant, promotion).
    """
    def __init__(self):
        # Board representation: 8x8 numpy array, each cell contains Piece object or None
        self.board = create_starting_board()

        # Game turn tracking: True = White to move, False = Black to move
        self.whiteToMove = True
        # Move history for undo functionality and game replay
        self.moveLog = []

        # Optimization caches for move generation
        self.attack_cache = {}
        self.cached_opp_moves = None
        self.cached_opp_turn = None
        
        # Dispatch table: maps piece kind to move generation function
        self.moveFunctions = {
            "p": self.getPawnMove,
            "N": self.getKnightMove,
            "B": self.getBishopMove,
            "R": self.getRookMove,
            "Q": self.getQueenMove,
            "K": self.getKingMove,
        }
        
        # King position tracking for quick check/checkmate detection and castling validation
        self.whiteKingLocation = (7, 4)  # Row 7, column 4 (e1 in algebraic notation)
        self.blackKingLocation = (0, 4)  # Row 0, column 4 (e8 in algebraic notation)
        
        # Game end states
        self.checkmate = False  # True if current player is in check and has no legal moves
        self.stalemate = False  # True if current player is not in check but has no legal moves
        
        # En passant: stores square where en passant capture is possible this turn
        self.enpassantPossible = ()  # Empty tuple means no en passant available
        self.enpassantPossibleLog = [self.enpassantPossible]  # History for undo
        
        # Castling rights: track if king/rook has moved (prevents illegal castling)
        self.currentCastlingRights = CastleRights(True, True, True, True)
        self.castleRightLog = [
            CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
        ]

    def makeMove(self, move, is_temp = False):
        """Execute move: update board, track history, handle special moves, update game state."""
        # Clear caches since board state is changing and cached values may no longer be valid
        if not is_temp:
            self.attack_cache.clear()
            self.cached_opp_moves = None
            self.cached_opp_turn = None

        # Update board: remove piece from start, place at end
        self.board[move.startRow, move.startCol] = None
        self.board[move.endRow, move.endCol] = move.pieceMoved
        # Record move in history for undo and game replay
        if not is_temp:
            self.moveLog.append(move)
        # Switch player turn
        self.whiteToMove = not self.whiteToMove
        
        # Update king location when king moves (needed for check detection and castling)
        if move.pieceMoved.code == "wK":
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved.code == "bK":
            self.blackKingLocation = (move.endRow, move.endCol)
        
        # Pawn promotion: convert pawn to chosen piece when reaching promotion rank
        if move.isPawnPromotion:
            if move.promote_to is not None:
                self.board[move.endRow, move.endCol] = move.promote_to(move.pieceMoved.color)
        
        # En passant: remove captured pawn when moving diagonally with no target
        if move.isEnpassantMove:
            self.board[move.startRow, move.endCol] = None
        
        # Update en passant availability: only for 2-square pawn advances
        if move.pieceMoved.kind == "p" and abs(move.startRow - move.endRow) == 2:
            self.enpassantPossible = ((move.startRow + move.endRow) // 2, move.startCol)
        else:
            self.enpassantPossible = ()
        
        # Castling: move rook to the other side of the king
        if move.isCastleMove:
            if move.endCol - move.startCol == 2:  # Kingside: king moves 2 squares right
                # Move rook from h-file to f-file
                self.board[move.endRow, move.endCol - 1] = self.board[move.endRow, move.endCol + 1]
                self.board[move.endRow, move.endCol + 1] = None
            elif move.endCol - move.startCol == -2:  # Queenside: king moves 2 squares left
                # Move rook from a-file to d-file
                self.board[move.endRow, move.endCol + 1] = self.board[move.endRow, move.endCol - 2]
                self.board[move.endRow, move.endCol - 2] = None
        
        # Record en passant state in history for undo
        if not is_temp:
            self.enpassantPossibleLog.append(self.enpassantPossible)
        # Update castling rights if king or rook moved
            self.updateCastleRights(move)
        # Record castling rights in history for undo
            self.castleRightLog.append(
                CastleRights(
                    self.currentCastlingRights.wks,
                    self.currentCastlingRights.bks,
                    self.currentCastlingRights.wqs,
                    self.currentCastlingRights.bqs,
                )
            )
        else:
        # For temp moves, still update castling rights, but don't log
            self.updateCastleRights(move)


    def undoMove(self, is_temp = False):
        """Reverse the last move: restore board, piece positions, game state, and castling rights."""
        # Clear caches since board state is changing and cached values may no longer be valid
        if not is_temp:
            self.attack_cache.clear()
            self.cached_opp_moves = None
            self.cached_opp_turn = None

        # Lightweight temp undo (used by getValidMoves)
        if is_temp:
            # Expect the caller to have set self._temp_move before calling makeMove(..., is_temp=True)
            move = getattr(self, "_temp_move", None)
            if move is None:
                # Defensive fallback: nothing to undo
                return

            # Restore pieces to original squares (lightweight)
            self.board[move.startRow, move.startCol] = move.pieceMoved
            self.board[move.endRow, move.endCol] = move.pieceCaptured

            # Restore player turn
            self.whiteToMove = not self.whiteToMove
            # Restore king locations saved per-candidate (fall back to current values if missing)
            self.whiteKingLocation = getattr(self, "_temp_king_w", self.whiteKingLocation)
            self.blackKingLocation = getattr(self, "_temp_king_b", self.blackKingLocation)
            # Restore en passant and castling rights saved per-candidate
            self.enpassantPossible = getattr(self, "_temp_enpassant", self.enpassantPossible)
            self.currentCastlingRights = getattr(self, "_temp_castle", self.currentCastlingRights)
            # Restore caches saved per-candidate (do not clear them)
            self.attack_cache = getattr(self, "_temp_attack_cache", self.attack_cache)
            self.cached_opp_moves = getattr(self, "_temp_cached_opp_moves", self.cached_opp_moves)
            self.cached_opp_turn = getattr(self, "_temp_cached_opp_turn", self.cached_opp_turn)
            # Clean up temporary attributes to avoid outdated state
            for attr in ("_temp_move", "_temp_enpassant", "_temp_castle", "_temp_king_w", "_temp_king_b"):
                if hasattr(self, attr):
                    delattr(self, attr)
            # Do NOT touch logs or caches and do not check checkmate/stalemate flags here\
            # This is what makes it LIGHTWEIGHT
            return

        # Safety check: don't undo if no moves have been made
        if len(self.moveLog) != 0:
            move = self.moveLog.pop()
        # Restore pieces to original squares
        self.board[move.startRow, move.startCol] = move.pieceMoved
        self.board[move.endRow, move.endCol] = move.pieceCaptured
        # Restore player turn
        self.whiteToMove = not self.whiteToMove            
        # Restore king positions when king has moved
        if move.pieceMoved.code == "wK":
            self.whiteKingLocation = (move.startRow, move.startCol)
        elif move.pieceMoved.code == "bK":
            self.blackKingLocation = (move.startRow, move.startCol)
         
        # Clear check/checkmate/stalemate flags (will be recalculated if needed)
        self.checkmate = False
        self.stalemate = False
            
        # Reverse en passant: restore captured pawn to correct square
        if move.isEnpassantMove:
            self.board[move.endRow, move.endCol] = None
            self.board[move.startRow, move.endCol] = move.pieceCaptured
            
        # Restore en passant state from history
        self.enpassantPossibleLog.pop()
        self.enpassantPossible = self.enpassantPossibleLog[-1]
            
        # Restore castling rights from history
        self.castleRightLog.pop()
        newRights = self.castleRightLog[-1]
        self.currentCastlingRights = CastleRights(
            newRights.wks, newRights.bks, newRights.wqs, newRights.bqs
        )
            
        # Reverse castling: restore rook to corner
        if move.isCastleMove:
            if move.endCol - move.startCol == 2:  # Kingside castling
                self.board[move.endRow, move.endCol + 1] = self.board[move.endRow, move.endCol - 1]
                self.board[move.endRow, move.endCol - 1] = None
            elif (move.endCol - move.startCol == -2):  # Queenside castling
                self.board[move.endRow, move.endCol - 2] = self.board[move.endRow, move.endCol + 1]
                self.board[move.endRow, move.endCol + 1] = None
            
        # Clear game end states
        self.checkmate = False
        self.stalemate = False

    def updateCastleRights(self, move):
        """Remove castling rights when king or rook moves, or when rook is captured."""
        # If king moves, both castling directions become impossible
        if move.pieceMoved.code == "wK":
            self.currentCastlingRights.wks = False
            self.currentCastlingRights.wqs = False
        elif move.pieceMoved.code == "bK":
            self.currentCastlingRights.bks = False
            self.currentCastlingRights.bqs = False
        
        # If rook moves from corner, that side's castling becomes impossible
        elif move.pieceMoved.code == "wR":
            if move.startRow == 7:
                if move.startCol == 0:  # Queenside rook moved
                    self.currentCastlingRights.wqs = False
                if move.startCol == 7:  # Kingside rook moved
                    self.currentCastlingRights.wks = False
        elif move.pieceMoved.code == "bR":
            if move.startRow == 0:
                if move.startCol == 0:  # Queenside rook moved
                    self.currentCastlingRights.bqs = False
                if move.startCol == 7:  # Kingside rook moved
                    self.currentCastlingRights.bks = False
        
        # If rook is captured, opponent loses castling on that side
        if move.pieceCaptured is not None and move.pieceCaptured.code == "wR":
            if move.endRow == 7:
                if move.endCol == 0:
                    self.currentCastlingRights.wqs = False
                elif move.endCol == 7:
                    self.currentCastlingRights.wks = False
        elif move.pieceCaptured is not None and move.pieceCaptured.code == "bR":
            if move.endRow == 0:
                if move.endCol == 0:
                    self.currentCastlingRights.bqs = False
                elif move.endCol == 7:
                    self.currentCastlingRights.bks = False

    def getValidMoves(self):
        """
        Generate all legal moves for current player (excluding moves that leave king in check).
        Detects checkmate and stalemate conditions.
        """
        # Profiling start for getValidMoves
        t0 = prof_start("getValidMoves")
        
        # Generate all pseudo-legal moves (ignoring check)
        moves = self.getAllPossibleMoves()
        
        # Filter out moves that leave own king in check
        # Iterate backwards to safely remove items while iterating
        for i in range(len(moves) - 1, -1, -1):
            move = moves[i]

            # Save per-candidate state so temp undo can restore it exactly
            self._temp_enpassant = self.enpassantPossible
            self._temp_castle = CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
            self._temp_king_w = self.whiteKingLocation
            self._temp_king_b = self.blackKingLocation
            # Save caches so temp moves don't mutate them
            self._temp_attack_cache = self.attack_cache
            self._temp_cached_opp_moves = self.cached_opp_moves
            self._temp_cached_opp_turn = self.cached_opp_turn
            # Determine mover color before applying the temp move
            mover_color = "w" if self.whiteToMove else "b"
            opponent_color = "b" if mover_color == "w" else "w"
            # Make temp move
            self._temp_move = move
            # Check whether the mover's king is attacked by the opponent
            if mover_color == "w":
                king_r, king_c = self.whiteKingLocation
            else:
                king_r, king_c = self.blackKingLocation
            # Store any illegal moves
            illegal = self.squareUnderAttack(king_r, king_c, opponent_color)
            # Undo temp move
            self.undoMove(is_temp=True)
            # Remove illegal moves
            if illegal:
                moves.remove(move)
                
        # Detect checkmate (no legal moves and king in check)
        if len(moves) == 0:
            if self.inCheck():
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.checkmate = False
            self.stalemate = False
                
        # Add castling moves if king hasn't moved
        if self.whiteToMove:
            self.getCastleMoves(self.whiteKingLocation[0], self.whiteKingLocation[1], moves)
        else:
            self.getCastleMoves(self.blackKingLocation[0], self.blackKingLocation[1], moves)

        prof_end("getValidMoves", t0)
        return moves

    def inCheck(self):
        """Check if current player's king is under attack."""
        if self.whiteToMove:
            return self.squareUnderAttack(self.whiteKingLocation[0], self.whiteKingLocation[1], "b")
        else:
            return self.squareUnderAttack(self.blackKingLocation[0], self.blackKingLocation[1], "w")

    def squareUnderAttack(self, r, c, attacker_color=None):
        """
        Return True if square (r, c) is attacked by attacker_color.
        If attacker_color is None, use the side opposite the current turn.
        """
        # Profiling start for squareUnderAttack
        t0 = prof_start("squareUnderAttack")
        # Determine attacker color if not provided
        if attacker_color is None:
            attacker_color = "w" if not self.whiteToMove else "b"
        # Small cache keyed by (r, c, attacker_color)
        key = (r, c, attacker_color)
        if key in self.attack_cache:
            prof_end("squareUnderAttack", t0)
            return self.attack_cache[key]
        # Pawn attacks
        pawn_dir = -1 if attacker_color == "w" else 1
        pr = r + pawn_dir
        for dc in (-1, 1):
            pc = c + dc
            if 0 <= pr < 8 and 0 <= pc < 8:
                p = self.board[pr, pc]
                if p is not None and p.color == attacker_color and p.kind == "p":
                    self.attack_cache[key] = True
                    prof_end("squareUnderAttack", t0)
                    return True
        # Knight attacks
        knight_offsets = (
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        )
        for dr, dc in knight_offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                p = self.board[rr, cc]
                if p is not None and p.color == attacker_color and p.kind == "N":
                    self.attack_cache[key] = True
                    prof_end("squareUnderAttack", t0)
                    return True
        # Diagonal attacks
        diag_dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for dr, dc in diag_dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                p = self.board[rr, cc]
                if p is None:
                    rr += dr
                    cc += dc
                    continue
                if p.color == attacker_color and (p.kind == "B" or p.kind == "Q"):
                    self.attack_cache[key] = True
                    prof_end("squareUnderAttack", t0)
                    return True
                break
        # Straight attacks
        ortho_dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for dr, dc in ortho_dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                p = self.board[rr, cc]
                if p is None:
                    rr += dr
                    cc += dc
                    continue
                if p.color == attacker_color and (p.kind == "R" or p.kind == "Q"):
                    self.attack_cache[key] = True
                    prof_end("squareUnderAttack", t0)
                    return True
                break
        # Adjacent attacks
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    p = self.board[rr, cc]
                    if p is not None and p.color == attacker_color and p.kind == "K":
                        self.attack_cache[key] = True
                        prof_end("squareUnderAttack", t0)
                        return True
        # No attackers found
        self.attack_cache[key] = False
        prof_end("squareUnderAttack", t0)
        return False

        # Opponent king adjacency
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    p = self.board[rr, cc]
                    if p is not None and p.color == attacker_color and p.kind == "K":
                        return True

        return False

    def getAllPossibleMoves(self):
        """Generate all pseudo-legal moves (not filtering out moves leaving king in check)."""
        # profiling start for getAllPossibleMoves
        t0 = prof_start("getAllPossibleMoves")
        moves = []
        # Iterate through all board squares
        for r in range(8):
            for c in range(8):
                piece = self.board[r, c]
                if piece is None:
                    continue
                # Check if piece belongs to current player
                if (piece.color == "w" and self.whiteToMove) or (
                    piece.color == "b" and not self.whiteToMove
                ):
                    # Use dispatch table to call appropriate move generator
                    self.moveFunctions[piece.kind](r, c, moves)
        prof_end("getAllPossibleMoves", t0)
        return moves

    def getPawnMove(self, r, c, moves):
        """Generate all pawn moves: one/two square advances, captures, en passant, promotion."""
        # Pawn direction: white moves up (row -1), black moves down (row +1)
        direction = -1 if self.whiteToMove else 1
        enemyColor = "b" if self.whiteToMove else "w"
        startRow = 6 if self.whiteToMove else 1
        
        # Forward movement: one square
        forward_row = r + direction
        if 0 <= forward_row < 8 and self.board[forward_row, c] is None:
            moves.append(Move((r, c), (forward_row, c), self.board))
            
            # Two square advance from starting position
            two_forward_row = r + 2 * direction
            if r == startRow and self.board[two_forward_row, c] is None:
                moves.append(Move((r, c), (two_forward_row, c), self.board))
        
        # Diagonal captures and en passant
        for delta_col in (-1, 1):
            end_col = c + delta_col
            if 0 <= end_col < 8 and 0 <= forward_row < 8:
                end_piece = self.board[forward_row, end_col]
                # Regular capture
                if end_piece is not None and end_piece.color == enemyColor:
                    moves.append(Move((r, c), (forward_row, end_col), self.board))
                # En passant capture
                elif (forward_row, end_col) == self.enpassantPossible:
                    moves.append(
                        Move((r, c), (forward_row, end_col), self.board, isEnpassantMove=True)
                    )

    def getKnightMove(self, r, c, moves):
        """Generate all knight moves: up to 8 L-shaped moves."""
        knightMoves = (
            (-1, -2), (-1, 2), (-2, -1), (-2, 1),
            (1, -2), (1, 2), (2, -1), (2, 1),
        )
        allyColor = "w" if self.whiteToMove else "b"
        # Check each L-shaped destination
        for dr, dc in knightMoves:
            endRow = r + dr
            endCol = c + dc
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow, endCol]
                # Can move to empty square or capture enemy piece
                if endPiece is None or endPiece.color != allyColor:
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    def getBishopMove(self, r, c, moves):
        """Generate all bishop moves: sliding diagonally until blocked."""
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        enemyColor = "b" if self.whiteToMove else "w"
        # Check each diagonal direction
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                # Stop if off board
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow, endCol]
                    # Empty square: add move and continue
                    if endPiece is None:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    # Enemy piece: add capture and stop
                    elif endPiece.color == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    # Ally piece: stop without adding
                    else:
                        break
                else:
                    break

    def getRookMove(self, r, c, moves):
        """Generate all rook moves: sliding horizontally/vertically until blocked."""
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
        enemyColor = "b" if self.whiteToMove else "w"
        # Check each orthogonal direction
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                # Stop if off board
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow, endCol]
                    # Empty square: add move and continue
                    if endPiece is None:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    # Enemy piece: add capture and stop
                    elif endPiece.color == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    # Ally piece: stop without adding
                    else:
                        break
                else:
                    break

    def getQueenMove(self, r, c, moves):
        """Generate all queen moves: combines bishop and rook movement."""
        self.getBishopMove(r, c, moves)
        self.getRookMove(r, c, moves)

    def getKingMove(self, r, c, moves):
        """Generate all king moves: up to 8 adjacent squares."""
        kingMoves = (
            (-1, 0), (0, -1), (1, 0), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        allyColor = "w" if self.whiteToMove else "b"
        enemyColor = "b" if allyColor == "w" else "w"
        original_king_pos = (r, c)
        # Check each adjacent square
        for dr, dc in kingMoves:
            endRow = r + dr
            endCol = c + dc
             # Skip off-board squares
            if not (0 <= endRow < 8 and 0 <= endCol < 8):
                continue
            # Skip squares occupied by our own piece
            endPiece = self.board[endRow, endCol]
            if endPiece is not None and endPiece.color == allyColor:
                continue
            # Temporarily move king on the board to test safety
            saved_src = self.board[r, c]
            saved_dest = self.board[endRow, endCol]
            self.board[r, c] = None
            self.board[endRow, endCol] = King(allyColor)
            # update king location for inCheck / attack helpers
            if allyColor == "w":
                self.whiteKingLocation = (endRow, endCol)
            else:
                self.blackKingLocation = (endRow, endCol)
            # Check if this destination is attacked by enemy
            attacked = self.squareUnderAttack(endRow, endCol, enemyColor)
            # Restore board and king location
            self.board[r, c] = saved_src
            self.board[endRow, endCol] = saved_dest
            if allyColor == "w":
                self.whiteKingLocation = original_king_pos
            else:
                self.blackKingLocation = original_king_pos
            # Only add the move if the destination is not attacked
            if not attacked:
                moves.append(Move((r, c), (endRow, endCol), self.board))

    def getCastleMoves(self, r, c, moves):
        """Add castling moves if king not in check and has castling rights."""
        # Can't castle if in check
        if self.squareUnderAttack(r, c):
            return
        # Check kingside (right) castling
        if (self.whiteToMove and self.currentCastlingRights.wks) or (
            not self.whiteToMove and self.currentCastlingRights.bks
        ):
            self.getKingSideCastleMoves(r, c, moves)
        # Check queenside (left) castling
        if (self.whiteToMove and self.currentCastlingRights.wqs) or (
            not self.whiteToMove and self.currentCastlingRights.bqs
        ):
            self.getQueenSideCastleMoves(r, c, moves)

    def getKingSideCastleMoves(self, r, c, moves):
        """Kingside (right) castling: king moves 2 squares right, f/g files must be empty and unattacked."""
        if self.board[r, c + 1] is None and self.board[r, c + 2] is None:
            if not self.squareUnderAttack(r, c + 1) and not self.squareUnderAttack(
                r, c + 2
            ):
                moves.append(Move((r, c), (r, c + 2), self.board, isCastleMove=True))

    def getQueenSideCastleMoves(self, r, c, moves):
        """Queenside (left) castling: king moves 2 squares left, b/c/d files must be empty and unattacked."""
        if (
            self.board[r, c - 1] is None
            and self.board[r, c - 2] is None
            and self.board[r, c - 3] is None
        ):
            if not self.squareUnderAttack(r, c - 1) and not self.squareUnderAttack(
                r, c - 2
            ):
                moves.append(Move((r, c), (r, c - 2), self.board, isCastleMove=True))


class CastleRights:
    """Track castling availability: wks/bks/wqs/bqs for kingside/queenside white/black."""
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks  # White kingside castling
        self.bks = bks  # Black kingside castling
        self.wqs = wqs  # White queenside castling
        self.bqs = bqs  # Black queenside castling


class Move:
    """Represents a chess move: start/end squares, piece moved/captured, special moves."""
    # Conversion tables between algebraic notation and array indices
    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "0": 0}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}
    fileToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    colsToFiles = {v: k for k, v in fileToCols.items()}

    def __init__(
        self, startSq, endSq, board, isEnpassantMove=False, isCastleMove=False
    ):
        # Parse square tuples into row/col coordinates
        self.startRow, self.startCol = startSq
        self.endRow, self.endCol = endSq
        # Piece information from board state
        self.pieceMoved = board[self.startRow, self.startCol]
        self.pieceCaptured = board[self.endRow, self.endCol]
        # Store the board
        self.board = board

        # Store all same-type piece positions at move time for PGN support
        self.sameTypePieceSquares = []
        for r in range(8):
            for c in range(8):
                p = board[r, c]
                if (p is not None and self.pieceMoved is not None and p.kind == self.pieceMoved.kind and p.color == self.pieceMoved.color):
                    self.sameTypePieceSquares.append((r, c))

        # En passant: flag indicates captured pawn on different square
        self.isEnpassantMove = isEnpassantMove
        if self.isEnpassantMove:
            self.pieceCaptured = Pawn("w") if self.pieceMoved.code == "bp" else Pawn("b")

        # Pawn promotion: flag indicates pawn reached promotion rank
        self.isPawnPromotion = (
            self.pieceMoved.code == "wp" and self.endRow == 0
        ) or (self.pieceMoved.code == "bp" and self.endRow == 7)

        # Castling: flag indicates king moved 2 squares (rook moves auto-handled)
        self.isCastleMove = isCastleMove

        # Capture detection: true if destination had enemy piece
        self.isCapture = self.pieceCaptured is not None

        # Promotion choice detection: detect what piece the player has chosen to promote their pawn to
        self.promote_to = None

        # Allow for either "+" to be written if a move resulted in check or "#" if a move resulted in checkmate
        self.postMoveSuffix = ""

        # Unique move identifier for transposition tables and killer move heuristic
        self.moveID = (
            self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        )

    def __eq__(self, other):
        """Two moves are equal if their moveIDs match (same source/destination)."""
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getChessNotation(self):
        """Return long algebraic notation: e2e4 (start + end squares)."""
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(
            self.endRow, self.endCol
        )

    def getRankFile(self, r, c):
        """Convert board coordinates (row, col) to algebraic notation (file + rank)."""
        return self.colsToFiles[c] + self.rowsToRanks[r]

    def __str__(self):
        """Return move notation with PGN support: O-O (kingside), O-O-O (queenside), or piece notation."""
        if self.isCastleMove:
            return "O-O" if self.endCol == 6 else "O-O-O"
        # Destination square for all moves
        endSquare = self.getRankFile(self.endRow, self.endCol)
        notation = ""
        # Pawn notation: just destination (or filex destination for captures)
        if self.pieceMoved.kind == "p":
            # Capture notation
            if self.isCapture:
                notation = self.colsToFiles[self.startCol] + "x" + endSquare
            else:
                notation = endSquare
            # Promotion
            if self.isPawnPromotion:
                promoted_letter = {Queen: "Q", Rook: "R", Bishop: "B", Knight: "N"}.get(self.promote_to, "Q")
                notation += "=" + promoted_letter
            # Add "+" or "#" symbols if move led to check or checkmate respectively
            return notation + self.postMoveSuffix
        
        # Default: no disambiguation
        disambig = ""
        # We need access to all legal moves from the same position.
        # The Move object already stores the board, but not the GameState.
        # So we detect disambiguation using the board directly.
        # We search for other pieces of the same type that can also move to this square.
        same_piece_moves = []
        # Get snapshot of board and check for identical pieces
        for (r, c) in self.sameTypePieceSquares:
            if (r, c) == (self.startRow, self.startCol): # Iterate to next square if the same square is being checked
                continue
            # Store all possible moves by piece
            if self.isValidForPGN(r, c):
                same_piece_moves.append((r, c))
        # Apply PGN disambiguation rules
        if same_piece_moves:
            same_file = any(c == self.startCol for (r, c) in same_piece_moves)
            same_rank = any(r == self.startRow for (r, c) in same_piece_moves)
            # Set disambig variable if detected
            if same_file and same_rank:
                disambig = self.colsToFiles[self.startCol] + self.rowsToRanks[self.startRow]
            elif same_file:
                disambig = self.rowsToRanks[self.startRow]
            else:
                disambig = self.colsToFiles[self.startCol]

        # Piece notation: piece symbol + disambiguation (if any) + x for captures + destination
        notation = self.pieceMoved.kind + disambig
        if self.isCapture:
            notation += "x"
        notation += endSquare
        # Add "+" or "#" symbols if move led to check or checkmate respectively
        return notation + self.postMoveSuffix
    
    def isValidForPGN(self, r, c):
        """Check if this move matches the movement pattern of the piece."""
        dr = self.endRow - r
        dc = self.endCol - c
        p = self.pieceMoved.kind

        if p == "N":  # Knight
            return (abs(dr), abs(dc)) in [(1, 2), (2, 1)]

        if p == "B":  # Bishop
            return abs(dr) == abs(dc)

        if p == "R":  # Rook
            return dr == 0 or dc == 0

        if p == "Q":  # Queen
            return dr == 0 or dc == 0 or abs(dr) == abs(dc)

        if p == "K":  # King
            return max(abs(dr), abs(dc)) == 1

        return False