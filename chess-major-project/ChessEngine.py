"""
Chess game state management and move generation engine.
Handles board representation, legal move generation, special moves (castling, en passant, promotion).
Maintains move history and game state (checkmate, stalemate, check).
"""

import numpy as np


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

    def makeMove(self, move):
        """Execute move: update board, track history, handle special moves, update game state."""
        # Update board: remove piece from start, place at end
        self.board[move.startRow, move.startCol] = None
        self.board[move.endRow, move.endCol] = move.pieceMoved
        # Record move in history for undo and game replay
        self.moveLog.append(move)
        # Switch player turn
        self.whiteToMove = not self.whiteToMove
        
        # Update king location when king moves (needed for check detection and castling)
        if move.pieceMoved.code == "wK":
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved.code == "bK":
            self.blackKingLocation = (move.endRow, move.endCol)
        
        # Pawn promotion: convert pawn to queen when reaching promotion rank
        if move.isPawnPromotion:
            self.board[move.endRow, move.endCol] = Queen(move.pieceMoved.color)
        
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
        self.enpassantPossibleLog.append(self.enpassantPossible)
        # Update castling rights if king or rook moved
        self.updateCastlRights(move)
        # Record castling rights in history for undo
        self.castleRightLog.append(
            CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
        )


    def undoMove(self):
        """Reverse the last move: restore board, piece positions, game state, and castling rights."""
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

    def updateCastlRights(self, move):
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
        # Save current en passant and castling state to restore after validity check
        tempEnpassantPossible = self.enpassantPossible
        tempCastleRights = CastleRights(
            self.currentCastlingRights.wks,
            self.currentCastlingRights.bks,
            self.currentCastlingRights.wqs,
            self.currentCastlingRights.bqs,
        )
        
        # Generate all pseudo-legal moves (ignoring check)
        moves = self.getAllPossibleMoves()
        
        # Filter out moves that leave own king in check
        # Iterate backwards to safely remove items while iterating
        for i in range(len(moves) - 1, -1, -1):
            # Make move temporarily
            self.makeMove(moves[i])
            # Switch perspective to check if opponent attacks our king
            self.whiteToMove = not self.whiteToMove
            # If king is under attack, this move is illegal
            if self.inCheck():
                moves.remove(moves[i])
            # Restore perspective
            self.whiteToMove = not self.whiteToMove
            # Undo temporary move
            self.undoMove()
        
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
            self.getCastleMoves(
                self.whiteKingLocation[0], self.whiteKingLocation[1], moves
            )
        else:
            self.getCastleMoves(
                self.blackKingLocation[0], self.blackKingLocation[1], moves
            )
        
        # Restore temporary game state
        self.enpassantPossible = tempEnpassantPossible
        self.currentCastlingRights = tempCastleRights
        return moves

    def inCheck(self):
        """Check if current player's king is under attack."""
        if self.whiteToMove:
            return self.squareUnderAttack(
                self.whiteKingLocation[0], self.whiteKingLocation[1]
            )
        else:
            return self.squareUnderAttack(
                self.blackKingLocation[0], self.blackKingLocation[1]
            )

    def squareUnderAttack(self, r, c):
        """Check if any opponent piece can attack square (r, c)."""
        # Temporarily switch to opponent's perspective
        self.whiteToMove = not self.whiteToMove
        # Generate all opponent's possible moves
        oppMoves = self.getAllPossibleMoves()
        # Switch back
        self.whiteToMove = not self.whiteToMove
        # Check if any opponent move attacks this square
        for move in oppMoves:
            if move.endRow == r and move.endCol == c:
                return True
        return False

    def getAllPossibleMoves(self):
        """Generate all pseudo-legal moves (not filtering out moves leaving king in check)."""
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
        # Check each adjacent square
        for dr, dc in kingMoves:
            endRow = r + dr
            endCol = c + dc
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow, endCol]
                # Can move to empty square or capture enemy piece
                if endPiece is None or endPiece.color != allyColor:
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
        """Return move notation: O-O (kingside), O-O-O (queenside), or piece notation."""
        if self.isCastleMove:
            return "O-O" if self.endCol == 6 else "O-O-O"
        # Destination square for all moves
        endSquare = self.getRankFile(self.endRow, self.endCol)
        # Pawn notation: just destination (or filex destination for captures)
        if self.pieceMoved.kind == "p":
            if self.isCapture:
                return self.colsToFiles[self.startCol] + "x" + endSquare
            else:
                return endSquare
        # Piece notation: piece symbol + x for captures + destination
        moveString = self.pieceMoved.kind
        if self.isCapture:
            moveString += "x"
        return moveString + endSquare
