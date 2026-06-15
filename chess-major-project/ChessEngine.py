"""
This is responsiwle for storing all the information about the current state
of the chess game. Also, be responsible for determining the valid moves at
the current state. And it'll keep a move log.
"""

import numpy as np


class Piece:
    def __init__(self, color, kind):
        self.color = color  # 'w' or 'b'
        self.kind = kind  # 'p', 'N', 'B', 'R', 'Q', 'K'

    @property
    def code(self):
        return f"{self.color}{self.kind}"

    def __str__(self):
        return self.code

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.color}')"


class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color, "p")


class Knight(Piece):
    def __init__(self, color):
        super().__init__(color, "N")


class Bishop(Piece):
    def __init__(self, color):
        super().__init__(color, "B")


class Rook(Piece):
    def __init__(self, color):
        super().__init__(color, "R")


class Queen(Piece):
    def __init__(self, color):
        super().__init__(color, "Q")


class King(Piece):
    def __init__(self, color):
        super().__init__(color, "K")


def create_starting_board():
    board = np.full((8, 8), None, dtype=object)
    board[0] = [Rook("b"), Knight("b"), Bishop("b"), Queen("b"), King("b"), Bishop("b"), Knight("b"), Rook("b")]
    board[1] = [Pawn("b") for _ in range(8)]
    board[6] = [Pawn("w") for _ in range(8)]
    board[7] = [Rook("w"), Knight("w"), Bishop("w"), Queen("w"), King("w"), Bishop("w"), Knight("w"), Rook("w")]
    return board


class GameState:
    def __init__(self):
        # this is a 2d representation of the board from white prespective
        # to gain some more speed, we might use numpy library instead
        # the representation is pretty easy:
        # the first character is about the piece color: b = black, w = white
        # and the second one is the piece standard notation:
        # K = King, Q = Queen, R = Rook, B = Bishop, N = Knight, p = pawn
        # finally, "--" is for empty squares
        self.board = create_starting_board()

        self.whiteToMove = True
        self.moveLog = []  # Move objects
        self.moveFunctions = {
            "p": self.getPawnMove,
            "N": self.getKnightMove,
            "B": self.getBishopMove,
            "R": self.getRockMove,
            "Q": self.getQueenMove,
            "K": self.getKingMove,
        }
        # to keep track of the kings locations becuase:
        # castling, checks, checkmates and stalemates
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)
        self.checkmate = False
        self.stalemate = False
        # the corrdinates where an enpassant capture is possible
        self.enpassantPossible = ()
        self.enpassantPossibleLog = [self.enpassantPossible]
        self.currentCastlingRights = CastleRights(True, True, True, True)
        self.castleRightLog = [
            CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
        ]

    """
    This functions takes a move as a parameter and executes it
    Note: this won't work now quite well for castling, pawn promotion and en-passant
    """

    def makeMove(self, move):
        self.board[move.startRow, move.startCol] = None
        self.board[move.endRow, move.endCol] = move.pieceMoved
        # log the move, so we can undo it later or print a PNG for the game
        self.moveLog.append(move)
        self.whiteToMove = not self.whiteToMove  # switch turns
        # update the both of the kings location after making a move
        if move.pieceMoved.code == "wK":
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved.code == "bK":
            self.blackKingLocation = (move.endRow, move.endCol)
        # about pawn promotions, we'll consider the queen promotion at first:
        if move.isPawnPromotion:
            self.board[move.endRow, move.endCol] = Queen(move.pieceMoved.color)
        # about enpassant move
        if move.isEnpassantMove:
            self.board[move.startRow, move.endCol] = None  # capturing the pawn
        # update the enpassantPossible variable
        # only for 2 square pawn advance
        if move.pieceMoved.kind == "p" and abs(move.startRow - move.endRow) == 2:
            self.enpassantPossible = ((move.startRow + move.endRow) // 2, move.startCol)
        else:
            self.enpassantPossible = ()
        # about castling
        if move.isCastleMove:
            # we need to check to see if it castles to left or right
            if move.endCol - move.startCol == 2:  # to the right: king side castle
                self.board[move.endRow, move.endCol - 1] = self.board[move.endRow, move.endCol + 1]
                self.board[move.endRow, move.endCol + 1] = None  # remove the old rook
            elif move.endCol - move.startCol == -2:  # to the left: queen side castle
                self.board[move.endRow, move.endCol + 1] = self.board[move.endRow, move.endCol - 2]
                self.board[move.endRow, move.endCol - 2] = None  # remove the old rook
        # update the enpassantPossibleLog
        self.enpassantPossibleLog.append(self.enpassantPossible)
        # update the castling rights whenever its a rook or a king move
        self.updateCastlRights(move)
        self.castleRightLog.append(
            CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
        )

    """ undo the last move made on the board """

    def undoMove(self):
        # first let's make sure that there's a move to undo
        if len(self.moveLog) != 0:
            move = self.moveLog.pop()
            self.board[move.startRow, move.startCol] = move.pieceMoved
            self.board[move.endRow, move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove  # switch turns
            # update the both of the kings location after undo a move
            if move.pieceMoved.code == "wK":
                self.whiteKingLocation = (move.startRow, move.startCol)
            elif move.pieceMoved.code == "bK":
                self.blackKingLocation = (move.startRow, move.startCol)
            # delete checkmate and stalemate states
            self.checkmate = False
            self.stalemate = False
            # undo the enpassant move
            if move.isEnpassantMove:
                self.board[move.endRow, move.endCol] = None
                self.board[move.startRow, move.endCol] = move.pieceCaptured
            self.enpassantPossibleLog.pop()
            self.enpassantPossible = self.enpassantPossibleLog[-1]
            # undo the castle rights
            # first get rid of the new castle rights from the move we're undoing
            self.castleRightLog.pop()
            # then set the currentCastlingRights to last one we have now on the log list
            newRights = self.castleRightLog[-1]
            self.currentCastlingRights = CastleRights(
                newRights.wks, newRights.bks, newRights.wqs, newRights.bqs
            )
            # undo the castle move
            if move.isCastleMove:
                # we need to check to see if it castles to left or right
                if move.endCol - move.startCol == 2:  # to the right: king side castle
                    self.board[move.endRow, move.endCol + 1] = self.board[move.endRow, move.endCol - 1]
                    self.board[move.endRow, move.endCol - 1] = None
                elif (
                    move.endCol - move.startCol == -2
                ):  # to the left: queen side castle
                    self.board[move.endRow, move.endCol - 2] = self.board[move.endRow, move.endCol + 1]
                    self.board[move.endRow, move.endCol + 1] = None
            # undo the checkmate and stalemate
            self.checkmate = False
            self.stalemate = False

    """ update the casle rights given a move """

    def updateCastlRights(self, move):
        # check if the king moved or the rook moved
        if move.pieceMoved.code == "wK":
            self.currentCastlingRights.wks = False
            self.currentCastlingRights.wqs = False
        elif move.pieceMoved.code == "bK":
            self.currentCastlingRights.bks = False
            self.currentCastlingRights.bqs = False
        elif move.pieceMoved.code == "wR":
            if move.startRow == 7:
                if move.startCol == 0:  # whites left rook
                    self.currentCastlingRights.wqs = False
                if move.startCol == 7:  # whites right rook
                    self.currentCastlingRights.wks = False
        elif move.pieceMoved.code == "bR":
            if move.startRow == 0:
                if move.startCol == 0:  # black left rook
                    self.currentCastlingRights.bqs = False
                if move.startCol == 7:  # black right rook
                    self.currentCastlingRights.bks = False
        # check if the rook is captured
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

    """ all moves considering even if the king is in check """

    def getValidMoves(self):
        tempEnpassantPossible = self.enpassantPossible
        tempCastleRights = CastleRights(
            self.currentCastlingRights.wks,
            self.currentCastlingRights.bks,
            self.currentCastlingRights.wqs,
            self.currentCastlingRights.bqs,
        )
        # the easy, but not efficient solution is:
        # 1. let's generate all the possible moves and don't worry about the kings state
        moves = self.getAllPossibleMoves()
        # 2. for each move found, make that move
        # when removing from the list, go backwords :)
        for i in range(len(moves) - 1, -1, -1):
            self.makeMove(moves[i])
            # 3. generate all opponent's moves
            # 4. for each of those moves, check if they attack your king
            # we need this as the makeMove() did swap the players once
            self.whiteToMove = not self.whiteToMove
            if self.inCheck():
                # 5. if they do, it's not a valid move
                moves.remove(moves[i])
            # we need this to return every thing as before
            self.whiteToMove = not self.whiteToMove
            self.undoMove()
        # do we have a checkmate |:) or stalemate (:|
        if len(moves) == 0:
            if self.inCheck():
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.checkmate = False
            self.stalemate = False
        # to generate castle moves
        if self.whiteToMove:
            self.getCastleMoves(
                self.whiteKingLocation[0], self.whiteKingLocation[1], moves
            )
        else:
            self.getCastleMoves(
                self.blackKingLocation[0], self.blackKingLocation[1], moves
            )
        self.enpassantPossible = tempEnpassantPossible
        self.currentCastlingRights = tempCastleRights
        return moves

    """ to determine if the current player is in check """

    def inCheck(self):
        if self.whiteToMove:
            return self.squareUnderAttack(
                self.whiteKingLocation[0], self.whiteKingLocation[1]
            )
        else:
            return self.squareUnderAttack(
                self.blackKingLocation[0], self.blackKingLocation[1]
            )

    """ to determine if the enemy can attack the square(r, c) """

    def squareUnderAttack(self, r, c):
        # first switch to the opponents move
        self.whiteToMove = not self.whiteToMove
        # generate all of its moves
        oppMoves = self.getAllPossibleMoves()
        self.whiteToMove = not self.whiteToMove  # switch the turns back
        # check if any of those moves is attacking my kings location
        for move in oppMoves:
            if move.endRow == r and move.endCol == c:  # now my king is under attack
                # note: we have done the move on our back end, so no need to undoMove() it
                return True
        return False  # none of my opponents move will be attacking my king

    """ all moves without considering checks """

    def getAllPossibleMoves(self):
        moves = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r, c]
                if piece is None:
                    continue
                if (piece.color == "w" and self.whiteToMove) or (
                    piece.color == "b" and not self.whiteToMove
                ):
                    self.moveFunctions[piece.kind](r, c, moves)
        return moves

    """ all moves for a pawn located at row:r and column:c
    then this move to the list """

    def getPawnMove(self, r, c, moves):
        direction = -1 if self.whiteToMove else 1
        enemyColor = "b" if self.whiteToMove else "w"
        startRow = 6 if self.whiteToMove else 1
        # move one square forward
        forward_row = r + direction
        if 0 <= forward_row < 8 and self.board[forward_row, c] is None:
            moves.append(Move((r, c), (forward_row, c), self.board))
            # move two squares from the starting rank
            two_forward_row = r + 2 * direction
            if r == startRow and self.board[two_forward_row, c] is None:
                moves.append(Move((r, c), (two_forward_row, c), self.board))
        # captures
        for delta_col in (-1, 1):
            end_col = c + delta_col
            if 0 <= end_col < 8 and 0 <= forward_row < 8:
                end_piece = self.board[forward_row, end_col]
                if end_piece is not None and end_piece.color == enemyColor:
                    moves.append(Move((r, c), (forward_row, end_col), self.board))
                elif (forward_row, end_col) == self.enpassantPossible:
                    moves.append(
                        Move((r, c), (forward_row, end_col), self.board, isEnpassantMove=True)
                    )

    """ all moves for a knight located at row:r and column:c
    then this move to the list """

    def getKnightMove(self, r, c, moves):
        knightMoves = (
            (-1, -2),
            (-1, 2),
            (-2, -1),
            (-2, 1),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )
        allyColor = "w" if self.whiteToMove else "b"
        for dr, dc in knightMoves:
            endRow = r + dr
            endCol = c + dc
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow, endCol]
                if endPiece is None or endPiece.color != allyColor:
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    """ all moves for a bishop located at row:r and column:c
    then this move to the list """

    def getBishopMove(self, r, c, moves):
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        enemyColor = "b" if self.whiteToMove else "w"
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow, endCol]
                    if endPiece is None:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece.color == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break

    """ all moves for a rock located at row:r and column:c
    then this move to the list """

    def getRockMove(self, r, c, moves):
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
        enemyColor = "b" if self.whiteToMove else "w"
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow, endCol]
                    if endPiece is None:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece.color == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break

    """ all moves for a queen located at row:r and column:c
    then this move to the list """

    def getQueenMove(self, r, c, moves):
        self.getBishopMove(r, c, moves)
        self.getRockMove(r, c, moves)

    """ all moves for a king located at row:r and column:c
    then this move to the list """

    def getKingMove(self, r, c, moves):
        kingMoves = (
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
        allyColor = "w" if self.whiteToMove else "b"
        for dr, dc in kingMoves:
            endRow = r + dr
            endCol = c + dc
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow, endCol]
                if endPiece is None or endPiece.color != allyColor:
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    """ generate all valid castle moves for king(r, c) and add them to the list of moves """

    def getCastleMoves(self, r, c, moves):
        if self.squareUnderAttack(r, c):
            return
        if (self.whiteToMove and self.currentCastlingRights.wks) or (
            not self.whiteToMove and self.currentCastlingRights.bks
        ):
            self.getKingSideCastleMoves(r, c, moves)
        if (self.whiteToMove and self.currentCastlingRights.wqs) or (
            not self.whiteToMove and self.currentCastlingRights.bqs
        ):
            self.getQueenSideCastleMoves(r, c, moves)

    def getKingSideCastleMoves(self, r, c, moves):
        if self.board[r, c + 1] is None and self.board[r, c + 2] is None:
            if not self.squareUnderAttack(r, c + 1) and not self.squareUnderAttack(
                r, c + 2
            ):
                moves.append(Move((r, c), (r, c + 2), self.board, isCastleMove=True))

    def getQueenSideCastleMoves(self, r, c, moves):
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
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs


class Move:
    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "0": 0}

    rowsToRanks = {v: k for k, v in ranksToRows.items()}

    fileToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    colsToFiles = {v: k for k, v in fileToCols.items()}

    def __init__(
        self, startSq, endSq, board, isEnpassantMove=False, isCastleMove=False
    ):
        self.startRow, self.startCol = startSq
        self.endRow, self.endCol = endSq
        self.pieceMoved = board[self.startRow, self.startCol]
        self.pieceCaptured = board[self.endRow, self.endCol]

        # enpassant move
        self.isEnpassantMove = isEnpassantMove
        if self.isEnpassantMove:
            self.pieceCaptured = Pawn("w") if self.pieceMoved.code == "bp" else Pawn("b")

        # pawn promotion move
        self.isPawnPromotion = (
            self.pieceMoved.code == "wp" and self.endRow == 0
        ) or (self.pieceMoved.code == "bp" and self.endRow == 7)

        # castle move
        self.isCastleMove = isCastleMove

        # see if the move was a capture move or not
        self.isCapture = self.pieceCaptured is not None

        # a unique id for each move in the range of 0 and 7777
        self.moveID = (
            self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        )
        # print(self.moveID) # for debugging

    """ overriding the equals method: maybe like copy or move constructors """

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getChessNotation(self):
        # this can be modified to be a more real chess notation
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(
            self.endRow, self.endCol
        )

    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]

    def __str__(self):
        if self.isCastleMove:
            return "O-O" if self.endCol == 6 else "O-O-O"
        endSquare = self.getRankFile(self.endRow, self.endCol)
        if self.pieceMoved.kind == "p":
            if self.isCapture:
                return self.colsToFiles[self.startCol] + "x" + endSquare
            else:
                return endSquare
        moveString = self.pieceMoved.kind
        if self.isCapture:
            moveString += "x"
        return moveString + endSquare
