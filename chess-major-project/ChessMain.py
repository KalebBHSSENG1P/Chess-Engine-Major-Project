"""
This is our main driver. It is responsible for handling the UI, game loop,
and connecting the chess engine with the AI.
"""

import os
import pygame as p
from multiprocessing import Process, Queue
from queue import Empty

from ChessEngine import GameState, Move
from SmartMoveFinder import ChessAI

# window and board layout constants
BOARD_WIDTH = BOARD_HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 270
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT
DIMENSION = 8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images")


class ChessApp:
    def __init__(self):
        p.init()
        p.display.set_caption("Chess")
        self.screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
        self.clock = p.time.Clock()
        self.move_log_font = p.font.SysFont("Arial", 20, False, False)
        self.images = {}
        self.load_images()
        self.reset_game()

    def reset_game(self):
        self.gs = GameState()
        self.valid_moves = self.gs.getValidMoves()
        self.move_made = False
        self.animate = False
        self.game_over = False
        self.sq_selected = ()
        self.player_clicks = []
        self.player_one = True
        self.player_two = False
        self.ai_thinking = False
        self.move_finder_process = None
        self.return_queue = None
        self.move_undone = False

    def load_images(self):
        pieces = ["wp", "wN", "wB", "wR", "wQ", "wK", "bp", "bN", "bB", "bR", "bQ", "bK"]
        for piece in pieces:
            image_file = os.path.join(IMAGE_PATH, piece + ".png")
            self.images[piece] = p.transform.scale(p.image.load(image_file), (SQ_SIZE, SQ_SIZE))

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            if self.move_made:
                self.update_after_move()
            self.handle_ai_move()
            self.draw_game_state()
            if self.gs.checkmate or self.gs.stalemate:
                game_over_text = (
                    "Stalemate"
                    if self.gs.stalemate
                    else "Black wins by checkmate"
                    if self.gs.whiteToMove
                    else "White wins by checkmate"
                )
                self.draw_end_game_text(game_over_text)
            self.clock.tick(MAX_FPS)
            p.display.flip()

    def handle_events(self):
        for event in p.event.get():
            if event.type == p.QUIT:
                return False
            if event.type == p.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event)
            elif event.type == p.KEYDOWN:
                self.handle_key_down(event)
        return True

    def handle_mouse_click(self, event):
        if self.game_over:
            return
        location = p.mouse.get_pos()
        col = location[0] // SQ_SIZE
        row = location[1] // SQ_SIZE
        if col >= DIMENSION:
            self.sq_selected = ()
            self.player_clicks = []
            return
        current_color = "w" if self.gs.whiteToMove else "b"
        if self.sq_selected == (row, col):
            self.sq_selected = ()
            self.player_clicks = []
        else:
            if len(self.player_clicks) == 0:
                selected_piece = self.gs.board[row, col]
                if selected_piece is None or selected_piece.color != current_color:
                    return
            self.sq_selected = (row, col)
            self.player_clicks.append(self.sq_selected)
        if len(self.player_clicks) == 2 and self.human_turn():
            self.try_player_move()

    def handle_key_down(self, event):
        if event.key == p.K_z:
            self.undo_move()
        elif event.key == p.K_r:
            self.reset_game()

    def human_turn(self):
        return (self.gs.whiteToMove and self.player_one) or (not self.gs.whiteToMove and self.player_two)

    def try_player_move(self):
        move = Move(self.player_clicks[0], self.player_clicks[1], self.gs.board)
        for valid_move in self.valid_moves:
            if move == valid_move:
                self.gs.makeMove(valid_move)
                self.move_made = True
                self.animate = True
                self.sq_selected = ()
                self.player_clicks = []
                return
        self.player_clicks = [self.sq_selected]

    def undo_move(self):
        self.gs.undoMove()
        self.sq_selected = ()
        self.player_clicks = []
        self.move_made = True
        self.animate = False
        self.game_over = False
        if self.ai_thinking and self.move_finder_process is not None:
            self.move_finder_process.terminate()
            self.ai_thinking = False
        self.move_undone = True

    def handle_ai_move(self):
        if self.game_over or self.human_turn() or self.move_undone:
            return
        if not self.ai_thinking:
            self.ai_thinking = True
            print("[ChessApp] AI thinking started")
            self.return_queue = Queue()
            self.move_finder_process = Process(
                target=ChessAI.find_best_move_minmax,
                args=(self.gs, self.valid_moves, self.return_queue),
            )
            self.move_finder_process.start()
        if self.return_queue is not None and not self.move_finder_process.is_alive():
            try:
                ai_move = self.return_queue.get_nowait()
            except Empty:
                ai_move = None
            if ai_move is None:
                print("[ChessApp] AI failed to return a move, choosing random fallback")
                ai_move = ChessAI.find_random_move(self.valid_moves)
            print(f"[ChessApp] AI selected move: {ai_move}")
            self.gs.makeMove(ai_move)
            self.move_made = True
            self.animate = True
            self.ai_thinking = False

    def update_after_move(self):
        if self.animate and self.gs.moveLog:
            self.animate_move(self.gs.moveLog[-1])
        self.valid_moves = self.gs.getValidMoves()
        self.move_made = False
        self.animate = False
        self.move_undone = False

    def draw_game_state(self):
        self.draw_board()
        self.highlight_squares()
        self.draw_pieces()
        self.draw_move_log()

    def draw_board(self):
        colors = [p.Color("white"), p.Color("gray")]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r + c) % 2]
                p.draw.rect(
                    self.screen,
                    color,
                    p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                )

    def highlight_squares(self):
        if self.sq_selected == ():
            return
        r, c = self.sq_selected
        selected_piece = self.gs.board[r, c]
        if selected_piece is None or selected_piece.color != ("w" if self.gs.whiteToMove else "b"):
            return
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)
        s.fill(p.Color("blue"))
        self.screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
        s.fill(p.Color("yellow"))
        for move in self.valid_moves:
            if move.startRow == r and move.startCol == c:
                self.screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))

    def draw_pieces(self):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.gs.board[r, c]
                if piece is not None:
                    self.screen.blit(
                        self.images[piece.code],
                        p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                    )

    def animate_move(self, move):
        dR = move.endRow - move.startRow
        dC = move.endCol - move.startCol
        frames_per_square = 10
        frame_count = (abs(dR) + abs(dC)) * frames_per_square
        for frame in range(frame_count + 1):
            r = move.startRow + dR * frame / frame_count
            c = move.startCol + dC * frame / frame_count
            self.draw_board()
            self.draw_pieces()
            end_color = p.Color("white") if (move.endRow + move.endCol) % 2 == 0 else p.Color("gray")
            end_rect = p.Rect(move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            p.draw.rect(self.screen, end_color, end_rect)
            if move.pieceCaptured is not None:
                capture_row = move.endRow if not move.isEnpassantMove else (move.endRow + 1 if move.pieceCaptured.color == "b" else move.endRow - 1)
                capture_rect = p.Rect(move.endCol * SQ_SIZE, capture_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                self.screen.blit(self.images[move.pieceCaptured.code], capture_rect)
            self.screen.blit(self.images[move.pieceMoved.code], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            p.display.flip()
            self.clock.tick(120)

    def draw_end_game_text(self, text):
        font = p.font.SysFont("Helvetica", 32, True, False)
        text_object = font.render(text, True, p.Color("Gray"))
        text_location = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(
            BOARD_WIDTH / 2 - text_object.get_width() / 2,
            BOARD_HEIGHT / 2 - text_object.get_height() / 2,
        )
        self.screen.blit(text_object, text_location)
        self.screen.blit(text_object, text_location.move(2, 2))

    def draw_move_log(self):
        move_log_rect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
        p.draw.rect(self.screen, p.Color("black"), move_log_rect)
        move_texts = []
        for i in range(0, len(self.gs.moveLog), 2):
            move_string = f"{i // 2 + 1}. {self.gs.moveLog[i]} "
            if i + 1 < len(self.gs.moveLog):
                move_string += str(self.gs.moveLog[i + 1])
            move_texts.append(move_string)
        padding = 5
        text_y = padding
        line_spacing = 5
        moves_per_row = 3
        for i in range(0, len(move_texts), moves_per_row):
            row_text = "  ".join(move_texts[i : i + moves_per_row])
            text_object = self.move_log_font.render(row_text, True, p.Color("white"))
            self.screen.blit(text_object, move_log_rect.move(padding, text_y))
            text_y += text_object.get_height() + line_spacing


if __name__ == "__main__":
    ChessApp().run()
