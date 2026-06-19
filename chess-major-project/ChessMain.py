"""
Main game driver: handles UI, game loop, user input, AI integration.
Uses pygame for graphics and ProcessPoolExecutor for background AI calculation.
"""

import os
import pygame as p
from concurrent.futures import ProcessPoolExecutor

from ChessEngine import GameState, Move
from SmartMoveFinder import ChessAI

from Debug import debug_print, prof_start, prof_end, prof_report

# Board and window dimensions
BOARD_WIDTH = BOARD_HEIGHT = 512  # 512x512 pixel chess board
MOVE_LOG_PANEL_WIDTH = 270  # Right sidebar showing move history
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT  # Sidebar height matches board
DIMENSION = 8  # 8x8 chess board
SQ_SIZE = BOARD_HEIGHT // DIMENSION  # Each square is 64x64 pixels
MAX_FPS = 15  # Frame rate cap for game loop
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images")  # Path to piece images


class ChessApp:
    """Main game application: manages game state, UI rendering, user input, and AI."""
    def __init__(self):
        # Initialize pygame and create game window
        p.init()
        p.display.set_caption("Chess")
        self.screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
        self.clock = p.time.Clock()
        # Font for move log display
        self.move_log_font = p.font.SysFont("Arial", 20, False, False)
        self.images = {}  # Dictionary to cache piece images
        self.load_images()
        # Persistent single-worker executor for background AI moves
        self.ai_executor = ProcessPoolExecutor(max_workers=1)
        self.ai_future = None
        self.reset_game()

    def reset_game(self):
        """Initialize new game: fresh board, clear UI state, set player modes."""
        # Game state: board, pieces, valid moves, check/checkmate detection
        self.gs = GameState()
        self.valid_moves = self.gs.getValidMoves()
        # UI state flags
        self.move_made = False  # Trigger board update/animation
        self.animate = False  # Enable move animation
        self.game_over = False  # Prevent moves after checkmate/stalemate
        # Move input: selected square and click coordinates
        self.sq_selected = ()  # Currently selected square (row, col)
        self.player_clicks = []  # Two-element list: [start_sq, end_sq]
        # Player modes: can be human (True) or AI (False)
        self.player_one = False  # White player: human by default
        self.player_two = False  # Black player: AI by default
        # AI calculation state
        self.ai_thinking = False  # Flag: AI is calculating
        self.ai_future = None  # Future object for AI result
        self.move_undone = False  # Flag: undo was just performed

    def load_images(self):
        """Load and cache all piece images scaled to square size."""
        # All 12 piece types: white and black for each piece
        pieces = ["wp", "wN", "wB", "wR", "wQ", "wK", "bp", "bN", "bB", "bR", "bQ", "bK"]
        for piece in pieces:
            image_file = os.path.join(IMAGE_PATH, piece + ".png")
            # Load image and scale to exact square size (64x64) for rendering
            self.images[piece] = p.transform.scale(p.image.load(image_file), (SQ_SIZE, SQ_SIZE))

    def run(self):
        """Main game loop: process events, update state, render, and maintain frame rate."""
        running = True
        while running:
            # Process user input and window events
            running = self.handle_events()
            # Update game state after move
            if self.move_made:
                self.update_after_move()
            # Check and execute AI move if not human's turn
            self.handle_ai_move()
            # Render board, pieces, and UI
            self.draw_game_state()
            # Display end-game message and prevent further moves
            if self.gs.checkmate or self.gs.stalemate:
                game_over_text = (
                    "Stalemate"
                    if self.gs.stalemate
                    else "Black wins by checkmate"
                    if self.gs.whiteToMove
                    else "White wins by checkmate"
                )
                self.draw_end_game_text(game_over_text)
            # Frame rate control: cap at MAX_FPS
            self.clock.tick(MAX_FPS)
            p.display.flip()

    def handle_events(self):
        """Process pygame events: quit, mouse clicks, keyboard input."""
        for event in p.event.get():
            # Window close button
            if event.type == p.QUIT:
                return False
            # Mouse click on board
            if event.type == p.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event)
            # Keyboard: Z for undo, R for reset
            elif event.type == p.KEYDOWN:
                self.handle_key_down(event)
        return True

    def handle_mouse_click(self, event):
        """Process mouse clicks: select piece and attempt move (2-click interface)."""
        # Ignore clicks after game ends
        if self.game_over:
            return
        # Convert pixel coordinates to board row/col
        location = p.mouse.get_pos()
        col = location[0] // SQ_SIZE
        row = location[1] // SQ_SIZE
        # Ignore clicks on move log panel (right side)
        if col >= DIMENSION:
            self.sq_selected = ()
            self.player_clicks = []
            return
        # Determine current player's color
        current_color = "w" if self.gs.whiteToMove else "b"
        # Clicking same square twice: deselect
        if self.sq_selected == (row, col):
            self.sq_selected = ()
            self.player_clicks = []
        else:
            # First click: must be player's own piece
            if len(self.player_clicks) == 0:
                selected_piece = self.gs.board[row, col]
                if selected_piece is None or selected_piece.color != current_color:
                    return
            # Record click and update selection
            self.sq_selected = (row, col)
            self.player_clicks.append(self.sq_selected)
        # Two-click move: validate and execute if legal
        if len(self.player_clicks) == 2 and self.human_turn():
            self.try_player_move()

    def handle_key_down(self, event):
        """Handle keyboard input: Z=undo move, R=reset game."""
        if event.key == p.K_z:
            self.undo_move()
        elif event.key == p.K_r:
            self.reset_game()

    def human_turn(self):
        """Check if it's a human player's turn (vs AI)."""
        return (self.gs.whiteToMove and self.player_one) or (not self.gs.whiteToMove and self.player_two)

    def try_player_move(self):
        """Validate player's move against legal moves and execute if valid."""
        # Validate that starting square still has a piece (board may have changed)
        start_piece = self.gs.board[self.player_clicks[0][0], self.player_clicks[0][1]]
        if start_piece is None:
            # Starting square is now empty: reset and abort
            self.sq_selected = ()
            self.player_clicks = []
            return
        # Create move object from click coordinates
        move = Move(self.player_clicks[0], self.player_clicks[1], self.gs.board)
        # Check if move matches any legal move
        for valid_move in self.valid_moves:
            if move == valid_move:
                # Execute move and trigger UI update/animation
                self.gs.makeMove(valid_move)
                self.move_made = True
                self.animate = True
                # Reset selection state
                self.sq_selected = ()
                self.player_clicks = []
                return
        # Move was invalid: keep first click, allow new destination
        self.player_clicks = [self.sq_selected]

    def undo_move(self):
        """Undo last move and reset game state for next turn."""
        # Revert last move in game state
        self.gs.undoMove()
        # Clear move selection UI
        self.sq_selected = ()
        self.player_clicks = []
        # Trigger board update (will recalculate valid moves)
        self.move_made = True
        self.animate = False  # No animation on undo
        # Clear game-over state to allow continued play
        self.game_over = False
        # Cancel AI calculation if in progress
        if self.ai_thinking and self.ai_future is not None:
            self.ai_thinking = False
        # Flag for AI to reset properly next turn
        self.move_undone = True

    def handle_ai_move(self):
        """Manage AI calculation: start calculation and apply result when ready."""
        # Skip if game ended, human's turn, or undo was just performed
        if self.game_over or self.human_turn() or self.move_undone:
            return
        # Start AI calculation if not already running
        if not self.ai_thinking:
            self.ai_thinking = True
            # Submit AI move calculation to background thread
            self.ai_future = self.ai_executor.submit(
                ChessAI.find_best_move_minmax,
                self.gs,
                self.valid_moves,
            )
        # Check if AI calculation is complete
        if self.ai_future is not None and self.ai_future.done():
            # Retrieve result with error handling
            try:
                ai_move = self.ai_future.result()
            except Exception:
                ai_move = None
            # Fallback to random move if AI failed, but only if moves are available
            if ai_move is None:
                if len(self.valid_moves) > 0:
                    ai_move = ChessAI.find_random_move(self.valid_moves)
                else:
                    # No valid moves: game is checkmate or stalemate
                    self.game_over = True
                    self.ai_thinking = False
                    self.ai_future = None
                    return
            # Execute AI move on board
            self.gs.makeMove(ai_move)
            self.move_made = True  # Trigger board update
            self.animate = True  # Show animation
            # Reset AI state for next turn
            self.ai_thinking = False
            self.ai_future = None

    def update_after_move(self):
        """Update game state after a move: animate, recalculate legal moves."""
        # Animate the last move that was made
        if self.animate and self.gs.moveLog:
            self.animate_move(self.gs.moveLog[-1])
        # Recalculate legal moves for next player
        self.valid_moves = self.gs.getValidMoves()
        # Reset state flags for next iteration
        self.move_made = False
        self.animate = False
        self.move_undone = False

    def draw_game_state(self):
        """Render entire game state: board, pieces, highlights, move log."""
        self.draw_board()  # Draw 8x8 checkered board
        self.highlight_squares()  # Highlight selected square and legal moves
        self.draw_pieces()  # Draw all pieces on board
        self.draw_move_log()  # Draw move history in right panel

    def draw_board(self):
        """Draw 8x8 checkerboard pattern (alternating white and gray squares)."""
        colors = [p.Color("white"), p.Color("gray")]
        # Iterate through all board squares
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                # Alternate colors based on row+col (standard chess coloring)
                color = colors[(r + c) % 2]
                # Draw square as filled rectangle
                p.draw.rect(
                    self.screen,
                    color,
                    p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                )

    def highlight_squares(self):
        """Highlight selected square (blue) and legal move destinations (yellow)."""
        # Skip if no square selected
        if self.sq_selected == ():
            return
        # Check if selected square has a valid piece
        r, c = self.sq_selected
        selected_piece = self.gs.board[r, c]
        if selected_piece is None or selected_piece.color != ("w" if self.gs.whiteToMove else "b"):
            return
        # Create semi-transparent overlay surface
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)  # 100/255 opacity
        # Highlight selected square in blue
        s.fill(p.Color("blue"))
        self.screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
        # Highlight all legal move destinations in yellow
        s.fill(p.Color("yellow"))
        for move in self.valid_moves:
            if move.startRow == r and move.startCol == c:
                self.screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))

    def draw_pieces(self):
        """Draw all pieces on board using cached piece images."""
        # Iterate through all board squares
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.gs.board[r, c]
                # Draw piece if square is occupied
                if piece is not None:
                    self.screen.blit(
                        self.images[piece.code],  # Look up image by piece code (e.g., 'wp')
                        p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                    )

    def animate_move(self, move):
        """Animate piece sliding from start to end square over multiple frames."""
        # Calculate distance and direction
        dR = move.endRow - move.startRow
        dC = move.endCol - move.startCol
        # Animation parameters: 10 frames per square distance
        frames_per_square = 10
        frame_count = (abs(dR) + abs(dC)) * frames_per_square
        # Interpolate position from start to end over frame_count frames
        for frame in range(frame_count + 1):
            r = move.startRow + dR * frame / frame_count
            c = move.startCol + dC * frame / frame_count
            # Redraw board and all pieces
            self.draw_board()
            self.draw_pieces()
            # Redraw destination square with correct board color
            end_color = p.Color("white") if (move.endRow + move.endCol) % 2 == 0 else p.Color("gray")
            end_rect = p.Rect(move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            p.draw.rect(self.screen, end_color, end_rect)
            # Draw captured piece (for en passant, different square than destination)
            if move.pieceCaptured is not None:
                capture_row = move.endRow if not move.isEnpassantMove else (move.endRow + 1 if move.pieceCaptured.color == "b" else move.endRow - 1)
                capture_rect = p.Rect(move.endCol * SQ_SIZE, capture_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                self.screen.blit(self.images[move.pieceCaptured.code], capture_rect)
            # Draw animating piece at interpolated position
            self.screen.blit(self.images[move.pieceMoved.code], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            p.display.flip()
            # High FPS for smooth animation
            self.clock.tick(120)

    def draw_end_game_text(self, text):
        """Display game-over message centered on board with shadow effect."""
        # Large bold font for end-game text
        font = p.font.SysFont("Helvetica", 32, True, False)
        text_object = font.render(text, True, p.Color("Gray"))
        # Center text on board
        text_location = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(
            BOARD_WIDTH / 2 - text_object.get_width() / 2,
            BOARD_HEIGHT / 2 - text_object.get_height() / 2,
        )
        # Draw text twice: shadow offset, then main text
        self.screen.blit(text_object, text_location)
        self.screen.blit(text_object, text_location.move(2, 2))

    def draw_move_log(self):
        """Display move history in right sidebar: pairs of moves numbered 1-N."""
        # Draw black background for move log panel
        move_log_rect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
        p.draw.rect(self.screen, p.Color("black"), move_log_rect)
        # Format moves: pair white and black moves with move number
        move_texts = []
        for i in range(0, len(self.gs.moveLog), 2):
            move_string = f"{i // 2 + 1}. {self.gs.moveLog[i]} "  # White's move
            if i + 1 < len(self.gs.moveLog):
                move_string += str(self.gs.moveLog[i + 1])  # Black's move (if exists)
            move_texts.append(move_string)
        # Layout parameters: pack 3 moves per row to save space
        padding = 5
        text_y = padding
        line_spacing = 5
        moves_per_row = 3
        # Render moves in rows
        for i in range(0, len(move_texts), moves_per_row):
            row_text = "  ".join(move_texts[i : i + moves_per_row])
            text_object = self.move_log_font.render(row_text, True, p.Color("white"))
            self.screen.blit(text_object, move_log_rect.move(padding, text_y))
            text_y += text_object.get_height() + line_spacing


# Entry point: create app and start main game loop
if __name__ == "__main__":
    ChessApp().run()
