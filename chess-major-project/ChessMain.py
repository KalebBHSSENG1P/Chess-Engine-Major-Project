"""
Main game driver: handles UI, game loop, user input, AI integration.
Uses pygame for graphics and ProcessPoolExecutor for background AI calculation.
"""

import os
import pygame as p
from concurrent.futures import ProcessPoolExecutor

from ChessEngine import GameState, Move
from SmartMoveFinder import ChessAI
from pyvidplayer2 import Video

from Debug import DEBUG, debug_print, prof_start, prof_end, prof_report
from ChessMenu import Menu

class ChessApp:
    """Main game application: manages game state, UI rendering, user input, and AI."""
    def __init__(self):
        # Initialize pygame and create game window
        p.init()
        p.mixer.init()
        p.display.set_caption("Chess")
        self.screen = p.display.set_mode((0, 0), p.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        self.clock = p.time.Clock()
        # Calculate size of board and move long panel
        self.BOARD_HEIGHT = self.BOARD_WIDTH = round(self.screen_height * 0.85)
        self.SQ_SIZE = self.BOARD_HEIGHT // 8
        self.MOVE_LOG_PANEL_WIDTH = 815
        self.MOVE_LOG_PANEL_HEIGHT = 1200
        self.BOARD_Y = (self.screen_height - self.BOARD_HEIGHT) // 2
        self.BOARD_X = 200
        self.DIMENSION = 8  # 8x8 chess board
        self.MAX_FPS = 15  # Frame rate cap for game loop
        self.IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images")  # Path to piece images
        # Load fonts
        self.move_log_font = p.font.SysFont("impact", 35, False, False)
        self.profile_name_font = p.font.SysFont("impact", 25, False, False)
        self.pieces_captured_font = p.font.SysFont("Segoe UI Symbol", 32)
        # Run loading functions
        self.images = {}  # Dictionary to cache piece images
        self.load_images()
        self.load_sounds()
        self.load_buttons()
        # Persistent single-worker executor for background AI moves
        self.ai_executor = ProcessPoolExecutor(max_workers=1)
        self.ai_future = None
        self.reset_game()
        # Set default values for game flags
        self.player_one = None # True for player, false for AI
        self.player_two = None 
        self.flip_board = None # True for black pieces at the bottom, false for white pieces at the bottom
        self.game_not_ended = True

    def reset_game(self):
        """Initialize new game: background, fresh board, clear UI state, set player modes."""
        # Play sound
        self.sound_button.play()
        # Game state: board, pieces, valid moves, check/checkmate detection
        self.gs = GameState()
        self.valid_moves = self.gs.getValidMoves()
        self.game_not_ended = True
        # UI state flags
        self.move_made = False  # Trigger board update/animation
        self.animate = False  # Enable move animation
        self.game_over = False  # Prevent moves after checkmate/stalemate
        # Move input: selected square and click coordinates
        self.sq_selected = ()  # Currently selected square (row, col)
        self.player_clicks = []  # Two-element list: [start_sq, end_sq]
        # AI calculation state
        self.ai_thinking = False  # Flag: AI is calculating
        self.ai_future = None  # Future object for AI result
        self.move_undone = False  # Flag: undo was just performed

    def load_images(self):
        """Load and cache all piece images scaled to square size and background image."""
        # All 12 piece types: white and black for each piece
        pieces = ["wp", "wN", "wB", "wR", "wQ", "wK", "bp", "bN", "bB", "bR", "bQ", "bK"]
        for piece in pieces:
            image_file = os.path.join(self.IMAGE_PATH, piece + ".png")
            # Load image and scale to exact square size (64x64) for rendering
            self.images[piece] = p.transform.scale(p.image.load(image_file), (self.SQ_SIZE, self.SQ_SIZE))
        # Load profile images
        p_ai_path = os.path.join(os.path.dirname(__file__), "images", "profile_ai.png")
        p_human_path  = os.path.join(os.path.dirname(__file__), "images", "profile_human.png")
        self.profile_ai = p.image.load(p_ai_path).convert_alpha()
        self.profile_human  = p.image.load(p_human_path).convert_alpha()
        self.profile_ai = p.transform.smoothscale(self.profile_ai, (72, 72))
        self.profile_human  = p.transform.smoothscale(self.profile_human, (72, 72))
        # Load background image
        bg_path = os.path.join(os.path.dirname(__file__), "images", "game_background.png")
        self.background = p.image.load(bg_path).convert()
        self.background = p.transform.smoothscale(self.background, (self.screen_width, self.screen_height))
        # Load end game images
        # Stalemate
        stalemate_path = os.path.join(os.path.dirname(__file__), "images", "stalemate.png")
        self.stalemate = p.image.load(stalemate_path).convert_alpha()
        self.stalemate = p.transform.smoothscale(self.stalemate, (self.screen_width, self.screen_height))
        # Black checkmate
        bcheckmate_path = os.path.join(os.path.dirname(__file__), "images", "checkmate_black.png")
        self.black_checkmate = p.image.load(bcheckmate_path).convert_alpha()
        self.black_checkmate = p.transform.smoothscale(self.black_checkmate, (self.screen_width, self.screen_height))
        # White checkmate
        wcheckmate_path = os.path.join(os.path.dirname(__file__), "images", "checkmate_white.png")
        self.white_checkmate = p.image.load(wcheckmate_path).convert_alpha()
        self.white_checkmate = p.transform.smoothscale(self.white_checkmate, (self.screen_width, self.screen_height))

    def load_sounds(self):
        self.sound_move = p.mixer.Sound("sounds/move.mp3")
        self.sound_check = p.mixer.Sound("sounds/check.mp3")
        self.sound_checkmate = p.mixer.Sound("sounds/checkmate.mp3")
        self.sound_castle = p.mixer.Sound("sounds/castle.mp3")
        self.sound_capture = p.mixer.Sound("sounds/capture.mp3")
        self.sound_illegal = p.mixer.Sound("sounds/illegal.mp3")
        self.sound_button = p.mixer.Sound("sounds/game_button.mp3")

    def load_buttons(self):
        # Set button dimensions
        button_width = 140
        button_height = 60
        button_x = self.screen_width - 500
        button_y = self.screen_height - 100
        # Instantiate buttons
        self.undo_button = p.Rect(button_x - 490, button_y, button_width, button_height)
        self.reset_button = p.Rect(button_x - 330, button_y, button_width, button_height)
        self.menu_button  = p.Rect(button_x - 170, button_y, button_width, button_height)
        self.exit_button  = p.Rect(button_x, button_y, button_width, button_height)

    def run(self):
        """Main game loop: process events, update state, render, and maintain frame rate."""
        running = True
        while running:
            # Process user input and window events
            running = self.handle_events()
            # Check for result of "menu" from handle_events()
            if running == "menu":
                return "menu"
            # Update game state after move
            if self.move_made:
                self.update_after_move()
            # Check and execute AI move if not human's turn
            self.handle_ai_move()
            # Render board, pieces, and UI
            self.draw_game_state()
            # Display end-game message and prevent further moves
            if (self.gs.checkmate or self.gs.stalemate) and self.game_not_ended == True:
                # Play sound
                self.sound_checkmate.play()
                # Set game_not_ended flag to False
                self.game_not_ended = False
            elif (self.gs.checkmate or self.gs.stalemate) and self.game_not_ended == False:
                # Overlay stalemate image
                if self.gs.stalemate:
                    self.screen.blit(self.stalemate, (0, 0))
                else:
                    # Overlay black checkmate iamge
                    if self.gs.whiteToMove:
                        self.screen.blit(self.black_checkmate, (0, 0))
                    # Overlay white checkmate image
                    else:
                        self.screen.blit(self.white_checkmate, (0, 0))
            # Frame rate control: cap at MAX_FPS
            self.clock.tick(self.MAX_FPS)
            p.display.flip()

    def handle_events(self):
        """Process pygame events: quit, mouse clicks, keyboard input."""
        for event in p.event.get():
            # Window close button
            if event.type == p.QUIT:
                return False
            # Mouse click on board
            if event.type == p.MOUSEBUTTONDOWN:
                result = self.handle_mouse_click(event)
                # Check for result of "menu" from handle_mouse_click()
                if result == "menu":
                    return "menu"
            # Keyboard: Z for undo, R for reset
            elif event.type == p.KEYDOWN:
                self.handle_key_down(event)
        return True

    def handle_mouse_click(self, event):
        """Process mouse clicks: select piece and attempt move (2-click interface)."""
        # Get mouse position
        mouse_x, mouse_y = p.mouse.get_pos()
        # Detect if any button was clicked
        result = self.handle_button_clicks(mouse_x, mouse_y)
        # Check if the menu button was clicked
        if result == "menu":
            return "menu"
        if result is not None:
            return # A button was clicked but it was not the menu button
        # Ignore clicks after game ends
        if self.game_over:
            return
        # Convert screen coordinates to board coordinates
        board_pos = self.screen_to_board(mouse_x, mouse_y)
        # Detect clicks outside board
        if board_pos is None:
            self.sq_selected = ()
            self.player_clicks = []
            return
        # Set rows and columns
        row, col = board_pos
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

    def handle_button_clicks(self, mouse_x, mouse_y):
        """Detect if player clicks mouse over button bounding box"""
        # Undo button
        if self.undo_button.collidepoint(mouse_x, mouse_y):
            self.undo_move()
            return
        # Reset button
        if self.reset_button.collidepoint(mouse_x, mouse_y):
            self.reset_game()
            return
        # Menu button
        if self.menu_button.collidepoint(mouse_x, mouse_y):
            # Initialize transition video for going back to menu
            transition = Video("images/back_to_menu_transition.mp4")
            # Play sound
            self.sound_button.play()
            # Start loop to play transition
            while True:
                for event in p.event.get():
                    if event.type == p.QUIT:
                        transition.close()
                        p.quit()
                        return
                # Advance the video internal frame timer
                transition.draw(self.screen, (0, 0)) 
                # Resize the active frame surface to match screen dimensions
                if transition.frame_surf is not None:
                    scaled_frame = p.transform.scale(transition.frame_surf, (self.screen_width, self.screen_height))
                    self.screen.blit(scaled_frame, (0, 0)) # Draw the correctly resized video
                p.display.update()
                self.clock.tick(30)
                # Check if the video natural end is met, if so stop playing video
                if not transition.active:
                    transition.close()
                    return "menu"
        # Quit button
        if self.exit_button.collidepoint(mouse_x, mouse_y):
            p.quit()
            quit()

    def handle_key_down(self, event):
        """Handle keyboard input: Z=undo move, R=reset game."""
        if event.key == p.K_z:
            self.undo_move()
        elif event.key == p.K_r:
            self.reset_game()

    def handle_sounds(self, move):
        """Handle what sounds to play depending on the type of move."""
        # Check
        if self.gs.inCheck():
            self.sound_check.play()
            return
        # Capture
        if move.isCapture:
            self.sound_capture.play()
            return
        # Castling
        if move.isCastleMove:
            self.sound_castle.play()
            return
        # Promotion
        if move.isPawnPromotion:
            self.sound_move.play()
            return
        # En passant
        if move.isEnpassantMove:
            self.sound_capture.play()
            return
        # Normal move
        self.sound_move.play()

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
                # Execute move, trigger UI update/animation, play sound
                self.gs.makeMove(valid_move)
                self.move_made = True
                self.animate = True
                self.handle_sounds(valid_move)
                # Reset selection state
                self.sq_selected = ()
                self.player_clicks = []
                return
        # Move was invalid: keep first click, allow new destination
        self.player_clicks = [self.sq_selected]
        if self.gs.board[self.player_clicks[0][0], self.player_clicks[0][1]] is None:
            self.sound_illegal.play()

    def undo_move(self):
        """Undo last move and reset game state for next turn."""
        # Play sound
        self.sound_button.play()
        # Revert last move in game state
        self.gs.undoMove()
        # Clear move selection UI
        self.sq_selected = ()
        self.player_clicks = []
        # Trigger board update (will recalculate valid moves)
        self.move_made = True
        self.animate = False  # No animation on undo
        # Clear game-over state and game-over flag to allow continued play
        self.game_over = False
        self.game_not_ended = True
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
            self.handle_sounds(ai_move) # Play sound
            # Reset AI state for next turn
            self.ai_thinking = False
            self.ai_future = None
            # Reset selected squares and player clicks
            self.sq_selected = ()
            self.player_clicks = []

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

    def screen_to_board(self, mouse_x, mouse_y):
        """Convert screen pixel coordinates to board (row, col)"""
        x = mouse_x - self.BOARD_X
        y = mouse_y - self.BOARD_Y
        # Detect clicks outside board
        if x < 0 or x >= self.BOARD_WIDTH or y < 0 or y >= self.BOARD_HEIGHT:
            return None
        # Calculate row and column sizes
        col = x // self.SQ_SIZE
        row = y // self.SQ_SIZE
        # Flip calculated row and column numbers of board is flipped
        if getattr(self, "flip_board", False):
            row = self.DIMENSION - 1 - row
            col = self.DIMENSION - 1 - col
        return row, col

    def board_to_screen(self, row, col):
        """Convert board (row, col) to screen pixel coordinates"""
        # Flip drawn rows and columns if the board is flipped
        if getattr(self, "flip_board", False):
            draw_col = self.DIMENSION - 1 - col
            draw_row = self.DIMENSION - 1 - row
        else:
            draw_col = col
            draw_row = row
        # Calculate x and y
        x = self.BOARD_X + draw_col * self.SQ_SIZE
        y = self.BOARD_Y + draw_row * self.SQ_SIZE
        return x, y

    def draw_game_state(self):
        """Render entire game state and background: board, pieces, highlights, move log."""
        self.screen.blit(self.background, (0, 0)) # Draw background first
        self.draw_board()  # Draw 8x8 checkered board
        self.highlight_squares()  # Highlight selected square and legal moves
        self.draw_pieces()  # Draw all pieces on board
        self.draw_move_log()  # Draw move history in right panel
        self.draw_elements() # Draw button boundaries and player/AI profile icons
        self.captured_pieces() # Track a list of captured pieces to display beside profile icon

    def draw_board(self):
        """Draw 8x8 checkerboard pattern (alternating white and gray squares)."""
        colors = [p.Color("white"), p.Color("gray")]
        # Iterate through all board squares
        for r in range(self.DIMENSION):
            for c in range(self.DIMENSION):
                # Alternate colors based on row+col (standard chess coloring)
                color = colors[(r + c) % 2]
                x, y = self.board_to_screen(r, c)
                # Draw square as filled rectangle
                p.draw.rect(
                    self.screen,
                    color,
                    p.Rect(x, y, self.SQ_SIZE, self.SQ_SIZE),
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
        s = p.Surface((self.SQ_SIZE, self.SQ_SIZE))
        s.set_alpha(100)  # 100/255 opacity
        # Highlight selected square in blue
        s.fill(p.Color("blue"))
        x, y = self.board_to_screen(r, c)
        self.screen.blit(s, (x, y))
        # Highlight all legal move destinations in yellow
        s.fill(p.Color("yellow"))
        for move in self.valid_moves:
            if move.startRow == r and move.startCol == c:
                dx, dy = self.board_to_screen(move.endRow, move.endCol)
                self.screen.blit(s, (dx, dy))

    def draw_pieces(self):
        """Draw all pieces on board using cached piece images."""
        # Iterate through all board squares
        for r in range(self.DIMENSION):
            for c in range(self.DIMENSION):
                piece = self.gs.board[r, c]
                # Draw piece if square is occupied
                if piece is not None:
                    x, y = self.board_to_screen(r, c)
                    self.screen.blit(
                        self.images[piece.code],  # Look up image by piece code (e.g., 'wp')
                        p.Rect(x, y, self.SQ_SIZE, self.SQ_SIZE),
                    )

    def animate_move(self, move):
        """Animate piece sliding from start to end square over multiple frames."""
        # Calculate distance and direction
        dR = move.endRow - move.startRow
        dC = move.endCol - move.startCol
        # Animation parameters: 4 frames per square distance
        frames_per_square = 4
        frame_count = (abs(dR) + abs(dC)) * frames_per_square
        # Interpolate position from start to end over frame_count frames
        for frame in range(frame_count + 1):
            r = move.startRow + dR * frame / frame_count
            c = move.startCol + dC * frame / frame_count
            # Redraw board and all pieces
            self.draw_board()
            self.draw_pieces()
            # Redraw destination square with correct board color
            end_x, end_y = self.board_to_screen(move.endRow, move.endCol)
            end_color = p.Color("white") if (move.endRow + move.endCol) % 2 == 0 else p.Color("gray")
            end_rect = p.Rect(end_x, end_y, self.SQ_SIZE, self.SQ_SIZE)
            p.draw.rect(self.screen, end_color, end_rect)
            # Draw captured piece (for en passant, different square than destination)
            if move.pieceCaptured is not None:
                if move.isEnpassantMove:
                    capture_row = move.endRow + 1 if move.pieceCaptured.color == "b" else move.endRow - 1
                    capture_col = move.endCol
                else:
                    capture_row = move.endRow
                    capture_col = move.endCol
                capture_x, capture_y = self.board_to_screen(capture_row, capture_col)
                capture_rect = p.Rect(capture_x, capture_y, self.SQ_SIZE, self.SQ_SIZE)
                self.screen.blit(self.images[move.pieceCaptured.code], capture_rect)
            # Draw animating piece at interpolated position
            px, py = self.board_to_screen(r, c)
            self.screen.blit(self.images[move.pieceMoved.code], p.Rect(px, py, self.SQ_SIZE, self.SQ_SIZE))
            p.display.flip()
            # High FPS for smooth animation
            self.clock.tick(120)

    def draw_move_log(self):
        """Display move history in right sidebar: pairs of moves numbered 1-N."""
        # Draw bounding rectangle for move log panel
        move_log_rect = p.Rect(self.BOARD_X + self.BOARD_WIDTH + 70, self.BOARD_Y + 15, self.MOVE_LOG_PANEL_WIDTH, self.MOVE_LOG_PANEL_HEIGHT)
        # Format moves: pair white and black moves with move number
        move_texts = []
        for i in range(0, len(self.gs.moveLog), 2):
            move_string = f"{i // 2 + 1}. {self.gs.moveLog[i]} "  # White's move
            if i + 1 < len(self.gs.moveLog):
                move_string += str(self.gs.moveLog[i + 1])  # Black's move (if exists)
            move_texts.append(move_string)
        move_texts = move_texts[-64:] # cap at 64 moves
        # Layout parameters: pack 5 moves per row to save space
        padding = 5
        text_y = padding
        line_spacing = 10
        moves_per_row = 4
        # Render moves in rows
        for i in range(0, len(move_texts), moves_per_row):
            row_text = "     ".join(move_texts[i : i + moves_per_row])
            text_object = self.move_log_font.render(row_text, True, p.Color("Gray"))
            self.screen.blit(text_object, move_log_rect.move(padding, text_y))
            text_y += text_object.get_height() + line_spacing

    def draw_elements(self):
        """Draw button bounding boxes, profiles, profile names and profile material"""
        # Buttons list
        buttons = [self.undo_button, self.reset_button, self.menu_button, self.exit_button]
        # Profile names
        name_one = self.profile_name_font.render("Player 1", True, p.Color("Gray"))
        name_two = self.profile_name_font.render("Player 2", True, p.Color("Gray"))
        name_three = self.profile_name_font.render("AI", True, p.Color("Gray"))
        name_four = self.profile_name_font.render("Player", True, p.Color("Gray"))
        # If Debug is true, highlight buttons with a transparent red
        if DEBUG == True:
            for rect in buttons:
                s = p.Surface((rect.width, rect.height), p.SRCALPHA)
                s.fill((255, 0, 0, 120))  # transparent red
                self.screen.blit(s, rect.topleft)
        # Draw profiles
        if self.player_one == True and self.player_two == True:
            self.screen.blit(self.profile_human, (200, 20)) # Render human profile
        elif self.player_one == False or self.player_two == False:
            self.screen.blit(self.profile_ai, (200, 20)) # Render AI profile
        self.screen.blit(self.profile_human, (200, 1350)) # Render human profile, the user at the bottom will always be a human
        # Draw profile names
        if self.player_one == True and self.player_two == True and self.flip_board == False: # Two player game with white at the bottom
            self.screen.blit(name_two, (280, 20))
            self.screen.blit(name_one, (280, 1350))
        elif self.player_one == True and self.player_two == True and self.flip_board == True: # Two player game with black at the bottom
            self.screen.blit(name_one, (280, 20))
            self.screen.blit(name_two, (280, 1350))
        elif self.player_one == False or self.player_two == False: # Player vs AI
            self.screen.blit(name_three, (280, 20))
            self.screen.blit(name_four, (280, 1350))
        # Draw captured pieces under profile names
        captured_white_pieces, captured_black_pieces = self.captured_pieces()
        # Top profile captured pieces
        if self.flip_board:
            # Black is at bottom, white is at top
            display_top = captured_black_pieces
            display_bottom = captured_white_pieces
        else:
            # White is at bottom, black is at top
            display_top = captured_white_pieces
            display_bottom = captured_black_pieces
        # Draw top display
        x = 280 # Dynamically change x so pieces do not overlap each other
        for u in display_top:
            surf = self.pieces_captured_font.render(u, True, p.Color("Gray58"))
            self.screen.blit(surf, (x, 50))
            x += surf.get_width() + 0.5
        # Draw bottom display
        x = 280 # Dynamically change x so pieces do not overlap each other
        for u in display_bottom:
            surf = self.pieces_captured_font.render(u, True, p.Color("Gray58"))
            self.screen.blit(surf, (x, 1380))
            x += surf.get_width() + 0.5

    def captured_pieces(self):
        """ Return two lists of pieces captured by white and pieces captured by black """
        # Algebraic chess notation to unicode chess pieces converter
        PIECE_TO_UNICODE = {"p": "♟︎", "N": "♞", "B": "♝", "R": "♜", "Q": "♛", "K": "♚"}
        # Sort priority: king > queen > rook > bishop > knight > pawn
        SORT_ORDER = {"K": 0, "Q": 1, "R": 2, "B": 3, "N": 4, "p": 5}
        # Generate lists
        captured_white_pieces = []
        captured_black_pieces = []
        # Fill lists
        for move in self.gs.moveLog:
            piece = move.pieceCaptured
            # If move lead to no pieces captured, do nothing
            if move.pieceCaptured is None:
                continue
            piece_type = piece.kind
            piece_icon = PIECE_TO_UNICODE[piece_type]
            # Fill appropriate captured lists
            if piece.color == "b":
                captured_black_pieces.append((SORT_ORDER[piece_type], piece_icon))
            else:
                captured_white_pieces.append((SORT_ORDER[piece_type], piece_icon))
        # Sort by the numeric priority
        captured_black_pieces.sort(key=lambda x: x[0])
        captured_white_pieces.sort(key=lambda x: x[0])
        # Return lists of unicode chess characters
        return [u for _, u in captured_white_pieces], [u for _, u in captured_black_pieces]



# Entry point: create app and start main game loop
if __name__ == "__main__":
    menu = Menu()
    while True:
        mode, color = menu.run_menu()
        app = ChessApp()
        # Play intro ONCE before game starts
        menu.play_intro()
        # Check menu settings
        app.flip_board = (color == "black")
        if mode == "pvp":
            app.player_one = True   # White = human
            app.player_two = True   # Black = human
        elif mode == "pvai":
            if color == "white":
                app.player_one = True   # White = human
                app.player_two = False  # Black = AI
            elif color == "black":
                app.player_one = False  # White = AI
                app.player_two = True   # Black = human
        # Run game
        running = app.run()
        # Stop game if result of "menu" is detected from handle_events()
        if running == "menu":
            continue