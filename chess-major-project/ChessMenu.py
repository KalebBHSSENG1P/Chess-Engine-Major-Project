"""
Draws the main menu using pygame and runs transitions between menu and game.
For videos to work, FFmpeg MUST be instantialized so pyvidplayer2 library can work.
"""

import pygame as p
import sys
import os
from pyvidplayer2 import Video
from Debug import DEBUG

class Menu:
    """Main menu application for game: manages animations for loading and beginning games as well."""
    def __init__(self):
        p.init()
        p.mixer.init()
        self.screen = p.display.set_mode((0, 0), p.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        self.clock = p.time.Clock()
        # set default colour picked to white
        self.selected_color = "white"
        # load UI button sound
        self.sound_button = p.mixer.Sound("sounds/menu_button.mp3")
        # Play start-up intro
        self.intro_played = False

    def run_menu(self):
        # Load background image
        bg_path = os.path.join(os.path.dirname(__file__), "images", "menu_background.png")
        background = p.image.load(bg_path).convert()
        background = p.transform.smoothscale(background, (self.width, self.height))
        # Load white selected button image
        w_select_path = os.path.join(os.path.dirname(__file__), "images", "selected_white.png")
        white_selected = p.image.load(w_select_path).convert()
        white_selected = p.transform.smoothscale(white_selected, (self.width, self.height))
        # Load black selected button image
        b_select_path = os.path.join(os.path.dirname(__file__), "images", "selected_black.png")
        black_selected = p.image.load(b_select_path).convert()
        black_selected = p.transform.smoothscale(black_selected, (self.width, self.height))

        # Button settings
        # Scaling
        button_width = int(self.width * 0.33)
        button_height = int(self.height * 0.1)
        button_small_width = int(self.width * 0.15)
        # Positioning
        button_x = int(self.width * 0.25)
        button_y = int(self.height * 0.33)
        # Spacing
        gap_y = int(self.height * 0.18)
        gap_x = int(self.width * 0.085)

        # centre bounding box to correct positions
        pvp_rect   = p.Rect(0, 0, button_width, button_height); pvp_rect.center   = (button_x, button_y)
        pvai_rect  = p.Rect(0, 0, button_width, button_height); pvai_rect.center  = (button_x, button_y + gap_y)
        white_rect = p.Rect(0, 0, button_small_width, button_height); white_rect.center = (button_x - gap_x, button_y + (gap_y * 2))
        black_rect = p.Rect(0, 0, button_small_width, button_height); black_rect.center = (button_x + gap_x, button_y + (gap_y * 2))
        exit_rect  = p.Rect(0, 0, button_width, button_height); exit_rect.center  = (button_x, button_y + (gap_y * 3))

        # Load intro video
        intro = Video("images/menu_intro.mp4") if not self.intro_played else None

        # Begin main pygame loop for menu screen
        while True:
            for event in p.event.get():
                if event.type == p.QUIT:
                    p.quit()
                    sys.exit()
                if event.type == p.MOUSEBUTTONDOWN and (intro is None or not intro.active):
                    # Return tuples to change self.player_one and self.player_two booleans in ChessApp
                    if pvp_rect.collidepoint(event.pos): # self.player_one = True, self.player_two = True
                        if self.selected_color == "white":
                            return ("pvp", "white")
                        elif self.selected_color == "black":
                            return ("pvp", "black")
                    if pvai_rect.collidepoint(event.pos):
                        if self.selected_color == "white":
                            return ("pvai", "white") # self.player_one = True, self.player_two = False
                        elif self.selected_color == "black":
                            return ("pvai", "black") # self.player_one = False, self.player_two = True
                    # Set selected_colour string to selected colour of button
                    if white_rect.collidepoint(event.pos):
                        self.sound_button.play()
                        self.selected_color = "white"
                    if black_rect.collidepoint(event.pos):
                        self.screen.blit(background, (0, 0))
                        self.sound_button.play()
                        self.selected_color = "black"
                    # Exit game if button is clicked
                    if exit_rect.collidepoint(event.pos):
                        p.quit()
                        sys.exit()

            # If video is still playing, draw it
            if intro is not None and intro.active:
                intro.draw(self.screen, (0, 0))
                if intro.frame_surf is not None:
                    scaled_frame = p.transform.scale(intro.frame_surf, (self.width, self.height))
                    self.screen.blit(scaled_frame, (0, 0)) # Draw the correctly resized video
                p.display.update()
                continue  # Skip drawing the menu until video ends
            elif intro is not None and not intro.active:
                intro.close()
                self.intro_played = True # Mark intro played flag as true so the intro does not play again
                intro = None

            # check if debug switch is true in Debug.py
            if DEBUG:
                debug_color = (255, 0, 0, 120)  # semi‑transparent red
            else:
                debug_color = (255, 0, 0, 0)  # fully-transparent red
            # Create a surface with alpha for transparency
            overlay = p.Surface((button_width, button_height), p.SRCALPHA)
            overlay_small = p.Surface((button_small_width, button_height), p.SRCALPHA)
            overlay.fill(debug_color)
            overlay_small.fill(debug_color)

            # Draw background menu
            self.screen.blit(background, (0, 0))

            # Draw highlight around selected option
            if self.selected_color == "white":
                self.screen.blit(white_selected, (0, 0))
            elif self.selected_color == "black":
                self.screen.blit(black_selected, (0, 0))

            # Draw bounding boxes for all option buttons
            self.screen.blit(overlay, pvp_rect)
            self.screen.blit(overlay, pvai_rect)
            self.screen.blit(overlay_small, white_rect)
            self.screen.blit(overlay_small, black_rect)
            self.screen.blit(overlay, exit_rect)
            p.display.flip()

    def play_intro(self):
            # Initialize videos
            intro = Video("images/game_start_intro.mp4")
            intro2 = Video("images/game_start_intro_black.mp4")
            while True:
                for event in p.event.get():
                    if event.type == p.QUIT:
                        intro.close()
                        p.quit()
                        return
                # Play intro (white POV) if self.player_one = True and self.player_two = False
                if self.selected_color == "white":
                    # Advance the video internal frame timer
                    intro.draw(self.screen, (0, 0)) 
                    # Resize the active frame surface to match screen dimensions
                    if intro.frame_surf is not None:
                        scaled_frame = p.transform.scale(intro.frame_surf, (self.width, self.height))
                        self.screen.blit(scaled_frame, (0, 0)) # Draw the correctly resized video
                    p.display.update()
                    self.clock.tick(30)
                    # Check if the video natural end is met, if so stop playing video
                    if not intro.active:
                        intro.close()
                        return
                # Play intro (black POV) if self.player_one == False and self.player_two == True
                elif self.selected_color == "black":
                    # Advance the video internal frame timer
                    intro2.draw(self.screen, (0, 0)) 
                    # Resize the active frame surface to match screen dimensions
                    if intro2.frame_surf is not None:
                        scaled_frame = p.transform.scale(intro2.frame_surf, (self.width, self.height))
                        self.screen.blit(scaled_frame, (0, 0)) # Draw the correctly resized video
                    p.display.update()
                    self.clock.tick(30)
                    # Check if the video natural end is met, if so stop playing video
                    if not intro2.active:
                        intro2.close()
                        return