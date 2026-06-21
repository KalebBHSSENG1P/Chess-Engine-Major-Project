import pygame as p
import sys
import os
from pyvidplayer import Video
from Debug import DEBUG

def run_menu():
    p.init()

    # Fullscreen window
    screen = p.display.set_mode((0, 0), p.FULLSCREEN)
    width, height = screen.get_size()

    # Load background image
    bg_path = os.path.join(os.path.dirname(__file__), "images", "menu_background.png")
    background = p.image.load(bg_path).convert()
    background = p.transform.smoothscale(background, (width, height))
    # Load white selected button image
    w_select_path = os.path.join(os.path.dirname(__file__), "images", "white_selected.png")
    white_selected = p.image.load(w_select_path).convert()
    white_selected = p.transform.smoothscale(white_selected, (width, height))
    # Load black selected button image
    b_select_path = os.path.join(os.path.dirname(__file__), "images", "black_selected.png")
    black_selected = p.image.load(b_select_path).convert()
    black_selected = p.transform.smoothscale(black_selected, (width, height))

    # set default colour picked to white
    selected_color = "white"

    # Button settings
    button_width = 800
    button_height = 150
    button_small_width = 350

    # centre bounding box to correct positions
    pvp_rect   = p.Rect(0, 0, button_width, button_height); pvp_rect.center   = (635, 470)
    pvai_rect  = p.Rect(0, 0, button_width, button_height); pvai_rect.center  = (635, 730)
    white_rect = p.Rect(0, 0, button_small_width, button_height); white_rect.center = (415, 990)
    black_rect = p.Rect(0, 0, button_small_width, button_height); black_rect.center = (865, 990)
    exit_rect  = p.Rect(0, 0, button_width, button_height); exit_rect.center  = (635, 1250)

    # Load intro video
    intro = Video("images/menu_intro.mp4")
    intro.set_size((width, height))

    # Begin main pygame loop for menu screen
    while True:
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()

            if event.type == p.MOUSEBUTTONDOWN and not intro.active:
                if pvp_rect.collidepoint(event.pos):
                    return ("pvp", None)
                if pvai_rect.collidepoint(event.pos):
                    if selected_color == "white":
                        return ("pvai", "white")
                    elif selected_color == "black":
                        return ("pvai", "black")
                if white_rect.collidepoint(event.pos):
                    selected_color = "white"
                if black_rect.collidepoint(event.pos):
                    screen.blit(background, (0, 0))
                    selected_color = "black"
                if exit_rect.collidepoint(event.pos):
                    p.quit()
                    sys.exit()

            if event.type == p.KEYDOWN:
                if event.key == p.K_ESCAPE:
                    p.quit()
                    sys.exit()

        # If video is still playing, draw it
        if intro.active:
            intro.draw(screen, (0, 0))
            p.display.update()
            continue  # Skip drawing the menu until video ends

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
        screen.blit(background, (0, 0))

        # Draw highlight around selected option
        if selected_color == "white":
            screen.blit(white_selected, (0, 0))
        elif selected_color == "black":
            screen.blit(black_selected, (0, 0))

        # Draw bounding boxes for all option buttons
        screen.blit(overlay, pvp_rect)
        screen.blit(overlay, pvai_rect)
        screen.blit(overlay_small, white_rect)
        screen.blit(overlay_small, black_rect)
        screen.blit(overlay, exit_rect)
        p.display.flip()