import os
import pygame

os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800
MENU_WIDTH = 300
MENU_HEIGHT = 300

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 40)
small_font = pygame.font.SysFont("comicsans", 24)


class MenuReturn(Exception):
    pass
