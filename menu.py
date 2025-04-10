import pygame
from constants import MENU_WIDTH, MENU_HEIGHT


def menu_loop_small():
    menu_screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("comicsans", 30)
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "AI"
                if event.key == pygame.K_2:
                    return "Human"
        menu_screen.fill((30, 30, 30))
        line1 = font.render("1 = AI Mode", True, (255, 255, 255))
        line2 = font.render("2 = Human Mode", True, (255, 255, 255))
        line3 = font.render("Close window to quit", True, (255, 255, 255))
        x1 = (MENU_WIDTH - line1.get_width()) // 2
        x2 = (MENU_WIDTH - line2.get_width()) // 2
        x3 = (MENU_WIDTH - line3.get_width()) // 2
        menu_screen.blit(line1, (x1, 80))
        menu_screen.blit(line2, (x2, 120))
        menu_screen.blit(line3, (x3, 180))
        pygame.display.update()
