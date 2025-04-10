import os
import pygame
import neat
from game_runner import game
from game_runner.menu import menu_loop_small
from game_runner.game import run_game
from game_runner.ai import run_ai


def main():
    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    while True:
        mode = menu_loop_small()
        game.GEN = 0
        if mode == "quit":
            break
        if mode == "AI":
            cfg = os.path.join(os.path.dirname(__file__), "config_feedforward.txt")
            config = neat.config.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                cfg
            )
            result = run_ai(config)
        elif mode == "Human":
            result = run_game([], [], ai=False)

        if result == "quit":
            break
        if result == "menu":
            continue


if __name__ == "__main__":
    main()
