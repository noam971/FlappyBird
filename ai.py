import neat
from constants import MenuReturn
from game import run_game


def ai_fitness_func(genomes, config):
    result = run_game(genomes, config, ai=True)
    if result == "quit":
        raise SystemExit
    if result == "menu":
        raise MenuReturn


def run_ai(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    try:
        p.run(ai_fitness_func, 50)
        return "died"
    except SystemExit:
        return "quit"
    except MenuReturn:
        return "menu"
