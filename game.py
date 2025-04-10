import pygame
import neat
import bird as Bird
import pipe as Pipe
import ground as Base
from constants import WIN_WIDTH, WIN_HEIGHT, BG_IMG, STAT_FONT, small_font

GEN = 0


def initialize_birds(genomes, config, ai):
    nets = []
    ge = []
    birds = []
    if ai:
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            birds.append(Bird.Bird(230, 350))
            g.fitness = 0
            ge.append(g)
    else:
        birds.append(Bird.Bird(230, 350))
    return nets, ge, birds


def initialize_world(ai):
    pipes = [Pipe.Pipe(700, ai)]
    base = Base.Base(700)
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    fps = 60 if ai else 30
    return pipes, base, win, clock, fps


def process_events(ai, birds):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "escape"
            if (not ai) and event.key == pygame.K_SPACE:
                if birds:
                    birds[0].jump()
    return None


def confirm_menu_popup(win):
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("comicsans", 32)
    overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    while True:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return "menu"
                if event.key == pygame.K_n:
                    return "resume"
        win.blit(overlay, (0, 0))
        lines = ["Return to Menu?", "Y = Yes", "N = No"]
        y_pos = 300
        for txt in lines:
            t = font.render(txt, True, (255, 255, 255))
            x_pos = (WIN_WIDTH - t.get_width()) // 2
            win.blit(t, (x_pos, y_pos))
            y_pos += 50
        pygame.display.update()


def update_birds(ai, birds, nets, ge, pipes, pipe_ind):
    for x, b in enumerate(birds):
        b.move()
        if ai:
            ge[x].fitness += 0.1
            output = nets[x].activate(
                (b.y,
                 abs(b.y - pipes[pipe_ind].height),
                 abs(b.y - pipes[pipe_ind].bottom))
            )
            if output[0] > 0.5:
                b.jump()


def update_pipes(ai, birds, nets, ge, pipes, score):
    add_pipe = False
    rem = []
    for p in pipes:
        for x, b in enumerate(birds):
            if p.collide(b):
                if ai:
                    ge[x].fitness -= 1
                    nets.pop(x)
                    ge.pop(x)
                birds.pop(x)
            if (not p.passed) and (p.x < b.x):
                p.passed = True
                add_pipe = True
        if (p.x + p.PIPE_TOP.get_width()) < 0:
            rem.append(p)
        p.move()

    if add_pipe:
        score += 1
        if ai:
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe.Pipe(600, True))
        else:
            pipes.append(Pipe.Pipe(600, False))

    for r in rem:
        pipes.remove(r)

    return score


def cleanup_birds(ai, birds, nets, ge, score):
    to_remove = []
    for x, b in enumerate(birds):
        off_screen = (700 <= b.y + b.img.get_height() or b.y < 0)
        if off_screen:
            to_remove.append(x)
        elif ai and score > 20:
            to_remove.append(x)
    for x in reversed(to_remove):
        if ai:
            nets.pop(x)
            ge.pop(x)
        birds.pop(x)


def draw_window(win, birds, pipes, base, score, gen, ai):
    win.blit(BG_IMG, (0, 0))
    for p in pipes:
        p.draw(win)
    score_text = STAT_FONT.render("Score:" + str(score), True, (255, 255, 255))
    win.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))
    menu_text = small_font.render("ESC to Menu", True, (255, 255, 255))
    win.blit(menu_text, (WIN_WIDTH - 10 - menu_text.get_width(), 10 + score_text.get_height() + 5))
    if ai:
        gen_text = STAT_FONT.render("Gen:" + str(gen), True, (255, 255, 255))
        count_text = STAT_FONT.render("Birds:" + str(len(birds)), True, (255, 255, 255))
        win.blit(gen_text, (10, 10))
        win.blit(count_text, (10, 50))
        for b in birds:
            color = getattr(b, "color", (255, 0, 0))
            pygame.draw.rect(win, color, (b.x, b.y, b.img.get_width(), b.img.get_height()), 2)
            if pipes:
                last_pipe = pipes[-1]
                pygame.draw.line(win, color, (b.x + b.img.get_width(), b.y),
                                 (last_pipe.x, last_pipe.top + last_pipe.PIPE_TOP.get_height()), 2)
                pygame.draw.line(win, color, (b.x + b.img.get_width(), b.y),
                                 (last_pipe.x, last_pipe.bottom), 2)
    base.draw(win)
    for b in birds:
        b.draw(win)
    pygame.display.update()


def run_game(genomes, config, ai=True):
    global GEN
    GEN += 1

    nets, ge, birds = initialize_birds(genomes, config, ai)
    pipes, base, win, clock, fps = initialize_world(ai)
    score = 0
    game_over = False

    while not game_over:
        clock.tick(fps)
        event_result = process_events(ai, birds)
        if event_result == "quit":
            return "quit"
        if event_result == "escape":
            choice = confirm_menu_popup(win)
            if choice == "quit":
                return "quit"
            if choice == "menu":
                return "menu"

        pipe_ind = 0
        if birds:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            game_over = True
            break

        update_birds(ai, birds, nets, ge, pipes, pipe_ind)
        score = update_pipes(ai, birds, nets, ge, pipes, score)
        cleanup_birds(ai, birds, nets, ge, score)
        if not birds:
            game_over = True
            break
        base.move()
        draw_window(win, birds, pipes, base, score, GEN, ai)

    if game_over:
        return "died"
    return "quit"
