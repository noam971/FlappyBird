import bird as Bird
import pipe as Pipe
import ground as Base
import pygame
import neat
import os
pygame.font.init()


WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("coimcsans", 50)


def draw_window(win, birds, pipes, base, score, gen, ai):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score:" + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    if ai:
        text = STAT_FONT.render("Gen:" + str(gen), 1, (255, 255, 255))
        win.blit(text, (10, 10))
        information_text = STAT_FONT.render("Num Of BIrds:" + str(len(birds)), 1, (255, 255, 255))
        win.blit(information_text, (10, 50))
        for bird in birds:
            pygame.draw.rect(win, bird.color, (bird.x, bird.y, bird.img.get_width(), bird.img.get_height()), 2)
            pygame.draw.line(win, bird.color, (bird.x + bird.img.get_width(), bird.y),
                             (pipes[-1].x, pipes[-1].top + pipes[-1].PIPE_TOP.get_height()), 2)
            pygame.draw.line(win, bird.color, (bird.x + bird.img.get_width(), bird.y),
                             (pipes[-1].x, pipes[-1].bottom), 2)

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def run_game(genomes, config, ai=True):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []
    if not ai:
        birds.append(Bird.Bird(230, 350))
    if ai:
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            birds.append(Bird.Bird(230, 350))
            g.fitness = 0
            ge.append(g)

    pipes = [Pipe.Pipe(700, ai)]
    base = Base.Base(700)
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    fps = 30
    if ai:
        fps = 60

    game_over = False
    run = True
    while run:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            if not ai:
                if event.type == pygame.KEYUP:
                    if not game_over:
                        if event.key == pygame.K_SPACE:
                            if len(birds) > 0:
                                birds[0].jump()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()

            if ai:
                ge[x].fitness += 0.1
                output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

                if output[0] > 0.5:
                    bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    if ai:
                        ge[x].fitness -= 1
                        nets.pop(x)
                        ge.pop(x)
                    birds.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

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

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 700 or bird.y < 0:
                if ai:
                    nets.pop(x)
                    ge.pop(x)
                birds.pop(x)

            if ai and score > 20:
                nets.pop(x)
                ge.pop(x)
                birds.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score, GEN, ai)
