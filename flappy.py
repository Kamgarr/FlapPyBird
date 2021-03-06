from itertools import cycle
import random
import sys
import numpy as np
from torch_net import network
from evolution import evolution
import argparse

import pygame
from pygame.locals import *

POP_SIZE = 100
GENS = 100
ELITE_SIZE = 30
TOURNAMENT_SIZE = 5
CROSS_OVER_PROB = 0.5
MUT_PROB = 0.8
MUT_PER_BIT = 0.01

FPS = 500
SCREENWIDTH = 288
SCREENHEIGHT = 512
VIEWHEIGHT = 144
# amount by which base can maximum shift to left
PIPEGAPSIZE = 500  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
TOPY = -100
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # purple
    (
        'assets/sprites/purplebird-upflap.png',
        'assets/sprites/purplebird-midflap.png',
        'assets/sprites/purplebird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-window.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/block-green.png',
)


class bird:
    indexGen = cycle([0, 1, 2, 1])
    image = 0
    active = True
    score = 0  # TODO use this as fitness
    rotation = 0
    loopIter = 0

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY = -9  # player's velocity along Y, default same as playerFlapped
    playerMaxVelY = 10  # max vel along Y, max descend speed
    playerMinVelY = -8  # min vel along Y, max ascend speed
    playerAccY = 1  # players downward accleration
    playerRot = 45  # player's rotation
    playerVelRot = 3  # angular speed
    playerRotThr = 20  # rotation threshold
    playerFlapAcc = -9  # players speed on flapping

    def __init__(self, id, x, y, images):
        self.id = id
        self.x = x
        self.y = y
        self.images = images

    def update(self, score, upperPipes, lowerPipes, flapped):

        # check for crash here
        crashTest = checkCrash(self, upperPipes, lowerPipes)
        if crashTest[0] or score >= 10000:
            self.active = False
            self.score = score
            return False

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not flapped:
            self.playerVelY += self.playerAccY
        if flapped:
            self.playerVelY = self.playerFlapAcc
            self.playerRot = 45

        playerHeight = IMAGES['player'][bird.image].get_height()
        self.y += min(self.playerVelY, BASEY - self.y - playerHeight)

        # playerIndex basex change
        if score % 3 == 0:
            self.image = next(self.indexGen)

        # Player rotation has a threshold
        self.rotation = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            self.rotation = self.playerRot

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", default=None, type=str, help="Save folder.")
    parser.add_argument("--resume", default=None, type=str, help="Weight file")
    parser.add_argument("--change_training", default=None, type=int, help="Generation when training of conv layers stops")
    parser.add_argument("--show_one", default=None, type=str, help="Weight file")
    args = parser.parse_args()

    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base_green.png').convert_alpha()
    IMAGES['top'] = pygame.image.load('assets/sprites/top_blue.png').convert_alpha()

    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[0]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    player_img = []
    for i in range(0, len(PLAYERS_LIST)):
        player_img.append((
            pygame.image.load(PLAYERS_LIST[i][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[i][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[i][2]).convert_alpha(),
        ))

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[0]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[0]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    startx, starty = int(SCREENWIDTH * 0.2), int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    # TODO Neural net and evolution definition
    # create network
    net = network([1, 1, SCREENWIDTH, VIEWHEIGHT], "C-3-4-2,T,C-2-4-2,T,P-2-2,C-2-4-2,T,F,D-64,R,D-2")

    if args.show_one:
        global FPS
        FPS = 50
        b = bird(id=i, x=startx, y=starty, images=player_img[0])
        mainGame([b], 0, net, np.loadtxt(args.show_one), False)
    else:
        # create population of weights
        evolve = evolution(MUT_PROB, MUT_PER_BIT, CROSS_OVER_PROB, ELITE_SIZE, TOURNAMENT_SIZE)

        if args.resume:
            elite = np.loadtxt(args.resume)
            new = np.random.uniform(-1, 1, (POP_SIZE - ELITE_SIZE, net.weight_size)).reshape(
                (POP_SIZE - ELITE_SIZE, net.weight_size))
            population = np.append(elite, new, axis=0)
        else:
            population = np.random.uniform(-1, 1, (POP_SIZE, net.weight_size)).reshape((POP_SIZE, net.weight_size))

        for generation in range(GENS):
            #evaluate population
            fitness = []
            for i in range(0, POP_SIZE):
                b = bird(id=i, x=startx, y=starty, images=player_img[0])
                mainGame([b], generation, net, population[i])
                fitness.append(b.score)
            #generate new population
            if args.change_training and args.change_training > generation:
                population = evolve(population, np.array(fitness), net.weight_size - net.dense_weight_size + 1)
            else:
                population = evolve(population, np.array(fitness))
            np.savetxt(args.save_folder + "/gen_" + str(generation), evolve.elite)
            np.savetxt(args.save_folder + "/gen_" + str(generation) + "_best", evolve.best)


def mainGame(birds, generation, network, weights, learning = True):
    score = 0
    if not learning:
        random.seed(12)

    # list of upper pipes
    upperPipes = []
    # list of lowerpipe
    lowerPipes = []

    newPipe1 = getRandomPipe()
    upperPipes.append({'x': SCREENWIDTH, 'y': newPipe1[0]['y']})
    lowerPipes.append({'x': SCREENWIDTH, 'y': newPipe1[1]['y']})

    pipeVelX = -4

    active_birds = len(birds)
    # use this cycle as metrics for fitness
    while True:
        score += 1

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        if (active_birds < 1):
            return

        for bird in birds:
            if not bird.active:
                continue
            # TODO call neural net here
            rgb = np.array(SCREEN.get_view('3'))
            gray_scale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            top = max(0, int(bird.y - VIEWHEIGHT / 2))
            view = gray_scale[:, top:top + VIEWHEIGHT]
            jump = network(np.expand_dims(view, 0), weights)

            if not bird.update(score, upperPipes, lowerPipes, jump):
                active_birds -= 1
                print("Generation: ", generation, "  bird: ", bird.id, "  score: ", score)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, -512+bird.y))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (0, BASEY))
        SCREEN.blit(IMAGES['top'], (0, TOPY))
        # print score so player overlaps the score
        # showScore(score)
        pygame.display.set_caption("Flappy Bird, score: " + str(score))

        for bird in birds:
            if (bird.active):
                playerSurface = pygame.transform.rotate(bird.images[bird.image], bird.rotation)
                SCREEN.blit(playerSurface, (bird.x, bird.y))
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
        playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    pipeHeight = IMAGES['pipe'][0].get_height()
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(pipeHeight, int(BASEY))
    # gapY += int(BASEY * 0.2)

    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(bird, upperPipes, lowerPipes):
    player = {}
    player['x'] = bird.x
    player['y'] = bird.y

    """returns True if player collders with base or pipes."""
    pi = bird.image
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    elif player['y'] + player['h'] <= -5:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


if __name__ == '__main__':
    main()
