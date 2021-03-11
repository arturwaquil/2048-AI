import sys
import pygame
from game import Game2048

pygame.init()

size = (400, 400)
screen = pygame.display.set_mode(size)

game = Game2048(4)
game_status = Game2048.IN_GAME

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                sys.exit()
            if event.key == pygame.K_LEFT:
                game_status = game.step(Game2048.LEFT)
                print("left")
            if event.key == pygame.K_UP:
                game_status = game.step(Game2048.UP)
                print("up")
            if event.key == pygame.K_RIGHT:
                game_status = game.step(Game2048.RIGHT)
                print("right")
            if event.key == pygame.K_DOWN:
                game_status = game.step(Game2048.DOWN)
                print("down")

    if game_status == Game2048.GAME_ENDED:
        break