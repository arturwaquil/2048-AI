import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "True"
import pygame
from game import Game2048


class GUI:

    bg_gray = (185, 169, 157)
    tile_gray = (205, 191, 179)

    def __init__(self, n):
        pygame.init()

        self.n = n
        self.size = (410, 410)
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.bg_gray)

        self.game = Game2048(self.n)
        self.game_status = Game2048.IN_GAME

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()
                    if event.key == pygame.K_LEFT:
                        game_status = self.game.step(Game2048.LEFT)
                        print("left")
                    if event.key == pygame.K_UP:
                        game_status = self.game.step(Game2048.UP)
                        print("up")
                    if event.key == pygame.K_RIGHT:
                        game_status = self.game.step(Game2048.RIGHT)
                        print("right")
                    if event.key == pygame.K_DOWN:
                        game_status = self.game.step(Game2048.DOWN)
                        print("down")

            self.paint_current_state(self.game.current_state())

            if self.game_status == Game2048.GAME_ENDED:
                break

    def paint_current_state(self, state):
        for i in range(self.n):
            for j in range(self.n):
                self.paint_tile((i, j))

    def paint_tile(self, position):
        tile = pygame.Surface((90, 90))
        tile.fill(self.tile_gray)
        self.screen.blit(tile, (10 + 100*position[0], 10 + 100*position[1]))
        pygame.display.flip()


if __name__ == "__main__":
    GUI(4).run()
