import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "True"
import pygame
from game import Game2048


class GUI:

    Colors = {
        "light_gray": (205, 191, 179),
        "medium_gray": (185, 169, 157),
        "dark_gray": (119, 109, 101),
        "white": (237, 226, 217),
    }

    def __init__(self, n):
        pygame.init()

        self.n = n
        self.size = (100*n+10, 100*n+10)
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.Colors["medium_gray"])

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
                    if event.key == pygame.K_UP:
                        game_status = self.game.step(Game2048.UP)
                    if event.key == pygame.K_RIGHT:
                        game_status = self.game.step(Game2048.RIGHT)
                    if event.key == pygame.K_DOWN:
                        game_status = self.game.step(Game2048.DOWN)

            self.paint_current_state(self.game.current_state())

            if self.game_status == Game2048.GAME_ENDED:
                break

    def paint_current_state(self, state):
        _, board = state

        for row in range(self.n):
            for col in range(self.n):
                self.paint_tile(board[row, col], (row, col))

    def paint_tile(self, value, position):
        tile = pygame.Surface((90, 90))

        screen_pos = (10 + 100*position[1], 10 + 100*position[0])

        if value == 0:
            tile.fill(self.Colors["light_gray"])
            self.screen.blit(tile, screen_pos)
        else:
            tile.fill(self.Colors["white"])
            font = pygame.font.Font(pygame.font.get_default_font(), 50)
            label = font.render(str(value), True, self.Colors["dark_gray"])
            self.screen.blit(tile, screen_pos)
            self.screen.blit(label, screen_pos)
        
        pygame.display.flip()


if __name__ == "__main__":
    GUI(4).run()
