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
        
        pygame.display.set_caption("2048")

        self.tile_side = 100
        self.margin = 10
        self.score_board_height = 40
        self.width = self.tile_side*n + self.margin
        self.height = self.width + self.score_board_height
        self.size = (self.width, self.height)
        
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
        score, board = state

        # Print score in the upper part of the window
        score_space = pygame.Surface((self.width, self.score_board_height))
        score_space.fill(self.Colors["medium_gray"])
        self.screen.blit(score_space, (0,0))
        font = pygame.font.Font(pygame.font.get_default_font(), self.score_board_height-self.margin)
        label = font.render("SCORE: " + str(score), True, self.Colors["dark_gray"])
        self.screen.blit(label, (self.margin,self.margin))
        pygame.display.flip()

        # Paint all tiles
        for row in range(self.n):
            for col in range(self.n):
                self.paint_tile(board[row, col], (row, col))

    def paint_tile(self, value, position):
        tile = pygame.Surface((self.tile_side-self.margin, self.tile_side-self.margin))

        row_pos = self.margin + self.tile_side*position[0] + self.score_board_height
        col_pos = self.margin + self.tile_side*position[1]
        screen_pos = (col_pos, row_pos)

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
