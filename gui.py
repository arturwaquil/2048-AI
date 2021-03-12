import sys
import math
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "True"
import pygame
from game import Game2048


class GUI:

    Colors = {
        "transparent_whitish": (238, 228, 218, 186),
        "gray_light": (205, 191, 179),
        "gray_medium": (185, 169, 157),
        "gray_dark": (119, 109, 101),
        "white": (237, 226, 217),
        "whitish": (238, 225, 201),
        "orange": (243, 178, 122),
        "orange_dark": (246, 150, 100),
        "redish": (247, 124, 95),
        "red": (247, 95, 59),
        "yellow_1": (237, 208, 115),
        "yellow_2": (237, 204, 98),
        "yellow_3": (237, 201, 80),
        "yellow_4": (237, 197, 63),
        "yellow_5": (237, 194, 46),
        "black": (60, 58, 51)
    }

    def __init__(self, n):
        pygame.init()

        # Info on dimensions
        self.n = n
        self.tile_side = 130
        self.margin = 10
        self.score_board_height = 40
        self.width = self.tile_side*n + self.margin
        self.height = self.width + self.score_board_height
        self.size = (self.width, self.height)
        
        # The game screen
        pygame.display.set_caption("2048")
        self.screen = pygame.display.set_mode(self.size)

        # Game core
        self.game = Game2048(self.n)

        self.on_end_screen = False

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()
                    elif event.key == pygame.K_n:
                        self.game.new_game()
                    elif event.key == pygame.K_LEFT:
                        self.game.step(Game2048.LEFT)
                    elif event.key == pygame.K_UP:
                        self.game.step(Game2048.UP)
                    elif event.key == pygame.K_RIGHT:
                        self.game.step(Game2048.RIGHT)
                    elif event.key == pygame.K_DOWN:
                        self.game.step(Game2048.DOWN)

            if self.game.in_game or not self.on_end_screen:
                self.paint_current_state(self.game.current_state())

            if not self.game.in_game:
                self.paint_end_screen()

    def paint_end_screen(self):
        if not self.on_end_screen:
            self.on_end_screen = True

            end_screen = pygame.Surface(self.size, pygame.SRCALPHA)
            end_screen.fill(self.Colors["transparent_whitish"])
            self.screen.blit(end_screen, (0,0))

            font = pygame.font.Font(pygame.font.get_default_font(), int(0.4*self.tile_side))
            label = font.render("Game over!", True, self.Colors["gray_dark"])
            label_rect = label.get_rect(center=(self.width/2,self.height/2))
            self.screen.blit(label, label_rect)

            font = pygame.font.Font(pygame.font.get_default_font(), int(0.2*self.tile_side))
            label = font.render("Press 'n' to try again", True, self.Colors["gray_dark"])
            label_rect = label.get_rect(center=(self.width/2,self.height/2+0.4*self.tile_side))
            self.screen.blit(label, label_rect)

            pygame.display.flip()

    def paint_current_state(self, state):
        score, board = state

        self.screen.fill(self.Colors["gray_medium"])

        # Print score in the upper part of the window
        score_space = pygame.Surface((self.width, self.score_board_height))
        score_space.fill(self.Colors["gray_medium"])
        self.screen.blit(score_space, (0,0))
        font = pygame.font.Font(pygame.font.get_default_font(), self.score_board_height-self.margin)
        label = font.render("SCORE: " + str(score), True, self.Colors["white"])
        self.screen.blit(label, (self.margin,self.margin))

        # Paint all tiles
        for row in range(self.n):
            for col in range(self.n):
                self.paint_tile(board[row, col], (row, col))

        pygame.display.flip()

    def paint_tile(self, value, position):
        tile = pygame.Surface((self.tile_side-self.margin, self.tile_side-self.margin))

        row_pos = self.tile_side*position[0] + self.margin + self.score_board_height
        col_pos = self.tile_side*position[1] + self.margin
        screen_pos = (col_pos, row_pos)

        if value == 0:
            tile.fill(self.Colors["gray_light"])
            self.screen.blit(tile, screen_pos)
        else:
            tile.fill(self.Colors[self.get_tile_color(value)])
            self.screen.blit(tile, screen_pos)
            font = pygame.font.Font(pygame.font.get_default_font(), self.get_label_font_size(value))
            label = font.render(str(value), True, self.Colors[self.get_label_color(value)])
            half_tile_side = (self.tile_side-self.margin)/2
            label_rect = label.get_rect(center=(col_pos+half_tile_side, row_pos+half_tile_side))
            self.screen.blit(label, label_rect)
        

    def get_tile_color(self, value):
        colors = ["white", "whitish", "orange", "orange_dark", "redish", "red", 
                  "yellow_1", "yellow_2", "yellow_3", "yellow_4", "yellow_5", "black"]
        return colors[min(int(math.log(value,2))-1, 11)]
    
    def get_label_color(self, value):
        return "gray_dark" if value in [2,4] else "white"
    
    def get_label_font_size(self, value):
        if value <= 64: return 60
        elif value <= 512: return 50
        elif value <= 2048: return 40
        else: return 30

if __name__ == "__main__":
    GUI(4).run()
