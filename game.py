import numpy as np


class Game2048:

    GAME_ENDED = -1

    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3

    def __init__(self, n):
        self.n = n
        self.score = 0
        self.board = np.zeros((n, n), int)

        self.insert_random()
        self.insert_random()

    # Insert new value in random position
    def insert_random(self):
        y = self.rand()
        x = self.rand()

        while self.board[y, x] != 0:
            y = self.rand()
            x = self.rand()

        self.board[y, x] = self.two_or_four()

    # Perform action, check if lost, insert new value
    def step(self, direction):
        board_changed = self.action(direction)

        if board_changed:
            if 0 not in self.board:
                return self.GAME_ENDED
            self.insert_random()

    # Move the pieces to the direction and unify when needed. The 
    # movement is always done to the left, so in the other directions 
    # the board is rotated and de-rotated.
    def action(self, direction):
        orig_board = self.board.copy()
        self.board = np.rot90(self.board, direction)

        for row in range(self.n):

            new_line = np.zeros((self.n,), int)
            new_index = 0

            line = self.board[row]
            line = line[line != 0]

            i = 0
            while i < len(line):
                # Copy single number
                if i == len(line)-1 or line[i] != line[i+1]:
                    new_line[new_index] = line[i]
                    new_index = new_index + 1
                    i = i + 1
                # Sum two consecutive equal numbers
                else:
                    new_line[new_index] = 2*line[i]
                    self.score = self.score + 2*line[i]
                    new_index = new_index + 1
                    i = i + 2
            
            self.board[row] = new_line

        self.board = np.rot90(self.board, -direction)

        return not (self.board == orig_board).all()

    def __str__(self):
        return str(self.board)

    ############ AUXILIARY FUNCTIONS ############

    # Random number in [0,n)
    def rand(self):
        return int(np.random.rand()*self.n)

    # 90% chance it's 2, 10% it's 4
    def two_or_four(self):
        return 2 if np.random.rand() > 0.1 else 4


game = Game2048(2)
print(game)
game.step(Game2048.RIGHT)
game.step(Game2048.LEFT)
game.step(Game2048.TOP)
game.step(Game2048.BOTTOM)