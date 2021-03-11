import numpy as np


class Game2048:

    GAME_ENDED = -1

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    last_move = "(none)"

    def __init__(self, n):
        self.n = n
        self.score = 0
        self.board = np.zeros((n, n), int)

        self.insert_random()
        self.insert_random()

    # Insert new value in random position
    def insert_random(self):
        while True:
            tup = (int(np.random.rand()*self.n), int(np.random.rand()*self.n))
            if self.board[tup] == 0:
                break

        self.board[tup] = 2 if np.random.rand() > 0.1 else 4

    # Perform action, check if lost, insert new value
    def step(self, direction):
        board_changed = self.action(direction)

        if board_changed:
            self.last_move = ["left", "up", "right", "down"][direction]
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
        return "\nSCORE: " + str(self.score) + \
               "\nLAST MOVE: " + self.last_move + \
               "\n" + str(self.board)
