import numpy as np


class Game2048:

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    last_move = "(none)"

    def __init__(self, n=4, seed=None):
        self.n = n
        self.seed(seed)
        self.new_game()

    def new_game(self):
        self.in_game = True
        self.score = 0
        self.board = np.zeros((self.n, self.n), int)

        self.insert_random()
        self.insert_random()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    # Insert new value in random position
    def insert_random(self):
        if 0 not in self.board: return

        while True:
            tup = (int(self.rng.random()*self.n), int(self.rng.random()*self.n))
            if self.board[tup] == 0:
                break

        self.board[tup] = 2 if self.rng.random() > 0.1 else 4

    # Perform action, check if lost, insert new value
    def step(self, direction):
        if not self.in_game: return
        
        board_changed = self.action(direction)

        if board_changed:
            self.last_move = ["left", "up", "right", "down"][direction]
            self.insert_random()
            if not self.moves_available():
                self.in_game = False
        
        return self.board, self.score, not self.in_game

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

        # Return true if board changed
        return (self.board != orig_board).any()
    
    # Return a list of booleans indicating for each direction 
    # (LEFT, UP, RIGHT, DOWN) if it's possible to make a move
    def possible_moves(self):
        orig_board = self.board.copy()

        left  = self.action(self.LEFT);  self.board = orig_board.copy()
        up    = self.action(self.UP);    self.board = orig_board.copy()
        right = self.action(self.RIGHT); self.board = orig_board.copy()
        down  = self.action(self.DOWN);  self.board = orig_board.copy()

        return [left, up, right, down]

    def moves_available(self):
        if 0 in self.board: return True

        temp_board = np.zeros((self.n+2, self.n+2), int)
        temp_board[1:-1,1:-1] = self.board

        for x in range(self.n):
            for y in range(self.n):
                temp_x = x+1
                temp_y = y+1

                tile = temp_board[temp_y, temp_x]

                if tile == temp_board[temp_y, temp_x-1]: return True    # left
                if tile == temp_board[temp_y-1, temp_x]: return True    # up
                if tile == temp_board[temp_y, temp_x+1]: return True    # right
                if tile == temp_board[temp_y+1, temp_x]: return True    # down

        return False

    def current_state(self):
        return self.score, self.board

    def __str__(self):
        return "\nSCORE: " + str(self.score) + \
               "\nLAST MOVE: " + self.last_move + \
               "\n" + str(self.board)
