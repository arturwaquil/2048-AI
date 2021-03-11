import numpy as np

class Game2048:

    def __init__(self, n):
        self.n = n
        self.score = 0
        self.board = np.zeros((n,n), int)

        self.insert_random()
        self.insert_random()

    def insert_random(self):
        y = self.rand()
        x = self.rand()

        while self.board[y][x] != 0:
            y = self.rand()
            x = self.rand()
        
        self.board[y][x] = self.two_or_four()

    # 90% chance it's 2, 10% it's 4
    def two_or_four(self):
        return 2 if np.random.rand() > 0.1 else 4

    # Random number in [0,n)
    def rand(self):
        return int(np.random.rand()*self.n)

    def __str__(self):
        return str(self.score) + "\n" + str(self.board)

print(Game2048(4))