from cgitb import small
import torch
import random
from collections import deque
import numpy as np

from game import BLOCK_SIZE, GameAI, Direction, Point

MAX_MEM = 10**6
BATCH_SZ = 10**3
LRATE = 10**-3

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEM)
        self.model = None
        self.trainer = None


    def getState(self, game):
        head = game.snake[0]
        Pleft = Point(head.x - BLOCK_SIZE, head.y)
        Pright = Point(head.x + BLOCK_SIZE, head.y)
        Pup = Point(head.x, head.y - BLOCK_SIZE)
        Pdown = Point(head.x, head.y + BLOCK_SIZE)

        Dleft = game.direction == Direction.LEFT
        Dright = game.direction == Direction.RIGHT
        Dup = game.direction == Direction.UP
        Ddown = game.direction == Direction.DOWN

        curr_state = [
            (Dright and game.is_collision(Pright)) or
            (Dleft and game.is_collision(Pleft)) or
            (Dup and game.is_collision(Pup)) or
            (Ddown and game.is_collision(Pdown)),

            (Dup and game.is_collision(Pright)) or
            (Ddown and game.is_collision(Pleft)) or
            (Dleft and game.is_collision(Pup)) or
            (Dright and game.is_collision(Pdown)),

            (Ddown and game.is_collision(Pright)) or
            (Dup and game.is_collision(Pleft)) or
            (Dright and game.is_collision(Pup)) or
            (Dleft and game.is_collision(Pdown)),

            Dleft,
            Dright,
            Dup,
            Ddown,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(curr_state, dtype=int)


    def writeMemory(self, state, action, reward, next_state, ok):
        self.memory.append((state, action, reward, next_state, ok))

    def train_long_term(self):
        if len(self.memory) > BATCH_SZ:
            sample_small = random.sample(self.memory, BATCH_SZ)
        else:
            sample_small = self.memory
        
        states, actions, rewards, next_states, oks = zip(*sample_small)
        self.trainer.train_step(states, actions, rewards, next_states, oks)

    def train_short_term(self, state, action, reward, next_state, ok):
        self.trainer.train_step(state, action, reward, next_state, ok)

    def calcAction(self, state):
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    scores = []
    scores_mean = []
    score_total = 0
    record = 0
    agent = Agent()
    game = GameAI()
    while True:
        state_old = agent.getState(game)
        move_final = agent.calcAction(state_old)
        reward, ok, score = game.play_step(move_final)
        state_new = agent.getState(game)

        agent.train_short_term(state_old, move_final, reward, state_new, ok)
        agent.writeMemory(state_old, move_final, reward, state_new, ok)

        if ok:
            game.reset()
            agent.n_games += 1
            agent.train_long_term()

            if score > record:
                record = score

            print(f"Game {agent.n_games} Score {score} Record {record}")


if __name__ == '__main__':
    train()