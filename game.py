import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('serif', 30)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 15
SPEED = 5

class GameAI:
    def __init__(self, w=1080, h=720):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Moving snake
        self._move(action)
        # update the head
        self.snake.insert(0, self.head)
        
        # Checking terminal conditions
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 50*len(self.snake):
            game_over = True
            reward = -5
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 5
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point == None:
            point = self.head

        # Boundary check
        if (point.x > self.w-BLOCK_SIZE) or (point.x < 0) or (point.y > self.h-BLOCK_SIZE) or (point.y < 0):
            return True
        # Self check
        if point in self.snake[1:]:
            return True

        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        i = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            dir = directions[i]
        elif np.array_equal(action, [1, 0, 0]):
            next_i = (i + 1) % 4
            dir = directions[next_i]
        else:
            next_i = (i - 1) % 4
            dir = directions[next_i]

        self.direction = dir 

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)