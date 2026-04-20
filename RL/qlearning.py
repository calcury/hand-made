import numpy as np
import pygame

GRID_SIZE = 10
CELL_SIZE = 40
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995


class GridWorld:
    def __init__(self):
        # 0: 空地, 1: 障碍物
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.grid[2:8, 2] = 1
        self.grid[5, 6:10] = 1
        self.reset()

    def reset(self):
        self.state = np.array([0, 0])
        return self.state_to_index(self.state)

    def state_to_index(self, state):
        return state[0] * GRID_SIZE + state[1]

    def step(self, action):
        new_state = self.state.copy()
        if action == 0 and self.state[0] > 0:
            new_state[0] -= 1
        elif action == 1 and self.state[0] < GRID_SIZE - 1:
            new_state[0] += 1
        elif action == 2 and self.state[1] > 0:
            new_state[1] -= 1
        elif action == 3 and self.state[1] < GRID_SIZE - 1:
            new_state[1] += 1

        if self.grid[new_state[0], new_state[1]] == 0:
            self.state = new_state

        goal = np.array([GRID_SIZE - 1, GRID_SIZE - 1])
        done = np.array_equal(self.state, goal)
        reward = 10 if done else -1

        return self.state_to_index(self.state), reward, done


Q_table = np.zeros((NUM_STATES, NUM_ACTIONS))


def get_action(state, Q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    return np.argmax(Q_table[state])


def update_q_table(state, action, reward, next_state, Q_table):
    best_next_q = np.max(Q_table[next_state])
    target = reward + GAMMA * best_next_q
    Q_table[state, action] = (1 - ALPHA) * \
        Q_table[state, action] + ALPHA * target
    return Q_table

# --- 4. 可视化 ---


def init_pygame():
    pygame.init()
    return pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))


def draw_state(screen, state_index, env):
    screen.fill((255, 255, 255))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if env.grid[r, c] == 1:
                pygame.draw.rect(screen, (100, 100, 100),
                                 (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, (0, 255, 0), ((GRID_SIZE-1) *
                     CELL_SIZE, (GRID_SIZE-1)*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    r, c = state_index // GRID_SIZE, state_index % GRID_SIZE
    pygame.draw.rect(screen, (0, 0, 255), (c*CELL_SIZE,
                     r*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()


screen = init_pygame()
clock = pygame.time.Clock()
env = GridWorld()
state = env.reset()
running = True
speed = 10000

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = get_action(state, Q_table, EPSILON)
    next_state, reward, done = env.step(action)
    Q_table = update_q_table(state, action, reward, next_state, Q_table)
    state = next_state

    if done:
        state = env.reset()
        speed *= 0.99
        EPSILON = max(0.01, EPSILON * EPSILON_DECAY)

    draw_state(screen, state, env)
    clock.tick(max(50, speed))

pygame.quit()
