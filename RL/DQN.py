import pygame
import numpy as np
import random
import math
import os


class UnicycleEnv:
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.gravity = 9.8
        self.dt = 0.02
        self.force_mag = 15.0
        self.pole_length = 1.0

    def reset(self):
        self.state = np.array([random.uniform(-0.05, 0.05), 0.0, 0.0, 0.0])
        return self.state

    def step(self, action_idx):
        force = (action_idx - 2) * self.force_mag
        theta, theta_dot, x, x_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        total_mass = 1.1
        p_len_mass = self.pole_length * 0.1

        temp = (force + p_len_mass * theta_dot**2 * sintheta) / total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / \
            (self.pole_length * (1.33 - 0.1 * costheta**2 / total_mass))
        x_acc = temp - p_len_mass * theta_acc * costheta / total_mass

        self.state[0] += theta_dot * self.dt
        self.state[1] += theta_acc * self.dt
        self.state[2] += x_dot * self.dt
        self.state[3] += x_acc * self.dt

        done = abs(self.state[0]) > 0.85 or abs(self.state[2]) > 2.4

        reward = self.get_reward(state, done)
        return self.state, reward, done

    def get_reward(self, state, done):
        theta, theta_dot, x, x_dot = state
        if done:
            return -100.0

        penalty = 0.1 * (abs(theta_dot) + abs(x_dot))
        score = (1.0 - abs(theta)/0.42) + (1.0 - abs(x)/2.4)

        return score - penalty


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)


class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        self.w1 = np.random.randn(input_size, hidden_size) * scale
        self.w2 = np.random.randn(hidden_size, output_size) * scale
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.input_x = x
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.output = np.dot(self.a1, self.w2) + self.b2
        return self.output

    def backward(self, target, learning_rate=0.00002):
        d_output = self.output - target
        d_hidden = np.dot(d_output, self.w2.T) * (self.z1 > 0)
        self.w2 -= learning_rate * np.dot(self.a1.T, d_output)
        self.w1 -= learning_rate * np.dot(self.input_x.T, d_hidden)


def get_normalized_state(state):
    return np.array([state[0]/0.42, state[1]/2.0, state[2]/2.4, state[3]/2.0])


class DQNAgent:
    def __init__(self):
        self.model = SimpleNet(4, 64, 5)
        self.target_net = SimpleNet(4, 64, 5)
        self.update_target_net()

    def update_target_net(self):
        self.target_net.w1, self.target_net.w2 = self.model.w1.copy(), self.model.w2.copy()

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 4)
        return np.argmax(self.model.forward(get_normalized_state(state).reshape(1, -1)))

    def train(self, buffer, batch_size=64):
        if len(buffer.buffer) < batch_size:
            return
        s, a, r, ns, d = buffer.sample(batch_size)
        s_norm = np.array([get_normalized_state(i) for i in s])
        ns_norm = np.array([get_normalized_state(i) for i in ns])

        target = self.model.forward(s_norm)
        max_next_q = np.max(self.target_net.forward(ns_norm), axis=1)
        target[np.arange(batch_size), a] = r + 0.99 * max_next_q * (1 - d)
        self.model.backward(target)


pygame.init()
screen = pygame.display.set_mode((800, 600))
env, agent, buffer = UnicycleEnv(), DQNAgent(), ReplayBuffer()
epsilon, step_count = 1.0, 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = env.state.copy()
    action = agent.act(state, epsilon)
    next_state, reward, done = env.step(action)
    buffer.push(state, action, reward, next_state, done)

    if step_count > 1000:
        agent.train(buffer)
    if step_count % 200 == 0:
        agent.update_target_net()

    screen.fill((255, 255, 255))
    theta, _, x, _ = state
    cart_x = int(400 + x * 100)
    pole_x = int(cart_x + 150 * math.sin(theta))
    pole_y = int(500 - 150 * math.cos(theta))
    pygame.draw.line(screen, (0, 0, 0), (cart_x, 500), (pole_x, pole_y), 6)
    pygame.draw.circle(screen, (0, 0, 255), (cart_x, 500), 15)
    pygame.display.flip()

    if done:
        env.reset()
        print(f"Episode Done, Step: {step_count}")

    step_count += 1
    epsilon = max(0.1, epsilon * 0.99999)
    if step_count % 1000 == 0:
        q_values = agent.model.forward(
            get_normalized_state(env.state).reshape(1, -1))
        print(f"Step {step_count}: Q-values: {q_values}")
    if step_count < 100000:
        pygame.time.Clock().tick(0)
    else:
        pygame.time.Clock().tick(60)

pygame.quit()
