import json
import os
import random
import torch
from collections import deque
from twisted.python import log

DQN_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DQN_MODEL_DIR, "hyperparameters.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
MAX_BUFFER_SIZE = cfg["MAX_BUFFER_SIZE"]


class ReplayBuffer:
    def __init__(self):
        self.max_size = MAX_BUFFER_SIZE
        self.buffer = deque(maxlen=self.max_size)

    def add(self, state, action, reward, next_state, next_action, terminal):
        self.buffer.append((state, action, reward, next_state, next_action, terminal))

    def sample(self, batch_size):
        k = min(batch_size, len(self.buffer))
        if k == 0:
            return None

        indices = sorted(random.sample(range(len(self.buffer)), k))
        indices_set = set(indices)
        batch = [self.buffer[i] for i in indices]

        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        next_action_list = []
        terminal_list = []

        for experiences in batch:
            state, action, reward, next_state, next_action, terminal = experiences
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            next_action_list.append(next_action)
            terminal_list.append(terminal)

        new_buffer = deque(maxlen=self.max_size)
        for i, x in enumerate(self.buffer):
            if i not in indices_set:
                new_buffer.append(x)
        self.buffer = new_buffer

        states = torch.stack(state_list)
        actions = torch.stack(action_list)
        rewards = torch.stack(reward_list)
        next_states = torch.stack(next_state_list)
        next_actions = torch.stack(next_action_list)
        terminals = torch.stack(terminal_list)

        return states, actions, rewards, next_states, next_actions, terminals

    def __len__(self) -> int:
        return len(self.buffer)
