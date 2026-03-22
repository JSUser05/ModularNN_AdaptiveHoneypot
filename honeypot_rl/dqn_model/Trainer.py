import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

DQN_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DQN_MODEL_DIR, "hyperparameters.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

GAMMA = cfg["GAMMA"]
LR = cfg["LR"]
USE_SARSA = cfg["USE_SARSA"]
TARGET_UPDATE_INTERVAL = cfg.get("TARGET_UPDATE_INTERVAL", 50)


class Training:
    def __init__(self, model, target_model=None):
        self.model = model
        if target_model is not None:
            self.target_model = target_model
        else:
            self.target_model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def batch_update(self, states, actions, rewards, next_states, next_actions, terminal):
        self.model.train()
        q_values = self.model(states)

        actions_unsqueezed = actions.unsqueeze(1)
        q_values_taken = q_values.gather(1, actions_unsqueezed).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)

            if USE_SARSA:
                q_next_taken = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values = rewards + GAMMA * q_next_taken * (1 - terminal)
            else:
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + GAMMA * max_next_q_values * (1 - terminal)

        loss = nn.functional.mse_loss(q_values_taken, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), q_values.detach()

    def update_target_network(self):
        if self.target_model is not self.model:
            self.target_model.load_state_dict(self.model.state_dict())
