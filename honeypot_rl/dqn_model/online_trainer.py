import json
import os
import copy
import torch

from honeypot_rl.dqn_model.replay_buffer import ReplayBuffer
from honeypot_rl.dqn_model.Trainer import Training
from honeypot_rl.dqn_model.policy_log import get_policy_log_path

DQN_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(DQN_MODEL_DIR, "trained_model.pt")
with open(os.path.join(DQN_MODEL_DIR, "hyperparameters.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

BATCH_SIZE = cfg["BATCH_SIZE"]
MIN_BUFFER_SIZE = cfg["MIN_BUFFER_SIZE"]
TARGET_UPDATE_INTERVAL = cfg.get("TARGET_UPDATE_INTERVAL", 50)
MODEL_SAVE_INTERVAL = cfg.get("MODEL_SAVE_INTERVAL", 1000)


class OnlineTrainer:

    def __init__(self, model):
        self.model = model
        self.model_path = TRAINED_MODEL_PATH
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.trainer = Training(model, target_model=self.target_model)
        self.replay_buffer = ReplayBuffer()

        self.current_episode = []
        self.episode_count = 0
        self.total_updates = 0

    def record_input(self, state, action, reward, next_state, next_action, terminal):
        action_tensor = torch.tensor(action, dtype=torch.long)
        next_action_tensor = torch.tensor(next_action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        if terminal:
            terminal_tensor = torch.tensor(1.0, dtype=torch.float32)
        else:
            terminal_tensor = torch.tensor(0.0, dtype=torch.float32)
        self.current_episode.append((state.clone(), action_tensor, reward_tensor, next_state.clone(), next_action_tensor, terminal_tensor))

    def end_episode(self, terminal_reward=None):
        if not self.current_episode:
            return

        n = len(self.current_episode)
        if terminal_reward is not None and n:
            spread = float(terminal_reward) / n
        else:
            spread = 0.0
        for i, inputs in enumerate(self.current_episode):
            state, action, reward, next_state, next_action, terminal = inputs
            r = float(reward) + spread
            if terminal_reward is not None:
                term = (i == n - 1)
            else:
                if hasattr(terminal, "item"):
                    t = terminal.item()
                else:
                    t = float(terminal)
                term = bool(t != 0)
            self.replay_buffer.add(state, action, torch.tensor(r, dtype=torch.float32), next_state, next_action, torch.tensor(1.0 if term else 0.0, dtype=torch.float32))

        self.episode_count += 1

        buffer_size = len(self.replay_buffer)

        if buffer_size >= MIN_BUFFER_SIZE:
            batch = self.replay_buffer.sample(BATCH_SIZE)
            if batch is not None:
                states, actions, rewards, next_states, next_actions, terminal = batch
                loss, q_values = self.trainer.batch_update(states, actions, rewards, next_states, next_actions, terminal)
                self.total_updates += 1
                if self.total_updates % TARGET_UPDATE_INTERVAL == 0:
                    self.trainer.update_target_network()
                try:
                    path = get_policy_log_path()
                    event = {"event": "policy_update", "episode": self.episode_count, "total_updates": self.total_updates, "loss": float(loss)}
                    line = json.dumps(event) + "\n"
                    dirpath = os.path.dirname(path)
                    os.makedirs(dirpath, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(line)
                except OSError:
                    pass
        if self.episode_count % MODEL_SAVE_INTERVAL == 0 and self.episode_count > 0:
            torch.save(self.model.state_dict(), self.model_path)

        self.current_episode = []
