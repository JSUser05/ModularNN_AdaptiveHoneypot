import json
import os
import random
import torch
from collections import deque

from cowrie.commands import __all__ as COMMANDS
from cowrie.rl.dqn_model.q_network import QNetwork
from cowrie.rl.dqn_model.online_trainer import OnlineTrainer
from cowrie.rl.dqn_model.policy_log import write_policy_decision
from cowrie.rl.reward.qrassh_reward import QRaSSHReward

# QRaSSH parameters and training model path
DQN_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(DQN_MODEL_DIR, "trained_model.pt")
with open(os.path.join(DQN_MODEL_DIR, "hyperparameters.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)


# builds all commands based on the cowrie commands listed in the commands folder
def build_command_ids():
    names = sorted(COMMANDS)
    ids = {n: i + 1 for i, n in enumerate(names)}
    ids["./"] = len(names) + 1
    next_id = len(names) + 2
    for name in ("exit", "cd", "df"):
        if name not in ids:
            ids[name] = next_id
            next_id += 1
    return ids, next_id


command_ids, first_unknown_id = build_command_ids()
DICTIONARY_SIZE = cfg["DICTIONARY_SIZE"]

INPUT_LENGTH = cfg["INPUT_LENGTH"]
EPSILON = cfg["EPSILON"]
USE_EPSILON_DECAY = cfg.get("USE_EPSILON_DECAY", False)
EPSILON_MIN = cfg.get("EPSILON_MIN", 0.1)
EPSILON_DECAY_START_EPISODE = cfg.get("EPSILON_DECAY_START_EPISODE", 500)
EPSILON_DECAY_STEPS = cfg.get("EPSILON_DECAY_STEPS", 1500)
NUM_ACTIONS = cfg["NUM_ACTIONS"]
ACTION_NAMES = cfg["ACTION_NAMES"]
REWARD_STANDARD = cfg["REWARD_STANDARD"]
REWARD_WEIGHT = cfg["REWARD_WEIGHT"]

DWNLD_CMDS = {"wget", "curl", "ftp"}


class agent:

    def __init__(self, network: torch.nn.Module | None = None, action_map: list[str] | None = None):
        if network is not None:
            self.model = network
        else:
            self.model = QNetwork()

        self.model_path = TRAINED_MODEL_PATH
        try:
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        except (FileNotFoundError, RuntimeError) as e:
            pass

        self.model.train()
        self.online_trainer = OnlineTrainer(self.model)
        self.command_history = deque(maxlen=INPUT_LENGTH)
        self.previous_state = None
        self.previous_action = None
        self.previous_command = None
        self.session_length = 0
        self.last_action_state = None
        self.last_action = None
        self.last_action_command = None
        self.last_s_id = None
        self.last_session_id = None

        self.reward = QRaSSHReward(reward_standard=REWARD_STANDARD, reward_weight=REWARD_WEIGHT)
        if action_map is not None:
            self.action_map = action_map
        else:
            self.action_map = ACTION_NAMES

    def __decide__(self, s_id: int):
        state = torch.tensor([[s_id]], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            q = self.model(state)
        self.model.train()

        a_id = int(q.argmax(dim=1).item())
        return a_id

    def current_epsilon(self):
        if not USE_EPSILON_DECAY:
            return EPSILON
        episode = getattr(self.online_trainer, "episode_count", 0)
        if episode < EPSILON_DECAY_START_EPISODE:
            return EPSILON
        t = episode - EPSILON_DECAY_START_EPISODE
        frac = min(1.0, t / EPSILON_DECAY_STEPS)
        frac = max(0.0, min(1.0, frac))
        return EPSILON - (EPSILON - EPSILON_MIN) * frac

    def get_action(self, command, session_id: str = ""):
        if session_id and session_id != self.last_session_id and self.previous_state is not None:
            self.reset_history()
        if session_id:
            self.last_session_id = session_id

        if command is None:
            cmd_name = None
            cmd_id = 0
        elif command.strip().startswith("./") or " ./" in command:
            cmd_id = command_ids.get("./", 0)
            cmd_name = "./"
        else:
            cmd_name = command.split()[0]
            if cmd_name in command_ids:
                cmd_id = command_ids[cmd_name]
            else:
                cmd_id = first_unknown_id

        self.command_history.append(cmd_id)

        # ******Handling the length of the state****** 
        if INPUT_LENGTH == 1:
            state_list = [cmd_id]
        else:
            if len(self.command_history) < INPUT_LENGTH:
                pad = INPUT_LENGTH - len(self.command_history)
                state_list = [0] * pad + list(self.command_history)
            else:
                state_list = list(self.command_history)

        state = torch.tensor(state_list, dtype=torch.long)

        if INPUT_LENGTH == 1:
            s_id = cmd_id
        else:
            s_id = state_list

        if isinstance(s_id, int):
            reward_s_id = s_id
        else:
            if s_id:
                reward_s_id = s_id[-1]
            else:
                reward_s_id = 0
        # ******Handling the length of the state****** 

        self.model.eval()
        with torch.no_grad():
            q = self.model(state.unsqueeze(0))
        self.model.train()
        a_id = int(q.argmax(dim=1).item())
        action_name = ACTION_NAMES[a_id]
        q_values = q[0].tolist()
        eps = self.current_epsilon()
        if random.random() < eps:
            a_id = random.randint(0, NUM_ACTIONS - 1)
            action_name = ACTION_NAMES[a_id]
            greedy = False
        else:
            greedy = True

        self.session_length += 1
        r = self.reward.__decide__(reward_s_id, a_id, terminal=False, session_length=self.session_length, command=cmd_name)
        if self.previous_state is not None:
            self.online_trainer.record_input(self.previous_state, self.previous_action, r, state, a_id, terminal=False)

        if command:
            raw_command = command
        else:
            raw_command = ""
        write_policy_decision(session_id=session_id, command=raw_command, action_id=a_id, action_name=action_name, q_values=q_values, greedy=greedy)

        if cmd_name != "exit":
            self.last_action_state = state
            self.last_action = a_id
            self.last_action_command = cmd_name
            self.last_s_id = reward_s_id

        self.previous_state = state
        self.previous_action = a_id
        self.previous_command = cmd_name

        if cmd_name == "exit":
            self.reset_history()

        return a_id

    def reset_history(self):
        r_terminal = None
        if self.last_action_state is not None and self.last_action is not None:
            if self.last_s_id is not None:
                sid = self.last_s_id
            else:
                sid = 0
            r_terminal = self.reward.__decide__(sid, self.last_action, terminal=True, session_length=self.session_length, command=self.last_action_command)
        elif self.previous_state is not None:
            sid = getattr(self, "last_s_id", None)
            if sid is None:
                sid = 0
            r_terminal = self.reward.__decide__(sid, self.previous_action, terminal=True, session_length=self.session_length, command=self.previous_command)
        self.online_trainer.end_episode(terminal_reward=r_terminal)
        self.command_history.clear()
        self.previous_state = None
        self.previous_action = None
        self.previous_command = None
        self.session_length = 0
        self.last_action_state = None
        self.last_action = None
        self.last_action_command = None
        self.last_s_id = None


agent_instance = None

def get_agent():
    global agent_instance
    if agent_instance is None:
        agent_instance = agent()
    return agent_instance


def decide_action(command, session_id: str = ""):
    return get_agent().get_action(command, session_id=session_id)


def reset_session():
    get_agent().reset_history()
