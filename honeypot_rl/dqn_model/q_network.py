import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DQN_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DQN_MODEL_DIR, "hyperparameters.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

DICTIONARY_SIZE = cfg["DICTIONARY_SIZE"]
EMBEDDED_DIM = cfg["EMBEDDED_DIM"]
INPUT_LENGTH = cfg["INPUT_LENGTH"]
HIDDEN_LAYER1 = cfg["HIDDEN_LAYER1"]
HIDDEN_LAYER2 = cfg["HIDDEN_LAYER2"]
DROPOUT = cfg["DROPOUT"]
NUM_ACTIONS = cfg["NUM_ACTIONS"]


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(DICTIONARY_SIZE, EMBEDDED_DIM)

        if INPUT_LENGTH == 1:
            self.fc1 = nn.Linear(EMBEDDED_DIM, HIDDEN_LAYER2)
        else:
            self.lstm = nn.LSTM(EMBEDDED_DIM, HIDDEN_LAYER1, batch_first=True, bidirectional=True)
            lstm_size = HIDDEN_LAYER1 * 2
            self.fc1 = nn.Linear(lstm_size, HIDDEN_LAYER2)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc_out = nn.Linear(HIDDEN_LAYER2, NUM_ACTIONS)

    def forward(self, x):
        x = self.embedding(x)

        if INPUT_LENGTH == 1:
            x = x.squeeze(1)
        else:
            x, _ = self.lstm(x)
            x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        q_values = self.fc_out(x)
        return q_values
