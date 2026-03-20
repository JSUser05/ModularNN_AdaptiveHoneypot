import torch


class Policy:
    def __init__(self, network: torch.nn.Module, action_map: list[str], input_length: int):
        self.network = network
        self.action_map = action_map
        self.input_length = input_length

    def add_state_id(self, s_id) -> torch.Tensor:
        if isinstance(s_id, int):
            ids = [s_id]
        else:
            ids = list[int](s_id)
        if len(ids) < self.input_length:
            pad = self.input_length - len(ids)
            ids = [0] * pad + ids
        else:
            ids = ids[-self.input_length:]
        return torch.tensor([ids], dtype=torch.long)

    def __decide__(self, s_id, inp) -> tuple:

        state = self.add_state_id(s_id)
        self.network.eval()
        with torch.no_grad():
            q = self.network(state)
        self.network.train()
        a_id = int(q.argmax(dim=1).item())
        result = self.action_map[a_id]
        return (a_id, result, q[0].tolist())
