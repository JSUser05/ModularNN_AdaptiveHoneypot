from .reward import Reward


class QRaSSHReward(Reward):

    def __init__(self, reward_standard: float = 1.0, reward_weight: float = 2.0):
        self.reward_standard = reward_standard
        self.reward_weight = reward_weight

    def __decide__(self, s_id: int, a_id: int, *, terminal: bool = False, session_length: int = 0, **kwargs):
        if terminal:
            return -2 * self.reward_standard + self.reward_weight * session_length
        return self.reward_standard
