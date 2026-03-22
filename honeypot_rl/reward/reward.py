from abc import ABC, abstractmethod


class Reward(ABC):
    @abstractmethod
    def __decide__(self, s_id: int, a_id: int, **kwargs):
        raise NotImplementedError
