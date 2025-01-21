import torch
from abc import ABC, abstractmethod

class MMDLossPenalty(ABC):

    def __init__(self, weight):
        self._weight = weight

    @abstractmethod
    def get_weighted_penalty(self, U):
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, float]:
        return {
            "penalty_weight": self._weight,
        }

class MMDLossNoPenalty(MMDLossPenalty):

    def __init__(self, weight=0):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        return 0

    def get_stats(self):
        return super().get_stats()

class MMDLossL2Penalty(MMDLossPenalty):

    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        u_l2 = torch.norm(U)
        return self._weight * u_l2

    def get_stats(self):
        return super().get_stats()

class MMDLossHuberPenalty(MMDLossPenalty):

    def __init__(self, weight, delta):
        super().__init__(weight)
        self.delta = delta

    def __huber_loss(self, x):
        if torch.abs(x) >= self.delta:
            return 0.5 * torch.pow(x, 2)
        else:
            return self.delta * (torch.abs(x) - 0.5 * self.delta)

    def get_weighted_penalty(self, U):
        u_l2 = torch.norm(U)
        return self._weight * self.__huber_loss(u_l2)


    def get_stats(self):
        return super().get_stats() | {
            "delta": self.delta,
        }

class MMDLossDiscreteExponentialPenalty(MMDLossPenalty):
    """
    Just like MMDLossDiscretePenalty, but values nearer to 1 are also penalized.
    """
    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        return self._weight * torch.mean(U.float() * torch.exp(U.float()) * (1 - U.float()))

    def get_stats(self):
        return super().get_stats()

class MMDLossDiscreteJenkeJenkePenalty(MMDLossPenalty):
    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        return self._weight * torch.mean((-U.float()) * (U.float() - 0.9) + U.float() / 5)

    def get_stats(self):
        return super().get_stats()

class MMDLossDiscretePenalty(MMDLossPenalty):

    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        return self._weight * torch.mean(U.float() * (1 - U.float()))

    def get_stats(self):
        return super().get_stats()

class MMDLossPenaltyJoin(MMDLossPenalty):

    def __init__(self, mmd_loss_1: MMDLossPenalty, mmd_loss_2: MMDLossPenalty, weight):
        super().__init__(weight)
        self.mmd_loss_1 = mmd_loss_1
        self.mmd_loss_2 = mmd_loss_2

    def get_weighted_penalty(self, U):
        l1 = self.mmd_loss_1.get_weighted_penalty(U)
        l2 = self.mmd_loss_2.get_weighted_penalty(U)
        return self._weight * (l1 + l2)

    def get_stats(self):
        return super().get_stats()  | self.mmd_loss_1.get_stats() | self.mmd_loss_2.get_stats() | {
            "mmd_l1": self.mmd_loss_1.__class__,
            "mmd_l2": self.mmd_loss_2.__class__,
        }
