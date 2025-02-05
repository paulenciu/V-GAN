import torch
import math

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

        l1_stats = {f"mmd_l1_{k}": v for k, v in self.mmd_loss_1.get_stats().items()}
        l2_stats = {f"mmd_l2_{k}": v for k, v in self.mmd_loss_2.get_stats().items()}

        return super().get_stats() | l1_stats | l2_stats | {
            "mmd_l1_type": self.mmd_loss_1.__class__.__name__,
            "mmd_l2_type": self.mmd_loss_2.__class__.__name__,
        }

class MMDSparsityPenalty(MMDLossPenalty):

    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        return self._weight * torch.mean(U.float())

    def get_stats(self):
        return super().get_stats()

class MMDDiversityPenalty(MMDLossPenalty):

    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        pairwise_dist = torch.cdist(U, U)
        return self._weight * -1 * torch.mean(pairwise_dist)

    def get_stats(self):
        return super().get_stats()

class KLDivergencePenalty(MMDLossPenalty):

    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        target_probs = 0.5 * (torch.distributions.Normal(0, 0.1).log_prob(U) +
                              torch.distributions.Normal(1, 0.1).log_prob(U))
        kl_loss = -torch.mean(target_probs)
        return self._weight * kl_loss

    def get_stats(self):
        return super().get_stats()

class MMDGMMLoss(MMDLossPenalty):
    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        lambda_gmm=0.1
        sigma = 0.1
        m_flat = U.view(-1)
        log_prob_0 = -0.5 * ((m_flat - 0) / sigma) ** 2
        log_prob_1 = -0.5 * ((m_flat - 1) / sigma) ** 2
        log_prob = torch.logsumexp(torch.stack([log_prob_0, log_prob_1]), dim=0) - math.log(2)
        return -lambda_gmm * torch.mean(log_prob)

    def get_stats(self):
        return super().get_stats()

class MMDTVPenalty(MMDLossPenalty):

    def __init__(self, weight):
        super().__init__(weight)

    def get_weighted_penalty(self, U):
        lambda_tv = 0.05
        h_diff = U[:, :, 1:, :] - U[:, :, :-1, :]
        w_diff = U[:, :, :, 1:] - U[:, :, :, :-1]
        return lambda_tv * (torch.mean(h_diff.abs()) + torch.mean(w_diff.abs()))

    def get_stats(self):
        return super().get_stats()


class MMDSigmoidPenalty(MMDLossPenalty):

    def __init__(self, weight, epsilon=1e-6):
        super().__init__(weight)
        self.epsilon = epsilon

    def get_weighted_penalty(self, U):
        numerator = (U - 0.25) ** 2 * (U - 0.75) ** 2
        denominator = U * (1 - U) * ((U - 0.5) ** 2 + self.epsilon)

        return self._weight * torch.mean(numerator / denominator)

    def get_stats(self):
        return super().get_stats() | {
            "epsilon": self.epsilon,
        }

class MMDSigmoidPenalty2(MMDLossPenalty):

    def __init__(self, weight, epsilon=100):
        super().__init__(weight)
        self.epsilon = epsilon

    def get_weighted_penalty(self, U):
        penalty = (U - 0.25) ** 2 * (U - 0.75) ** 2 * self.epsilon
        return self._weight * penalty

    def get_stats(self):
        return super().get_stats() | {
            "epsilon": self.epsilon,
        }