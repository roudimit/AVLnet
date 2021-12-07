from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch as th
import numpy as np

class MMS_loss(th.nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * th.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = th.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss

class AMM_loss(th.nn.Module):
    def __init__(self):
        super(AMM_loss, self).__init__()

    def forward(self, S, alpha=0.5):
        ...
        # print("S shape", S.shape)
        # print("eye", th.eye(S.size(0)).to(S.device))
        # print("inverse eye", 1 - th.eye(S.size(0)).to(S.device))
        # TODO: margin needs to be different for each i
        # Sii - (1/B-1)sum(Sij) is the same as summing over Sij where all not on diagonal have * -1/(B-1)
        # can we get inverse of eye?
        identity = th.eye(S.size(0)).to(S.device)
        weights = identity - (1 / (S.size(0)-1)) * (1 - identity)
        # print("S", S)
        # print("weights", weights)
        weighted_S = S * weights
        # print("weighted S", weighted_S)
        margins = alpha * th.sum(weighted_S, dim=1)
        # print("margins", margins)
        deltas = margins * identity
        # replace with th.diag?
        # print("deltas", deltas)
        S = S - deltas
        target = th.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss


