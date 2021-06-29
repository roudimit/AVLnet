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