from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

def compute_metrics(x, eval_lang_retrieval=False, eval_msrvtt=False):
    if eval_lang_retrieval:
        print("Retrieving language given input video clips")
        x = x.T
    else:
        print("Retrieving video clips given input language")
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    test_set_size = x.shape[0] if not eval_msrvtt else 1000 
    if eval_msrvtt: print("MSR-VTT: counting {} missing test clips as mistakes".format(1000 - x.shape[0]))
    metrics['R1'] = float(np.sum(ind == 0)) / test_set_size
    metrics['R5'] = float(np.sum(ind < 5)) / test_set_size
    metrics['R10'] = float(np.sum(ind < 10)) / test_set_size
    metrics['MR'] = np.median(ind) + 1
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count