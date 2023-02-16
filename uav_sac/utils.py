import math
import torch
import numpy as np
import os
import random

repo_path = "/mnt/gpfs2_16m/pscratch/bxi224_uksr/pytorch-soft-actor-critic"  # TODO: fix

def max_file_search(file_format_code):
    '''
    messy binary search for files prefixed with numbers
    '''
    low = 1
    high = 10000
    mid = (high + low) // 2
    while True:
        if low == mid or high == mid:
            return mid
            # return file_format_code.format(mid)
        isfile = os.path.isfile(file_format_code.format(mid))
        if isfile:
            low = mid
        else:
            high = mid
        mid = (high + low) // 2


def load_random_animation():
    '''
    parse how many animations we have and pick one from the group, 
    TODO: do some format assertions here so that we can have description errors
    '''
    animations_path = os.path.join(repo_path, "animations", "{}_plume.npy")
    pick = random.randint(1,max_file_search(animations_path))
    animation = np.load(animations_path.format(pick))
    animation.flags.writeable = False # lock animation don't want to change it
    return animation


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
