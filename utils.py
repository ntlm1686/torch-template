import torch
import yaml
import numpy as np
from easydict import EasyDict
from dataloaders import *


def get_dataset(config):
    if config.dataset == "CIFAR-100":
        return cifar100('train'), cifar100('valid')
    elif config.dataset == "CIFAR-10":
        return cifar10('train'), cifar10('valid')
    else:
        raise ValueError("Unknown dataset")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def parse_config(path):
    """Parses a config file into a dictionary"""
    with open(path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

def get_parameter_names(model, ignore):
    result = []
    for name, child in model.named_children():
        result += [f"{name}.{n}" for n in get_parameter_names(child, ignore)
                   if not isinstance(child, tuple(ignore))]
    result += list(model._parameters.keys())
    return result

def get_run_name(config):
    return f"{config.dataset}-{config.arch}"

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
