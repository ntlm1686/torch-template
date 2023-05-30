# 'train' 'valid'

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy


train_transform_list = [transforms.RandomCrop(size=(32, 32), padding=4)]
train_transform_list.append(
    transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
test_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize(
        [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    )])
train_transform = transforms.Compose(
    train_transform_list +
    [transforms.ToTensor(),
        transforms.Normalize(
        [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    )])


def _cifar100(split: str):
    train = True if split == 'train' else False
    transform = train_transform if train else test_transform
    return torchvision.datasets.CIFAR100(root='./data', train=train,
                                         download=True, transform=transform)
