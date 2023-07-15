import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split


def get_dataset(dir, name):
    if name == 'mnist':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=trans)
        eval_dataset = datasets.MNIST(dir, train=False, transform=trans)

    elif name == 'femnist':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_dataset = datasets.FashionMNIST(dir, train=True, download=True, transform=trans)
        eval_dataset = datasets.FashionMNIST(dir, train=False, transform=trans)
    elif name == 'svhn':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_dataset = datasets.SVHN(root=dir, download=True, transform=trans, split='train')
        eval_dataset = datasets.SVHN(root=dir, download=True, transform=trans, split='test')
    elif name == 'eurosat':
        trans = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize([128,128]),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        dataset = datasets.EuroSAT(root=dir,download=True, transform=trans)
        length = len(dataset)
        train_len = 14000
        eval_len = length - train_len
        train_dataset, eval_dataset = random_split(dataset,[train_len,eval_len])

    elif name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    print(len(train_dataset))
    return train_dataset, eval_dataset
