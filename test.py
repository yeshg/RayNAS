# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from nas import *
from darts.cnn.model import NetworkCIFAR
from darts.cnn import genotypes


# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

def get_data_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    from filelock import FileLock
    with FileLock(os.path.expanduser("~/data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root="/data/daiyaanarfeen/cifar10",
            train=True,
            download=True,
            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=96, shuffle=True)
    valset = torchvision.datasets.CIFAR10(
        root="/data/daiyaanarfeen/cifar10", train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def train(model, criterion, optimizer, train_loader, device=torch.device("cpu")):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)[0]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=torch.device("cpu")):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)[0]
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

model = NetworkCIFAR(36, 10, 20, False, genotypes.DARTS)
model = model.cuda()
arch = sample_arch(4)
weights = get_weights_from_arch(arch, 4)
set_model_weights(model, weights)
model.drop_path_prob = 0.0
trainset, testset = get_data_loaders()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 600.0)
for i in range(10):
    train(model, nn.CrossEntropyLoss().cuda(), optimizer, trainset, torch.device('cuda'))
    print(test(model, testset, torch.device('cuda')))
    scheduler.step()
