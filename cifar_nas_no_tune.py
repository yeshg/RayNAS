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

# EPOCH_SIZE = 12
# TEST_SIZE = 6


def train(model, criterion, optimizer, train_loader, device=torch.device("cpu")):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, logits_aux = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        print(f'batch_idx: {batch_idx}\t loss: {loss}')

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
    print(correct/total)
    return correct / total

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
    with FileLock(os.path.expanduser("~/data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root="~/data",
            train=True,
            download=True,
            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valset = torchvision.datasets.CIFAR10(
        root="~/data", train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)
    return train_loader, test_loader

class train_cifar:

    def __init__(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)
#        self.model = Network(36, 10, 20, self.criterion, steps=4, multiplier=4)
        self.model = NetworkCIFAR(36, 10, config["layers"], False, genotypes.DARTS)
        print(torch.cuda.max_memory_allocated(self.device))
        self.model = self.model.to(self.device)
        print(torch.cuda.max_memory_allocated(self.device))
        self.weights = get_weights_from_arch(config['arch'], 4)
        set_model_weights(self.model, self.weights)
        self.model.drop_path_prob = 0.0

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 600.0)

    def _train(self):
        train(self.model, self.criterion, self.optimizer, self.train_loader, self.device)
        self.scheduler.step()
        acc = test(self.model, self.test_loader, self.device)
        print(f'mean_accuracy: {acc}')

    def _save(self, checkpoint):
        return self.model.state_dict()

    def _restore(self, checkpoint):
        self.model.load_state_dict(checkpoint, strict=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--layers", default=20, type=int, help="Number of layers in model")
    args = parser.parse_args()

    trainable = train_cifar(config={
        "arch": sample_arch(4),
        "use_gpu": args.cuda,
        "layers": args.layers
    })

    for epoch in range(EPOCH_SIZE):
        trainable._train()
