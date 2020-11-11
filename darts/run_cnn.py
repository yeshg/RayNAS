import os
import numpy as np
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

import ray
from ray import tune
from ray.tune import track
from ray.tune import Trainable
from ray.tune.logger import Logger
from ray.tune.schedulers import AsyncHyperBandScheduler, PopulationBasedTraining

# DARTS imports
from darts.data import Image
from darts.arch import *
from darts.cnn import genotypes
# from darts.cnn.model import NetworkCIFAR, NetworkImageNet  # TODO: Make this work for ImageNet, MNIST
from darts.cnn.model_search import Network

# Change these values if you want the training to run quicker or slower.
# EPOCH_SIZE = 512
# TEST_SIZE = 256

def train(model, criterion, optimizer, train_loader, device=torch.device("cpu")):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=torch.device("cpu")):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

class train_cnn(Trainable):

    def setup(self, config):
        args = config["args"]
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        dataset = Image(args)
        self.train_loader, self.test_loader = dataset.train, dataset.valid
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)
        self.model = Network(36, 10, config["layers"], self.criterion, steps=4, multiplier=4, stem_multiplier=3)
        self.model = self.model.to(self.device)
        self.weights = get_weights_from_arch(config['arch'], 4)
        set_model_weights(self.model, self.weights)
        self.model.drop_path_prob = 0.0

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 600.0)

        self.save_checkpoint(self.logdir)

    def step(self):
        train(self.model, self.criterion, self.optimizer, self.train_loader, self.device)
        self.scheduler.step()
        acc = test(self.model, self.test_loader, self.device)
        return {'mean_accuracy': acc}

    def save_checkpoint(self, tmp_checkpoint_dir):
        # Save model weights
        model_path = os.path.join(tmp_checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        # Save genotype
        genotype_path = os.path.join(tmp_checkpoint_dir, "genotype.pkl")
        with open(genotype_path, "wb") as f:
            pickle.dump(self.model.genotype(), f)
        return self.model.state_dict()

    def load_checkpoint(self, tmp_checkpoint_dir):
        # Load matching model weights
        model_path = os.path.join(tmp_checkpoint_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path), strict=False)

def run_experiment(args):

    if args.smoke_test:
        args.layers = 2

    if args.ray_address:
        ray.init(address=args.ray_address)

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy"
    )
    sched = PopulationBasedTraining(
        time_attr='time_total_s',
        metric='mean_accuracy',
        mode='max',
        perturbation_interval=5.0,
        custom_explore_fn=lambda c: {'arch': perturb_arch(c['arch'], 4), 'use_gpu': c['use_gpu']}
    )

    analysis = tune.run(
        train_cnn,
        name="darts",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.95,
            "training_iteration": 2 if args.smoke_test else 100
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1  # int(args.cuda) * 0.5
        },
        num_samples=1 if args.smoke_test else 50,
        config={
            "args": args,
            "arch": tune.sample_from(lambda _: sample_arch(4)),
            "layers": args.layers  # can use a flag to make this variable per tune worker later on
        }
    )

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    parser = argparse.ArgumentParser(description="DARTS: CNN NAS")
    parser.add_argument("--smoke-test", default=False, action="store_true", help="Finish quickly for testing")
    parser.add_argument("--layers", default=20, type=int, help="Number of layers in model")
    parser.add_argument("--ray-address", default=None, help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enables GPU training")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Name of dataset")
    args = parser.parse_args()

    args.smoke_test = True

    run_experiment(args)
