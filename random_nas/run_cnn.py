#!/usr/bin/env python
from random_nas.operations import *
from random_nas.sampler import ArchitectureSampler
from random_nas.model import NasNet
from random_nas.data import Image

import torch.optim as optim

# import ray

def run_experiment(args):
    # ray.init(address=args.ray_address, num_cpus=6 if args.smoke_test else None)

    ## Creating an architecture sampler
    controller = ArchitectureSampler(
        h_ops=[avgpool3x3, maxpool3x3, conv3x3, conv5x5, sep3x3, sep5x5],
        com_ops=[add, concat],
        B=5,
        viz=True
    )

    ## Sampling an architecture
    cell_nodes = controller.sample()
    controller.view_graph()
    for i in range(len(cell_nodes[1])):
        print(f"cell id = {cell_nodes[1][i].id}")

    ## Creating NasNet model
    model = NasNet(
        N=4,
        initial_filters=32,
        repeats=2,
        decisions=cell_nodes
    )
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ## Load dataset
    dataset = Image(args)

    for i, data in enumerate(dataset.train, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        print(outputs.size())
        exit()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    parser = argparse.ArgumentParser()
    # parser.add_argument("--use-gpu", action="store_true", default=False, help="enables CUDA training")
    parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Name of dataset to use")
    parser.add_argument("--batch_size",  default=64, type=int, help="Training Batch Size")
    parser.add_argument("--dataset_workers", default=2, type=int, help="cpus for torch DataLoaders")
    parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    args = parser.parse_args()

    args.smoke_test = True

    run_experiment(args)
