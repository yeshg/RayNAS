import os
from filelock import FileLock

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_image_data_loaders(args):
    if args.dataset == 'cifar10':
        """
        input_size : [args.batch_size, 3, 32, 32]
        label_size : [args.batch_size]
        """
        Dataset = datasets.CIFAR10

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
    elif args.dataset == 'mnist':
        Dataset = datasets.MNIST
    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')

    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            Dataset(
                root='~/data',
                train=True,
                download=True,
                transform=transform_train
            ),
            batch_size=64,
            shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        Dataset(
            root='~/data',
            train=False,
            download=False,
            transform=transform_test
        ),
        batch_size=64,
        shuffle=False
    )

    return train_loader, test_loader

# TODO: common func for RNN datasets
def get_text_data_loaders(args):
    raise NotImplementedError
