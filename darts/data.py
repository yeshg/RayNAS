# Source: https://github.com/carpedm20/ENAS-pytorch/blob/master/data/text.py
import os
from filelock import FileLock

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import collections

class Image(object):
    def __init__(self, args):
        if args.dataset == 'cifar10':
            Dataset = datasets.CIFAR10

            mean = [0.49139968, 0.48215827, 0.44653124]
            std = [0.24703233, 0.24348505, 0.26158768]

            normalize = transforms.Normalize(mean, std)

            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif args.dataset == 'MNIST':
            Dataset = datasets.MNIST
        else:
            raise NotImplementedError(f'Unknown dataset: {args.dataset}')

        self.train = torch.utils.data.DataLoader(
            Dataset(root='~/data', train=True, transform=transform, download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        self.valid = torch.utils.data.DataLoader(
            Dataset(root='~/data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        self.test = self.valid

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = collections.Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1

        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.num_tokens = len(self.dictionary)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
