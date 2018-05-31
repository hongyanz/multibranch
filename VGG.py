#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import CIFAR10

ARGS = None


class Vgg9(nn.Module):

    def __init__(self, width):
        super(Vgg9, self).__init__()
        self.conv1 = nn.Conv2d(3, width * 1, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(width * 1, width * 1, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(width * 1)
        self.bn2 = nn.BatchNorm2d(width * 1)

        self.conv3 = nn.Conv2d(width * 1, width * 2, kernel_size=3, bias=False, padding=1)
        self.conv4 = nn.Conv2d(width * 2, width * 2, kernel_size=3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 2)
        self.bn4 = nn.BatchNorm2d(width * 2)

        self.conv5 = nn.Conv2d(width * 2, width * 4, kernel_size=3, bias=False, padding=1)
        self.conv6 = nn.Conv2d(width * 4, width * 4, kernel_size=3, bias=False, padding=1)
        self.conv7 = nn.Conv2d(width * 4, width * 4, kernel_size=3, bias=False, padding=1)
        self.bn5 = nn.BatchNorm2d(width * 4)
        self.bn6 = nn.BatchNorm2d(width * 4)
        self.bn7 = nn.BatchNorm2d(width * 4)

        self.fc1 = nn.Linear(width * 64, width * 4, bias=False)
        self.fc2 = nn.Linear(width * 4, 10)
        self.bn8 = nn.BatchNorm1d(width * 4)

    def forward(self, x):  # pylint: disable=W0221
        def _func(x, func, bn, activation=F.relu):
            out = func(x)
            if bn is not None:
                out = bn(out)
            if activation is not None:
                out = activation(out)
            return out
        out = x
        out = _func(out, self.conv1, self.bn1)
        out = _func(out, self.conv2, self.bn2)
        out = F.max_pool2d(out, kernel_size=2)
        out = _func(out, self.conv3, self.bn3)
        out = _func(out, self.conv4, self.bn4)
        out = F.max_pool2d(out, kernel_size=2)
        out = _func(out, self.conv5, self.bn5)
        out = _func(out, self.conv6, self.bn6)
        out = _func(out, self.conv7, self.bn7)
        out = F.max_pool2d(out, kernel_size=2)
        out = out.view(-1, self.fc1.in_features)
        out = _func(out, self.fc1, self.bn8)
        out = _func(out, self.fc2, bn=None, activation=None)
        return out


def vgg_9(width=32):
    return Vgg9(width=width)


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--n_paths', type=int, required=True)
    # Training
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--width', type=int, default=None)
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    # Parsing
    args = parser.parse_args()
    random.seed(a=args.seed)
    np.random.seed(seed=args.seed)
    torch.manual_seed(seed=args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=args.seed)
    return args


def build_dataset():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    trans_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_data = CIFAR10(root="./data/raw_cifar10", train=True, download=True, transform=trans_train)
    valid_data = CIFAR10(root="./data/raw_cifar10", train=False, download=True, transform=trans_valid)
    train_loader = DataLoader(train_data, pin_memory=True, batch_size=ARGS.batch_size, shuffle=True, num_workers=ARGS.n_workers)
    valid_loader = DataLoader(valid_data, pin_memory=True, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.n_workers)
    return train_loader, valid_loader


class Model(nn.Module):

    def __init__(self, n_paths, make_sub_model):
        super(Model, self).__init__()
        modules = nn.ModuleList()
        for _ in range(n_paths):
            modules.append(make_sub_model())
        self.paths = modules

    def forward(self, x): # pylint: disable=W0221
        mbs = x.shape[0]
        outs = []
        for path in self.paths:
            outs.append(path(x).view(mbs, 1, -1))
        outs = torch.cat(outs, dim=1).mean(dim=1) # pylint: disable=E1101
        outs = outs.view(mbs, -1)
        return outs


def main():
    # Build dataset
    train_data, valid_data = build_dataset()
    # Build model
    model = Model(ARGS.n_paths, vgg_9)

    optim = torch.optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[150, 225, 275], gamma=0.1)

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"# of parameters: {n_params}")

    def do_eval(name, data):
        model.eval()
        total_samples = 0.0
        total_loss = 0.0
        total_correct = 0.0
        for data_x, data_y in data:
            data_x = Variable(data_x, volatile=True)
            data_y = Variable(data_y, volatile=True)
            logits = model(data_x)
            _, pred_y = logits.data.max(dim=1)

            loss = F.cross_entropy(logits, data_y, size_average=False)

            total_samples += data_y.shape[0]
            total_loss += loss.data[0]
            total_correct += pred_y.eq(data_y.data).cpu().sum()
        total_loss /= total_samples
        total_correct /= total_samples
        print(f"{name} loss {total_loss:.5f} accuracy {total_correct * 100.0:.5f}")
        model.train()
        return total_correct

    # Start training
    global_step = 0
    for _ in range(ARGS.n_epochs):
        scheduler.step()
        for data_x, data_y in tqdm(train_data):
            data_x, data_y = Variable(data_x), Variable(data_y)
            optim.zero_grad()
            logits = model(data_x)
            loss = F.cross_entropy(logits, data_y)
            loss.backward()
            optim.step()
            global_step += 1
        do_eval("train", train_data)
        do_eval("valid", valid_data)


if __name__ == "__main__":
    ARGS = parse_args()
    main()
