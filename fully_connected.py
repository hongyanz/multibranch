#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange


ARGS = None


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--n_paths', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    # Training
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    # Calculating loss & plotting
    parser.add_argument('--range', type=float, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    # Book keeping
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--save_surface_to', type=str, required=True)
    parser.add_argument('--save_loss_to', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    # Misc
    parser.add_argument('--seed', type=int, required=True)
    # Parsing
    args = parser.parse_args()
    # Replace "{run_id}"
    for key in vars(args):
        value = getattr(args, key)
        if isinstance(value, str) and "{run_id}" in value:
            setattr(args, key, value.replace("{run_id}", args.run_id))
    # Initialize
    random.seed(a=args.seed)
    np.random.seed(seed=args.seed)
    torch.manual_seed(seed=args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=args.seed)
    return args


class Model(nn.Module):

    def __init__(self, input_size, n_paths, hidden_size, n_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        modules = nn.ModuleList()
        for _ in range(n_paths):
            modules.append(nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_classes),
            ))
        self.paths = modules

    def forward(self, x): # pylint: disable=W0221
        mbs = x.shape[0] # mini-batch size
        x = x.view(mbs, -1)
        outs = []
        for path in self.paths:
            outs.append(path(x).view(mbs, 1, -1))
        outs = torch.cat(outs, dim=1).mean(dim=1)  # pylint: disable=E1101
        outs = outs.view(mbs, -1)
        return outs

    def get_loss(self, data_x, data_y, volatile=False):
        data_x = Variable(data_x, volatile=volatile, requires_grad=False)
        data_y = Variable(data_y, volatile=volatile, requires_grad=False)
        logits = self.__call__(data_x)
        loss = F.multi_margin_loss(logits, data_y)
        return loss


def get_param_pos(model):
    pos_list = []
    for name, param in model.named_parameters():
        length = param.data.view(-1).size(0)
        pos_list.append((name, length))
    pos_list.sort()
    pos, last_pos = {}, 0
    for name, length in pos_list:
        pos[name] = (last_pos, last_pos + length)
        last_pos += length
    assert "len" not in pos
    pos["len"] = last_pos
    return pos


def to_param_vector(model: nn.Module, pos: dict):
    res = np.zeros(pos["len"], dtype="float32")
    for name, param in model.named_parameters():
        param = param.data.view(-1)
        st_pos, ed_pos = pos[name]
        res[st_pos: ed_pos] = param.numpy()
    res = torch.from_numpy(res) # pylint: disable=E1101
    return res


def make_model():
    model = Model(input_size=784, n_paths=ARGS.n_paths, hidden_size=ARGS.hidden_size, n_classes=10)
    param_pos = get_param_pos(model)
    return model, param_pos


def load_data(path):
    data = np.load(path)
    data_x = torch.from_numpy(data["data_x"])  # pylint: disable=E1101
    data_y = torch.from_numpy(data["data_y"])  # pylint: disable=E1101
    return data_x, data_y


def make_data():
    return load_data("./data/mnist.npz")


def train_model(model_path: str, make_model, make_data, args):
    model, param_pos = make_model()
    model.train()
    data_x, data_y = make_data()
    data_size = data_x.shape[0]
    shuffled_indices = list(range(data_size))
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    global_step, last_loss, decay_times = 0, None, 3
    done = False
    for _ in trange(args.n_epochs):
        np.random.shuffle(shuffled_indices)
        for st_idx in range(0, data_size, args.batch_size):
            ed_idx = min(st_idx + args.batch_size, data_size)
            indices = shuffled_indices[st_idx: ed_idx]
            # Get loss
            loss = model.get_loss(data_x[indices], data_y[indices])
            pyloss = loss.data[0]
            # Update model
            optim.zero_grad()
            loss.backward()
            optim.step()
            global_step += 1
            # Loss is 0, or explodes
            if pyloss < 1e-5 or np.isnan(pyloss) or np.isinf(pyloss):
                done = True
                break
            # Check every 100 times
            if global_step % 100 != 0:
                continue
            # None last_loss, don't decay
            if last_loss is None:
                last_loss = pyloss
                continue
            # Loss is decreasing, don't decay
            if (last_loss + 1e-5) >= pyloss:
                continue
            # Used up all decay_times, break
            if decay_times == 0:
                done = True
                break
            # Do the decay
            decay_times -= 1
            for param_group in optim.param_groups:
                if 'lr' in param_group:
                    param_group['lr'] *= 0.1
            # Update last_loss
            last_loss = pyloss
        if done:
            break
    torch.save(model.state_dict(), model_path)
    return to_param_vector(model, param_pos)


def load_param_vector(model, pos, vec):
    for name, param in model.named_parameters():
        st_pos, ed_pos = pos[name]
        loaded = vec[st_pos: ed_pos].view(*param.shape)
        param.data.copy_(loaded)


class Runner(object):

    def __init__(self, make_model, make_data, vectors):
        def _make_tensor(x): # pylint: disable=C0103
            x = x.astype("float32")
            x = torch.from_numpy(x) # pylint: disable=E1101
            return x
        self.model, self.pos = make_model()
        self.data_x, self.data_y = make_data()
        self.vectors = list(map(_make_tensor, vectors))
        self.buffer = self.vectors[0].new(*self.vectors[0].shape)

    def compute(self, x_len, y_len):
        theta_0, axis_x, axis_y = self.vectors
        self.model.eval()
        self.buffer                \
            .copy_(theta_0)        \
            .add_(x_len, axis_x)   \
            .add_(y_len, axis_y)
        load_param_vector(self.model, self.pos, self.buffer)

        if isinstance(self.data_x, list):
            losses = []
            for data_x, data_y in zip(self.data_x, self.data_y):
                losses.append(self.model.get_loss(data_x, data_y, volatile=True).data[0])
            loss = sum(losses) / len(losses)
        else:
            loss = self.model.get_loss(self.data_x, self.data_y, volatile=True).data[0]
        return loss


def main():

    def _get_weights():
        weights = []
        for idx in range(3):
            weights.append(train_model(
                model_path=ARGS.model_path.format(idx=idx),
                make_model=make_model,
                make_data=make_data,
                args=ARGS,
            ).view(1, -1))
        return weights

    def _project(weights):
        theta_0 = torch.cat(weights, dim=0).mean(dim=0)  # pylint: disable=E1101
        axis_x = (weights[0] - weights[1]).view(-1)
        axis_y = (weights[0] - weights[2]).view(-1)
        axis_x = axis_x / axis_x.norm(p=2) * weights[-1].norm(p=2)
        axis_y = axis_y / axis_y.norm(p=2) * weights[-1].norm(p=2)
        theta_0 = theta_0.numpy()
        axis_x = axis_x.numpy()
        axis_y = axis_y.numpy()
        return theta_0, axis_x, axis_y

    weights = _get_weights()
    theta_0, axis_x, axis_y = _project(weights)
    runner = Runner(make_model, make_data, [theta_0, axis_x, axis_y])
    grid_x = np.linspace(-ARGS.range, ARGS.range, num=ARGS.resolution).tolist()
    grid_y = np.linspace(-ARGS.range, ARGS.range, num=ARGS.resolution).tolist()
    for idx, (x_len, y_len) in enumerate(product(grid_x, grid_y)):
        print(idx, x_len, y_len, runner.compute(x_len, y_len))


ARGS = parse_args()
if __name__ == "__main__":
    main()
