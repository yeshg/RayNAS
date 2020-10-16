import numpy as np
import torch
import copy
from torch.autograd import Variable

from nas.darts.cnn import genotypes
from nas.darts.cnn.model_search import Network

def sample_arch(steps):
    k = sum(1 for i in range(steps) for n in range(2+i))
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = steps

    normal = []
    reduction = []
    for i in range(n_nodes):
        ops = np.random.choice(range(num_ops), 4)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
        normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
        reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

    normal, reduction = [(int(n[0]), int(n[1])) for n in normal], [(int(r[0]), int(r[1])) for r in reduction]

    return normal, reduction

def perturb_arch(arch, steps):
    new_arch = copy.deepcopy(arch)
    num_ops = len(genotypes.PRIMITIVES)

    cell_ind = np.random.choice(2)
    step_ind = np.random.choice(steps)
    nodes_in = np.random.choice(step_ind+2, 2, replace=False)
    ops = np.random.choice(range(num_ops), 2)

    new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
    new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])

    new_arch = tuple([[(int(a[0]), int(a[1])) for a in cell] for cell in new_arch])

    return new_arch

def get_weights_from_arch(arch, steps):
    k = sum(1 for i in range(steps) for n in range(2+i))
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = steps

    alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
    alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

    offset = 0
    for i in range(n_nodes):
        normal1 = arch[0][2*i]
        normal2 = arch[0][2*i+1]
        reduce1 = arch[1][2*i]
        reduce2 = arch[1][2*i+1]
        alphas_normal[offset+normal1[0], normal1[1]] = 1
        alphas_normal[offset+normal2[0], normal2[1]] = 1
        alphas_reduce[offset+reduce1[0], reduce1[1]] = 1
        alphas_reduce[offset+reduce2[0], reduce2[1]] = 1
        offset += (i+2)

    arch_parameters = [
        alphas_normal,
        alphas_reduce,
    ]
    return arch_parameters

def set_model_weights(model, weights):
    model.alphas_normal = weights[0]
    model.alphas_reduce = weights[1]
    model._arch_parameters = [model.alphas_normal, model.alphas_reduce]
