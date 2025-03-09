# Copyright 2025 Anonymized Authors

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""
This script contains all helper methods that were introduced throughout the 
notebooks. 

Usage: 
To import a method m in a notebook please use:

import sys
sys.path.append('..')
from utils.nas_utils import m
 
"""


from nasbench.api import ModelSpec, NASBench
import copy
import random
import numpy as np
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.func import grad
from torch import vmap
from torch.autograd import Function
import time 
import torch.nn.functional as F
from typing import Callable, Any,List



class NASBenchConstants:
    # Useful constants
    INPUT = "input"
    OUTPUT = "output"
    CONV3X3 = "conv3x3-bn-relu"
    CONV1X1 = "conv1x1-bn-relu"
    MAXPOOL3X3 = "maxpool3x3"
    NUM_VERTICES = 7
    MAX_EDGES = 9
    EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
    OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed
    ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
    ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix
    OPS_IND = list(range(len(ALLOWED_OPS)))
    TRIU = np.triu_indices(NUM_VERTICES, k=1)
    SPEC = ModelSpec

def plot_data(data, color, label, max_budget=5000000,  which = 2, gran=10000):
  """Computes the mean and IQR fixed time steps."""

  # which = 2 is test, which = 1 is valid
  xs = range(0, max_budget+1, gran)
  mean = [0.0]
  per25 = [0.0]
  per75 = [0.0]
  
  repeats = len(data)
  pointers = [1 for _ in range(repeats)]
  cur = gran
  while cur < max_budget+1:
    all_vals = []
    for repeat in range(repeats):
      while (pointers[repeat] < len(data[repeat][0]) and 
             data[repeat][0][pointers[repeat]] < cur):
        pointers[repeat] += 1
      prev_time = data[repeat][0][pointers[repeat]-1]
      prev_test = data[repeat][which][pointers[repeat]-1]
      next_time = data[repeat][0][pointers[repeat]]
      next_test = data[repeat][which][pointers[repeat]]
      assert prev_time < cur and next_time >= cur

      # Linearly interpolate the test between the two surrounding points
      cur_val = ((cur - prev_time) / (next_time - prev_time)) * (next_test - prev_test) + prev_test
      
      all_vals.append(cur_val)
      
    all_vals = sorted(all_vals)
    mean.append(sum(all_vals) / float(len(all_vals)))
    per25.append(all_vals[int(0.25 * repeats)])
    per75.append(all_vals[int(0.75 * repeats)])
      
    cur += gran
    
  plt.plot(xs, mean, color=color, label=label, linewidth=2)
  plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)

def random_spec(nasbench: NASBench):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(NASBenchConstants.ALLOWED_EDGES,
                                  size=(NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(NASBenchConstants.ALLOWED_OPS, size=NASBenchConstants.NUM_VERTICES).tolist()
        ops[0] = NASBenchConstants.INPUT
        ops[-1] = NASBenchConstants.OUTPUT
        spec = _global_spec_class(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec

def mutate_spec(old_spec, nasbench: NASBench, mutation_rate=1.0):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)

        # In expectation, V edges flipped (note that most end up being pruned).
        edge_mutation_prob = mutation_rate / NASBenchConstants.NUM_VERTICES
        for src in range(0, NASBenchConstants.NUM_VERTICES - 1):
            for dst in range(src + 1, NASBenchConstants.NUM_VERTICES):
                if random.random() < edge_mutation_prob:
                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

        # In expectation, one op is resampled.
        op_mutation_prob = mutation_rate / NASBenchConstants.OP_SPOTS
        for ind in range(1, NASBenchConstants.NUM_VERTICES - 1):
            if random.random() < op_mutation_prob:
                available = [
                    o for o in nasbench.config["available_ops"]
                    if o != new_ops[ind]
                ]
                new_ops[ind] = random.choice(available)

        new_spec = _global_spec_class(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return new_spec

def mutation(nasbench: NASBench, population, mutation_rate=0.5):
    """Mutates each individual of a population."""

    mutated = []
    for p in population:
        new_spec = mutate_spec(p, nasbench, mutation_rate)
        mutated.append(new_spec)
    return mutated

def softmax(x, axis=-1):
    """Implements softmax function with specifiable axis."""
    # Subtract the maximum value along the specified axis for numerical stability
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

def selection_la(parents, nasbench):
    """Selects n parent pairs lookig ahead to see high performing children"""

    psize = len(parents)
    training_time_spent, total_epochs_spent = nasbench.get_budget_counters()
    ground_truth = np.zeros((psize,psize))
    for a in range(psize):
        for b in range(psize):
            p1 = parents[a][1]
            p2 = parents[b][1]
            child = crossparents(p1,p2)
            try:
                data = nasbench.query(child)
                ground_truth[a,b] = data["validation_accuracy"]
            except:   
                ground_truth[a,b] = 0

    nasbench.training_time_spent = training_time_spent
    nasbench.total_epochs_spent = total_epochs_spent

    top_indices = np.argsort(ground_truth.flatten())[-psize:]
    row_indices, col_indices = np.meshgrid(np.arange(psize),np.arange(psize))
    mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])
    pairings = mapping[top_indices]

    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents



def selection_s(parents, S):
    """Selects n parent pairs of a population with size n using matrix S."""

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Extract the accuracy each tuple
    accuracies = [p[0] for p in parents]

    # Convert the list of accuracies to a extended numpy array
    x = np.array(accuracies)[np.newaxis,:]

    y = softmax((x @ S)[:,0,:])

    # Find the indices of the two largest values in each row
    indices = np.argsort(y, axis=1)[:, -2:]

    # add parents to the list of pairing tuples
    size = indices.shape[0]
    paired_parents = []
    for i in range(size):
        p1 = parents[indices[i,0]]
        p2 = parents[indices[i,1]]
        paired_parents.append((p1,p2))
    return paired_parents

def selection_random(parents):
    """Selects n random parent pairs of a population neural network."""
    paired_parents = []
    for _ in range(len(parents)):
        i1 = np.random.randint(len(parents))
        i2 = np.random.randint(len(parents))
        p1 = parents[i1]
        p2 = parents[i2]
        paired_parents.append((p1,p2)) 
    return paired_parents

def tournament_selection(population,  population_size=50, tournament_size=10):
    """Selects n parent pairs of a population with size n using tournament."""

    parents = []
    for _ in range(population_size):
        candidates = random_combination(population, tournament_size)
        p1 = sorted(candidates, key=lambda i: i[0])[-1]
        p2 = sorted(candidates, key=lambda i: i[0])[-2]
        parents.append((p1,p2))
    return parents

def greedy_selection(parents):
    """Selects greedy n parent pairs"""
    p_size = len(parents)
    population = np.asarray([p[0] for p in parents])

    summed_population = []
    for p1 in range(p_size):
        for p2 in range(p_size):
            if p1 < p2: # only upper triangle
                sum = population[p1] + population[p2]
            else:
                sum = 0
            summed_population.append(sum)

    samples = np.argsort(summed_population)[-p_size:]
    row_indices, col_indices = np.meshgrid(np.arange(p_size),np.arange(p_size))
    mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])
    pairings = mapping[samples]

    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents


def selection_pp(parents, model):
    """Selects n parent pairs of a population neural network."""

    # this sampling expects the model to have output dimension psize and instead
    # of already giving probabilities of pairings, this gives a probability of 
    # choosing this parent 

    psize = len(parents)
    accuracies = np.asarray([p[0] for p in parents])

    model_input = tf.cast(tf.constant([accuracies]), tf.int32)
    model_output =  model(model_input).numpy()[0]

    probabilities = model_output
    indices = np.arange(psize)

    number_of_parents = 2
    paired_parents = []
    for _ in range(psize):
        s= np.random.choice(indices, size=number_of_parents, replace=False, p=probabilities)
        p1 = parents[s[0]]
        p2 = parents[s[1]]
        paired_parents.append((p1,p2))
    return paired_parents

def selection_nn(parents, model):
    """Selects n parent pairs of a population neural network."""

    psize = len(parents)
    accuracies = np.asarray([p[0] for p in parents])

    model_input = tf.cast(tf.constant([accuracies]), tf.float32)
    probabilities =  model(model_input).numpy()[0]


    if np.count_nonzero(probabilities) <= len(parents):
        probabilities = np.ones(psize*psize)/(psize*psize)

    row_indices, col_indices = np.meshgrid(np.arange(psize),np.arange(psize))
    mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])
    indices = np.arange(psize*psize)  

    # Sample without replacement using specified probabilities
    samples = np.random.choice(indices, size=psize, replace=False, p=probabilities)
    pairings = mapping[samples]

    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents

def selection_t(parents, model, device):
    """Selects n parent pairs of a torch population neural network."""

    psize = len(parents)
    inputs = np.zeros((1,2,psize,psize))
    for a in range(psize):
        for b in range(psize):
            inputs[0,0,a,b] = parents[a][0]
            inputs[0,1,a,b] = parents[b][0]

    x = torch.tensor(inputs,dtype=torch.float32).to(device)
    output = model(x)
    logits = torch.flatten(output)
    _,topk = torch.topk(logits,32)

    indices = np.asarray(torch.flatten(topk.cpu()))

    row_indices, col_indices = np.meshgrid(np.arange(psize),np.arange(psize))
    flat_mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])

    pairings = flat_mapping[indices]
    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents

def selection_nn_sampling(parents, model, resolution=1):
    """Selects n parent pairs of a population neural network."""

    psize = len(parents)
    accuracies = np.asarray([p[0] for p in parents]).reshape(1, -1)
    tf_acc = tf.cast(tf.constant(accuracies*resolution), tf.int32)
    model_output =  model(tf_acc).numpy()[0]    
    probabilities = model_output

    row_indices, col_indices = np.meshgrid(np.arange(psize),np.arange(psize))
    mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])
    indices = np.arange(psize*psize)  

    # Sample without replacement using specified probabilities
    samples = np.random.choice(indices, size=psize, replace=False, p=probabilities)
    pairings = mapping[samples]

    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents


def selection_ud(parents):
    """Selects n parent pairs on uniform distribution"""
    
    psize = len(parents)
    probabilities = np.ones(psize*psize)
    row_indices, col_indices = np.meshgrid(np.arange(psize),np.arange(psize))
    mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])
    indices = np.arange(psize*psize)  

    # Sample without replacement using specified probabilities
    samples = np.random.choice(indices, size=psize, replace=False, p=probabilities)
    pairings = mapping[samples]

    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents


def crossvertical(m1,m2):
    """Crosses two  2D matrices at crossover point."""

    crossover_point = np.random.randint(1, NASBenchConstants.NUM_VERTICES-1)
    crossover_point = 3

    # Perform crossover
    offspring_matrix = np.copy(m1)
    offspring_matrix[:crossover_point, :] = m2[:crossover_point, :]

    return offspring_matrix


def crossparents(p1,p2):
    """Crosses two parents p1 and p2 and returns p1 if unsuccessful."""

    if p1.original_matrix.shape != p1.original_matrix.shape:
        print("shape not similar")
        return p1
    m = crossvertical(p1.original_matrix,p2.original_matrix)
    new_spec = _global_spec_class(m, p1.original_ops)
    return new_spec

def crossover(parents):
    """Crosses each pair of a population of parent pairs."""

    crossed_parents = []
    for p in parents:
        crossed = crossparents(p[0][1],p[1][1])
        crossed_parents.append(crossed)
    return crossed_parents


def v_crossvertical(m1,m2, crossover_point):
    """Crosses two  2D matrices at crossover point."""
    offspring_matrix = np.copy(m1)
    offspring_matrix[:crossover_point, :] = m2[:crossover_point, :]

    return offspring_matrix

def v_crossops(ops1,ops2,crossover_point):
    """Crosses two  operation lists at crossover point."""

    combined_operations = ops1[:crossover_point] + ops2[crossover_point:]
    return combined_operations

def v_crossparents(nasb, p1,p2):
    """Crosses two parents p1 and p2 and returns p1 if unsuccessful."""

    if p1.original_matrix.shape != p1.original_matrix.shape:
        print("shape not similar")
        return p1
    
    for _ in range(5): # 5 tries
        i = random.randint(1,NASBenchConstants.NUM_VERTICES-1)
        m = v_crossvertical(p1.original_matrix,p2.original_matrix, i)
        o = v_crossops(p1.original_ops, p2.original_ops, i)
        new_spec = _global_spec_class(m, o)
        if nasb.is_valid(new_spec):
            return new_spec
    return p1

def v_crossover(nasb, parents):
    """Crosses each pair of a population of parent pairs."""

    crossed_parents = []
    for p in parents:
        crossed = v_crossparents(nasb, p[0][1],p[1][1])
        crossed_parents.append(crossed)
    return crossed_parents


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def init_pop(nasbench, max_time_budget, p_size=50):
    """Initializes a population of given psize"""

    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    population = []  # (validation, spec) tuples

    # For the first population_size individuals, seed the population with
    # randomly generated cells.
    for _ in range(p_size):
        spec = random_spec(nasbench)
        data = nasbench.query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data["validation_accuracy"], spec))

        if data["validation_accuracy"] > best_valids[-1]:
            best_valids.append(data["validation_accuracy"])
            best_tests.append(data["test_accuracy"])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break
    return population, times, best_valids, best_tests

def run_xevolution(
    nasbench: NASBench,
    max_time_budget=5e6,
    selection_algorithm: Callable[[List], Any] = None,
    initial_population = [],
    itimes = [0.0],
    ibest_valids = [0.0],
    ibest_tests = [0.0],
    p_size=50,
    mutation_algorithm: Callable[[NASBench, List], Any] = mutation,
    crossover_algorithm: Callable[[NASBench, List], Any] = lambda x,y: crossover(y)
):
    """Run a single roll-out of evolution to a fixed time budget."""

    itime = copy.deepcopy(itimes[-1])
    population = []  # (validation, spec) tuples
    if len(initial_population) == 0:
        population, times, best_valids, best_tests = init_pop(nasbench, max_time_budget, p_size)
    else:
        times = copy.deepcopy(itimes)
        best_valids = copy.deepcopy(ibest_valids)
        best_tests = copy.deepcopy(ibest_tests)
        population = copy.deepcopy(initial_population)

    # After the population is seeded, proceed with evolving the population.
    while True:
        selected_candidates = selection_algorithm(population)  
        crossed_parents = crossover_algorithm(nasbench, selected_candidates)
        children = mutation_algorithm(nasbench, crossed_parents)

        # generational approach: children replace old population
        population = []
        for child in children: 
            data = nasbench.query(child)
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent+itime)
            population.append((data["validation_accuracy"], child))

            if data["validation_accuracy"] > best_valids[-1]:
                best_valids.append(data["validation_accuracy"])
                best_tests.append(data["test_accuracy"])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])

        if time_spent+itime> max_time_budget:
            break

    return times, best_valids, best_tests


def run_random_search(nasbench, max_time_budget=5e6):
  """Run a single roll-out of random search to a fixed time budget."""
  nasbench.reset_budget_counters()
  times, best_valids, best_tests = [0.0], [0.0], [0.0]
  while True:
    spec = random_spec(nasbench)
    data = nasbench.query(spec)

    # It's important to select models only based on validation accuracy, test
    # accuracy is used only for comparing different search trajectories.
    if data['validation_accuracy'] > best_valids[-1]:
      best_valids.append(data['validation_accuracy'])
      best_tests.append(data['test_accuracy'])
    else:
      best_valids.append(best_valids[-1])
      best_tests.append(best_tests[-1])

    time_spent, _ = nasbench.get_budget_counters()
    times.append(time_spent)
    if time_spent > max_time_budget:
      # Break the first time we exceed the budget.
      break

  return times, best_valids, best_tests


def run_revolution_search(
    nasbench: NASBench,
    max_time_budget=5e6,
    population_size=50,
    tournament_size=10,
    mutation_rate=0.5
):
    """Run a single roll-out of regularized evolution to a fixed time budget."""

    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    population = []  # (validation, spec) tuples

    # For the first population_size individuals, seed the population with
    # randomly generated cells.
    for _ in range(population_size):
        spec = random_spec(nasbench)
        data = nasbench.query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data["validation_accuracy"], spec))

        if data["validation_accuracy"] > best_valids[-1]:
            best_valids.append(data["validation_accuracy"])
            best_tests.append(data["test_accuracy"])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break
    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        new_spec = mutate_spec(best_spec, nasbench, mutation_rate)

        data = nasbench.query(new_spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual.
        population.append((data["validation_accuracy"], new_spec))
        population.pop(0)

        if data["validation_accuracy"] > best_valids[-1]:
            best_valids.append(data["validation_accuracy"])
            best_tests.append(data["test_accuracy"])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests


def load_nasbench(path=None):
    """Loads nasbench dataset from data."""
    if path is None:
        path = os.path.join("..","generated", "nasbench_only108.tfrecord") 

    physical_devices = tf.config.experimental.list_physical_devices("GPU")

    if len(physical_devices) > 0:
        print("Using GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU.")

    return NASBench(path)



# please use SpecOneHot instead for true one hot encoding
class Spec(ModelSpec):
    """Spec extension to support flattening of adjacency matrix and operations."""

    def __init__(self, matrix, ops):
        super().__init__(matrix=matrix, ops=ops)
        self.lookup = {op: i for i, op in enumerate(NASBenchConstants.ALLOWED_OPS)}
        self.test_decode_encode()
        self.flat = torch.tensor(self.to_flat())


    def to_flat(self):
        ops = self.original_ops[1:-1] # omit first and last one 
        mat = self.original_matrix
        ops_ind = [self.lookup[s] for s in ops]
        flat = self.encode(mat, ops_ind)
        return flat
    
    def get_ops_from_indices(self, ops_ind):
        ops = np.asarray(NASBenchConstants.ALLOWED_OPS)[ops_ind]
        ops = np.concatenate(([NASBenchConstants.INPUT], ops, [NASBenchConstants.OUTPUT]))
        return ops.tolist()

    def encode(self, matrix,ops_ind):
        # TRIU only uses upper triangle matrix indices including diagonal
        flattened =  np.concatenate((matrix[NASBenchConstants.TRIU],ops_ind))
        return flattened

    def decode(self, flat):
        matrix = np.zeros((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES), dtype=flat.dtype)
        matrix[NASBenchConstants.TRIU] = flat[:-NASBenchConstants.NUM_VERTICES+2]
        ops_ind = flat[-NASBenchConstants.NUM_VERTICES+2:]
        return matrix, ops_ind

    def test_decode_encode(self):
        matrix = np.asarray(list(range(NASBenchConstants.NUM_VERTICES*NASBenchConstants.NUM_VERTICES))).reshape((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops_ind = np.random.choice(NASBenchConstants.OPS_IND, size=NASBenchConstants.NUM_VERTICES-2)
        flat = self.encode(matrix,ops_ind)
        m,o = self.decode(flat)
        assert np.array_equal(matrix,m) 
        assert np.array_equal(ops_ind,o) 

    def spec_from_flat(self, flat):
        matrix, ops_ind = self.decode(flat)
        ops = self.get_ops_from_indices(ops_ind)
        spec = ModelSpec(matrix=matrix, ops=ops)
        return spec

# utils for using nasbench dataset

class SpecMixed(ModelSpec):
    """Spec extension to support categorical one-hot for operations."""

    flat_matrix = ((NASBenchConstants.NUM_VERTICES-1+1)*(NASBenchConstants.NUM_VERTICES-1)//2)
    input_size = ((NASBenchConstants.NUM_VERTICES-1+1)*(NASBenchConstants.NUM_VERTICES-1)//2) + len(NASBenchConstants.ALLOWED_OPS)*(NASBenchConstants.NUM_VERTICES - 2)
    name_lu = {op: np.eye(3)[i] for i, op in enumerate(NASBenchConstants.ALLOWED_OPS)}
    onehot_lu = list(name_lu.values())

    def __init__(self, matrix, ops):
        super().__init__(matrix=matrix, ops=ops)
        self.flat = self.to_flat()

    def to_flat(self):
        ops = self.original_ops[1:-1] # omit first and last one 
        mat = self.original_matrix
        ops_onehot = np.concatenate([SpecMixed.name_lu[s] for s in ops])
        flat = self.encode(mat, ops_onehot)
        return torch.tensor(flat, dtype=torch.float)

    @staticmethod
    def encode(matrix,ops_onehot):
        # TRIU only uses upper triangle matrix indices including diagonal
        flattened =  np.concatenate((matrix[NASBenchConstants.TRIU],ops_onehot))
        return flattened
    
    @staticmethod
    def decode(flat):
        matrix = np.zeros((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES), dtype=flat.dtype)
        matrix[NASBenchConstants.TRIU] = flat[:SpecMixed.flat_matrix]
        assert np.all(np.triu(matrix) == matrix)

        ops_onehot = flat[SpecMixed.flat_matrix:].reshape((NASBenchConstants.NUM_VERTICES - 2, len(NASBenchConstants.ALLOWED_OPS)))
        assert np.sum(ops_onehot) == NASBenchConstants.NUM_VERTICES - 2

        indices = np.argmax(ops_onehot, 1)
        return matrix, indices

    @staticmethod
    def test_decode_encode():
        matrix = np.asarray(list(range(NASBenchConstants.NUM_VERTICES*NASBenchConstants.NUM_VERTICES))).reshape((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops_ind = np.random.choice(NASBenchConstants.OPS_IND, size=NASBenchConstants.NUM_VERTICES-2)
        ops_onehot = np.concatenate([SpecMixed.onehot_lu[i] for i in ops_ind])
        flat = SpecMixed.encode(matrix,ops_onehot)
        m,o = SpecMixed.decode(flat)
        assert np.array_equal(matrix,m) 
        assert np.array_equal(ops_ind,o) 

    @staticmethod
    def spec_from_flat(flat):
        matrix, ops_ind = SpecMixed.decode(flat)
        ops = np.asarray(NASBenchConstants.ALLOWED_OPS)[ops_ind]
        ops = np.concatenate(([NASBenchConstants.INPUT], ops, [NASBenchConstants.OUTPUT])).tolist()
        spec = SpecMixed(matrix=matrix, ops=ops)
        return spec
    


class SpecOneHot(ModelSpec):
    """Spec extension to support categorical one-hot for operations and adjacency matrix."""

    flat_matrix = ((NASBenchConstants.NUM_VERTICES-1+1)*(NASBenchConstants.NUM_VERTICES-1)//2) * 2
    input_size = ((NASBenchConstants.NUM_VERTICES-1+1)*(NASBenchConstants.NUM_VERTICES-1)//2)*2 + len(NASBenchConstants.ALLOWED_OPS)*(NASBenchConstants.NUM_VERTICES - 2)
  
    name_lu = {op: np.eye(3)[i] for i, op in enumerate(NASBenchConstants.ALLOWED_OPS)}
    onehot_lu = list(name_lu.values())

    def __init__(self, matrix, ops):
        super().__init__(matrix=matrix, ops=ops)
        # self.flat = self.to_flat() # only when using instead of when initializing

    def to_flat(self):
        ops = self.original_ops[1:-1] # omit first and last one 
        mat = self.original_matrix
        ops_onehot = np.concatenate([SpecOneHot.name_lu[s] for s in ops])
        flat = self.encode(mat, ops_onehot)
        return torch.tensor(flat, dtype=torch.float)

    @staticmethod
    def encode(matrix,ops_onehot):
        # TRIU only uses upper triangle matrix indices including diagonal
        onehot_matrix = np.eye(2)[matrix[NASBenchConstants.TRIU]].flatten()
        flattened =  np.concatenate((onehot_matrix,ops_onehot))
        return flattened
    
    @staticmethod
    def decode(flat):
        matrix = np.zeros((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES), dtype=flat.dtype)
        onehot_matrix = flat[:SpecOneHot.flat_matrix].reshape((-1,2))
        entries = np.argmax(onehot_matrix, 1)
        matrix[NASBenchConstants.TRIU] = entries
        assert np.all(np.triu(matrix) == matrix)

        ops_onehot = flat[SpecOneHot.flat_matrix:].reshape((NASBenchConstants.NUM_VERTICES - 2, len(NASBenchConstants.ALLOWED_OPS)))
        assert np.sum(ops_onehot) == NASBenchConstants.NUM_VERTICES - 2

        indices = np.argmax(ops_onehot, 1)
        return matrix, indices

    @staticmethod
    def test_decode_encode():
        rando = np.random.randint(2, size=NASBenchConstants.NUM_VERTICES*NASBenchConstants.NUM_VERTICES)
        matrix = rando.reshape((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops_ind = np.random.choice(NASBenchConstants.OPS_IND, size=NASBenchConstants.NUM_VERTICES-2)
        ops_onehot = np.concatenate([SpecOneHot.onehot_lu[i] for i in ops_ind])
        flat = SpecOneHot.encode(matrix,ops_onehot)
        m,o = SpecOneHot.decode(flat)
        assert np.array_equal(matrix,m) 
        assert np.array_equal(ops_ind,o) 

    @staticmethod
    def spec_from_flat(flat):
        matrix,indices = SpecOneHot.decode(flat)
        ops = np.asarray(NASBenchConstants.ALLOWED_OPS)[indices]
        cops = np.concatenate(([NASBenchConstants.INPUT], ops, [NASBenchConstants.OUTPUT])).tolist()      
        spec = SpecOneHot(matrix=matrix, ops=cops)
        return spec


    @staticmethod
    def test_time_spec_from_flat(r=1):
        # init a flat sample
        rando = np.random.randint(2, size=NASBenchConstants.NUM_VERTICES*NASBenchConstants.NUM_VERTICES)
        matrix = rando.reshape((NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops_ind = np.random.choice(NASBenchConstants.OPS_IND, size=NASBenchConstants.NUM_VERTICES-2)
        ops_onehot = np.concatenate([SpecOneHot.onehot_lu[i] for i in ops_ind])
        flat = SpecOneHot.encode(matrix,ops_onehot)

        # now time
        sss = time.time()
        for _ in range(r):
            matrix,indices = SpecOneHot.decode(flat)
        print(f"decode { time.time()-sss}") # about 0.0004

        s = time.time()
        for _ in range(r):
            ops = np.asarray(NASBenchConstants.ALLOWED_OPS)[indices]
        print(f"ops { time.time()-s}") # about 0.0004
        s = time.time()
        
        for _ in range(r):
            cops = np.concatenate(([NASBenchConstants.INPUT], ops, [NASBenchConstants.OUTPUT])).tolist()
        print(f"concat { time.time()-s}") # about 0.0004
        s = time.time()
        
        for _ in range(r):
            spec = SpecOneHot(matrix=matrix, ops=cops)
        print(f"new class { time.time()-s}") # about 0.0004
        print(f"total { time.time()-sss}") # about 0.0004
    

_global_spec_class = ModelSpec

def set_spec_class(spec_class):
    """This function sets the spec class for this whole file."""

    global _global_spec_class
    _global_spec_class = spec_class




def guided_mutation(nasb, device, population, mode = 0):
    """Guided mutation using population distribution to sample from."""


    psize = len(population)
    inputs = [x.flat for x in population]
    mutation_inputs = torch.stack(inputs).to(device)
    summed = torch.sum(mutation_inputs, dim=0)

    # mutation inputs of shape (psize, embedded_dim)
    # embedded dimension is flattened matrix of length spec.flat_matrix
    # plus 3 times 5 operations in onehot encodings. 
    
    probs_mat_1 = summed[:SpecMixed.flat_matrix]/psize # get population mass of ones for flat matrix
    probs_mat_0 = 1 - probs_mat_1 # get population mass of zeros for flat matrix

    probs_ops_1 = summed[SpecMixed.flat_matrix:]/psize # get population mass of ones for operations
    probs_ops_0 = 1 - probs_ops_1

    if mode==1:
        probs_ops = probs_ops_1
    elif mode==0:
        probs_ops = probs_ops_0
    else:
        print("mode can either be 0 or 1")


    children = []
    for i in range(psize):
        while True:
            child_spec = mutation_inputs[i,:].long().cpu().detach().numpy()

            mat_index_1 = torch.multinomial(probs_mat_0, 1, replacement=True)[0]
            mat_index_0 = torch.multinomial(probs_mat_1, 1, replacement=True)[0]
            ops_index = torch.multinomial(probs_ops, 1, replacement=True)[0]+SpecMixed.flat_matrix

            base = (ops_index - (ops_index%3)).cpu().numpy()
            remove = [base, base+1, base+2] 
            child_spec[remove] = 0
            child_spec[ops_index] = 1


            child_spec[mat_index_1] = 1
            child_spec[mat_index_0] = 0

            spec = SpecMixed.spec_from_flat(child_spec)
            if nasb.is_valid(spec):
                children.append(spec)
                break
    return children


def guided_mutation_true_onehot(nasb, device, population, mode=1):
    """population of psize individuals determines new mutated children"""
    trials = 100
    psize = len(population)
    # first initializations are neglibile take about one loop duration
    inputs = [x.to_flat() for x in population]
    mutation_inputs = torch.stack(inputs).to(device)

    # mutation inputs of shape (psize, embedded_dim)
    # embedded dimension is flattened matrix of length spec.flat_matrix
    # plus 3 times 5 operations in onehot encodings.
    summed = torch.sum(mutation_inputs, dim=0) # sum up along population dimension
       
    probs_1 = summed/psize 
    probs_0 = 1 - probs_1    
        
    if mode==1:
        probs = probs_1
    elif mode==0:
        probs = probs_0
    else:
        print("mode can either be 0 or 1")

    children = [] # these psize children are about to be repopulated
    for i in range(psize):
        for t in range(trials):

            # sample these indices from their respective probability vector
            ops_index = torch.multinomial(probs[SpecOneHot.flat_matrix:], 1, replacement=True)[0]+SpecOneHot.flat_matrix
            mat_index = torch.multinomial(probs[:SpecOneHot.flat_matrix], 1, replacement=True)[0]

            # get flattened spec from child
            child_spec = mutation_inputs[i,:].long().cpu().detach().numpy()

            if mat_index>=SpecOneHot.flat_matrix or ops_index < SpecOneHot.flat_matrix:
                print("error")


            # basically clear the 3 involved indices to 0 and set new one to 1
            base_ops = (ops_index - (ops_index%3)).cpu().numpy()
            remove_ops = [base_ops, base_ops+1, base_ops+2] 
            child_spec[remove_ops] = 0
            child_spec[ops_index] = 1

            # basically clear the 2 involved indices to 0 and set new one to 1
            base_mat = (mat_index - (mat_index%2)).cpu().numpy()
            remove_mat = [base_mat, base_mat+1] 
            child_spec[remove_mat] = 0
            child_spec[mat_index] = 1                    

            # create new spec
            spec = SpecOneHot.spec_from_flat(child_spec)  # 0.00012 seconds

            if nasb.is_valid(spec):
                children.append(spec)
                break
        if t == trials-1:
            print(f" {trials} trials were not sufficient to determine a mutated child for member {i}")
    return children


