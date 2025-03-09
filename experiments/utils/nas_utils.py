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
import time 
from typing import Callable, Any, List
import subprocess

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
 
def random_spec(nasbench: NASBench):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(NASBenchConstants.ALLOWED_EDGES,
                                  size=(NASBenchConstants.NUM_VERTICES, NASBenchConstants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(NASBenchConstants.ALLOWED_OPS, size=NASBenchConstants.NUM_VERTICES).tolist()
        ops[0] = NASBenchConstants.INPUT
        ops[-1] = NASBenchConstants.OUTPUT
        spec = SpecOneHot(matrix=matrix, ops=ops)
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

        new_spec = SpecOneHot(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return new_spec

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

def crossover(nasb, parents):
    """Performs vertical crossover on parent pairs, ensuring valid offspring."""
    
    def cross(p1, p2):
        if p1.original_matrix.shape != p2.original_matrix.shape:
            print("Shape mismatch")
            return p1

        for _ in range(5):
            i = random.randint(1, NASBenchConstants.NUM_VERTICES - 1)
            m = np.copy(p1.original_matrix)
            m[:i, :] = p2.original_matrix[:i, :]
            o = p1.original_ops[:i] + p2.original_ops[i:]
            new_spec = SpecOneHot(m, o)
            if nasb.is_valid(new_spec):
                return new_spec
        return p1

    return [cross(p[0][1], p[1][1]) for p in parents]

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
    selection_algorithm: Callable[[List], Any],
    crossover_algorithm: Callable[[NASBench, List], Any],
    mutation_algorithm: Callable[[NASBench, List], Any],
    max_time_budget=5e6,
    p_size=50,    
):
    """Run a single roll-out of evolution to a fixed time budget."""
    population, times, best_valids, best_tests = init_pop(nasbench, max_time_budget, p_size)
    while True:
        # generational approach: children replace old population
        selected_candidates = selection_algorithm(population)  
        crossed_parents = crossover_algorithm(nasbench, selected_candidates)
        children = mutation_algorithm(nasbench, crossed_parents)
        population = []
        for child in children: 
            data = nasbench.query(child)
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)
            population.append((data["validation_accuracy"], child))
            if data["validation_accuracy"] > best_valids[-1]:
                best_valids.append(data["validation_accuracy"])
                best_tests.append(data["test_accuracy"])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])

        if time_spent> max_time_budget:
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
        return flat.astype(int)

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

def run_algorithm(algname, dataset, budget, loops): 
    algorithms = {

        # custom scripts like this
        "GENE": os.path.join("4_GENE.py"),

        # baselines provided from autodl lib
        "regularized evolution": os.path.join("..", "thirdparty", "autodl", "exps", "NATS-algos", "regularized_ea.py"),
        "random": os.path.join("..", "thirdparty", "autodl", "exps", "NATS-algos", "random_wo_share.py"),
        "reinforce": os.path.join("..", "thirdparty", "autodl", "exps", "NATS-algos", "reinforce.py"),
        "bohb": os.path.join("..", "thirdparty", "autodl", "exps", "NATS-algos", "bohb.py"),
        
    }
    # Use os.path.join to construct the save_dir
    save_dir = os.path.join("..", "output", "search")

    print(f"Running algorithm {algname} on {dataset}...")
    command = [
        "python", 
        algorithms[algname],
        "--save_dir", save_dir, 
        "--dataset", dataset,
        "--search_space", "tss",
        "--time_budget", str(budget),
        "--loops_if_rand", str(loops),

    ]

    if algname== "regularized evolution" or algname== "regularized evolution gm": 
        command += ["--ea_cycles", "200","--ea_population", "20","--ea_sample_size", "10"]

    if algname=="bohb":
        command += ["--num_samples", "4", "--random_fraction", "0.0", "--bandwidth_factor","3"]

    if algname=="reinforce":
        command += ["--learning_rate", "0.01"]


    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # if no outputs are saved, check this command
    # print(result)