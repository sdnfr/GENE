##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
# Copyright 2025 Anonymized Authors

##################################################################
# Guiding Exploration and Exploitation for Neural Architecture Search #
##################################################################

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""
This script provides the GENE implementation for NATS-Bench.

Requirements: 

- This notebook requires that torch, tensorflow and numpy be installed within the 
Python environment you are running this script in. 

- This notebook requires the submodule autodl. See setup in README
 
"""
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, get_search_spaces
from nats_bench import create, search_space_info


class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)


def random_topology_func(op_names, max_nodes=4):
    # Return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return random_architecture


def random_size_func(info):
    # Return a random architecture
    def random_architecture():
        channels = []
        for i in range(info["numbers"]):
            channels.append(str(random.choice(info["candidates"])))
        return ":".join(channels)

    return random_architecture


def mutate_topology_func(op_names):
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """

    def mutate_topology_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch

    return mutate_topology_func


def mutate_size_func(info):
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """

    def mutate_size_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        child_arch = child_arch.split(":")
        index = random.randint(0, len(child_arch) - 1)
        child_arch[index] = str(random.choice(info["candidates"]))
        return ":".join(child_arch)

    return mutate_size_func

def selection_g(parents):
    """Selects greedy n parent pairs"""
    p_size = len(parents)
    row_indices, col_indices = np.meshgrid(np.arange(p_size),np.arange(p_size))
    mapping = np.reshape(np.stack([col_indices,row_indices],axis=-1), [-1,2])

    summed_population = []
    for p1 in range(p_size):
        for p2 in range(p_size):
            if p1 < p2: # only upper triangle
                sum = parents[p1].accuracy + parents[p2].accuracy
            else:
                sum = 0
            summed_population.append(sum)

    samples = np.argsort(summed_population)[-p_size:]
    pairings = mapping[samples]

    paired_parents = []
    for i in range(len(pairings)):
        p1 = parents[pairings[i,0]]
        p2 = parents[pairings[i,1]]
        paired_parents.append((p1,p2))

    return paired_parents

def crossover(pairings):
    def crossparents(p1,p2):
        child = deepcopy(p1)
        for i in range(2, len(child.arch.nodes)):
            child.arch.nodes[i] = p2.arch.nodes[i]
        child.accuracy = None
        if child.arch.check_valid():
            return child
        return p1
    crossed_parents = []
    for p in pairings:
        crossed = crossparents(p[0],p[1])
        crossed_parents.append(crossed)
    return crossed_parents

def to_onehot(arch, info):
    onehot = torch.zeros(info["edges"],info["len_ops"])
    total_counter = 0
    for i, node in enumerate(arch.nodes): # 0,1,2
        for k in range(i+1): #0  01  012
            o = info["operation_mapping"][node[k][0]]
            onehot[total_counter] = o     
            total_counter += 1
    return onehot.view(-1)

def arch_from_onehot(onehot, info):
    ops_onehot = onehot.reshape(info["edges"],info["len_ops"])
    indices = np.argmax(ops_onehot, 1)
    ops = np.asarray(info["op_names"])[indices]

    genotypes = []
    total_index = 0
    for i in range(1, info["num_nodes"]):
        xlist = []
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            xlist.append((ops[total_index], j))
            total_index += 1
        genotypes.append(tuple(xlist))
    return CellStructure(genotypes)


def mutation(population, info):
    psize = len(population)
    trials = 100
    inputs = [to_onehot(x.arch, info) for x in population]
    mutation_inputs = torch.stack(inputs)

    # mutation inputs of shape (psize, embedded_dim)
    # embedded dimension is flattened matrix of length FLAT_MATRIX
    # plus 3 times 5 operations in onehot encodings. 
    summed = torch.sum(mutation_inputs, dim=0) # sum up along population dimension
    probs_1 = summed/psize # get population mass of ones for flat matrix
    probs_0 = 1 - probs_1 # get population mass of zeros for flat matrix


    children = [] # these psize children are about to be repopulated
    for i in range(psize):
        for t in range(trials):

            # sample these indices from their respective probability vector
            ops_index = torch.multinomial(probs_0, 1, replacement=True).numpy()[0]

            # get flattened spec from child
            child_spec = mutation_inputs[i,:]

            # basically clear the 3 involved indices to 0 and set new one to 1
            base = (ops_index - (ops_index%info["len_ops"]))
            remove = []
            for k in range(info["len_ops"]):
                remove.append(base+k)
            child_spec[remove] = 0
            child_spec[ops_index] = 1

            # create new spec
            arch = arch_from_onehot(child_spec, info) 

            if arch.check_valid():
                children.append(arch)
                break
        if t == trials-1:
            print(f" {trials} trials were not sufficient to determine a mutated child for member {i}")
    return children

def generic_mutation(mutate_arch, population):
    mutated_population = []
    for p in population:
        mutated_arch = mutate_arch(p.arch)
        mutated_population.append(mutated_arch)
    return mutated_population



def guided_mutation(
    cycles,
    population_size,
    sample_size,
    time_budget,
    random_arch,
    mutate_arch,
    api,
    use_proxy,
    dataset,
    info,
):
    """Algorithm for guided mutation evolution.

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each tournament.
      time_budget: the upper bound of searching cost

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    api.reset_time()
    history, total_time_cost = (
        [],
        [],
    )  # Not used by the algorithm, only used to report results.
    current_best_index = []
    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_arch()
        model.accuracy, _, _, total_cost = api.simulate_train_eval(
            model.arch, dataset, hp="12" if use_proxy else api.full_train_epochs
        )
        # Append the info
        population.append(model)
        history.append((model.accuracy, model.arch))
        total_time_cost.append(total_cost)
        current_best_index.append(
            api.query_index_by_arch(max(history, key=lambda x: x[0])[1])
        )

    # generational approach: children replace old population
    while True:
        # greedy selection
        selected_candidates = selection_g(population)
        crossed_parents = crossover(selected_candidates)
        children = mutation(crossed_parents, info)
        # children = generic_mutation(mutate_arch, crossed_parents)
        # Create the child model and store it.

        population = collections.deque()
        for child in children:
            child_model = Model()
            child_model.arch = child
            child_model.accuracy, _, _, total_cost = api.simulate_train_eval(
                child_model.arch, dataset, hp="12" if use_proxy else api.full_train_epochs
            )
            # Append the info
            population.append(child_model)
            history.append((child_model.accuracy, child_model.arch))
            current_best_index.append(
                api.query_index_by_arch(max(history, key=lambda x: x[0])[1])
            )
            total_time_cost.append(total_cost)

        if total_time_cost[-1] > time_budget:
            break

    return history, current_best_index, total_time_cost


def main(xargs, api):
    torch.set_num_threads(4)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    if xargs.search_space == "tss":
        random_arch = random_topology_func(search_space)
        mutate_arch = mutate_topology_func(search_space)
    else:
        random_arch = random_size_func(search_space)
        mutate_arch = mutate_size_func(search_space)

    x_start_time = time.time()
    logger.log("{:} use api : {:}".format(time_string(), api))
    logger.log(
        "-" * 30
        + " start searching with the time budget of {:} s".format(xargs.time_budget)
    )

    info = search_space_info("nats-bench", xargs.search_space)
    info["edges"] = info["num_nodes"]*(info["num_nodes"]-1)//2
    info["len_ops"] = len(info["op_names"])
    operation_mapping ={}
    for i,operation in enumerate(info["op_names"]):
        operation_mapping[operation] = torch.eye(len(info["op_names"]))[i]
    info["operation_mapping"] = operation_mapping


    history, current_best_index, total_times = guided_mutation(
        200,
        50,
        3,
        xargs.time_budget,
        random_arch,
        mutate_arch,
        api,
        xargs.use_proxy > 0,
        xargs.dataset,
        info,
    )
    logger.log(
        "{:} guided_mutation finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).".format(
            time_string(), len(history), total_times[-1], time.time() - x_start_time
        )
    )
    best_arch = max(history, key=lambda x: x[0])[1]
    logger.log("{:} best arch is {:}".format(time_string(), best_arch))

    info = api.query_info_str_by_arch(
        best_arch, "200" if xargs.search_space == "tss" else "90"
    )



    logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, current_best_index, total_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Guiding Exploration and Exploitation")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        choices=["tss", "sss"],
        help="Choose the search space.",
    )
    # hyperparameters for GENE optional
    # parser.add_argument("--ea_cycles", type=int, help="The number of cycles in EA.")
    # parser.add_argument("--ea_population", type=int, help="The population size in EA.")
    # parser.add_argument("--ea_sample_size", type=int, help="The sample size in EA.")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--use_proxy",
        type=int,
        default=1,
        help="Whether to use the proxy (H0) task or not.",
    )
    #
    parser.add_argument(
        "--loops_if_rand", type=int, default=500, help="The total runs for evaluation."
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()

    api = create(None, args.search_space, fast_mode=True, verbose=False)

    args.save_dir = os.path.join(
        "{:}-{:}".format(args.save_dir, args.search_space),
        "{:}-T{:}{:}".format(
            args.dataset, args.time_budget, "" if args.use_proxy > 0 else "-FULL"
        ),
        "GENE",
    )
    print("save-dir : {:}".format(args.save_dir))
    print("xargs : {:}".format(args))

    if args.rand_seed < 0:
        save_dir, all_info = None, collections.OrderedDict()
        for i in range(args.loops_if_rand):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, args.loops_if_rand))
            args.rand_seed = random.randint(1, 100000)
            save_dir, all_archs, all_total_times = main(args, api)
            all_info[i] = {"all_archs": all_archs, "all_total_times": all_total_times}
        save_path = save_dir / "results.pth"
        print("save into {:}".format(save_path))
        torch.save(all_info, save_path)
    else:
        main(args, api)

# api = create(None, 'tss', fast_mode=True, verbose=False)
# info = search_space_info("nats-bench", "tss")

# info["edges"] = info["num_nodes"]*(info["num_nodes"]-1)//2
# info["len_ops"] = len(info["op_names"])
# operation_mapping ={}
# for i,operation in enumerate(info["op_names"]):
#     operation_mapping[operation] = torch.eye(len(info["op_names"]))[i]
# info["operation_mapping"] = operation_mapping

# print(info["op_names"])
# search_space = get_search_spaces("tss", "nats-bench")

# random_arch = random_topology_func(search_space)
# mutate_arch = mutate_topology_func(search_space)
# guided_mutation(200, 20, 3, 2000, random_arch, mutate_arch,api,False,"cifar10", info)