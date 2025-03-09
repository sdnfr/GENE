# Copyright 2025 Anonymized Authors

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""
This script contains some plotting helpers for the NASBench 101 dataset. 

Usage: 

# for jupyter notebooks
import sys; sys.path.append('..')
from utils.plotting import plot_all

config = {
   "budget" : int(1e6),
   "limits" : (0.938, 0.943),
   "n" : 1000,
   "confidence_intervall" : False,
   "pvalue" : 0.05,
   "significant_areas": False,
   "dataset" : "test",
}

# make sure colors are named for tableu-colorblind
m=[
    [gevolution_data_1e7,"greedy selection","Blue"],
    [cevolution_data_1e7,"vanilla evolution","Dark Orange"],
    [revolution_data_1e7,"regularized evolution","Dark Gray"],
    [random_data_1e7,"random","Light Orange"],
]


#############################################################
# single plot: 

fig, ax = plt.subplots()
plot_all([m[0], m[1], m[2], m[3]], config, ax)
plt.show()

#############################################################
# multiple plots: 

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

plot_all([m[0], m[1]], config, ax=axs[0])
plot_all([m[0], m[2]], config, ax=axs[1])
plot_all([m[1], m[2]], config, ax=axs[2])
plt.tight_layout()
plt.show()
#############################################################

"""
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_run(data, color, label, config, ax=None, gran=10000):
  """Computes the mean and std"""

  # which = 2 is test, which = 1 is valid
  which = 2
  if config["dataset"] == "test":
    which = 2
  elif config["dataset"] == "validation": 
    which = 1
  xs = range(0, config["budget"]+1, gran)
  mean = [0.0]
  std = [0.0]
  per25 = [0.0]
  per75 = [0.0]
  ci_lower = [0.0]
  ci_upper = [0.0]
  repeats = len(data)
  pointers = [1 for _ in range(repeats)]
  
  cur = gran
  while cur < config["budget"]+1:
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


    all_vals = sorted(all_vals)
    cur_mean = sum(all_vals) / float(len(all_vals))
    cur_std = np.std(all_vals)
    std.append(cur_std)
    
    mean.append(sum(all_vals) / float(len(all_vals)))
    per25.append(all_vals[int(0.25 * repeats)])
    per75.append(all_vals[int(0.75 * repeats)])

    if config["confidence_intervall"] is True:

      # Calculate the confidence interval
      sem = cur_std / np.sqrt(repeats)  # Standard Error
      confidence = 1 - config["pvalue"]
      t_critical = t.ppf(confidence + (1 - confidence) / 2, repeats - 1)
      margin_of_error = t_critical * sem
        
      ci_lower.append(cur_mean - margin_of_error)
      ci_upper.append(cur_mean + margin_of_error)
    
    cur += gran

  if config["confidence_intervall"] is True:
    ax.fill_between(xs, ci_lower, ci_upper, alpha=0.5, linewidth=0, facecolor=color)
  ax.plot(xs, mean, color=color, label=label, linewidth=2)
  # plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)

  return mean, std


def plot_significance(means, stds, n, red, green, ax, pvalue_threshold):
    comparisons = len(means)

    all_pvalues = []
    for a in range(comparisons):
        for b in range(comparisons):

            if a <=b :
                continue
            pvalues = []
            for mean1, std1, mean2, std2 in zip(means[a], stds[a], means[b], stds[b]):
                
                se1 = std1 / np.sqrt(n)
                se2 = std2 / np.sqrt(n)

                if se1==0 and se2==0:
                    pvalues.append(0.0)
                    continue
                t_stat = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)
                
                df = ((se1**2 + se2**2)**2) / (((se1**2)**2 / (n-1)) + ((se2**2)**2 / (n-1)))
                
                p_value = 2 * t.sf(np.abs(t_stat), df)
                pvalues.append(p_value)
            # plt.plot(range(len(pvalues)), pvalues, label=f"comparing {b} and {a}", linewidth=2)
            all_pvalues.append(pvalues)
    # plt.show()
    for idx in range(len(all_pvalues[0])):
        color = green
        for k in range(len(all_pvalues)):
            p = all_pvalues[k][idx]
            if p > pvalue_threshold: # at least one red
                color = red

        if color == green:
            continue
        ax.vlines(x=idx*10000, ymin=0.9, ymax=0.943, alpha=0.1,color=color, linewidth=2)

    # this line is just for plotting the legend label once
    ax.vlines(x=idx*10000, ymin=1, ymax=1.1,color=red, linewidth=2, label=f"pvalue > {pvalue_threshold}")



def plot_all(exp, ax=None):
    # tableau_colorblind10
    tc = {
        'Dark Blue': '#006BA4',
        'Blue': '#5F9ED1',
        'Light Blue': '#A2C8EC',
        'Dark Orange': '#FF800E',
        'Red Orange': '#C85200',
        'Light Orange': '#FFBC79',
        'Very Dark Gray': '#595959',
        'Dark Gray': '#898989',
        'Light Gray': '#ABABAB',
        'Very Light Gray': '#CFCFCF',
        "Green" : '#228B22',
        "Red" : '#FF4500' 
    }

    # clear_output(wait=True)  # Clear the previous output

    means, stds = [],[]
    for key,value in exp["data"].items():
        label = key
        data = value[0]
        color = value[1]

        mean, std = plot_run(data, tc[color], label, exp["config"], ax)
        means.append(mean)
        stds.append(std)


    if exp["config"]["significant_areas"]:
        plot_significance(means, stds, exp["config"]["n"], tc['Light Gray'], tc['Green'], ax, exp["config"]["pvalue"])


    ax.legend(loc='lower right')
    ax.set_ylim(exp["config"]["limits"][0], exp["config"]["limits"][1])
    ax.set_xlabel('total training time spent (seconds)')
    ax.set_ylabel('accuracy')
    return means, stds

def color_scheme():
    # tableau_colorblind10
    tc = {
        'Dark Blue': '#006BA4',
        'Blue': '#5F9ED1',
        'Light Blue': '#A2C8EC',
        'Dark Orange': '#FF800E',
        'Red Orange': '#C85200',
        'Light Orange': '#FFBC79',
        'Very Dark Gray': '#595959',
        'Dark Gray': '#898989',
        'Light Gray': '#ABABAB',
        'Very Light Gray': '#CFCFCF'
    }

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot a line for each color
    x_pos = 1

    # Plot a vertical line for each color shifted by 1 unit to the right
    for color_name, color in tc.items():
        ax.plot([x_pos, x_pos], [0, 1], color=color, label=color_name)
        x_pos += 1  # Move to the right for the next line

    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend(loc='lower right')
    ax.set_xlim(0,16)

    # Show plot
    plt.show()

def fetch_data(root_dir=None, search_space="tss", dataset=None, algorithms=["REA","REINFORCE", "RANDOM", "GM", "GE", "BOHB"]):
    if root_dir is None:
        root_dir = os.path.join("..", "output", "search")
    ss_dir = "{:}-{:}".format(root_dir, search_space)
    alg2name, alg2path = OrderedDict(), OrderedDict()
    for alg in algorithms:
        alg2name[alg] = alg
    for alg, name in alg2name.items():
        alg2path[alg] = os.path.join(ss_dir, dataset, name, "results.pth")
        assert os.path.isfile(alg2path[alg]), "invalid path : {:}".format(alg2path[alg])
    alg2data = OrderedDict()
    for alg, path in alg2path.items():
        data = torch.load(path)
        for index, info in data.items():
            info["time_w_arch"] = [
                (x, y) for x, y in zip(info["all_total_times"], info["all_archs"])
            ]
            for j, arch in enumerate(info["all_archs"]):
                assert arch != -1, "invalid arch from {:} {:} {:} ({:}, {:})".format(
                    alg, search_space, dataset, index, j
                )
        alg2data[alg] = data
    return alg2data


def query_performance(api, data, dataset, ticket):
    results, is_size_space = [], api.search_space_name == "size"
    for i, info in data.items():
        time_w_arch = sorted(info["time_w_arch"], key=lambda x: abs(x[0] - ticket))
        time_a, arch_a = time_w_arch[0]
        time_b, arch_b = time_w_arch[1]
        info_a = api.get_more_info(
            arch_a, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
        )
        info_b = api.get_more_info(
            arch_b, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
        )
        accuracy_a, accuracy_b = info_a["test-accuracy"], info_b["test-accuracy"]
        interplate = (time_b - ticket) / (time_b - time_a) * accuracy_a + (
            ticket - time_a
        ) / (time_b - time_a) * accuracy_b
        results.append(interplate)
    return np.mean(results), np.std(results)


def show_valid_test(api, data, dataset):
    valid_accs, test_accs, is_size_space = [], [], api.search_space_name == "size"
    for i, info in data.items():
        time, arch = info["time_w_arch"][-1]
        if dataset == "cifar10":
            xinfo = api.get_more_info(
                arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
            )
            test_accs.append(xinfo["test-accuracy"])
            xinfo = api.get_more_info(
                arch,
                dataset="cifar10-valid",
                hp=90 if is_size_space else 200,
                is_random=False,
            )
            valid_accs.append(xinfo["valid-accuracy"])
        else:
            xinfo = api.get_more_info(
                arch, dataset=dataset, hp=90 if is_size_space else 200, is_random=False
            )
            valid_accs.append(xinfo["valid-accuracy"])
            test_accs.append(xinfo["test-accuracy"])
    valid_str = "{:.2f}$\pm${:.2f}".format(np.mean(valid_accs), np.std(valid_accs))
    test_str = "{:.2f}$\pm${:.2f}".format(np.mean(test_accs), np.std(test_accs))
    return valid_str, test_str

def visualize_curve(api, exp, dataset, ax, search_space="tss"):
    def sub_plot_fn(ax, dataset, exp):
        xdataset, max_time = dataset.split("-T")
        algorithms_labels = [d[0] for d in exp["data"].items()]
        algorithms = [d[1][0] for d in exp["data"].items()]
        alg2data = fetch_data(search_space=search_space, dataset=dataset, algorithms=algorithms)
        total_tickets = 150
        time_tickets = [
            float(i) / total_tickets * int(max_time) for i in range(total_tickets)
        ]
        tc = {
            'Dark Blue': '#006BA4',
            'Blue': '#5F9ED1',
            'Light Blue': '#A2C8EC',
            'Dark Orange': '#FF800E',
            'Red Orange': '#C85200',
            'Light Orange': '#FFBC79',
            'Very Dark Gray': '#595959',
            'Dark Gray': '#898989',
            'Light Gray': '#ABABAB',
            'Very Light Gray': '#CFCFCF',
            "Green" : '#228B22',
            "Red" : '#FF4500' 
        }
        colors = [tc[d[1][1]] for d in exp["data"].items()]

        ax.set_xlim(0,int(max_time))
        
        ax.set_ylim(
            exp["config"]["limits"][xdataset][0], exp["config"]["limits"][xdataset][1]
        )

        xs = [x for x in time_tickets]

        for idx, (alg, data) in enumerate(alg2data.items()):
            accuracies = []
            ci_lower = []
            ci_upper = []
            repeats = len(data)
            for ticket in time_tickets:
                cur_mean, cur_std = query_performance(api, data, xdataset, ticket)
                accuracies.append(cur_mean)
                if exp["config"]["confidence_intervall"] is True:

                    # Calculate the confidence interval
                    sem = cur_std / np.sqrt(repeats)  # Standard Error
                    confidence = 1 - exp["config"]["pvalue"]
                    t_critical = t.ppf(confidence + (1 - confidence) / 2, repeats - 1)
                    margin_of_error = t_critical * sem
                        
                    ci_lower.append(cur_mean - margin_of_error)
                    ci_upper.append(cur_mean + margin_of_error)
                    
            ax.plot(
                xs,
                accuracies,
                c=colors[idx],
                label="{:}".format(algorithms_labels[idx]),
            )
            if exp["config"]["confidence_intervall"] is True:
                ax.fill_between(xs, ci_lower, ci_upper, alpha=0.5, linewidth=0, facecolor=colors[idx])

            ax.set_ylabel("accuracy")
            ax.set_xlabel('total training time spent (seconds)')

            name2label = {
                "cifar10": "CIFAR-10",
                "cifar100": "CIFAR-100",
                "ImageNet16-120": "ImageNet-16-120",
            }
            ax.set_title("NATS-Bench results on {:}".format(name2label[xdataset]))
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((1,4))
            ax.xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))


        ax.legend(loc=4)
    sub_plot_fn(ax, dataset, exp)
    print("sub-plot {:} on {:} done.".format(dataset, search_space))