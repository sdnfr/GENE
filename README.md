# GENE
Code implementation for "GENE: Guiding Exploration and Exploitation for Neural Architecture Search"

---

## Contents

- [Usage](#usage)
- [Repository Structure](#structure)
- [Setup](#automatic-setup)
- [License](#license)
- [Contact](#contact)
---

## Usage
The experiments are designed in an incrementally walk-through manner. For a thorough understanding, follow the labelled order and go through the experiments step by step. Use the provided setup script for an easy installation of all benchmarks. 

This repository contains the experiments as described in section 4 of the paper "GENE: Guiding Exploration and Exploitation in Neural Architecture Search". The experiments correspond to the respective sections:
1 → 4.1, 2 → 4.2, 3 → 4.3, and so on. Start with an overview at experiments/1_datasets.ipynb. 

## Structure

The structure of this repository is organized as follows:

```
├── experiments
|  └── 1_datasets.ipynb
|  └── 2_guiding_on_best_models.ipynb
|  └── 3_benchmarking_GENE.ipynb
|  └── 4_generalization_NATS.ipynb
|  └── 4_GENE.py
|  └── 5_1_ablation_greedy_selection.ipynb
|  └── 5_2_ablation_guided_mutation.ipynb
|  └── 5_3_ablation_mixed_encoding.ipynb
|  └── 5_4_ablation_operation_bias.ipynb
|  └── utils
|
├── generated
├── outputs
| 
├── thirdparty
|
├── setup.sh
|
├── requirements.txt
├── LICENSE
└── README.md
```

Here is a quick overview of the respective experiments: 
- 1_datasets: a quick introduction into the util functions to familiarize with the code.
- 2_guiding_on_best_models: using average of top models from NAS-Bench-101 to guide mutation.
- 3_benchmarking_GENE: benchmarking the proposed GENE algorithm on NAS-Bench-101 against regularized evolution.
- 4_generalization_NATS: benchmarking the proposed GENE algorithm on NATS-Bench against regularized evolution and reinforce.
- 4_GENE: algorithm implementation for NATS-Bench.
- 5_1_ablation_greedy_selection: this ablates the greedy selection in combination with different mutation methods. 
- 5_2_ablation_guided_mutation: this ablates the guided mutation in combination with different selection methods. 
- 5_3_ablation_mixed_encoding: this ablates the different combinations of sampling from probs vector for NAS-Bench-101.
- 5_4_ablation_operation_bias: this showcases the operation bias of NAS-Bench-101.
- utils: This folder contains code that is reused in multiple notebooks. We introduce most functions in 1_datasets, which are then reused from the utils folder to reduce boilerplate code in the notebooks. 

Other folders include custom generated files and models, as well as downloaded datasets:
- generated: The default location for downloaded datasets.
- output: The default location of NATS-run outputs.


## Automatic Setup

The provided plug-and-play installation script will automatically install all Python dependencies and both NAS-Bench-101 and NATS-Bench. You will only need this small setup:

Make sure to have python 3.11 installed for ideal performance.

```bash
git clone <repository_url> GENE
cd GENE
source setup.sh
```

If the repository can not be cloned, just unzip the archive instead of the first line.


## Manual Setup

If the installation script does not work or you prefer to manually install it yourself, try to manually install via requirements.txt. We recommend the automatic setup. 

```bash
python -m venv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

We used tensorflow 2.15 with cuda11 and torch 2.3.
There is a dependency conflict using jax and tensorflow together. 
jax requires ml_dtypes==0.4.0 and tensorflow requires ml_dtypes==0.2.0. 
An installation of first tensorflow and then jax shows warnings but if errors
persists
In case there is a jax version mismatch, try 
pip install --upgrade jax jaxlib
pip install jax==0.5.1 jaxlib==0.5.1


To generate your own data using NAS-Bench-101 and NATS-Bench, you need to download and install the required tabular benchmarks. (The automatic setup will install this for you)

#### NAS-Bench-101 Setup

1. Download the `.tfrecord` file from Google Drive to the `./generated` folder. Follow the instructions [here](https://github.com/google-research/nasbench?tab=readme-ov-file#download-the-dataset) to download either the full version or the smaller version of the dataset.
Alternatively, you can use the provided script to download: 

```bash
python ./experiments/utils/download_nasbench.py
```

2. After downloading, the directory structure should resemble the following:

```
├── generated/
|  └── nasbench_only108.tfrecord
```

3. Since there is no maintained NAS-Bench-101 repository that supports TensorFlow 2, the `nasbench-TF2` package is used. It needs to be manually installed via:

```bash
pip install -i https://test.pypi.org/simple/ nasbench-TF2
```

Note: TensorFlow 2.15.0 is required to support NAS-Bench-101.

#### NATS-Bench Setup

1. Install the `nats_bench` package for evaluation:

```bash
pip install nats_bench
```
2. Set environment variable `$TORCH_HOME` for autodl library to look for benchmark.

3. Download the dataset into the `$TORCH_HOME` directory. Instructions can be found [here](https://github.com/D-X-Y/AutoDL-Projects?tab=readme-ov-file#requirements-and-preparation), feel free to use the provided download script via 
```bash
python ./experiments/utils/download_natsbench.py
```

4. Additionally, to run algorithms such as REINFORCE or regularized evolution, you need the AutoDL repository. Install it as follows:

```bash
git submodule update --init --recursive
cd ./thirdparty/autodl
pip install .
```

if submodule does not already exist:
```bash
mkdir thirdparty
cd thirdparty
git clone https://github.com/D-X-Y/AutoDL-Projects.git autodl
cd ./thirdparty/autodl
pip install .
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE). 

## Contact

For any questions, suggestions, or discussions related to this project, please use the issues tab on the GitHub repository. This helps to keep track of all the queries and allows others to benefit from the discussion.
