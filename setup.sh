#!/bin/bash

python -m venv ./.venv
source ./.venv/bin/activate

# install only top requirements
pip install tensorflow==2.15.0
pip install matplotlib
pip install scipy
pip install -i https://test.pypi.org/simple/ nasbench-TF2
pip install ml_dtypes==0.4.0
pip install seaborn
pip install nats_bench
pip install torchvision
pip install gdown


# make user generated folders
mkdir -p "output"
mkdir -p "generated"

# download nasbench script
python ./experiments/utils/download_nasbench.py
# download natsbench script
python ./experiments/utils/download_natsbench.py

# init submodule
git submodule update --init --recursive
cd ./thirdparty/autodl
pip install .
cd ../..