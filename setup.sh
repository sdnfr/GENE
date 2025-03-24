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

# Install ipykernel to make the virtual environment available as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"


# make user generated folders
mkdir -p "output"
mkdir -p "generated"
mkdir -p "thirdparty"

# download nasbench script
python ./experiments/utils/download_nasbench.py

# init submodule
# Check if the repository is inside a Git submodule (when not cloned but installed from zip)
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

git submodule update --init --recursive

if [ ! -d "./thirdparty/autodl/.git" ]; then
    echo "Cloning missing AutoDL submodule..."
    git clone https://github.com/D-X-Y/AutoDL-Projects.git thirdparty/autodl
fi

cd ./thirdparty/autodl
pip install .
cd ../..

# Ask user if they want to set it permanently
read -p " Do you want me to set TORCH_HOME permanently in your ~/.bashrc? (required for NATS-Bench) (y/n): " answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    # Check if it already exists in the file
    if grep -q "TORCH_HOME" ~/.bashrc; then
        echo " It looks like TORCH_HOME is already set in ~/.bashrc."
        echo "You might want to review it manually."
        echo "Please still download NATS-Bench via this command:"
        echo "python ./experiments/utils/download_natsbench.py"
    else
        echo "Running: export TORCH_HOME=\"$(pwd)/generated\" >> ~/.bashrc"
        echo "export TORCH_HOME=\"$(pwd)/generated\"" >> ~/.bashrc
        source ~/.bashrc
        source ./.venv/bin/activate
        python ./experiments/utils/download_natsbench.py
    fi
else
    echo "Please add TORCH_HOME permanently to your path"
    echo "Additionally, download NATS-Bench via this command:"
    echo "python ./experiments/utils/download_natsbench.py"
fi
