@echo off
REM Create a virtual environment
@REM python -m venv .venv

REM Activate the virtual environment (Windows)
call .venv\Scripts\activate

REM Install dependencies
pip install tensorflow==2.15.0
pip install matplotlib
pip install scipy
pip install -i https://test.pypi.org/simple/ nasbench-TF2
pip install ml_dtypes==0.4.0
pip install seaborn
pip install nats_bench
pip install torchvision
pip install gdown

REM Create user-generated folders
mkdir "output"
mkdir "generated"

REM Download NAS-Bench script
python ./experiments/utils/download_nasbench.py

REM Download NATS-Bench script
python ./experiments/utils/download_natsbench.py

REM Initialize Git submodules
git submodule update --init --recursive

REM Install third-party autodl
cd thirdparty\autodl
pip install .
cd ..\..
