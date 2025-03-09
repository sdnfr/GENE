# Copyright 2025 Anonymized Authors

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""
This script downloads the NAS-Bench-101 dataset

Usage: 

python ./experiments/utils/download_nasbench.py
 
"""

import os
import urllib.request

DATASET_PATH = os.path.join("generated", "nasbench_only108.tfrecord")
FILE_NAME = "nasbench_only108"
URL = "https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord"

if __name__ == "__main__":

    # Check if the 'generated' directory exists, if not, create it
    if not os.path.exists("generated"):
        os.mkdir("generated")
        print("Directory 'generated' created.")
    else:
        print("Directory 'generated' already exists.")

    # Check if the file exists before downloading
    if not os.path.exists(DATASET_PATH):
        urllib.request.urlretrieve(URL, DATASET_PATH)
        print(f"File '{DATASET_PATH}' downloaded.")
    else:
        print(f"File '{DATASET_PATH}' already exists.")