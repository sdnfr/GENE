# Copyright 2025 Anonymized Authors

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""
This script downloads the NATS-Bench topology dataset

Usage: 

python ./experiments/utils/download_natsbench.py
 
"""
# Copyright 2025 Anonymized Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""
This script downloads the NATS-Bench topology dataset

Usage: 

python ./experiments/utils/download_natsbench.py
"""

import os
import sys
import gdown  # Import gdown
import tarfile  # Import tarfile for extraction

# Updated FILE_ID based on the new link
FILE_ID = "17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def main():
    # Step 1: Check TORCH_HOME environment variable
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        print("Error: TORCH_HOME environment variable is not set.")
        print("Please set TORCH_HOME to your desired directory.")
        sys.exit(1)
    else:
        print(f"TORCH_HOME found: {torch_home}")

    # Step 2: Create the full path to target directory
    target_dir = torch_home  # Target directory is TORCH_HOME itself

    # Step 4: Download the archive
    archive_filename = "NATS-tss-v1_0-3ffb9-simple.tar"  # Correct filename
    download_path = os.path.join(target_dir, archive_filename)

    if not os.path.exists(download_path):
        print(f"Downloading file to {download_path}...")
        # Use gdown to download the file directly from Google Drive
        gdown.download(DOWNLOAD_URL, download_path, quiet=False)
        print(f"Download complete: {download_path}")
    else:
        print(f"File already exists: {download_path}")

    # Step 5: Extract the TAR file
    extracted_dir_name = "NATS-tss-v1_0-3ffb9-simple"  # Directory name inside the tar
    extracted_path = os.path.join(target_dir, extracted_dir_name)

    if not os.path.exists(extracted_path):
        print(f"Extracting TAR file to {target_dir}...")
        with tarfile.open(download_path, "r:tar") as tar:
            tar.extractall(path=target_dir)  # Extract to TORCH_HOME directly
        print(f"Extraction complete: {extracted_path}")
    else:
        print(f"Directory already exists: {extracted_path}")

if __name__ == "__main__":
    main()
