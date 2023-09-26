#!/bin/bash

# This requires sudo rights on the machine, contact admins to install
sudo apt update
sudo apt install python3 ocl-icd-opencl-dev opencl-headers make g++ python3-matplotlib python3-numpy

python3 -m pip install -r requirements.txt
