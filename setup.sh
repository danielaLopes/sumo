#!/bin/bash

# This requires sudo rights on the machine, contact admins to install

if [ "$1" == "packages"]
then
  sudo apt update
  sudo apt install python3 ocl-icd-opencl-dev opencl-headers make g++ python3-matplotlib python3-numpy
  echo "Installing python requirements ..."
  python3 -m pip install -r requirements.txt
fi

echo "Downloading OSTrain, OSValidate, and OSTest extracted features ..."
curl -o extracted_features.tar.gz -L "https://zenodo.org/record/8369700/files/extracted_features.tar.gz?download=1"
gunzip extracted_features.tar.gz
tar -xf extracted_features.tar


echo "Obtaining subsetsumopencl2d.so ..."
cd sumo/sumo_pipeline/session_correlation
make torpedosubsetsumopencl2d.so
