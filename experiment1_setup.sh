#!/bin/bash

if [ ! -d "extracted_features" ]; then
    echo "Downloading OSTrain, OSValidate, and OSTest extracted features ..."
    curl -o extracted_features.tar.gz -L "https://zenodo.org/record/8369700/files/extracted_features.tar.gz?download=1"
    gunzip extracted_features.tar.gz
    tar -xf extracted_features.tar
fi


echo "Obtaining subsetsumopencl2d.so ..."
cd sumo/sumo_pipeline/session_correlation
make torpedosubsetsumopencl2d.so
