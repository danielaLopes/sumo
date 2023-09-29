#!/bin/bash

mkdir experiment2

cd sumo_pipeline/session_correlation
python3 app.py partial-coverage OSTest

cp results/figures/*.{pdf,png} ../../experiment2