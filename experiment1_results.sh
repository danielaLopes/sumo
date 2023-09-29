#!/bin/bash

mkdir experiment1

cd sumo_pipeline/session_correlation
python3 app.py plot-paper-results OSTest

cp results/figures_paper/*.{pdf,png} ../../experiment1
