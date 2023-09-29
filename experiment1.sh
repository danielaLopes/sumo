#!/bin/bash

mkdir experiment1

cd sumo_pipeline/session_correlation
python3 app.py correlate-sessions OSTest

cp results/figures_paper/*.{pdf,png} ../../experiment1

