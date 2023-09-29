#!/bin/bash

cd sumo_pipeline/session_correlation
make torpedosubsetsumopencl2d.so
python3 app.py correlate-sessions OSTest



