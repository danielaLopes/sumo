#!/bin/bash

python deepcoffea/model.py --delta 2 \
                           --win_size 3 \
                           --n_wins 7 \
                           --threshold 15 \
                           --tor_len 200 \
                           --exit_len 300 \
                           --n_test 0 \
                           --data_root datasets/datasets_20230521_train_deepcoffea