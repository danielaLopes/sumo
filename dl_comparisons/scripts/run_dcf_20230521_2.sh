#!/bin/bash

python deepcoffea/model.py --delta 3 \
                           --win_size 5 \
                           --n_wins 5 \
                           --threshold 20 \
                           --tor_len 300 \
                           --exit_len 500 \
                           --n_test 0 \
                           --data_root datasets/datasets_20230521_train_deepcoffea