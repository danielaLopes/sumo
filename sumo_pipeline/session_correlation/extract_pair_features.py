import logging
import sys
import os
from os import listdir
from os.path import isdir
import numpy as np
import tqdm
import pickle
import random as rd
import collections
from scipy.stats import kurtosis, skew
from scapy.all import *
import query_sumo_dataset
import sumo_features


def extract_pairs_features(data_folder, dataset_name, timeSamplingInterval):
    print("======== EXTRACTING PATHS ========\n")
    dataset = query_sumo_dataset.SumoDataset(data_folder)
    alexa_paths = dataset.alexa_sessions_paths_all()
    print("===== len(alexa_paths) before removing failed sessions", len(alexa_paths))
    client_paths_with_alexa = dataset.client_sessions_paths_all()
    print("===== len(client_paths_with_alexa)", len(client_paths_with_alexa))
    client_paths = dataset.client_sessions_paths_to_oses_all()
    print("===== len(client_paths) before removing failed sessions", len(client_paths))

    onion_paths = dataset.onion_sessions_paths_all()
    print("===== len(onion_paths) before removing failed sessions", len(onion_paths))

    try:
        session_paths_to_remove = dataset.filter_sessions_with_failed_requests()
    except query_sumo_dataset.MissingFailedRequestsLogException:
        print("No failed_requests.log files in the dataset")

    alexa_paths = [x for x in alexa_paths if x not in session_paths_to_remove]
    client_paths = [x for x in client_paths if x not in session_paths_to_remove]
    
    # TODO: Improve this
    failed_session_ids = []
    onion_paths_without_failed = []
    for client_path in session_paths_to_remove:
        session_id = query_sumo_dataset.get_session_id_from_path(client_path)
        failed_session_ids.append(session_id)
    for onion_path in onion_paths:
        session_id = query_sumo_dataset.get_session_id_from_path(onion_path)
        if session_id not in failed_session_ids:
            onion_paths_without_failed.append(onion_path)
    onion_paths = onion_paths_without_failed

    print("===== len(alexa_paths) after removing failed sessions", len(alexa_paths))
    print("===== len(client_paths) after removing failed sessions", len(client_paths))
    print("===== len(onion_paths) after removing failed sessions", len(onion_paths))

    #dataset.get_matching_sessions_clients_oses(session_paths_to_remove)
    #exit()
    
    print("\n======== EXTRACTING FEATURES ========\n")
    features = sumo_features.SumoFeatures(timeSamplingInterval, dataset_name, data_folder)


    try:
        features.extract(alexa_paths, client_paths, onion_paths)
    except sumo_features.MissingInventoryException:
        print("No inventory.cfg file in the dataset, not possible to get clients and oses IPs!")



    