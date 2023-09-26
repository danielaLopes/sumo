from http.client import CONTINUE
import dpkt
import socket
import numpy as np
#import pickle
import dill as pickle
import os
from tqdm import tqdm
from tqdm.contrib import tzip
import glob
import time
from joblib import Parallel, delayed
import multiprocessing
import random
from typing import List, Dict, Tuple, Literal

import flows


# TODO: Adjust as needed
#NUM_CORES = int(multiprocessing.cpu_count() / 2)
NUM_CORES = multiprocessing.cpu_count()


def get_extracted_features_folder(top_path, target_path, target_folder, session_id):
    return f"{top_path}/{target_path}/{target_folder}/{session_id}/"

def generate_buckets_epochs(initial_ts, last_ts, time_sampling_interval):
    initial_bucket = 0
    last_bucket = int(((last_ts - initial_ts) * 1000) // time_sampling_interval) + 1
    
    dictionary = {i: 0 for i in range(initial_bucket, last_bucket + 1)}

    return dictionary

def get_bucketized_packet_count(times_in, initial_ts, last_ts, time_sampling_interval):
    packet_count_in_dict = generate_buckets_epochs(initial_ts, last_ts, time_sampling_interval)
    for i in range(0, len(times_in)):
        ts = times_in[i] # time in milliseconds

        relative_ts = ts-initial_ts
        bucket = relative_ts * 1000 // time_sampling_interval
        if relative_ts < 0 and relative_ts > -1:
            bucket = 0
        
        packet_count_in_dict[bucket] += 1
    return list(packet_count_in_dict.values())

def get_tses_and_buckets_epochs(test_pairs, time_sampling_interval):
    tses = []
    buckets_clients = {}
    buckets_oses = {}

    for test_pair in test_pairs['correlated']['samples']:
        client_capture = test_pair['clientFolder'].split("/")[-1]   
        session_id = client_capture.split("_request")[0]
        for i in range(0, len(test_pair['clientFlow']['sizesIn'])):
            if session_id not in buckets_clients:
                buckets_clients[session_id] = {'initialTses': [], 'finalTses': []}
            buckets_clients[session_id]['initialTses'].append(min(test_pair['clientFlow']['timesInAbs']))
            buckets_clients[session_id]['finalTses'].append(max(test_pair['clientFlow']['timesInAbs']))
            
            tses.append(test_pair['clientFlow']['timesInAbs'][i])

        onion_capture = test_pair['hsFolder'].split("/")[-1]
        onion_session_id = onion_capture.split('_request')[0]
        for i in range(0, len(test_pair['hsFlow']['sizesOut'])):
            if onion_session_id not in buckets_oses:
                buckets_oses[onion_session_id] = {'initialTses': [], 'finalTses': []}
            buckets_oses[onion_session_id]['initialTses'].append(min(test_pair['hsFlow']['timesOutAbs']))
            buckets_oses[onion_session_id]['finalTses'].append(max(test_pair['hsFlow']['timesOutAbs']))

            tses.append(test_pair['hsFlow']['timesOutAbs'][i])

    for session_id, data in buckets_clients.items():
        client_initial_ts = min(data['initialTses'])
        client_final_ts = max(data['finalTses'])
        buckets_clients[session_id] = {'initialTs': client_initial_ts, 'finalTs': client_final_ts, 'buckets': generate_buckets_epochs(client_initial_ts, client_final_ts, time_sampling_interval)}

    for onion_session_id, data in buckets_oses.items():
        osInitialTs = min(data['initialTses'])
        osFinalTs = max(data['finalTses'])
        buckets_oses[onion_session_id] = {'initialTs': osInitialTs, 'finalTs': osFinalTs, 'buckets': generate_buckets_epochs(osInitialTs, osFinalTs, time_sampling_interval)}

    return min(tses), max(tses), buckets_clients, buckets_oses


def get_tses_and_buckets_epochs_requests(test_pairs, time_sampling_interval):
    tses = []
    buckets_clients = {}
    buckets_oses = {}

    for test_pair in test_pairs['correlated']['samples']:
        
        client_capture = test_pair['clientFolder'].split("/")[-1]   
        request_id = client_capture.split("_client")[0]
        for i in range(0, len(test_pair['clientFlow']['sizesIn'])):
            if request_id not in buckets_clients:
                buckets_clients[request_id] = {'initialTses': [], 'finalTses': []}
            buckets_clients[request_id]['initialTses'].append(min(test_pair['clientFlow']['timesInAbs']))
            buckets_clients[request_id]['finalTses'].append(max(test_pair['clientFlow']['timesInAbs']))
            
            tses.append(test_pair['clientFlow']['timesInAbs'][i])

        onion_capture = test_pair['hsFolder'].split("/")[-1]
        onion_request_id = onion_capture.split('_hs')[0]
        for i in range(0, len(test_pair['hsFlow']['sizesOut'])):
            if onion_request_id not in buckets_oses:
                buckets_oses[onion_request_id] = {'initialTses': [], 'finalTses': []}
            buckets_oses[onion_request_id]['initialTses'].append(min(test_pair['hsFlow']['timesOutAbs']))
            buckets_oses[onion_request_id]['finalTses'].append(max(test_pair['hsFlow']['timesOutAbs']))

            tses.append(test_pair['hsFlow']['timesOutAbs'][i])

    for request_id, data in buckets_clients.items():
        client_initial_ts = min(data['initialTses'])
        client_final_ts = max(data['finalTses'])
        buckets_clients[request_id] = {'initialTs': client_initial_ts, 'finalTs': client_final_ts, 'buckets': generate_buckets_epochs(client_initial_ts, client_final_ts, time_sampling_interval)}

    for onion_request_id, data in buckets_oses.items():
        osInitialTs = min(data['initialTses'])
        osFinalTs = max(data['finalTses'])
        buckets_oses[onion_request_id] = {'initialTs': osInitialTs, 'finalTs': osFinalTs, 'buckets': generate_buckets_epochs(osInitialTs, osFinalTs, time_sampling_interval)}

    return min(tses), max(tses), buckets_clients, buckets_oses


def process_correlated_pair_full_pipeline(client_file_path: str, 
                                          time_sampling_interval: int, 
                                          top_path: str) -> (str, 
                                                        flows.ClientFlow, 
                                                        flows.OnionFlow):
    client_folder_dict = pickle.load(open(client_file_path, 'rb'))
    onion_extracted_features_folder = get_extracted_features_folder(top_path, "onion", client_folder_dict['hsFolder'], client_folder_dict['clientSessionId'].split('_client.pcap')[0] + '_hs.pcap')
    onion_file_path = f"{onion_extracted_features_folder}folderDict.pickle"

    client_capture = client_folder_dict['clientSessionId']
    session_id = client_capture.split("_client")[0]
    initial_ts = client_folder_dict['clientMetaStats']['initialTimestamp']
    last_ts = client_folder_dict['clientMetaStats']['lastTimestamp']
    packet_count_in = get_bucketized_packet_count(client_folder_dict['clientFlow']['timesInAbs'], initial_ts, last_ts, time_sampling_interval)
    client_flow = flows.ClientFlow(initial_ts, last_ts, packet_count_in, client_folder_dict['clientFlow']['timesInAbs'])

    if os.path.isfile(onion_file_path):
        onion_folder_dict = pickle.load(open(onion_file_path, 'rb'))
        onion_flow = flows.OnionFlow(onion_folder_dict['hsMetaStats']['initialTimestamp'], onion_folder_dict['hsMetaStats']['lastTimestamp'], onion_folder_dict['hsFlow']['timesOutAbs'])
    else:
        return session_id, client_flow, None

    return session_id, client_flow, onion_flow

def get_session_duration(file_path):
    folderDict = pickle.load(open(file_path, 'rb'))
    if '_client.pcap' in file_path:
        allAbsTimes = folderDict['clientFlow']['timesOutAbs'] + folderDict['clientFlow']['timesInAbs']
    else:
        allAbsTimes = folderDict['hsFlow']['timesOutAbs'] + folderDict['hsFlow']['timesInAbs']
    absoluteInitialTime = min(allAbsTimes)
    maxAbsoluteTime = max(allAbsTimes)
    session_duration = maxAbsoluteTime - absoluteInitialTime

    return session_duration

def process_features_epochs_sessions_full_pipeline(dataset_name: str, 
                                                   time_sampling_interval: int,
                                                   buckets_per_window: int,
                                                   buckets_overlap: int, 
                                                   epoch_size: int, 
                                                   epoch_tolerance: int, 
                                                   min_session_durations: list[float], 
                                                   early_stop: int = None) -> (Dict[Tuple[str, str], flows.FlowPair], 
                                                                            Dict[str, flows.ClientFlow], 
                                                                            Dict[str, flows.OnionFlow],
                                                                            int,
                                                                            int,
                                                                            Dict[float, int],
                                                                            Dict[float, int]):
    possible_request_combinations: Dict[Tuple[str, str], flows.FlowPair] = {} # {(client_session_id, onion_session_id): FlowPair, ...}
    client_flows: Dict[str, flows.ClientFlow] = {} # {client_session_id: ClientFlow, ...}
    onion_flows: Dict[str, flows.OnionFlow] = {} # {onion_session_id: OnionFlow, ...}

    missed_client_flows_full_pipeline = 0
    missed_os_flows_full_pipeline = 0
    missed_client_flows_per_duration_full_pipeline = {}
    missed_os_flows_per_duration_full_pipeline = {}
    for min_dur in min_session_durations:
        missed_client_flows_per_duration_full_pipeline[min_dur] = 0
        missed_os_flows_per_duration_full_pipeline[min_dur] = 0

    dataset_name = dataset_name.split('_full_pipeline')[0]
    top_path = f"/mnt/nas-shared/torpedo/extracted_features/extracted_features_{dataset_name}"
    #os_flows_full_pipeline = pickle.load(open('../source_separation/full_pipeline_features/os_features_source_separation_thr_0.002577161882072687_{}.pickle'.format(dataset_name), 'rb'))
    os_flows_full_pipeline = pickle.load(open('../source_separation/full_pipeline_features/os_features_source_separation_thr_0.0010103702079504728_{}.pickle'.format(dataset_name), 'rb'))
    #os_flows_full_pipeline = pickle.load(open('../source_separation/full_pipeline_features/os_features_source_separation_thr_0.0016687361057847738_{}.pickle'.format(dataset_name), 'rb'))
    client_flows_full_pipeline = pickle.load(open('../target_separation/full_pipeline_features/client_features_target_separation_thr_0.9_{}.pickle'.format(dataset_name), 'rb'))
    client_session_ids_full_pipeline = list(client_flows_full_pipeline.keys())
    onion_session_ids_full_pipeline = list(os_flows_full_pipeline.keys())

    client_file_paths = list(glob.iglob(os.path.join(top_path+'/client', '**/folderDict.pickle'), recursive=True))

    results = []
    for _, client_file_path in tqdm(enumerate(client_file_paths), desc="Getting pairs features ..."):
        client_session_pcap_name = client_file_path.split('/')[-2]
        if client_session_pcap_name not in client_session_ids_full_pipeline:
            if 'alexa' not in client_session_pcap_name:
                missed_client_flows_full_pipeline += 1
                duration = get_session_duration(client_file_path)
                for min_dur in min_session_durations:
                    if duration >= min_dur:
                        missed_client_flows_per_duration_full_pipeline[min_dur] += 1
            continue

        client_folder_dict = pickle.load(open(client_file_path, 'rb'))
        onion_extracted_features_folder = get_extracted_features_folder(top_path, "onion", client_folder_dict['hsFolder'], client_folder_dict['clientSessionId'].split('_client.pcap')[0] + '_hs.pcap')
        onion_file_path = f"{onion_extracted_features_folder}folderDict.pickle"

        results.append(process_correlated_pair_full_pipeline(client_file_path, time_sampling_interval, top_path))

    for session_id, client_dict, onion_dict in results:
        if session_id is not None and client_dict is not None:
            client_flows[session_id] = client_dict
        if session_id is not None and onion_dict is not None:
            onion_flows[session_id] = onion_dict
    
    for client_session_id in client_session_ids_full_pipeline:
        if "_client.pcap" in client_session_id:
            session_id = client_session_id.split("_client.pcap")[0]
        else: # "_hs.pcap" in client_session_id
            session_id = client_session_id.split("_hs.pcap")[0]

        if session_id in client_flows:
            continue
        
        # Onion-side flow was mistaken as a client-side flow in the filtering phase
        if '_hs.pcap' in client_session_id:
            client_session_id_split = client_session_id.split("_")
            client_folder = client_session_id_split[1].split("-ostest-")[0]
            client_extracted_features_folder = get_extracted_features_folder(top_path, "onion", client_folder, client_session_id)
            client_file_path = f"{client_extracted_features_folder}folderDict.pickle"
            client_folder_dict = pickle.load(open(client_file_path, 'rb'))
            packet_count_in = get_bucketized_packet_count(client_folder_dict['hsFlow']['timesOutAbs'], client_folder_dict['hsMetaStats']['initialTimestamp'], client_folder_dict['hsMetaStats']['lastTimestamp'], time_sampling_interval)
            client_flows[session_id] = flows.ClientFlow(client_folder_dict['hsMetaStats']['initialTimestamp'], client_folder_dict['hsMetaStats']['lastTimestamp'], packet_count_in, client_folder_dict['hsFlow']['timesOutAbs'])

    onion_file_paths = list(glob.iglob(os.path.join(top_path+'/onion', '**/folderDict.pickle'), recursive=True))
    for onion_file_path in onion_file_paths:
        onion_session_pcap_name = onion_file_path.split('/')[-2]
        #if onion_session_id not in onion_flows
        onion_session_id = onion_session_pcap_name.split('_hs.pcap')
        if onion_session_pcap_name not in onion_session_ids_full_pipeline:
            missed_os_flows_full_pipeline += 1
            duration = get_session_duration(onion_file_path)
            for min_dur in min_session_durations:
                if duration >= min_dur:
                    missed_os_flows_per_duration_full_pipeline[min_dur] += 1

    for onion_session_id in onion_session_ids_full_pipeline:
        if "_client.pcap" in onion_session_id:
            session_id = onion_session_id.split("_client.pcap")[0]
        else: # "_hs.pcap" in onion_session_id:
            session_id = onion_session_id.split("_hs.pcap")[0]

        if session_id in onion_flows:
            continue
        
        # Client-side flow was mistaken as an onion-side flow in the filtering phase
        if '_client.pcap' in onion_session_id:
            onion_session_id_split = onion_session_id.split("_")
            client_folder = onion_session_id_split[0]
            onion_extracted_features_folder = get_extracted_features_folder(top_path, "client", client_folder, onion_session_id)
            onion_file_path = f"{onion_extracted_features_folder}folderDict.pickle"
            onion_folder_dict = pickle.load(open(onion_file_path, 'rb'))
            onion_flows[session_id] = flows.OnionFlow(onion_folder_dict['clientMetaStats']['initialTimestamp'], onion_folder_dict['clientMetaStats']['lastTimestamp'], onion_folder_dict['clientFlow']['timesInAbs'])

    client_keys = list(client_flows.keys())
    for client_session_id in client_keys:
        possible_request_combinations.update(process_uncorrelated_pair(client_flows[client_session_id], 
                                                                       onion_flows, 
                                                                       client_session_id, 
                                                                       time_sampling_interval,
                                                                       buckets_per_window,
                                                                       buckets_overlap,
                                                                       epoch_size, 
                                                                       epoch_tolerance))

    return (
        possible_request_combinations, 
        client_flows, onion_flows, 
        missed_client_flows_full_pipeline, 
        missed_os_flows_full_pipeline, 
        missed_client_flows_per_duration_full_pipeline, 
        missed_os_flows_per_duration_full_pipeline
    )

def process_correlated_pair(client_file_path: str, 
                            time_sampling_interval: int, 
                            top_path: str) -> (str, 
                                          flows.ClientFlow, 
                                          flows.OnionFlow):
    client_folder_dict = pickle.load(open(client_file_path, 'rb'))
    onion_extracted_features_folder = get_extracted_features_folder(top_path, "onion", client_folder_dict['hsFolder'], client_folder_dict['clientSessionId'].split('_client.pcap')[0] + '_hs.pcap')
    onion_file_path = f"{onion_extracted_features_folder}folderDict.pickle"
    # This prevents also getting alexa flows
    if not os.path.isfile(onion_file_path):
        #if 'alexa' not in onion_file_path:
        #    print("----> onion_file_path", onion_file_path)
        return None, None, None
    onion_folder_dict = pickle.load(open(onion_file_path, 'rb'))

    client_capture = client_folder_dict['clientSessionId']
    session_id = client_capture.split("_client")[0]
    initial_ts = client_folder_dict['clientMetaStats']['initialTimestamp']
    last_ts = client_folder_dict['clientMetaStats']['lastTimestamp']
    packet_count_in = get_bucketized_packet_count(client_folder_dict['clientFlow']['timesInAbs'], initial_ts, last_ts, time_sampling_interval)

    client_flow = flows.ClientFlow(initial_ts, last_ts, packet_count_in, client_folder_dict['clientFlow']['timesInAbs'])
    onion_flow = flows.OnionFlow(onion_folder_dict['hsMetaStats']['initialTimestamp'], onion_folder_dict['hsMetaStats']['lastTimestamp'], onion_folder_dict['hsFlow']['timesOutAbs'])

    return session_id, client_flow, onion_flow

def inner_process_uncorrelated_pair(client_flow: flows.ClientFlow, 
                                    onion_flow: flows.OnionFlow, 
                                    client_session_id: str, 
                                    onion_session_id: str, 
                                    time_sampling_interval: int, 
                                    epoch_size: int, 
                                    epoch_tolerance: int, 
                                    session_buckets: int,
                                    session_windows: int,
                                    initial_epoch: int, 
                                    last_epoch: int) -> Dict[Tuple[str, str], flows.FlowPair]:
    os_initial_epoch = onion_flow.first_ts // epoch_size
    os_last_epoch = (onion_flow.last_ts // epoch_size) + 1

    possible_request_combinations_tmp: Dict[Tuple[str, str], flows.FlowPair] = {} # {(client_session_id, onion_session_id): FlowPair, ...}
    
    #(StartDate1 <= EndDate2) and (StartDate2 <= EndDate1)
    # Both flows overlap in epoch times, so we consider them a possible combination
    if ((os_initial_epoch >= initial_epoch - epoch_tolerance) and (os_initial_epoch <= initial_epoch + epoch_tolerance)) and ((os_last_epoch >= last_epoch - epoch_tolerance) and (os_last_epoch <= last_epoch + epoch_tolerance)):
        key = (client_session_id, onion_session_id)
        label = 0
        if client_session_id == onion_session_id:
            label = 1
        else:
            # remove duplicated captures from same OS
            onion_name = onion_session_id.split("_")[1]
            if onion_name in client_session_id:
                return None
        
        possible_request_combinations_tmp[key] = flows.FlowPair(client_flow, 
                                                                onion_flow, 
                                                                time_sampling_interval, 
                                                                session_buckets, 
                                                                session_windows, 
                                                                label)

    return possible_request_combinations_tmp


def process_uncorrelated_pair(client_flow: flows.ClientFlow, 
                              onion_flows: list[flows.OnionFlow], 
                              client_session_id: str, 
                              time_sampling_interval: int, 
                              buckets_per_window: int,
                              buckets_overlap: int,
                              epoch_size: int, 
                              epoch_tolerance: int) -> Dict[Tuple[str, str], flows.FlowPair]:
    initial_epoch = client_flow.first_ts // epoch_size
    last_epoch = (client_flow.last_ts // epoch_size) + 1

    possible_request_combinations_tmp: Dict[Tuple[str, str], flows.FlowPair] = {} # {(client_session_id, onion_session_id): FlowPair, ...}

    onion_keys = list(onion_flows.keys())
    for onion_session_id in onion_keys:
        result = inner_process_uncorrelated_pair(client_flow, 
                                                 onion_flows[onion_session_id], 
                                                 client_session_id, 
                                                 onion_session_id, 
                                                 time_sampling_interval, 
                                                 epoch_size, 
                                                 epoch_tolerance, 
                                                 buckets_per_window,
                                                 buckets_overlap,
                                                 initial_epoch, 
                                                 last_epoch)
        # Here we place only the bucket range from the OS that makes sense to compare with the client
        if result is not None:
            possible_request_combinations_tmp.update(result)
        
    return possible_request_combinations_tmp


def process_features_epochs_sessions(dataset_name: str, 
                                     time_sampling_interval: int, 
                                     buckets_per_window: int,
                                     buckets_overlap: int,
                                     epoch_size: int, 
                                     epoch_tolerance: int, 
                                     early_stop: int = None) -> (Dict[Tuple[str, str], flows.FlowPair], 
                                                         Dict[str, flows.ClientFlow], 
                                                         Dict[str, flows.OnionFlow]):
    possible_request_combinations: Dict[Tuple[str, str], flows.FlowPair] = {} # {(client_session_id, onion_session_id): FlowPair, ...}
    client_flows: Dict[str, flows.ClientFlow] = {} # {client_session_id: ClientFlow, ...}
    onion_flows: Dict[str, flows.OnionFlow] = {} # {onion_session_id: OnionFlow, ...}

    top_path = f"/mnt/nas-shared/torpedo/extracted_features/extracted_features_{dataset_name}"
    client_file_paths = list(glob.iglob(os.path.join(top_path+'/client', '**/folderDict.pickle'), recursive=True))

    #print("\n==== len(client_file_paths)", len(client_file_paths))
    # TODO: PARALLELIZE
    results = []
    for test_idx, client_file_path in tqdm(enumerate(client_file_paths), desc="Getting pairs features ..."):
        results.append(process_correlated_pair(client_file_path, time_sampling_interval, top_path))
    """
    with tqdm(total=len(client_file_paths), desc="Getting correlated pairs features ...") as pbar:
        results = Parallel(n_jobs=NUM_CORES)(delayed(process_correlated_pair)(client_file_path[0], time_sampling_interval, top_path) for client_file_path in tzip(client_file_paths, leave=False))
        pbar.update()
    """
    count_eliminated = 0
    for session_id, client_dict, onion_dict in results:
        if session_id is None:
            count_eliminated += 1
            continue
        if onion_dict is None and 'alexa' not in session_id: 
            count_eliminated += 1
        if session_id is not None and client_dict is not None and onion_dict is not None:
            client_flows[session_id] = client_dict
            onion_flows[session_id] = onion_dict

    #print("=== Finished gathering data on correlated pairs {} : {}".format(len(client_flows), len(onion_flows)))

    # Now we have a list of all possible client-side sessions and os-side sessions
    # and their respetive start and end times. So, now we group all possible
    # combinations per epoch
    client_keys = list(client_flows.keys())
    for client_session_id in client_keys:
        possible_request_combinations.update(process_uncorrelated_pair(client_flows[client_session_id], 
                                                                       onion_flows, 
                                                                       client_session_id, 
                                                                       time_sampling_interval, 
                                                                       buckets_per_window,
                                                                       buckets_overlap,
                                                                       epoch_size, 
                                                                       epoch_tolerance))
    #with tqdm(total=len(clientsKeys), desc="Getting non correlated pairs features ...") as pbar:
        #results = Parallel(n_jobs=NUM_CORES, timeout=1000)(delayed(process_uncorrelated_pair)(client_flows[client_session_id[0]], onion_flows, time_sampling_interval, epoch_size, epoch_tolerance) for client_session_id in tzip(clientsKeys, leave=False))
        #results = Parallel(n_jobs=NUM_CORES)(delayed(process_uncorrelated_pair)(client_flows[client_session_id[0]], onion_flows, time_sampling_interval, epoch_size, epoch_tolerance) for client_session_id in tzip(clientsKeys, leave=False))
        #pbar.update()

    #for result in results:
        # Here we place only the bucket range from the OS that makes sense to compare with the client
        #possible_request_combinations.update(result)

    #print("=== Finished gathering data on uncorrelated pairs {}".format(len(possible_request_combinations)))

    return possible_request_combinations, client_flows, onion_flows

def select_random_items(lst, percentage):
    """Selects a random percentage of items from a list without repetition.

    Args:
        lst: The list to select items from.
        percentage: The percentage of items to select.

    Returns:
        A list of the randomly selected items.
    """

    total_items = len(lst)
    covered_items = int(total_items * percentage)
    selected_items = random.sample(lst, covered_items)

    return selected_items

def filter_alexas(client_file_paths):
    new_client_file_paths = []
    for client_file_path in client_file_paths:
        if 'alexa' not in client_file_path:
            new_client_file_paths.append(client_file_path)
    return new_client_file_paths
            
def process_features_epochs_sessions_by_eu_country(dataset_name: str, 
                                                   coverage_percentage: float, 
                                                   time_sampling_interval: int,
                                                   buckets_per_window: int,
                                                   buckets_overlap: int, 
                                                   epoch_size: int, 
                                                   epoch_tolerance: int, 
                                                   early_stop: int = None) -> (Dict[Tuple[str, str], flows.FlowPair], 
                                                                            Dict[str, flows.ClientFlow], 
                                                                            Dict[str, flows.OnionFlow]):
    possible_request_combinations: Dict[Tuple[str, str], flows.FlowPair] = {} # {(client_session_id, onion_session_id): FlowPair, ...}
    client_flows: Dict[str, flows.ClientFlow] = {} # {client_session_id: ClientFlow, ...}
    onion_flows: Dict[str, flows.OnionFlow] = {} # {onion_session_id: OnionFlow, ...}

    top_path = f"/mnt/nas-shared/torpedo/extracted_features/extracted_features_{dataset_name}"
    client_file_paths = list(glob.iglob(os.path.join(top_path+'/client', '**/folderDict.pickle'), recursive=True))
    client_file_paths = filter_alexas(client_file_paths)
    onion_file_paths = list(glob.iglob(os.path.join(top_path+'/onion', '**/folderDict.pickle'), recursive=True))

    #print("\n==== len(client_file_paths) FULL COVERAGE", len(client_file_paths))
    #print("\n==== len(onion_file_paths) FULL COVERAGE", len(onion_file_paths))

    client_file_paths = select_random_items(client_file_paths, coverage_percentage)
    onion_file_paths = select_random_items(onion_file_paths, coverage_percentage)

    #print("\n==== len(client_file_paths) PARTIAL COVERAGE", len(client_file_paths))
    #print("\n==== len(onion_file_paths) PARTIAL COVERAGE", len(onion_file_paths))
    # TODO: PARALLELIZE
    results = []
    for test_idx, client_file_path in tqdm(enumerate(client_file_paths), desc="Getting pairs features ..."):
        results.append(process_correlated_pair(client_file_path, 
                                            time_sampling_interval, 
                                            top_path))
    """
    with tqdm(total=len(client_file_paths), desc="Getting correlated pairs features ...") as pbar:
        results = Parallel(n_jobs=NUM_CORES)(delayed(process_correlated_pair)(client_file_path[0], time_sampling_interval, top_path) for client_file_path in tzip(client_file_paths, leave=False))
        pbar.update()
    """
    count_eliminated = 0
    for session_id, client_dict, onion_dict in results:
        if session_id is None:
            count_eliminated += 1
            continue
        if onion_dict is None and 'alexa' not in session_id: 
            count_eliminated += 1
        if session_id is not None and client_dict is not None and onion_dict is not None:
            client_flows[session_id] = client_dict
            onion_flows[session_id] = onion_dict

    print("len(onion_flows) BEFORE", len(onion_flows))
    # Process possibly remaining os traces that were not covered
    for i, onion_file_path in enumerate(onion_file_paths):
        onion_session_id = onion_file_path.split("/")[-2].split("_hs.pcap")[0]
        if onion_session_id not in onion_flows:
            onion_folder_dict = pickle.load(open(onion_file_path, 'rb'))
            onion_dict = {'rtts': [onion_folder_dict['hsMetaStats']['initialTimestamp'], onion_folder_dict['hsMetaStats']['lastTimestamp']], 'yPacketTimesOutOnion': onion_folder_dict['hsFlow']['timesOutAbs']}
            #onion_flows.update(onion_dict)
            onion_flows[onion_session_id] = onion_dict
    print("len(onion_flows) AFTER", len(onion_flows))

    print("=== Finished gathering data on correlated pairs {} : {}".format(len(client_flows), len(onion_flows)))

    # Now we have a list of all possible client-side sessions and os-side sessions
    # and their respetive start and end times. So, now we group all possible
    # combinations per epoch
    client_keys = list(client_flows.keys())
    for client_session_id in client_keys:
        possible_request_combinations.update(process_uncorrelated_pair(client_flows[client_session_id], 
                                                                       onion_flows, 
                                                                       client_session_id, 
                                                                       time_sampling_interval,
                                                                       buckets_per_window,
                                                                       buckets_overlap,
                                                                       epoch_size, 
                                                                       epoch_tolerance))
    #with tqdm(total=len(clientsKeys), desc="Getting non correlated pairs features ...") as pbar:
        #results = Parallel(n_jobs=NUM_CORES, timeout=1000)(delayed(process_uncorrelated_pair)(client_flows[client_session_id[0]], onion_flows, time_sampling_interval, epoch_size, epoch_tolerance) for client_session_id in tzip(clientsKeys, leave=False))
        #results = Parallel(n_jobs=NUM_CORES)(delayed(process_uncorrelated_pair)(client_flows[client_session_id[0]], onion_flows, time_sampling_interval, epoch_size, epoch_tolerance) for client_session_id in tzip(clientsKeys, leave=False))
        #pbar.update()

    #print("=== Finished gathering data on uncorrelated pairs {}".format(len(possible_request_combinations)))

    return possible_request_combinations, client_flows, onion_flows


# Groups requests features into full sessions, to be used with older datasets
def process_features_epochs_requests(test_pairs, time_sampling_interval, epoch_size, epoch_tolerance, earlyStop=None):
    possible_request_combinations = {}
    client_flows = {}
    onion_flows = {}

    counter = 0

    # absolute initial and final tses of the whole experience
    initial_ts_experience, last_ts_experience, buckets_clients, buckets_oses = get_tses_and_buckets_epochs(test_pairs, time_sampling_interval)

    print("=== Finished organizing buckets")
    for test_pair in test_pairs['correlated']['samples']:
        if earlyStop is not None and counter == earlyStop: 
            break

        client_capture = test_pair['clientFolder'].split("/")[-1]   
        session_id = client_capture.split("_request")[0]
        onion_capture = test_pair['hsFolder'].split("/")[-1]

        yPacketCountInDict = {}
        for bucket in buckets_clients[session_id]['buckets']:
            yPacketCountInDict[bucket] = 0

        for i in range(0, len(test_pair['clientFlow']['sizesIn'])):
            initial_ts = buckets_clients[session_id]['initialTs']
            ts = test_pair['clientFlow']['timesInAbs'][i] # time in milliseconds

            relative_ts = ts-initial_ts
            bucket = relative_ts * 1000 // time_sampling_interval
            
            yPacketCountInDict[bucket] += 1

        yPacketCountIn = list(yPacketCountInDict.values())

        allAbsTimes = test_pair['clientFlow']['timesOutAbs'] + test_pair['clientFlow']['timesInAbs']
        absoluteInitialTime = min(allAbsTimes)
        maxAbsoluteTime = max(allAbsTimes)

        if session_id not in client_flows:
            client_flows[session_id] = {'rtts': [absoluteInitialTime, maxAbsoluteTime], 'yPacketCountIn': yPacketCountIn, 'request_ids': [client_capture], 'yPacketTimesIn': test_pair['clientFlow']['timesInAbs']}
        else:
            if client_capture not in client_flows[session_id]['request_ids']:
                client_flows[session_id]['rtts'] += [absoluteInitialTime, maxAbsoluteTime]
                client_flows[session_id]['yPacketCountIn'] = np.add(client_flows[session_id]['yPacketCountIn'], yPacketCountIn)
                client_flows[session_id]['request_ids'] += [client_capture]
                client_flows[session_id]['yPacketTimesIn'] += test_pair['clientFlow']['timesInAbs']

        # onion part
        yPacketTimesOutOnion = []
        for i in range(0, len(test_pair['hsFlow']['sizesOut'])):
            yPacketTimesOutOnion.append(test_pair['hsFlow']['timesOutAbs'][i])

        allAbsTimesOnion = test_pair['hsFlow']['timesOutAbs'] + test_pair['hsFlow']['timesInAbs']
        absoluteInitialTimeOnion = min(allAbsTimesOnion)
        maxAbsoluteTimeOnion = max(allAbsTimesOnion)
        
        if session_id not in onion_flows:
            onion_flows[session_id] = {'rtts': [absoluteInitialTimeOnion, maxAbsoluteTimeOnion], 'yPacketTimesOutOnion': yPacketTimesOutOnion, 'request_ids': [onion_capture]}
        else:
            if onion_capture not in onion_flows[session_id]['request_ids']:
                onion_flows[session_id]['rtts'] += [absoluteInitialTimeOnion, maxAbsoluteTimeOnion]
                onion_flows[session_id]['yPacketTimesOutOnion'] += yPacketTimesOutOnion
                onion_flows[session_id]['request_ids'] += [onion_capture]

        counter += 1

    print("=== Finished gathering data on correlated pairs")
    counter = 0
    # Now we have a list of all possible client-side sessions and os-side sessions
    # and their respetive start and end times. So, now we group all possible
    # combinations per epoch
    for client_session_id in client_flows.keys():

        initial_epoch = buckets_clients[client_session_id]['initialTs'] // epoch_size
        last_epoch = (buckets_clients[client_session_id]['finalTs'] // epoch_size) + 1

        # Check which OSes are within the same epochs
        for onion_session_id in onion_flows.keys():

            os_initial_epoch = buckets_oses[onion_session_id]['initialTs'] // epoch_size
            os_last_epoch = (buckets_oses[onion_session_id]['finalTs'] // epoch_size) + 1
            
            #(StartDate1 <= EndDate2) and (StartDate2 <= EndDate1)
            # Both flows overlap in epoch times, so we consider them a possible combination
            #if (os_initial_epoch <= last_epoch) and (initial_epoch <= os_last_epoch):
            if ((os_initial_epoch >= initial_epoch - epoch_tolerance) and (os_initial_epoch <= initial_epoch + epoch_tolerance)) and ((os_last_epoch >= last_epoch - epoch_tolerance) and (os_last_epoch <= last_epoch + epoch_tolerance)):
                key = (client_session_id, onion_session_id)
                label = 0
                if client_session_id == onion_session_id:
                    label = 1
                else:
                    # remove duplicated captures from same OS
                    onion_name = onion_session_id.split("_")[1]
                    if onion_name in client_session_id:
                        continue
                    
                
                initial_session_ts = min(buckets_clients[client_session_id]['initialTs'], buckets_oses[onion_session_id]['initialTs'])
                final_session_ts = max(buckets_clients[client_session_id]['finalTs'], buckets_oses[onion_session_id]['finalTs'])
                buckets_session = generate_buckets_epochs(initial_session_ts, final_session_ts, time_sampling_interval)

                yPacketCountOutOnionDict = {}
                yPacketCountInDict = {}
                for bucket in buckets_session:
                    yPacketCountOutOnionDict[bucket] = 0
                    yPacketCountInDict[bucket] = 0
                

                # onion
                for i in range(0, len(onion_flows[onion_session_id]['yPacketTimesOutOnion'])):
                    ts = onion_flows[onion_session_id]['yPacketTimesOutOnion'][i] # time in milliseconds

                    relative_ts = ts - initial_session_ts
                    bucket = relative_ts * 1000 // time_sampling_interval
                        
                    yPacketCountOutOnionDict[bucket] += 1
                
                # client
                for i in range(0, len(client_flows[client_session_id]['yPacketTimesIn'])):
                    ts = client_flows[client_session_id]['yPacketTimesIn'][i] # time in milliseconds

                    relative_ts = ts - initial_session_ts
                    bucket = relative_ts * 1000 // time_sampling_interval
                        
                    yPacketCountInDict[bucket] += 1

                # Here we place only the bucket range from the OS that makes sense to compare with the client
                possible_request_combinations[key] = {'yPacketCountOutOnion': list(yPacketCountOutOnionDict.values()), \
                                                        'yPacketCountIn': list(yPacketCountInDict.values()), 'label': label}

        counter += 1

    return possible_request_combinations, client_flows, onion_flows


def process_features_epochs_requests_test_deepcoffea_our_dataset_march(test_pairs, time_sampling_interval, earlyStop=None):
    possible_request_combinations = {}
    client_flows = {}
    onion_flows = {}

    counter = 0

    # absolute initial and final tses of the whole experience
    initial_ts_experience, last_ts_experience, buckets_clients, buckets_oses = get_tses_and_buckets_epochs_requests(test_pairs, time_sampling_interval)

    print("=== Finished organizing buckets")
    for test_pair in test_pairs['correlated']['samples']:
        if earlyStop is not None and counter == earlyStop: 
            break

        client_capture = test_pair['clientFolder'].split("/")[-1]   
        request_id = client_capture.split("_client")[0]
        onion_capture = test_pair['hsFolder'].split("/")[-1]

        yPacketCountInDict = {}
        for bucket in buckets_clients[request_id]['buckets']:
            yPacketCountInDict[bucket] = 0

        for i in range(0, len(test_pair['clientFlow']['sizesIn'])):
            initial_ts = buckets_clients[request_id]['initialTs']
            ts = test_pair['clientFlow']['timesInAbs'][i] # time in milliseconds

            relative_ts = ts-initial_ts
            bucket = relative_ts * 1000 // time_sampling_interval
            
            yPacketCountInDict[bucket] += 1

        yPacketCountIn = list(yPacketCountInDict.values())

        allAbsTimes = test_pair['clientFlow']['timesOutAbs'] + test_pair['clientFlow']['timesInAbs']
        absoluteInitialTime = min(allAbsTimes)
        maxAbsoluteTime = max(allAbsTimes)

        client_flows[request_id] = {'rtts': [absoluteInitialTime, maxAbsoluteTime], 'yPacketCountIn': yPacketCountIn, 'request_ids': [client_capture], 'yPacketTimesIn': test_pair['clientFlow']['timesInAbs']}

        # onion part
        yPacketTimesOutOnion = []
        for i in range(0, len(test_pair['hsFlow']['sizesOut'])):
            yPacketTimesOutOnion.append(test_pair['hsFlow']['timesOutAbs'][i])

        allAbsTimesOnion = test_pair['hsFlow']['timesOutAbs'] + test_pair['hsFlow']['timesInAbs']
        absoluteInitialTimeOnion = min(allAbsTimesOnion)
        maxAbsoluteTimeOnion = max(allAbsTimesOnion)
        
        onion_flows[request_id] = {'rtts': [absoluteInitialTimeOnion, maxAbsoluteTimeOnion], 'yPacketTimesOutOnion': yPacketTimesOutOnion, 'request_ids': [onion_capture]}

        counter += 1

    print("=== Finished gathering data on correlated pairs")
    counter = 0

    test_samples_file = open('d1.0_ws1.6_nw5_thr10_tl200_el300_nt500_test_files.txt', 'r')
    test_samples = test_samples_file.readlines()
    test_samples_file.close()

    count_correlated = 0
    for clientSample in test_samples:
        clientSample = clientSample.replace("\n", "")
        clientrequest_id = clientSample.split('/')[-1]
        for osSample in test_samples:
            osSample = osSample.replace("\n", "")
            osrequest_id = osSample.split('/')[-1]

            key = (clientrequest_id, osrequest_id)
            label = 0
            if clientrequest_id == osrequest_id:
                label = 1
                count_correlated += 1
            else:
                # remove duplicated captures from same OS
                onion_name = osrequest_id.split("_")[1]
                if onion_name in clientrequest_id:
                    continue
                
            
            initial_session_ts = min(buckets_clients[clientrequest_id]['initialTs'], buckets_oses[osrequest_id]['initialTs'])
            final_session_ts = max(buckets_clients[clientrequest_id]['finalTs'], buckets_oses[osrequest_id]['finalTs'])
            buckets_session = generate_buckets_epochs(initial_session_ts, final_session_ts, time_sampling_interval)

            yPacketCountOutOnionDict = {}
            yPacketCountInDict = {}
            for bucket in buckets_session:
                yPacketCountOutOnionDict[bucket] = 0
                yPacketCountInDict[bucket] = 0
                

            # onion
            for i in range(0, len(onion_flows[osrequest_id]['yPacketTimesOutOnion'])):
                ts = onion_flows[osrequest_id]['yPacketTimesOutOnion'][i] # time in milliseconds

                relative_ts = ts - initial_session_ts
                bucket = relative_ts * 1000 // time_sampling_interval
                    
                yPacketCountOutOnionDict[bucket] += 1
            
            # client
            for i in range(0, len(client_flows[clientrequest_id]['yPacketTimesIn'])):
                ts = client_flows[clientrequest_id]['yPacketTimesIn'][i] # time in milliseconds

                relative_ts = ts - initial_session_ts
                bucket = relative_ts * 1000 // time_sampling_interval
                    
                yPacketCountInDict[bucket] += 1


            # Here we place only the bucket range from the OS that makes sense to compare with the client
            possible_request_combinations[key] = {'yPacketCountOutOnion': list(yPacketCountOutOnionDict.values()), \
                                                    'yPacketCountIn': list(yPacketCountInDict.values()), 'label': label}

        counter += 1

    print("\n+++++ count_correlated", count_correlated)

    return possible_request_combinations, client_flows, onion_flows


def process_features_epochs_requests_test_dataset_deepcoffea(time_sampling_interval, earlyStop=None):
    possible_request_combinations = {}
    client_flows = {}
    onion_flows = {}
    buckets_clients = {}
    buckets_oses = {}

    counter = 0

    test_pairs = []
    for file_idx, file in enumerate(os.listdir("ml_experiments/datasets/CrawlE_Proc_for_DC/")):
        if file_idx == 1:
            break
        dataset_chunk = pickle.load(open("ml_experiments/datasets/CrawlE_Proc_for_DC/"+file, 'rb'))
        for pair in dataset_chunk:
            test_pairs.append(pair)

            ts_accm_client = 0
            client_capture = 'client{}'.format(counter)  
            tses_in_client = pair['here'][0]['<-']
            tses_accm_client = []
            packets_in_client = pair['here'][1]['<-']
            for i in range(0, len(packets_in_client)):
                if client_capture not in buckets_clients:
                    buckets_clients[client_capture] = {'initialTses': [], 'finalTses': []}
                ts_accm_client += tses_in_client[i]
                tses_accm_client.append(ts_accm_client)
            buckets_clients[client_capture]['initialTses'].append(min(tses_accm_client))
            buckets_clients[client_capture]['finalTses'].append(max(tses_accm_client))
            buckets_clients[client_capture]['tses'] = tses_accm_client


            ts_accm_os = 0
            osCapture = 'client{}'.format(counter)  
            tses_in_os = pair['there'][0]['->']
            tses_accm_os = []
            packets_in_os = pair['there'][1]['->']
            for i in range(0, len(packets_in_os)):
                if osCapture not in buckets_oses:
                    buckets_oses[osCapture] = {'initialTses': [], 'finalTses': []}
                ts_accm_os += tses_in_os[i]
                tses_accm_os.append(ts_accm_os)
            buckets_oses[osCapture]['initialTses'].append(min(tses_accm_os))
            buckets_oses[osCapture]['finalTses'].append(max(tses_accm_os))
            buckets_oses[osCapture]['tses'] = tses_accm_os


            counter += 1

    for request_id, data in buckets_clients.items():
        client_initial_ts = min(data['initialTses'])
        client_final_ts = max(data['finalTses'])
        buckets_clients[request_id]['initialTs'] = client_initial_ts
        buckets_clients[request_id]['finalTs'] = client_final_ts
        buckets_clients[request_id]['buckets'] = generate_buckets_epochs(client_initial_ts,client_final_ts, time_sampling_interval)

    for onion_request_id, data in buckets_oses.items():
        osInitialTs = min(data['initialTses'])
        osFinalTs = max(data['finalTses'])
        buckets_oses[onion_request_id]['initialTs'] = osInitialTs
        buckets_oses[onion_request_id]['finalTs'] = osFinalTs
        buckets_oses[onion_request_id]['buckets'] = generate_buckets_epochs(osInitialTs, osFinalTs, time_sampling_interval)

    counter = 0
    print("=== Finished organizing buckets")
    for test_pair in test_pairs:
        if earlyStop is not None and counter == earlyStop: 
            break
  
        request_id = 'client{}'.format(counter)  

        yPacketCountInDict = {}

        for bucket in buckets_clients[request_id]['buckets']:
            yPacketCountInDict[bucket] = 0

        packets_in_client = test_pair['here'][1]['<-']
        for i in range(0, len(packets_in_client)):
            initial_ts = buckets_clients[request_id]['initialTs']
            ts = buckets_clients[request_id]['tses'][i] # time in milliseconds

            relative_ts = ts-initial_ts
            bucket = relative_ts * 1000 // time_sampling_interval
            
            yPacketCountInDict[bucket] += 1

        yPacketCountIn = list(yPacketCountInDict.values())

        allAbsTimes = buckets_clients[request_id]['tses']
        absoluteInitialTime = min(allAbsTimes)
        maxAbsoluteTime = max(allAbsTimes)

        client_flows[request_id] = {'rtts': [absoluteInitialTime, maxAbsoluteTime], 'yPacketCountIn': yPacketCountIn, 'request_ids': [client_capture], 'yPacketTimesIn': buckets_clients[request_id]['tses']}

        # onion part
        packets_in_os = test_pair['there'][1]['->']
        yPacketTimesOutOnion = []
        for i in range(0, len(packets_in_os)):
            yPacketTimesOutOnion.append(buckets_oses[request_id]['tses'][i])

        allAbsTimesOnion = buckets_oses[request_id]['tses']
        absoluteInitialTimeOnion = min(allAbsTimesOnion)
        maxAbsoluteTimeOnion = max(allAbsTimesOnion)
        
        onion_flows[request_id] = {'rtts': [absoluteInitialTimeOnion, maxAbsoluteTimeOnion], 'yPacketTimesOutOnion': buckets_oses[request_id]['tses']}

        counter += 1

    print("=== Finished gathering data on correlated pairs", counter)
    counter = 0


    count_correlated = 0
    for client_idx, clientrequest_id in enumerate(client_flows.keys()):
        for onion_idx, osrequest_id in enumerate(onion_flows.keys()):
            key = (clientrequest_id, osrequest_id)
            label = 0
            if clientrequest_id == osrequest_id:
                label = 1
                count_correlated += 1
                   
            initial_session_ts = min(buckets_clients[clientrequest_id]['initialTs'], buckets_oses[osrequest_id]['initialTs'])
            final_session_ts = max(buckets_clients[clientrequest_id]['finalTs'], buckets_oses[osrequest_id]['finalTs'])
            buckets_session = generate_buckets_epochs(initial_session_ts, final_session_ts, time_sampling_interval)

            yPacketCountOutOnionDict = {}
            yPacketCountInDict = {}
            for bucket in buckets_session:
                yPacketCountOutOnionDict[bucket] = 0
                yPacketCountInDict[bucket] = 0
                

            # onion
            for i in range(0, len(onion_flows[osrequest_id]['yPacketTimesOutOnion'])):
                ts = onion_flows[osrequest_id]['yPacketTimesOutOnion'][i] # time in milliseconds

                relative_ts = ts - initial_session_ts
                bucket = relative_ts * 1000 // time_sampling_interval
                    
                yPacketCountOutOnionDict[bucket] += 1


            # client
            for i in range(0, len(client_flows[clientrequest_id]['yPacketTimesIn'])):
                ts = client_flows[clientrequest_id]['yPacketTimesIn'][i] # time in milliseconds

                relative_ts = ts - initial_session_ts
                bucket = relative_ts * 1000 // time_sampling_interval
                    
                yPacketCountInDict[bucket] += 1


            # Here we place only the bucket range from the OS that makes sense to compare with the client
            possible_request_combinations[key] = {'yPacketCountOutOnion': list(yPacketCountOutOnionDict.values()), \
                                                    'yPacketCountIn': list(yPacketCountInDict.values()), 'label': label}
            
        counter += 1

    return possible_request_combinations, client_flows, onion_flows