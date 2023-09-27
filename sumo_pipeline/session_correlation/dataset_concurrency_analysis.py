import pickle
import os
from joblib import Parallel, delayed
import multiprocessing
import glob

import query_sumo_dataset
import concurrency_analysis
from constants import *


def getConcurrencyOverlapPercentage(initialTimestamp1, initialTimestamp2, lastTimestamp1, lastTimestamp2, concurrentRequests, osCapture1, osCapture2):
    duration1 = lastTimestamp1 - initialTimestamp1
    
    if initialTimestamp1 <= initialTimestamp2 and initialTimestamp2 < lastTimestamp1:

        concurrentRequests[osCapture1]['concurrent'] += 1
        concurrentRequests[osCapture1]['concurrentCaptures'] += [osCapture2]

        if initialTimestamp1 < lastTimestamp2 and lastTimestamp2 <= lastTimestamp1:
            duration2 = lastTimestamp2 - initialTimestamp2
        else:
            duration2 = lastTimestamp1 - initialTimestamp2

        concurrentRequests[osCapture1]['overlapPercentages'] += [duration2 / duration1]

    elif initialTimestamp1 < lastTimestamp2 and lastTimestamp2 <= lastTimestamp1:

        concurrentRequests[osCapture1]['concurrent'] += 1 
        concurrentRequests[osCapture1]['concurrentCaptures'] += [osCapture2]

        if initialTimestamp1 <= initialTimestamp2 and initialTimestamp2 < lastTimestamp1:
            duration2 = lastTimestamp2 - initialTimestamp2
        else:
            duration2 = lastTimestamp2 - initialTimestamp1
        
        concurrentRequests[osCapture1]['overlapPercentages'] += [duration2 / duration1]

    elif initialTimestamp2 <= initialTimestamp1 and lastTimestamp2 >= lastTimestamp1:

        concurrentRequests[osCapture1]['concurrent'] += 1 
        concurrentRequests[osCapture1]['concurrentCaptures'] += [osCapture2]
        concurrentRequests[osCapture1]['overlapPercentages'] += [1] # osCapture2 always overlaps completely osCapture1 in this case!

    return concurrentRequests


def getSessionConcurrencyAtOSes(base_dir, datasetBasePath, dataset_name):
    concurrentRequests = {}
    sessionsPerOS = {}

    concurrency_file = 'os_request_concurrency_{}.dat'.format(dataset_name)

    dataset = query_sumo_dataset.SumoDataset(datasetBasePath)
    onion_paths = dataset.onion_session_paths()

    for path in onion_paths:
        onionName = query_sumo_dataset.get_onion_name(path)
        startTime, endTime = concurrency_analysis.getCaptureStartEndTimes(path)
        session_id = path.split('/')[-1].split('_hs.pcap')[0]

        if onionName not in sessionsPerOS:
            sessionsPerOS[onionName] = {}

        
        #if path not in sessionsPerOS[onionName]:
        if session_id not in sessionsPerOS[onionName]:
            #sessionsPerOS[onionName][path] = {'startTime': startTime, 'endTime': endTime}
            sessionsPerOS[onionName][session_id] = {'startTime': startTime, 'endTime': endTime}

    #print("-> after first for")
    for onionName in sessionsPerOS:
        for osCapture1 in sessionsPerOS[onionName]:

            initialTimestamp1 = sessionsPerOS[onionName][osCapture1]['startTime']
            lastTimestamp1 = sessionsPerOS[onionName][osCapture1]['endTime']

            if osCapture1 not in concurrentRequests:
                concurrentRequests[osCapture1] = {'initialTimestamp': initialTimestamp1, 'lastTimestamp': lastTimestamp1, 'concurrent': 0, 'concurrentCaptures': [], 'overlapPercentages': []}

            # Iterates all other captures in the same OS and checks if they were concurrent
            for osCapture2 in sessionsPerOS[onionName]:
                if osCapture1 == osCapture2:
                    continue 

                initialTimestamp2 = sessionsPerOS[onionName][osCapture2]['startTime']
                lastTimestamp2 = sessionsPerOS[onionName][osCapture2]['endTime']

                # Check if request starts or ends concurrently with current client request
                concurrentRequests = getConcurrencyOverlapPercentage(initialTimestamp1, initialTimestamp2, lastTimestamp1, lastTimestamp2, concurrentRequests, osCapture1, osCapture2)

    pickle.dump(concurrentRequests, open(base_dir+concurrency_file, 'wb'))

    return concurrentRequests


def get_session_concurrency_at_onions_from_features(dataset_name):
    concurrentRequests = {}
    sessionsPerOS = {}

    concurrency_file = get_session_concurrency_at_onion_file_name(dataset_name)
    topPath = get_captures_folder(dataset_name)

    onion_file_paths = list(glob.iglob(os.path.join(topPath+'/onion', '**/folderDict.pickle'), recursive=True))
    for test_idx, onion_file_path in enumerate(onion_file_paths):
        onionFolderDict = pickle.load(open(onion_file_path, 'rb'))
        onionName = onionFolderDict['hsSessionId'].split('_')[1]
        startTime, endTime = onionFolderDict['hsMetaStats']['initialTimestamp'], onionFolderDict['hsMetaStats']['lastTimestamp']
        session_id = onionFolderDict['hsSessionId'].split("_hs")[0]
        
        if onionName not in sessionsPerOS:
            sessionsPerOS[onionName] = {}
        
        if session_id not in sessionsPerOS[onionName]:
            sessionsPerOS[onionName][session_id] = {'startTime': startTime, 'endTime': endTime}

    for onionName in sessionsPerOS:
        for osCapture1 in sessionsPerOS[onionName]:

            initialTimestamp1 = sessionsPerOS[onionName][osCapture1]['startTime']
            lastTimestamp1 = sessionsPerOS[onionName][osCapture1]['endTime']

            if osCapture1 not in concurrentRequests:
                concurrentRequests[osCapture1] = {'initialTimestamp': initialTimestamp1, 'lastTimestamp': lastTimestamp1, 'concurrent': 0, 'concurrentCaptures': [], 'overlapPercentages': []}

            # Iterates all other captures in the same OS and checks if they were concurrent
            for osCapture2 in sessionsPerOS[onionName].keys():
                if osCapture1 == osCapture2:
                    continue 

                initialTimestamp2 = sessionsPerOS[onionName][osCapture2]['startTime']
                lastTimestamp2 = sessionsPerOS[onionName][osCapture2]['endTime']

                # Check if request starts or ends concurrently with current client request
                concurrentRequests = getConcurrencyOverlapPercentage(initialTimestamp1, initialTimestamp2, lastTimestamp1, lastTimestamp2, concurrentRequests, osCapture1, osCapture2)
                
    pickle.dump(concurrentRequests, open(concurrency_file, 'wb'))

    return concurrentRequests