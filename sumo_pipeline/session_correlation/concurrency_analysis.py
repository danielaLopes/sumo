import dpkt
import socket
import traceback
from scapy.all import *
import numpy as np

import sumo_pipeline.extract_raw_pcap_features.sumo_features as sumo_features


def getConcurrencyOverlapPercentage(request_id, initialTimestamp1, initialTimestamp2, lastTimestamp1, lastTimestamp2, concurrentRequests, osCapture1, osCapture2):
    duration1 = lastTimestamp1 - initialTimestamp1
    
    if initialTimestamp1 <= initialTimestamp2 and initialTimestamp2 < lastTimestamp1:

        concurrentRequests[request_id]['concurrent'] += 1
        concurrentRequests[request_id]['concurrentCaptures'] += [osCapture2]

        if initialTimestamp1 < lastTimestamp2 and lastTimestamp2 <= lastTimestamp1:
            duration2 = lastTimestamp2 - initialTimestamp2
        else:
            duration2 = lastTimestamp1 - initialTimestamp2

        concurrentRequests[request_id]['overlapPercentages'] += [duration2 / duration1]

    elif initialTimestamp1 < lastTimestamp2 and lastTimestamp2 <= lastTimestamp1:

        concurrentRequests[request_id]['concurrent'] += 1 
        concurrentRequests[request_id]['concurrentCaptures'] += [osCapture2]

        if initialTimestamp1 <= initialTimestamp2 and initialTimestamp2 < lastTimestamp1:
            duration2 = lastTimestamp2 - initialTimestamp2
        else:
            duration2 = lastTimestamp2 - initialTimestamp1
        
        concurrentRequests[request_id]['overlapPercentages'] += [duration2 / duration1]

    elif initialTimestamp2 <= initialTimestamp1 and lastTimestamp2 >= lastTimestamp1:

        concurrentRequests[request_id]['concurrent'] += 1 
        concurrentRequests[request_id]['concurrentCaptures'] += [osCapture2]
        concurrentRequests[request_id]['overlapPercentages'] += [1] # osCapture2 always overlaps completely osCapture1 in this case!

    #elif initialTimestamp2 == initialTimestamp1 and lastTimestamp2 == lastTimestamp1:
        # Captures are 100% concurrent! Same timestamps! 
        # Happens for 21march_2022 dataset with client-iowa-1-new_os-sao-paulo-1-new_ppcentrend4erspk_100_3_session_42_request_2_hs.pcap and client-saopaulo-1-new_os-sao-paulo-1-new_ppcentrend4erspk_100_3_session_37_request_3_hs.pcap

        #concurrentRequests[request_id]['concurrent'] += 1 
        #concurrentRequests[request_id]['concurrentCaptures'] += [osCapture2]
        #concurrentRequests[request_id]['overlapPercentages'] += [1]

    return concurrentRequests


def getConcurrencyOverlapPercentageClient(request_id, initialTimestamp1, initialTimestamp2, lastTimestamp1, lastTimestamp2, concurrentRequests, osName, clientCapture1, clientCapture2):
    duration1 = lastTimestamp1 - initialTimestamp1
    
    if initialTimestamp1 <= initialTimestamp2 and initialTimestamp2 < lastTimestamp1:
        if initialTimestamp1 < lastTimestamp2 and lastTimestamp2 <= lastTimestamp1:
            duration2 = lastTimestamp2 - initialTimestamp2
        else:
            duration2 = lastTimestamp1 - initialTimestamp2

        if osName in clientCapture2:
            concurrentRequests[request_id]['concurrent'] += 1
            concurrentRequests[request_id]['concurrentCaptures'] += [clientCapture2]
            concurrentRequests[request_id]['overlapPercentages'] += [duration2 / duration1]
        concurrentRequests[request_id]['concurrentCapturesMultipleOSes'] += [clientCapture2]


    elif initialTimestamp1 < lastTimestamp2 and lastTimestamp2 <= lastTimestamp1:
        if initialTimestamp1 <= initialTimestamp2 and initialTimestamp2 < lastTimestamp1:
            duration2 = lastTimestamp2 - initialTimestamp2
        else:
            duration2 = lastTimestamp2 - initialTimestamp1

        if osName in clientCapture2:
            concurrentRequests[request_id]['concurrent'] += 1
            concurrentRequests[request_id]['concurrentCaptures'] += [clientCapture2]
            concurrentRequests[request_id]['overlapPercentages'] += [duration2 / duration1]
        concurrentRequests[request_id]['concurrentCapturesMultipleOSes'] += [clientCapture2]

    elif initialTimestamp2 <= initialTimestamp1 and lastTimestamp2 >= lastTimestamp1:
        if osName in clientCapture2:
            concurrentRequests[request_id]['concurrent'] += 1
            concurrentRequests[request_id]['concurrentCaptures'] += [clientCapture2]
            concurrentRequests[request_id]['overlapPercentages'] += [1] # osCapture2 always overlaps completely osCapture1 in this case!
        concurrentRequests[request_id]['concurrentCapturesMultipleOSes'] += [clientCapture2]

    #elif initialTimestamp2 == initialTimestamp1 and lastTimestamp2 == lastTimestamp1:
        # Captures are 100% concurrent! Same timestamps! 
        # Happens for 21march_2022 dataset with client-iowa-1-new_os-sao-paulo-1-new_ppcentrend4erspk_100_3_session_42_request_2_hs.pcap and client-saopaulo-1-new_os-sao-paulo-1-new_ppcentrend4erspk_100_3_session_37_request_3_hs.pcap

        #concurrentRequests[request_id]['concurrent'] += 1 
        #concurrentRequests[request_id]['concurrentCaptures'] += [osCapture2]
        #concurrentRequests[request_id]['overlapPercentages'] += [1]

    return concurrentRequests


def getCaptureStartEndTimes(fileName):
    try:
        cap = PcapReader(fileName)
    except Exception as e:
        print("Problem parsing pcap {}".format(fileName))
        print(e)
        #continue
        return

    absoluteInitialTime = -1
    maxAbsoluteTime = -1

    #Read one by one
    packets = []
    i = 0
    for i, pkt in enumerate(cap):
        ts = np.float64(pkt.time)
        size = pkt.wirelen

        try:
            if pkt.haslayer(TCP):
                src_ip_addr_str = pkt[IP].src
                dst_ip_addr_str = pkt[IP].dst
                
                dport = pkt[TCP].dport
                sport = pkt[TCP].sport

                if sumo_features.skip_packets(dport, sport):
                    continue

                # Record first absolute time of all packets
                if absoluteInitialTime == -1:
                    absoluteInitialTime = ts
                elif absoluteInitialTime > ts:
                    absoluteInitialTime = ts

                if maxAbsoluteTime < ts:
                    maxAbsoluteTime = ts

        except Exception as e:
            print(e) #Uncomment to check what error is showing up when reading the pcap
            #Skip this corrupted packet
            traceback.print_exc()
            continue

    return absoluteInitialTime, maxAbsoluteTime

