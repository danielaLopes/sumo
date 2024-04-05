from scipy.stats import kurtosis, skew
from scapy.all import *
import numpy as np
import traceback
import tqdm
from tqdm.contrib import tzip
import dill as pickle
import multiprocessing
from ansible.inventory.manager import InventoryManager
import random as rd
import os
import pandas as pd
from joblib import Parallel, delayed

from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")

# Surpress RuntimeWarning messages
import warnings
warnings.filterwarnings("ignore")


LABEL_IS_CLIENT_SIDE = 0
LABEL_IS_OS_SIDE = 1
LABEL_IS_NOT_SESSION_TO_OS = 0
LABEL_IS_SESSION_TO_OS = 1

NUM_CORES = multiprocessing.cpu_count()


class MissingInventoryException(Exception):
    "The dataset does not contain file inventory.cfg"  

class InvalidPcapException(Exception):
    def __init__(self, path):  
        message = "No data could be read!"         
        super().__init__(f'{message} : {path}')

class StatisticalFeaturesException(Exception):
    def __init__(self, path):  
        message = "Problem extracting statistical features from packet sizes and timings!"         
        super().__init__(f'{message} : {path}')


def RoundToNearest(n, m):
    r = n % m
    return n + m - r if r + r >= m else n - r


def flattenList(list1):
    return [item for sublist in list1 for item in sublist]


def filter_empty_ack(pkt):
    if pkt[TCP].flags & 0x10 and len(pkt[TCP].payload) == 0:
        ip_len = pkt[IP].len
        ip_hdr_len = pkt[IP].ihl * 4
        tcp_hdr_len = pkt[TCP].dataofs * 4
        payload_len = ip_len - ip_hdr_len - tcp_hdr_len
        if payload_len == 0:
            return True
        
    return False


def skip_packets(dport, sport):
    """Return True or False

    True if this packet should be skipped or False if this packet should be kept
    """

    # Internal communications between client Docker containers to use Tor socket
    if(dport == 9050 or sport == 9050):
        return True

    # Do not target packets produced due to synchronizing REST calls
    if(dport == 5005 or sport == 5005):
        return True

    # skip HTTP packets to manage Google Cloud instances
    if(dport == 80 or sport == 80):
        return True

    # Internal communications between onion Docker containers to host the onion website
    if(dport == 8080 or sport == 8080):
        return True

    return False


def get_headers():
    f_names_stats = []

    f_names_stats.append('TotalPackets')
    f_names_stats.append('totalPacketsIn')
    f_names_stats.append('totalPacketsOut')
    f_names_stats.append('totalBytes')
    f_names_stats.append('totalBytesIn')
    f_names_stats.append('totalBytesOut')
    f_names_stats.append('minPacketSize')
    f_names_stats.append('maxPacketSize')
    f_names_stats.append('meanPacketSizes')        
    f_names_stats.append('stdevPacketSizes')
    f_names_stats.append('variancePacketSizes')
    f_names_stats.append('kurtosisPacketSizes')
    f_names_stats.append('skewPacketSizes')
    f_names_stats.append('p10PacketSizes')
    f_names_stats.append('p20PacketSizes')
    f_names_stats.append('p30PacketSizes')
    f_names_stats.append('p40PacketSizes')
    f_names_stats.append('p50PacketSizes')
    f_names_stats.append('p60PacketSizes')
    f_names_stats.append('p70PacketSizes')
    f_names_stats.append('p80PacketSizes')
    f_names_stats.append('p90PacketSizes')
    f_names_stats.append('minPacketSizeIn')
    f_names_stats.append('maxPacketSizeIn')
    f_names_stats.append('meanPacketSizesIn')
    f_names_stats.append('stdevPacketSizesIn')
    f_names_stats.append('variancePacketSizesIn')
    f_names_stats.append('skewPacketSizesIn')
    f_names_stats.append('kurtosisPacketSizesIn')
    f_names_stats.append('p10PacketSizesIn')
    f_names_stats.append('p20PacketSizesIn')
    f_names_stats.append('p30PacketSizesIn')
    f_names_stats.append('p40PacketSizesIn')
    f_names_stats.append('p50PacketSizesIn')
    f_names_stats.append('p60PacketSizesIn')
    f_names_stats.append('p70PacketSizesIn')
    f_names_stats.append('p80PacketSizesIn')
    f_names_stats.append('p90PacketSizesIn')
    f_names_stats.append('minPacketSizeOut')
    f_names_stats.append('maxPacketSizeOut')
    f_names_stats.append('meanPacketSizesOut')
    f_names_stats.append('stdevPacketSizesOut')
    f_names_stats.append('variancePacketSizesOut')
    f_names_stats.append('skewPacketSizesOut')
    f_names_stats.append('kurtosisPacketSizesOut')
    f_names_stats.append('p10PacketSizesOut')
    f_names_stats.append('p20PacketSizesOut')
    f_names_stats.append('p30PacketSizesOut')
    f_names_stats.append('p40PacketSizesOut')
    f_names_stats.append('p50PacketSizesOut')
    f_names_stats.append('p60PacketSizesOut')
    f_names_stats.append('p70PacketSizesOut')
    f_names_stats.append('p80PacketSizesOut')
    f_names_stats.append('p90PacketSizesOut')
    f_names_stats.append('maxIPT')
    f_names_stats.append('minIPT')
    f_names_stats.append('meanPacketTimes')
    f_names_stats.append('stdevPacketTimes')
    f_names_stats.append('variancePacketTimes')
    f_names_stats.append('kurtosisPacketTimes')
    f_names_stats.append('skewPacketTimes')
    f_names_stats.append('p10PacketTimes')
    f_names_stats.append('p20PacketTimes')
    f_names_stats.append('p30PacketTimes')
    f_names_stats.append('p40PacketTimes')
    f_names_stats.append('p50PacketTimes')
    f_names_stats.append('p60PacketTimes')
    f_names_stats.append('p70PacketTimes')
    f_names_stats.append('p80PacketTimes')
    f_names_stats.append('p90PacketTimes')
    f_names_stats.append('minPacketTimesIn')
    f_names_stats.append('maxPacketTimesIn')
    f_names_stats.append('meanPacketTimesIn')
    f_names_stats.append('stdevPacketTimesIn')
    f_names_stats.append('variancePacketTimesIn')
    f_names_stats.append('skewPacketTimesIn')
    f_names_stats.append('kurtosisPacketTimesIn')
    f_names_stats.append('p10PacketTimesIn')
    f_names_stats.append('p20PacketTimesIn')
    f_names_stats.append('p30PacketTimesIn')
    f_names_stats.append('p40PacketTimesIn')
    f_names_stats.append('p50PacketTimesIn')
    f_names_stats.append('p60PacketTimesIn')
    f_names_stats.append('p70PacketTimesIn')
    f_names_stats.append('p80PacketTimesIn')
    f_names_stats.append('p90PacketTimesIn')
    f_names_stats.append('minPacketTimesOut')
    f_names_stats.append('maxPacketTimesOut')
    f_names_stats.append('meanPacketTimesOut')
    f_names_stats.append('stdevPacketTimesOut')
    f_names_stats.append('variancePacketTimesOut')
    f_names_stats.append('skewPacketTimesOut')
    f_names_stats.append('kurtosisPacketTimesOut')
    f_names_stats.append('p10PacketTimesOut')
    f_names_stats.append('p20PacketTimesOut')
    f_names_stats.append('p30PacketTimesOut')
    f_names_stats.append('p40PacketTimesOut')
    f_names_stats.append('p50PacketTimesOut')
    f_names_stats.append('p60PacketTimesOut')
    f_names_stats.append('p70PacketTimesOut')
    f_names_stats.append('p80PacketTimesOut')
    f_names_stats.append('p90PacketTimesOut')
    f_names_stats.append('out_totalBursts')
    f_names_stats.append('out_maxBurst')
    f_names_stats.append('out_meanBurst')
    f_names_stats.append('out_stdevBurst')
    f_names_stats.append('out_varianceBurst')
    f_names_stats.append('out_kurtosisBurst')
    f_names_stats.append('out_skewBurst')
    f_names_stats.append('out_p10Burst')
    f_names_stats.append('out_p20Burst')
    f_names_stats.append('out_p30Burst')
    f_names_stats.append('out_p40Burst')
    f_names_stats.append('out_p50Burst')
    f_names_stats.append('out_p60Burst')
    f_names_stats.append('out_p70Burst')
    f_names_stats.append('out_p80Burst')
    f_names_stats.append('out_p90Burst')
    f_names_stats.append('out_maxBurstBytes')
    f_names_stats.append('out_minBurstBytes')
    f_names_stats.append('out_meanBurstBytes')
    f_names_stats.append('out_stdevBurstBytes')
    f_names_stats.append('out_varianceBurstBytes')
    f_names_stats.append('out_kurtosisBurstBytes')
    f_names_stats.append('out_skewBurstBytes')
    f_names_stats.append('out_p10BurstBytes')
    f_names_stats.append('out_p20BurstBytes')
    f_names_stats.append('out_p30BurstBytes')
    f_names_stats.append('out_p40BurstBytes')
    f_names_stats.append('out_p50BurstBytes')
    f_names_stats.append('out_p60BurstBytes')
    f_names_stats.append('out_p70BurstBytes')
    f_names_stats.append('out_p80BurstBytes')
    f_names_stats.append('out_p90BurstBytes')
    f_names_stats.append('in_totalBursts')
    f_names_stats.append('in_maxBurst')
    f_names_stats.append('in_meanBurst')
    f_names_stats.append('in_stdevBurst')
    f_names_stats.append('in_varianceBurst')
    f_names_stats.append('in_kurtosisBurst')
    f_names_stats.append('in_skewBurst')
    f_names_stats.append('in_p10Burst')
    f_names_stats.append('in_p20Burst')
    f_names_stats.append('in_p30Burst')
    f_names_stats.append('in_p40Burst')
    f_names_stats.append('in_p50Burst')
    f_names_stats.append('in_p60Burst')
    f_names_stats.append('in_p70Burst')
    f_names_stats.append('in_p80Burst')
    f_names_stats.append('in_p90Burst')
    f_names_stats.append('in_maxBurstBytes')
    f_names_stats.append('in_minBurstBytes')
    f_names_stats.append('in_meanBurstBytes')
    f_names_stats.append('in_stdevBurstBytes')
    f_names_stats.append('in_varianceBurstBytes')
    f_names_stats.append('in_kurtosisBurstBytes')
    f_names_stats.append('in_skewBurstBytes')
    f_names_stats.append('in_p10BurstBytes')
    f_names_stats.append('in_p20BurstBytes')
    f_names_stats.append('in_p30BurstBytes')
    f_names_stats.append('in_p40BurstBytes')
    f_names_stats.append('in_p50BurstBytes')
    f_names_stats.append('in_p60BurstBytes')
    f_names_stats.append('in_p70BurstBytes')
    f_names_stats.append('in_p80BurstBytes')
    f_names_stats.append('in_p90BurstBytes')
    f_names_stats.append('Class')
    f_names_stats.append('Capture')

    return f_names_stats


class SessionFeatures:
    clientFolder : str
    hsFolder : str
    sessionId : str
    #origin : str
    #destination : str 
    onionUrl : str 
    clearwebAddress: str
    first_ts : float
    last_ts : float
    packetTimes : list[float]
    packetTimesIn : list[float]
    packetTimesOut : list[float]
    packetTimesInRel : list[float]
    packetTimesOutRel : list[float]
    packetTimesInAbs : list[float]
    packetTimesOutAbs : list[float]
    packetSizes : list[int]
    packetSizesIn : list[int]
    packetSizesOut : list[int]
    special_features : dict[str, object]
    machine_ip : str
    special_features = dict[str, object]
    out_bursts_packets : list[int]
    out_burst_sizes : list[int]
    out_burst_times : list[int]
    in_bursts_packets : list[int]
    in_burst_sizes : list[int]
    in_burst_times : list[int]
    totalPackets : int
    totalPacketsIn : int
    totalPacketsOut : int
    totalBytes : int
    totalBytesIn : int
    totalBytesOut : int
    bin_dict = dict[str, object]
    bin_dict2 = dict[str, object]
    binWidth : int

    def __init__(self):
        self.first_ts = 0
        self.last_ts = 0
        self.packetTimes = []
        self.packetTimesIn = []
        self.packetTimesInRel = [0]
        self.packetTimesOut = []
        self.packetTimesOutRel = [0]
        self.packetTimesInAbs = []
        self.packetTimesOutAbs = []
        self.packetSizes = []
        self.packetSizesIn = []
        self.packetSizesOut = []
        self.specialFeatures = {}
        self.out_bursts_packets = []
        self.out_burst_sizes = []
        self.out_burst_times = []
        self.in_bursts_packets = []
        self.in_burst_sizes = []
        self.in_burst_times = []
        self.totalPackets = 0
        self.totalPacketsIn = 0
        self.totalPacketsOut = 0
        self.totalBytes = 0
        self.totalBytesIn = 0
        self.totalBytesOut = 0
        self.bin_dict = {}
        self.bin_dict2 = {}
        self.binWidth = 5

    def process_packets(self, path, ips, isClient = True, isAlexa = True):
        try:
            # The capture file is not entirely loaded into memory at the same time, 
            # only a packet each time, making it more memory efficient
            cap = PcapReader(path)
        except Exception as e:
            print("Problem parsing pcap {}".format(path))
            print(e)
            traceback.print_exc()
            #continue
            raise InvalidPcapException(path)

        pcap_name = path.split('/')[-1]
        #print("pcap_name", pcap_name)
        #print("re.search(r'client-[a-zA-Z]+-[0-9]+-client[0-9]+', pcap_name)", re.search(r'client-[a-zA-Z]+-[0-9]+-client[0-9]+', pcap_name))
        if isClient:
            # TODO: change this
            #machine_name = re.search(r'client-[a-zA-Z]+-[0-9]+', pcap_name).group()
            machine_name = re.search(r'client(?:-[a-zA-Z]+)?-[a-zA-Z]+-[0-9]+', pcap_name).group()
        else:
            machine_name = re.search(r'os(?:-[a-zA-Z]+)?-[a-zA-Z]+(?:-[0-9]+)?', pcap_name).group()
        self.machine_ip = "172." + str(ips[machine_name])

        self.clientFolder = re.search(r'client(?:-[a-zA-Z]+)?-[a-zA-Z]+-[0-9]+-client[0-9]+', pcap_name).group()
        #print("self.clientFolder", self.clientFolder)
        #self.clientFolder = pcap_name
        #print("self.clientFolder", self.clientFolder)
        if isAlexa:
            self.hsFolder = None
            self.clearwebAddress = re.search(r'alexa_([a-zA-Z0-9-]+\.[a-z]+)', pcap_name).group(1)
        else:
            self.hsFolder = re.search(r'os[0-9]+-os-[a-zA-Z]+(?:-[0-9]+)?', pcap_name).group()
            self.onionUrl = re.search(r'[a-z0-9]{56}', pcap_name).group()

        self.sessionId = pcap_name.split('/')[-1]


        prev_ts_in = 0
        prev_ts_out = 0

        # Generate the set of all possible bins
        for i in range(0, 100000, self.binWidth):
            self.bin_dict[i] = 0
            self.bin_dict2[i] = 0

        out_current_burst = 0
        out_current_burst_start = 0
        out_current_burst_size = 0

        in_current_burst = 0
        in_current_burst_size = 0

        prev_ts = 0
        counter = 0

        for i, pkt in enumerate(cap):
            ts = np.float64(pkt.time)
            size = pkt.wirelen

            try:
                # Target TCP communication
                if pkt.haslayer(TCP):
                    src_ip_addr_str = pkt[IP].src
                    dst_ip_addr_str = pkt[IP].dst
                    
                    dport = pkt[TCP].dport
                    sport = pkt[TCP].sport

                    if skip_packets(dport, sport):
                        continue

                    # Filter packets with empty TCP ACK payload
                    if filter_empty_ack(pkt):
                        continue
                    
                    # Record initial and last timestamps
                    if(self.first_ts == 0):
                        self.first_ts = ts
                    if ts < self.first_ts:
                        self.first_ts = ts
                    if self.last_ts < ts:
                        self.last_ts = ts

                    if(self.machine_ip in dst_ip_addr_str):
                        self.packetSizesIn.append(size)

                        if (prev_ts_in != 0):         
                            ts_difference_in = max(0, ts - prev_ts_in)
                            self.packetTimesIn.append(ts_difference_in) 
                        else:
                            self.packetTimesIn.append(0)

                        prev_ts_in = ts

                        self.packetTimesInAbs.append(ts)
                        self.packetTimesInRel.append(ts - self.first_ts)

                        self.totalPacketsIn += 1
                        binned = RoundToNearest(size, self.binWidth)
                        self.bin_dict2[binned] += 1

                        #print("<- out_current_burst", out_current_burst)
                        if (out_current_burst != 0):
                            if (out_current_burst > 1):
                                self.out_bursts_packets.append(out_current_burst)  # packets on burst
                                self.out_burst_sizes.append(out_current_burst_size)  # total bytes on burst
                                self.out_burst_times.append(ts - out_current_burst_start)
                            out_current_burst = 0
                            out_current_burst_size = 0
                            out_current_burst_start = 0
                        if (in_current_burst == 0):
                            in_current_burst_start = ts
                        in_current_burst += 1
                        in_current_burst_size += size              

                    # If machine is sender
                    elif(self.machine_ip in src_ip_addr_str):
                        self.packetSizesOut.append(size)
                        
                        if (prev_ts_out != 0):
                            ts_difference_out = max(0, ts - prev_ts_out)
                            self.packetTimesOut.append(ts_difference_out)

                        else:
                            self.packetTimesOut.append(0)

                        prev_ts_out = ts

                        self.packetTimesOutAbs.append(ts)
                        self.packetTimesOutRel.append(ts - self.first_ts)


                        self.totalPacketsOut += 1
                        binned = RoundToNearest(size, self.binWidth)
                        self.bin_dict[binned] += 1
                        if (out_current_burst == 0):
                            out_current_burst_start = ts
                        out_current_burst += 1
                        out_current_burst_size += size
                        
                        if (in_current_burst != 0):
                            #print("-> in_current_burst", in_current_burst)
                            if (in_current_burst > 1):
                                self.in_bursts_packets.append(out_current_burst)  # packets on burst
                                self.in_burst_sizes.append(out_current_burst_size)  # total bytes on burst
                                self.in_burst_times.append(ts - out_current_burst_start)
                            self.in_current_burst = 0
                            self.in_current_burst_size = 0

                    # Bytes transmitted statistics
                    self.totalBytes += pkt.wirelen
                    self.totalPackets += 1

                    if (self.machine_ip in src_ip_addr_str):
                        self.totalBytesOut += size
                    else:
                        self.totalBytesIn += size

                    # Packet Size statistics
                    self.packetSizes.append(size)

                    # Packet Times statistics
                    if (prev_ts != 0):
                        ts_difference = max(0, ts - prev_ts)
                        self.packetTimes.append(ts_difference * 1000)

                    prev_ts = ts
                
            except Exception as e:
                print("Corrupted packet")
                print(repr(e))
                print(e)
                traceback.print_exc()

            counter += 1


    def bucketize_packets(self):
        # TODO: PASS ALL THE LOGIC INSIDE feature_collection_new.py TO THIS CLASS!!
        pass


    def get_statistical_features(self, path, label_source, capture):
        # TODO: Keep changing all instances of written_header_stats and so on with source, and for the target copy if it's a client flow
        
        try:
            ##########################################################
            # Statistical indicators for packet sizes (total)
            meanPacketSizes = np.mean(self.packetSizes)
            stdevPacketSizes = np.std(self.packetSizes)
            variancePacketSizes = np.var(self.packetSizes)
            kurtosisPacketSizes = kurtosis(self.packetSizes)
            skewPacketSizes = skew(self.packetSizes)
            maxPacketSize = np.amax(self.packetSizes)
            minPacketSize = np.amin(self.packetSizes)
            p10PacketSizes = np.percentile(self.packetSizes, 10)
            p20PacketSizes = np.percentile(self.packetSizes, 20)
            p30PacketSizes = np.percentile(self.packetSizes, 30)
            p40PacketSizes = np.percentile(self.packetSizes, 40)
            p50PacketSizes = np.percentile(self.packetSizes, 50)
            p60PacketSizes = np.percentile(self.packetSizes, 60)
            p70PacketSizes = np.percentile(self.packetSizes, 70)
            p80PacketSizes = np.percentile(self.packetSizes, 80)
            p90PacketSizes = np.percentile(self.packetSizes, 90)

        except Exception as e:
            print("Error in block 1 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            #continue
            raise StatisticalFeaturesException(path)

        try:
            ##########################################################
            # Statistical indicators for packet sizes (in)
            meanPacketSizesIn = np.mean(self.packetSizesIn)
            stdevPacketSizesIn = np.std(self.packetSizesIn)
            variancePacketSizesIn = np.var(self.packetSizesIn)
            kurtosisPacketSizesIn = kurtosis(self.packetSizesIn)
            skewPacketSizesIn = skew(self.packetSizesIn)
            maxPacketSizeIn = np.amax(self.packetSizesIn)
            minPacketSizeIn = np.amin(self.packetSizesIn)
            p10PacketSizesIn = np.percentile(self.packetSizesIn, 10)
            p20PacketSizesIn = np.percentile(self.packetSizesIn, 20)
            p30PacketSizesIn = np.percentile(self.packetSizesIn, 30)
            p40PacketSizesIn = np.percentile(self.packetSizesIn, 40)
            p50PacketSizesIn = np.percentile(self.packetSizesIn, 50)
            p60PacketSizesIn = np.percentile(self.packetSizesIn, 60)
            p70PacketSizesIn = np.percentile(self.packetSizesIn, 70)
            p80PacketSizesIn = np.percentile(self.packetSizesIn, 80)
            p90PacketSizesIn = np.percentile(self.packetSizesIn, 90)
        
        except Exception as e:
            print("Error in block 2 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            #continue
            raise StatisticalFeaturesException(path)

        try:    
            ##########################################################
            # Statistical indicators for packet sizes (out)
            meanPacketSizesOut = np.mean(self.packetSizesOut)
            stdevPacketSizesOut = np.std(self.packetSizesOut)
            variancePacketSizesOut = np.var(self.packetSizesOut)
            kurtosisPacketSizesOut = kurtosis(self.packetSizesOut)
            skewPacketSizesOut = skew(self.packetSizesOut)
            maxPacketSizeOut = np.amax(self.packetSizesOut)
            minPacketSizeOut = np.amin(self.packetSizesOut)
            p10PacketSizesOut = np.percentile(self.packetSizesOut, 10)
            p20PacketSizesOut = np.percentile(self.packetSizesOut, 20)
            p30PacketSizesOut = np.percentile(self.packetSizesOut, 30)
            p40PacketSizesOut = np.percentile(self.packetSizesOut, 40)
            p50PacketSizesOut = np.percentile(self.packetSizesOut, 50)
            p60PacketSizesOut = np.percentile(self.packetSizesOut, 60)
            p70PacketSizesOut = np.percentile(self.packetSizesOut, 70)
            p80PacketSizesOut = np.percentile(self.packetSizesOut, 80)
            p90PacketSizesOut = np.percentile(self.packetSizesOut, 90)

        except Exception as e:
            print("Error in block 3 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            #continue
            raise StatisticalFeaturesException(path)

        try:
            ##################################################################
            # Statistical indicators for Inter-Packet Times (total)
            meanPacketTimes = np.mean(self.packetTimes)
            stdevPacketTimes = np.std(self.packetTimes)
            variancePacketTimes = np.var(self.packetTimes)
            kurtosisPacketTimes = kurtosis(self.packetTimes)
            skewPacketTimes = skew(self.packetTimes)
            maxIPT = np.amax(self.packetTimes)
            minIPT = np.amin(self.packetTimes)
            p10PacketTimes = np.percentile(self.packetTimes, 10)
            p20PacketTimes = np.percentile(self.packetTimes, 20)
            p30PacketTimes = np.percentile(self.packetTimes, 30)
            p40PacketTimes = np.percentile(self.packetTimes, 40)
            p50PacketTimes = np.percentile(self.packetTimes, 50)
            p60PacketTimes = np.percentile(self.packetTimes, 60)
            p70PacketTimes = np.percentile(self.packetTimes, 70)
            p80PacketTimes = np.percentile(self.packetTimes, 80)
            p90PacketTimes = np.percentile(self.packetTimes, 90)
        
        except Exception as e:
            print("Error in block 4 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            #continue
            raise StatisticalFeaturesException(path)

        try:
            ##################################################################
            # Statistical indicators for Inter-Packet Times (in)
            meanPacketTimesIn = np.mean(self.packetTimesIn)
            stdevPacketTimesIn = np.std(self.packetTimesIn)
            variancePacketTimesIn = np.var(self.packetTimesIn)
            kurtosisPacketTimesIn = kurtosis(self.packetTimesIn)
            skewPacketTimesIn = skew(self.packetTimesIn)
            maxPacketTimesIn = np.amax(self.packetTimesIn)
            minPacketTimesIn = np.amin(self.packetTimesIn)
            p10PacketTimesIn = np.percentile(self.packetTimesIn, 10)
            p20PacketTimesIn = np.percentile(self.packetTimesIn, 20)
            p30PacketTimesIn = np.percentile(self.packetTimesIn, 30)
            p40PacketTimesIn = np.percentile(self.packetTimesIn, 40)
            p50PacketTimesIn = np.percentile(self.packetTimesIn, 50)
            p60PacketTimesIn = np.percentile(self.packetTimesIn, 60)
            p70PacketTimesIn = np.percentile(self.packetTimesIn, 70)
            p80PacketTimesIn = np.percentile(self.packetTimesIn, 80)
            p90PacketTimesIn = np.percentile(self.packetTimesIn, 90)

        except Exception as e:
            print("Error in block 5 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            #continue
            raise StatisticalFeaturesException(path)

        try:
            ##################################################################
            # Statistical indicators for Inter-Packet Times (out)
            meanPacketTimesOut = np.mean(self.packetTimesOut)
            stdevPacketTimesOut = np.std(self.packetTimesOut)
            variancePacketTimesOut = np.var(self.packetTimesOut)
            kurtosisPacketTimesOut = kurtosis(self.packetTimesOut)
            skewPacketTimesOut = skew(self.packetTimesOut)
            maxPacketTimesOut = np.amax(self.packetTimesOut)
            minPacketTimesOut = np.amin(self.packetTimesOut)
            p10PacketTimesOut = np.percentile(self.packetTimesOut, 10)
            p20PacketTimesOut = np.percentile(self.packetTimesOut, 20)
            p30PacketTimesOut = np.percentile(self.packetTimesOut, 30)
            p40PacketTimesOut = np.percentile(self.packetTimesOut, 40)
            p50PacketTimesOut = np.percentile(self.packetTimesOut, 50)
            p60PacketTimesOut = np.percentile(self.packetTimesOut, 60)
            p70PacketTimesOut = np.percentile(self.packetTimesOut, 70)
            p80PacketTimesOut = np.percentile(self.packetTimesOut, 80)
            p90PacketTimesOut = np.percentile(self.packetTimesOut, 90)
        
        except Exception as e:
            print("Error in block 6 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            #continue
            raise StatisticalFeaturesException(path)

        try:
            ########################################################################
            # Statistical indicators for Outgoing bursts
            out_totalBursts = len(self.out_bursts_packets)
            out_meanBurst = np.mean(self.out_bursts_packets)
            out_stdevBurst = np.std(self.out_bursts_packets)
            out_varianceBurst = np.var(self.out_bursts_packets)
            out_maxBurst = np.amax(self.out_bursts_packets)
            out_kurtosisBurst = kurtosis(self.out_bursts_packets)
            out_skewBurst = skew(self.out_bursts_packets)
            out_p10Burst = np.percentile(self.out_bursts_packets, 10)
            out_p20Burst = np.percentile(self.out_bursts_packets, 20)
            out_p30Burst = np.percentile(self.out_bursts_packets, 30)
            out_p40Burst = np.percentile(self.out_bursts_packets, 40)
            out_p50Burst = np.percentile(self.out_bursts_packets, 50)
            out_p60Burst = np.percentile(self.out_bursts_packets, 60)
            out_p70Burst = np.percentile(self.out_bursts_packets, 70)
            out_p80Burst = np.percentile(self.out_bursts_packets, 80)
            out_p90Burst = np.percentile(self.out_bursts_packets, 90)

        except Exception as e:
            print("Error in block 7 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            
            #raise StatisticalFeaturesException(path)
            out_totalBursts = 0
            out_meanBurst = 0
            out_stdevBurst = 0
            out_varianceBurst = 0
            out_maxBurst = 0
            out_kurtosisBurst = 0
            out_skewBurst = 0
            out_p10Burst = 0
            out_p20Burst = 0
            out_p30Burst = 0
            out_p40Burst = 0
            out_p50Burst = 0
            out_p60Burst = 0
            out_p70Burst = 0
            out_p80Burst = 0
            out_p90Burst = 0

        try:
            ########################################################################
            # Statistical indicators for Outgoing bytes (sliced intervals)
            out_meanBurstBytes = np.mean(self.out_burst_sizes)
            out_stdevBurstBytes = np.std(self.out_burst_sizes)
            out_varianceBurstBytes = np.var(self.out_burst_sizes)
            out_kurtosisBurstBytes = kurtosis(self.out_burst_sizes)
            out_skewBurstBytes = skew(self.out_burst_sizes)
            out_maxBurstBytes = np.amax(self.out_burst_sizes)
            out_minBurstBytes = np.amin(self.out_burst_sizes)
            out_p10BurstBytes = np.percentile(self.out_burst_sizes, 10)
            out_p20BurstBytes = np.percentile(self.out_burst_sizes, 20)
            out_p30BurstBytes = np.percentile(self.out_burst_sizes, 30)
            out_p40BurstBytes = np.percentile(self.out_burst_sizes, 40)
            out_p50BurstBytes = np.percentile(self.out_burst_sizes, 50)
            out_p60BurstBytes = np.percentile(self.out_burst_sizes, 60)
            out_p70BurstBytes = np.percentile(self.out_burst_sizes, 70)
            out_p80BurstBytes = np.percentile(self.out_burst_sizes, 80)
            out_p90BurstBytes = np.percentile(self.out_burst_sizes, 90)

        except Exception as e:
            print("Error in block 8 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            
            #raise StatisticalFeaturesException(path)

            out_meanBurstBytes = 0
            out_stdevBurstBytes = 0
            out_varianceBurstBytes = 0
            out_kurtosisBurstBytes = 0
            out_skewBurstBytes = 0
            out_maxBurstBytes = 0
            out_minBurstBytes = 0
            out_p10BurstBytes = 0
            out_p20BurstBytes = 0
            out_p30BurstBytes = 0
            out_p40BurstBytes = 0
            out_p50BurstBytes = 0
            out_p60BurstBytes = 0
            out_p70BurstBytes = 0
            out_p80BurstBytes = 0
            out_p90BurstBytes = 0

        try:
            ########################################################################
            # Statistical indicators for Incoming bursts
            in_totalBursts = len(self.in_bursts_packets)
            in_meanBurst = np.mean(self.in_bursts_packets)
            in_stdevBurst = np.std(self.in_bursts_packets)
            in_varianceBurst = np.var(self.in_bursts_packets)
            in_maxBurst = np.amax(self.in_bursts_packets)
            in_kurtosisBurst = kurtosis(self.in_bursts_packets)
            in_skewBurst = skew(self.in_bursts_packets)
            in_p10Burst = np.percentile(self.in_bursts_packets, 10)
            in_p20Burst = np.percentile(self.in_bursts_packets, 20)
            in_p30Burst = np.percentile(self.in_bursts_packets, 30)
            in_p40Burst = np.percentile(self.in_bursts_packets, 40)
            in_p50Burst = np.percentile(self.in_bursts_packets, 50)
            in_p60Burst = np.percentile(self.in_bursts_packets, 60)
            in_p70Burst = np.percentile(self.in_bursts_packets, 70)
            in_p80Burst = np.percentile(self.in_bursts_packets, 80)
            in_p90Burst = np.percentile(self.in_bursts_packets, 90)

        except Exception as e:
            print("Error in block 9 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
            
            in_totalBursts = 0
            in_meanBurst = 0
            in_stdevBurst = 0
            in_varianceBurst = 0
            in_maxBurst = 0
            in_kurtosisBurst = 0
            in_skewBurst = 0
            in_p10Burst = 0
            in_p20Burst = 0
            in_p30Burst = 0
            in_p40Burst = 0
            in_p50Burst = 0
            in_p60Burst = 0
            in_p70Burst = 0
            in_p80Burst = 0
            in_p90Burst = 0
            #raise StatisticalFeaturesException(path)

        try:
            ########################################################################
            # Statistical indicators for Incoming burst bytes (sliced intervals)
            in_meanBurstBytes = np.mean(self.in_burst_sizes)
            in_stdevBurstBytes = np.std(self.in_burst_sizes)
            in_varianceBurstBytes = np.var(self.in_burst_sizes)
            in_kurtosisBurstBytes = kurtosis(self.in_burst_sizes)
            in_skewBurstBytes = skew(self.in_burst_sizes)
            in_maxBurstBytes = np.amax(self.in_burst_sizes)
            in_minBurstBytes = np.amin(self.in_burst_sizes)
            in_p10BurstBytes = np.percentile(self.in_burst_sizes, 10)
            in_p20BurstBytes = np.percentile(self.in_burst_sizes, 20)
            in_p30BurstBytes = np.percentile(self.in_burst_sizes, 30)
            in_p40BurstBytes = np.percentile(self.in_burst_sizes, 40)
            in_p50BurstBytes = np.percentile(self.in_burst_sizes, 50)
            in_p60BurstBytes = np.percentile(self.in_burst_sizes, 60)
            in_p70BurstBytes = np.percentile(self.in_burst_sizes, 70)
            in_p80BurstBytes = np.percentile(self.in_burst_sizes, 80)
            in_p90BurstBytes = np.percentile(self.in_burst_sizes, 90)
        
        except Exception as e:
            print("Error in block 10 when processing " + path)
            print("Skipping sample")
            print(e)
            traceback.print_exc()
           
            #raise StatisticalFeaturesException(path)

            in_meanBurstBytes = 0
            in_stdevBurstBytes = 0
            in_varianceBurstBytes = 0
            in_kurtosisBurstBytes = 0
            in_skewBurstBytes = 0
            in_maxBurstBytes = 0
            in_minBurstBytes = 0
            in_p10BurstBytes = 0
            in_p20BurstBytes = 0
            in_p30BurstBytes = 0
            in_p40BurstBytes = 0
            in_p50BurstBytes = 0
            in_p60BurstBytes = 0
            in_p70BurstBytes = 0
            in_p80BurstBytes = 0
            in_p90BurstBytes = 0

        f_values_stats = []

        od_dict = collections.OrderedDict(sorted(self.bin_dict.items(), key=lambda t: float(t[0])))
        bin_list = []
        for i in od_dict:
            bin_list.append(od_dict[i])

        od_dict2 = collections.OrderedDict(sorted(self.bin_dict2.items(), key=lambda t: float(t[0])))
        bin_list2 = []
        for i in od_dict2:
            bin_list2.append(od_dict2[i])

        ###################################################################
        # Global Packet Features
        f_values_stats.append(self.totalPackets)
        f_values_stats.append(self.totalPacketsIn)
        f_values_stats.append(self.totalPacketsOut)
        f_values_stats.append(self.totalBytes)
        f_values_stats.append(self.totalBytesIn)
        f_values_stats.append(self.totalBytesOut)

        ###################################################################
        # Packet Length Features
        f_values_stats.append(minPacketSize)
        f_values_stats.append(maxPacketSize)
        f_values_stats.append(meanPacketSizes)
        f_values_stats.append(stdevPacketSizes)
        f_values_stats.append(variancePacketSizes)
        f_values_stats.append(kurtosisPacketSizes)
        f_values_stats.append(skewPacketSizes)

        f_values_stats.append(p10PacketSizes)
        f_values_stats.append(p20PacketSizes)
        f_values_stats.append(p30PacketSizes)
        f_values_stats.append(p40PacketSizes)
        f_values_stats.append(p50PacketSizes)
        f_values_stats.append(p60PacketSizes)
        f_values_stats.append(p70PacketSizes)
        f_values_stats.append(p80PacketSizes)
        f_values_stats.append(p90PacketSizes)

        ###################################################################
        # Packet Length Features (in)
        f_values_stats.append(minPacketSizeIn)
        f_values_stats.append(maxPacketSizeIn)
        f_values_stats.append(meanPacketSizesIn)
        f_values_stats.append(stdevPacketSizesIn)
        f_values_stats.append(variancePacketSizesIn)
        f_values_stats.append(skewPacketSizesIn)
        f_values_stats.append(kurtosisPacketSizesIn)

        f_values_stats.append(p10PacketSizesIn)
        f_values_stats.append(p20PacketSizesIn)
        f_values_stats.append(p30PacketSizesIn)
        f_values_stats.append(p40PacketSizesIn)
        f_values_stats.append(p50PacketSizesIn)
        f_values_stats.append(p60PacketSizesIn)
        f_values_stats.append(p70PacketSizesIn)
        f_values_stats.append(p80PacketSizesIn)
        f_values_stats.append(p90PacketSizesIn)

        ###################################################################
        # Packet Length Features (out)
        f_values_stats.append(minPacketSizeOut)
        f_values_stats.append(maxPacketSizeOut)
        f_values_stats.append(meanPacketSizesOut)
        f_values_stats.append(stdevPacketSizesOut)
        f_values_stats.append(variancePacketSizesOut)
        f_values_stats.append(skewPacketSizesOut)
        f_values_stats.append(kurtosisPacketSizesOut)

        f_values_stats.append(p10PacketSizesOut)
        f_values_stats.append(p20PacketSizesOut)
        f_values_stats.append(p30PacketSizesOut)
        f_values_stats.append(p40PacketSizesOut)
        f_values_stats.append(p50PacketSizesOut)
        f_values_stats.append(p60PacketSizesOut)
        f_values_stats.append(p70PacketSizesOut)
        f_values_stats.append(p80PacketSizesOut)
        f_values_stats.append(p90PacketSizesOut)

        ###################################################################
        # Packet Timing Features
        f_values_stats.append(maxIPT)
        f_values_stats.append(minIPT)
        f_values_stats.append(meanPacketTimes)
        f_values_stats.append(stdevPacketTimes)
        f_values_stats.append(variancePacketTimes)
        f_values_stats.append(kurtosisPacketTimes)
        f_values_stats.append(skewPacketTimes)

        f_values_stats.append(p10PacketTimes)
        f_values_stats.append(p20PacketTimes)
        f_values_stats.append(p30PacketTimes)
        f_values_stats.append(p40PacketTimes)
        f_values_stats.append(p50PacketTimes)
        f_values_stats.append(p60PacketTimes)
        f_values_stats.append(p70PacketTimes)
        f_values_stats.append(p80PacketTimes)
        f_values_stats.append(p90PacketTimes)

        ###################################################################
        # Packet Timing Features (in)
        f_values_stats.append(minPacketTimesIn)
        f_values_stats.append(maxPacketTimesIn)
        f_values_stats.append(meanPacketTimesIn)
        f_values_stats.append(stdevPacketTimesIn)
        f_values_stats.append(variancePacketTimesIn)
        f_values_stats.append(skewPacketTimesIn)
        f_values_stats.append(kurtosisPacketTimesIn)

        f_values_stats.append(p10PacketTimesIn)
        f_values_stats.append(p20PacketTimesIn)
        f_values_stats.append(p30PacketTimesIn)
        f_values_stats.append(p40PacketTimesIn)
        f_values_stats.append(p50PacketTimesIn)
        f_values_stats.append(p60PacketTimesIn)
        f_values_stats.append(p70PacketTimesIn)
        f_values_stats.append(p80PacketTimesIn)
        f_values_stats.append(p90PacketTimesIn)

        ###################################################################
        # Packet Timing Features (out)
        f_values_stats.append(minPacketTimesOut)
        f_values_stats.append(maxPacketTimesOut)
        f_values_stats.append(meanPacketTimesOut)
        f_values_stats.append(stdevPacketTimesOut)
        f_values_stats.append(variancePacketTimesOut)
        f_values_stats.append(skewPacketTimesOut)
        f_values_stats.append(kurtosisPacketTimesOut)

        f_values_stats.append(p10PacketTimesOut)
        f_values_stats.append(p20PacketTimesOut)
        f_values_stats.append(p30PacketTimesOut)
        f_values_stats.append(p40PacketTimesOut)
        f_values_stats.append(p50PacketTimesOut)
        f_values_stats.append(p60PacketTimesOut)
        f_values_stats.append(p70PacketTimesOut)
        f_values_stats.append(p80PacketTimesOut)
        f_values_stats.append(p90PacketTimesOut)

        #################################################################
        # Outgoing Packet number of Bursts features
        f_values_stats.append(out_totalBursts)
        f_values_stats.append(out_maxBurst)
        f_values_stats.append(out_meanBurst)
        f_values_stats.append(out_stdevBurst)
        f_values_stats.append(out_varianceBurst)
        f_values_stats.append(out_kurtosisBurst)
        f_values_stats.append(out_skewBurst)

        f_values_stats.append(out_p10Burst)
        f_values_stats.append(out_p20Burst)
        f_values_stats.append(out_p30Burst)
        f_values_stats.append(out_p40Burst)
        f_values_stats.append(out_p50Burst)
        f_values_stats.append(out_p60Burst)
        f_values_stats.append(out_p70Burst)
        f_values_stats.append(out_p80Burst)
        f_values_stats.append(out_p90Burst)

        #################################################################
        # Outgoing Packet Bursts data size features
        f_values_stats.append(out_maxBurstBytes)
        f_values_stats.append(out_minBurstBytes)
        f_values_stats.append(out_meanBurstBytes)
        f_values_stats.append(out_stdevBurstBytes)
        f_values_stats.append(out_varianceBurstBytes)
        f_values_stats.append(out_kurtosisBurstBytes)
        f_values_stats.append(out_skewBurstBytes)

        f_values_stats.append(out_p10BurstBytes)
        f_values_stats.append(out_p20BurstBytes)
        f_values_stats.append(out_p30BurstBytes)
        f_values_stats.append(out_p40BurstBytes)
        f_values_stats.append(out_p50BurstBytes)
        f_values_stats.append(out_p60BurstBytes)
        f_values_stats.append(out_p70BurstBytes)
        f_values_stats.append(out_p80BurstBytes)
        f_values_stats.append(out_p90BurstBytes)

        #################################################################
        # Incoming Packet number of Bursts features
        f_values_stats.append(in_totalBursts)
        f_values_stats.append(in_maxBurst)
        f_values_stats.append(in_meanBurst)
        f_values_stats.append(in_stdevBurst)
        f_values_stats.append(in_varianceBurst)
        f_values_stats.append(in_kurtosisBurst)
        f_values_stats.append(in_skewBurst)

        f_values_stats.append(in_p10Burst)
        f_values_stats.append(in_p20Burst)
        f_values_stats.append(in_p30Burst)
        f_values_stats.append(in_p40Burst)
        f_values_stats.append(in_p50Burst)
        f_values_stats.append(in_p60Burst)
        f_values_stats.append(in_p70Burst)
        f_values_stats.append(in_p80Burst)
        f_values_stats.append(in_p90Burst)

        #################################################################
        # Incoming Packet Bursts data size features
        f_values_stats.append(in_maxBurstBytes)
        f_values_stats.append(in_minBurstBytes)
        f_values_stats.append(in_meanBurstBytes)
        f_values_stats.append(in_stdevBurstBytes)
        f_values_stats.append(in_varianceBurstBytes)
        f_values_stats.append(in_kurtosisBurstBytes)
        f_values_stats.append(in_skewBurstBytes)

        f_values_stats.append(in_p10BurstBytes)
        f_values_stats.append(in_p20BurstBytes)
        f_values_stats.append(in_p30BurstBytes)
        f_values_stats.append(in_p40BurstBytes)
        f_values_stats.append(in_p50BurstBytes)
        f_values_stats.append(in_p60BurstBytes)
        f_values_stats.append(in_p70BurstBytes)
        f_values_stats.append(in_p80BurstBytes)
        f_values_stats.append(in_p90BurstBytes)
        
        # Write Stats csv
        f_values_stats.append(label_source)
        f_values_stats.append(capture)

        return f_values_stats


class SumoFeatures:

    #sessionFeaturesCollection : dict[str, SessionFeatures]
    # Each pair is a combination of 2 SessionFeatures objects, one for the client session,
    # and a second one for the OS session
    pairsFolders : dict[str, object]
    dataset_name : str
    datasetPath : str
    ips : dict[str, str]
    clientPath : str
    onionPath : str
    topPath : str

    def __init__(self, dataset_name, datasetPath):
        self.pairsFolders = {}
        self.dataset_name = dataset_name
        self.datasetPath = datasetPath
        self.ips = {}
        self.clientPath = "client"
        self.onionPath = "onion"
        self.topPath = f"/mnt/nas-shared/torpedo/extracted_features_{dataset_name}"


    def get_ips_from_inventory(self):
        inventory_path = self.datasetPath + '../inventory.cfg'
        print("\n#### inventory_path", inventory_path)
        if not os.path.isfile(inventory_path):
            raise MissingInventoryException
        inventory_manager = InventoryManager(loader=None, sources=inventory_path)
        hosts = inventory_manager.get_hosts()
        for host in hosts:
            if host.get_name().startswith('client-') or host.get_name().startswith('os-'):
                self.ips[host.get_name()] = host.vars['static_docker_ip']

        print("self.ips", self.ips)


    def set_pairsFolders(self, pairsFolders):
        self.pairsFolders = pairsFolders


    def get_extracted_features_folder(self, targetPath, targetFolder, sessionId):
        return "%s/%s/%s/%s/"%(self.topPath, targetPath, targetFolder, sessionId)


    def process_alexa_features(self, session_path):
        #print("--- Session {}".format(session_path))
        session_features = SessionFeatures()
        try:
            session_features.process_packets(session_path, self.ips, isClient=True, isAlexa=True)
        except InvalidPcapException:
            print("InvalidPcapException")
            return None
        except Exception as e:
            traceback.print_exc()
            return None

        if len(session_features.packetTimesInAbs) == 0 or len(session_features.packetSizesIn) == 0:
            return None

        try:
            f_values_stats = session_features.get_statistical_features(session_path, LABEL_IS_CLIENT_SIDE, session_path.split('/')[-1])
        except StatisticalFeaturesException:
            print("StatisticalFeaturesException")
            traceback.print_exc()
            return None
        except Exception as e:
            traceback.print_exc()
            return None
        
        folderDict = {}
        folderDict['clientFolder'] = session_features.clientFolder
        folderDict['hsFolder'] = session_features.hsFolder
        folderDict['clientSessionId'] = session_features.sessionId
        folderDict['clearwebAddress'] = session_features.clearwebAddress
        folderDict['clientMetaStats']  = {}
        folderDict['clientMetaStats']['initialTimestamp'] = session_features.first_ts
        folderDict['clientMetaStats']['lastTimestamp'] = session_features.last_ts
        folderDict['clientFlow'] = {}
        folderDict['clientFlow']['timesIn'] = session_features.packetTimesIn
        folderDict['clientFlow']['timesOut'] = session_features.packetTimesOut
        folderDict['clientFlow']['timesInRel'] = session_features.packetTimesInRel
        folderDict['clientFlow']['timesOutRel'] = session_features.packetTimesOutRel
        folderDict['clientFlow']['timesInAbs'] = session_features.packetTimesInAbs
        folderDict['clientFlow']['timesOutAbs'] = session_features.packetTimesOutAbs
        folderDict['clientFlow']['sizesIn'] = session_features.packetSizesIn
        folderDict['clientFlow']['sizesOut'] = session_features.packetSizesOut
        folderDict['clientFeatures'] = session_features.specialFeatures

        extracted_features_folder = self.get_extracted_features_folder(self.clientPath, session_features.clientFolder, session_features.sessionId)
        
        #print("====== ALEXA client folder {}".format(extracted_features_folder))
            
        if not os.path.exists(extracted_features_folder):
            #print("==== CREATE extracted_features_folder {}".format(extracted_features_folder))
            os.makedirs(extracted_features_folder)

        pickle.dump(folderDict, open(extracted_features_folder + "folderDict.pickle", "wb"))

        del folderDict

        # Label indicates that session is not issued to an OS, but to a clearweb site
        return f_values_stats, LABEL_IS_NOT_SESSION_TO_OS


    def extract_alexa_features_parallel(self, alexaCaptures, arff_path_stats_source):
        print("\n--- ALEXA")
    #def extract_alexa_features_parallel(self, alexaCaptures):
        #alexa_captures_part = alexaCaptures[ : 100]
        with tqdm.tqdm(total=len(alexaCaptures)) as pbar:
            results = Parallel(n_jobs=NUM_CORES)(delayed(self.process_alexa_features)(session_path[0]) for session_path in tzip(alexaCaptures, leave=False))
        #with tqdm.tqdm(total=len(alexa_captures_part)) as pbar:
        #    results = Parallel(n_jobs=NUM_CORES)(delayed(self.process_alexa_features)(session_path[0]) for session_path in tzip(alexa_captures_part, leave=False))
            pbar.update()

        f_values_stats = [result[0] for result in results if result is not None]
        labels_target = [result[2] for result in results if result is not None]

        stats_source_pd_alexa = pd.DataFrame(f_values_stats)
        stats_source_pd_alexa.to_csv(arff_path_stats_source, mode='a', index=False, header=False)
        
        return [result for result in labels_target if result is not None]


    def process_client_features(self, session_path):
        #print("--- Session {}".format(session_path))
        session_features = SessionFeatures()
        try:
            # TODO: Remove self.ips from the arguments
            session_features.process_packets(session_path, self.ips, isClient=True, isAlexa=False)
        except InvalidPcapException:
            print("InvalidPcapException")
            return None
        except Exception as e:
            traceback.print_exc()
            return None

        try:
            f_values_stats = session_features.get_statistical_features(session_path, LABEL_IS_CLIENT_SIDE, session_path.split('/')[-1])
        except StatisticalFeaturesException:
            print("StatisticalFeaturesException")
            traceback.print_exc()
            return None
        except Exception as e:
            traceback.print_exc()
            return None

        if len(session_features.packetTimesInAbs) == 0 or len(session_features.packetSizesIn) == 0:
            return None
            
        folderDict = {}
        folderDict['clientFolder'] = session_features.clientFolder
        folderDict['hsFolder'] = session_features.hsFolder
        folderDict['clientSessionId'] = session_features.sessionId
        folderDict['onionAddress'] = session_features.onionUrl
        folderDict['clientMetaStats']  = {}
        folderDict['clientMetaStats']['initialTimestamp'] = session_features.first_ts
        folderDict['clientMetaStats']['lastTimestamp'] = session_features.last_ts
        folderDict['clientFlow'] = {}
        folderDict['clientFlow']['timesIn'] = session_features.packetTimesIn
        folderDict['clientFlow']['timesOut'] = session_features.packetTimesOut
        folderDict['clientFlow']['timesInRel'] = session_features.packetTimesInRel
        folderDict['clientFlow']['timesOutRel'] = session_features.packetTimesOutRel
        folderDict['clientFlow']['timesInAbs'] = session_features.packetTimesInAbs
        folderDict['clientFlow']['timesOutAbs'] = session_features.packetTimesOutAbs
        folderDict['clientFlow']['sizesIn'] = session_features.packetSizesIn
        folderDict['clientFlow']['sizesOut'] = session_features.packetSizesOut
        folderDict['clientFeatures'] = session_features.specialFeatures

        extracted_features_folder = self.get_extracted_features_folder(self.clientPath, session_features.clientFolder, session_features.sessionId)

        #print("====== client folder {}".format(extracted_features_folder))
            
        if not os.path.exists(extracted_features_folder):
            #print("==== CREATE extracted_features_folder {}".format(extracted_features_folder))
            os.makedirs(extracted_features_folder)

        pickle.dump(folderDict, open(extracted_features_folder + "folderDict.pickle", "wb"))

        del folderDict

        # Label indicates that session is issued to an OS
        return f_values_stats, LABEL_IS_SESSION_TO_OS

    
    def extract_client_features_parallel(self, clientCaptures, arff_path_stats_source, arff_path_stats_target, labels_target_alexa):
        print("\n--- CLIENT")
        labels_target = labels_target_alexa
        #client_captures_part = clientCaptures[ : 100]
        #print("\n--- clientCaptures[ : 100]",  clientCaptures[ : 100])
        with tqdm.tqdm(total=len(clientCaptures)) as pbar:
            results = Parallel(n_jobs=NUM_CORES)(delayed(self.process_client_features)(session_path[0]) for session_path in tzip(clientCaptures, leave=False))
        #with tqdm.tqdm(total=len(client_captures_part)) as pbar:
        #    results = Parallel(n_jobs=NUM_CORES)(delayed(self.process_client_features)(session_path[0]) for session_path in tzip(client_captures_part, leave=False))
            pbar.update()

        print("AFTER CLIENT RESULTS!", len(results))

        f_values_stats = [result[0] for result in results if result is not None]
        labels_target += [result[2] for result in results if result is not None]


        stats_source_pd_client = pd.DataFrame(f_values_stats)
        stats_source_pd_client.to_csv(arff_path_stats_source, mode='a', index=False, header=False)

        # Copy extracted source separation features to the target features
        stats_target_pd = pd.read_csv(arff_path_stats_source, low_memory=False)
        
        # This is required so that the index is not confused with a wrong length of the list
        labels_series = pd.Series(labels_target)
        stats_target_pd['Class'] = labels_series

        # Write target separation features with only the alexa and client features
        stats_target_pd.to_csv(arff_path_stats_target)

    
    def process_onion_features(self, session_path):
        session_features = SessionFeatures()
        try:
            session_features.process_packets(session_path, self.ips, isClient=False, isAlexa=False)
        except InvalidPcapException:
            print("InvalidPcapException")
            return None
        except Exception as e:
            traceback.print_exc()
            return None

        try:
            f_values_stats = session_features.get_statistical_features(session_path, LABEL_IS_OS_SIDE, session_path.split('/')[-1])
        except StatisticalFeaturesException:
            print("StatisticalFeaturesException")
            traceback.print_exc()
            return None
        except Exception as e:
            traceback.print_exc()
            return None

        if len(session_features.packetTimesOutAbs) == 0 or len(session_features.packetSizesOut) == 0:
            return None

        #print("====== OS client folder {}".format(session_path))

        folderDict = {}
        folderDict['hsSessionId'] = session_features.sessionId
        folderDict['hsMetaStats']  = {}
        folderDict['hsMetaStats']['initialTimestamp'] = session_features.first_ts
        folderDict['hsMetaStats']['lastTimestamp'] = session_features.last_ts
        folderDict['hsFlow'] = {}
        folderDict['hsFlow']['timesIn'] = session_features.packetTimesIn
        folderDict['hsFlow']['timesOut'] = session_features.packetTimesOut
        folderDict['hsFlow']['timesInRel'] = session_features.packetTimesInRel
        folderDict['hsFlow']['timesOutRel'] = session_features.packetTimesOutRel
        folderDict['hsFlow']['timesInAbs'] = session_features.packetTimesInAbs
        folderDict['hsFlow']['timesOutAbs'] = session_features.packetTimesOutAbs
        folderDict['hsFlow']['sizesIn'] = session_features.packetSizesIn
        folderDict['hsFlow']['sizesOut'] = session_features.packetSizesOut
        folderDict['hsFeatures'] = session_features.specialFeatures

        extracted_features_folder = self.get_extracted_features_folder(self.onionPath, session_features.hsFolder, session_features.sessionId)
        
        if not os.path.exists(extracted_features_folder):
            #print("==== CREATE extracted_features_folder {}".format(extracted_features_folder))
            os.makedirs(extracted_features_folder)
        
        pickle.dump(folderDict, open(extracted_features_folder + "folderDict.pickle", "wb"))

        del folderDict

        return f_values_stats


    def extract_onion_features_parallel(self, onionCaptures, arff_path_stats_source):
        print("\n--- ONIONS")
    #def extract_onion_features_parallel(self, onionCaptures):
        #onion_captures_part = onionCaptures[ : 100]
        with tqdm.tqdm(total=len(onionCaptures)) as pbar:
            results = Parallel(n_jobs=NUM_CORES)(delayed(self.process_onion_features)(session_path[0]) for session_path in tzip(onionCaptures, leave=False))
        #with tqdm.tqdm(total=len(onion_captures_part)) as pbar:
        #    results = Parallel(n_jobs=NUM_CORES)(delayed(self.process_onion_features)(session_path[0]) for session_path in tzip(onion_captures_part, leave=False))
            pbar.update()

        f_values_stats = [result[0] for result in results if result is not None]

        stats_source_pd_onion = pd.DataFrame(f_values_stats)
        stats_source_pd_onion.to_csv(arff_path_stats_source, mode='a', index=False, header=False)
    

    def extract(self, alexaCaptures, clientCaptures, onionCaptures):
        self.get_ips_from_inventory()
        if not os.path.exists(self.topPath):
            os.makedirs(self.topPath)
        # client-small-ostrain-5-client4_os4-os-small-ostrain-4_shopskagjfzhrt4isgdaahmq2br75nrmicxar2xhhj7tpdjwycuv4hid_session_911_client.pcap
        arff_path_stats_source = f'{self.topPath}/stats_source_separation.csv'

        arff_path_stats_target = f'{self.topPath}/stats_target_separation.csv'

        stats_columns = get_headers()
        stats_source_pd = pd.DataFrame(columns=stats_columns)
        stats_source_pd.to_csv(arff_path_stats_source, mode='w', index=False)

        labels_target = self.extract_alexa_features_parallel(alexaCaptures, arff_path_stats_source)
        self.extract_client_features_parallel(clientCaptures, arff_path_stats_source, arff_path_stats_target, labels_target)
        self.extract_onion_features_parallel(onionCaptures, arff_path_stats_source)