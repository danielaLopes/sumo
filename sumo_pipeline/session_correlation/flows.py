from typing import List, Dict, Tuple, Literal
from datetime import datetime


class Flow:
    first_ts: datetime
    last_ts: datetime

    def __init__(self, first_ts, last_ts):
        self.first_ts = first_ts
        self.last_ts = last_ts

    def get_duration(self):
        return self.last_ts - self.first_ts

class ClientFlow(Flow):
    packet_count_in: list[int] # Is already the bucketized version
    packet_times_in: list[datetime]

    def __init__(self, first_ts, last_ts, packet_count_in, packet_times_in):
        super().__init__(first_ts, last_ts)
        self.packet_count_in = packet_count_in
        self.packet_times_in = packet_times_in

class OnionFlow(Flow):
    # In the onion side, we cannot have the bucketized 
    # version right away because it'll depend on the 
    # client flow that it is combined with
    packet_times_out: list[datetime]

    def __init__(self, first_ts, last_ts, packet_times_out):
        super().__init__(first_ts, last_ts)
        self.packet_times_out = packet_times_out

def generate_buckets_epochs(initial_ts, last_ts, time_sampling_interval):
    initial_bucket = 0
    last_bucket = int(((last_ts - initial_ts) * 1000) // time_sampling_interval) + 1
    
    dictionary = {i: 0 for i in range(initial_bucket, last_bucket + 1)}

    return dictionary

def get_windows_count(n_buckets, buckets_per_window, buckets_overlap):
    if buckets_overlap == 0:
        return (n_buckets - buckets_per_window) // buckets_per_window + 1
    else:
        return (n_buckets - buckets_per_window) // buckets_overlap + 1

class FlowPair:
    bucketized_client_packets_in: list[int]
    bucketized_onion_packets_out: list[int]
    session_buckets: int
    session_windows: int
    label: Literal[0, 1]

    def __init__(self, client_flow: ClientFlow, 
                 onion_flow: OnionFlow, 
                 time_sampling_interval: int,
                 buckets_per_window: int,
                 buckets_overlap: int,
                 label: Literal[0, 1]):
        initial_session_ts = min(client_flow.first_ts, onion_flow.first_ts)
        final_session_ts = max(client_flow.last_ts, onion_flow.last_ts)

        bucketized_onion_packets_out_dict = generate_buckets_epochs(initial_session_ts, final_session_ts, time_sampling_interval)
        bucketized_client_packets_in_dict = generate_buckets_epochs(initial_session_ts, final_session_ts, time_sampling_interval)
        
        # onion
        onion_buckets = [(ts - initial_session_ts) * 1000 // time_sampling_interval for ts in onion_flow.packet_times_out]
        for onion_bucket in onion_buckets:
            bucketized_onion_packets_out_dict[onion_bucket] += 1

        # client
        client_buckets = [(ts - initial_session_ts) * 1000 // time_sampling_interval for ts in client_flow.packet_times_in]
        for client_bucket in client_buckets:
            bucketized_client_packets_in_dict[client_bucket] += 1

        self.bucketized_client_packets_in = list(bucketized_client_packets_in_dict.values())
        self.bucketized_onion_packets_out = list(bucketized_onion_packets_out_dict.values())
        
        self.session_buckets = len(self.bucketized_client_packets_in)
        self.session_windows = get_windows_count(self.session_buckets, buckets_per_window, buckets_overlap)

        self.label = label
