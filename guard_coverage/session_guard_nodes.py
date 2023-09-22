
# %%
import joblib
import tqdm
from scapy.all import *
import glob
from ansible.inventory.manager import InventoryManager
import requests

import query_sumo_dataset

# %%
CAPTURES_FOLDER = '/mnt/nas-shared/torpedo/datasets_20230521/OSTest/experiment_results/'
#ONION_FOLDER = CAPTURES_FOLDER+'TrafficCapturesClient'
#CLIENT_FOLDER = CAPTURES_FOLDER+'TrafficCapturesOnion'

RESULTS_FILE = "./results/data/guard_nodes.joblib"

# %%
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

# %%
def filter_empty_ack(pkt):
    if pkt[TCP].flags & 0x10 and len(pkt[TCP].payload) == 0:
        ip_len = pkt[IP].len
        ip_hdr_len = pkt[IP].ihl * 4
        tcp_hdr_len = pkt[TCP].dataofs * 4
        payload_len = ip_len - ip_hdr_len - tcp_hdr_len
        if payload_len == 0:
            return True
        
    return False

# %%
class MissingInventoryException(Exception):
    "The dataset does not contain file inventory.cfg"  

class InvalidPcapException(Exception):
    def __init__(self, path):  
        message = "No data could be read!"         
        super().__init__(f'{message} : {path}')

class NoGuardFoundException(Exception):
    def __init__(self, path):  
        message = "Could not retrieve guard nodes from session!"         
        super().__init__(f'{message} : {path}')

class RelayGeolocationException(Exception):
    def __init__(self, message):
        super().__init__(message)

# %%
# Limit request rate
from ratelimit import limits, RateLimitException, sleep_and_retry
from backoff import on_exception, expo
ONE_MINUTE = 60
#MAX_CALLS_PER_MINUTE = 30
MAX_CALLS_PER_MINUTE = 15

@on_exception(expo, RateLimitException, max_tries=20)
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def get_geolocation(ip_address):
    url = f"http://ip-api.com/json/{ip_address}" # Limited by 45 requests per minute
    response = requests.get(url)
    if response.status_code != 200:
        raise RelayGeolocationException(f"Could not fetch geolocation for {ip_address} due to {response.status_code} error")
    data = response.json()

    return data['countryCode'], data['isp'], data['as']

# %%
class Session:
    session_id: str
    client_guard_ip: str
    onion_guard_ip: str
    client_guard_country_code: str
    onion_guard_country_code: str
    client_guard_as: str
    onion_guard_as: str
    client_guard_isp: str
    onion_guard_isp: str
    client_machine_ip: str
    onion_machine_ip: str
    client_capture_path: str
    onion_capture_path: str

    def __init__(self, session_id, client_capture_path, onion_capture_path, ips):
        self.session_id = session_id
        self.client_capture_path = client_capture_path
        self.onion_capture_path = onion_capture_path

        self.get_guard_node(client_capture_path, ips, isClient=True)
        self.get_guard_node(onion_capture_path, ips, isClient=False)

        self.client_guard_country_code, self.client_guard_isp, self.client_guard_as = get_geolocation(self.client_guard_ip)
        self.onion_guard_country_code, self.onion_guard_isp, self.onion_guard_as = get_geolocation(self.onion_guard_ip)

    def __repr__(self):
        return f"Client guard:\n \
                    \tip {self.client_guard_ip}; country: {self.client_guard_country_code}; as: {self.client_guard_as}; isp: {self.client_guard_isp}\n \
                    Onion guard:\n \
                    \tip {self.onion_guard_ip}; country: {self.onion_guard_country_code}; as: {self.onion_guard_as}; isp: {self.onion_guard_isp}\n"

    """
    Our policy to determine the guard node ip is to choose the IP 
    with which were exchanged more packets

    """
    def get_guard_node(self, path, ips, isClient=True):
        try:
            # The capture file is not entirely loaded into memory at the same time, 
            # only a packet each time, making it more memory efficient
            cap = PcapReader(path)
        except Exception as e:
            print("Problem parsing pcap {}".format(path))
            print(e)
            traceback.print_exc()
            raise InvalidPcapException(path)

        pcap_name = path.split('/')[-1]

        if isClient:
            machine_name = re.search(r'client(?:-[a-zA-Z]+)?-[a-zA-Z]+-[0-9]+', pcap_name).group()
            self.client_machine_ip = "172." + str(ips[machine_name])
            ip_address = self.client_machine_ip
        else:
            machine_name = re.search(r'os(?:-[a-zA-Z]+)?-[a-zA-Z]+(?:-[0-9]+)?', pcap_name).group()
            self.onion_machine_ip = "172." + str(ips[machine_name])
            ip_address = self.onion_machine_ip

        # ip: counter
        guards = {}

        for i, pkt in enumerate(cap):
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
                    
                    # in packets
                    if ip_address in dst_ip_addr_str:
                        if src_ip_addr_str not in guards:
                            guards[src_ip_addr_str] = 0
                        guards[src_ip_addr_str] += 1

                    # out packets
                    elif ip_address in src_ip_addr_str:
                        if dst_ip_addr_str not in guards:
                            guards[dst_ip_addr_str] = 0
                        guards[dst_ip_addr_str] += 1
                
            except Exception as e:
                print("Corrupted packet")
                print(repr(e))
                print(e)
                traceback.print_exc()

        if isClient:
            if len(guards) == 0:
                raise NoGuardFoundException(path)
            else:
                self.client_guard_ip = max(guards, key=guards.get)
        else:
            if len(guards) == 0:
                raise NoGuardFoundException(path)
            else:
                self.onion_guard_ip = max(guards, key=guards.get)



# %%
def get_ips_from_inventory():
    ips = {}

    inventory_path = CAPTURES_FOLDER + '../inventory.cfg'
    print("\n#### inventory_path", inventory_path)
    if not os.path.isfile(inventory_path):
        raise MissingInventoryException
    inventory_manager = InventoryManager(loader=None, sources=inventory_path)
    hosts = inventory_manager.get_hosts()
    for host in hosts:
        if host.get_name().startswith('client-') or host.get_name().startswith('os-'):
            ips[host.get_name()] = host.vars['static_docker_ip']

    print("ips", ips)

    return ips

def main():
    dataset = query_sumo_dataset.SumoDataset(CAPTURES_FOLDER)
    client_paths = dataset.client_sessions_paths_to_oses_all()
    onion_paths = dataset.onion_sessions_paths_all()

    try:
        session_paths_to_remove = dataset.filter_sessions_with_failed_requests()
    except query_sumo_dataset.MissingFailedRequestsLogException:
        print("No failed_requests.log files in the dataset")

    client_paths = [x for x in client_paths if x not in session_paths_to_remove]

    failed_session_ids = []
    onion_paths_without_failed = {}
    for client_path in session_paths_to_remove:
        session_id = query_sumo_dataset.get_session_id_from_path(client_path)
        failed_session_ids.append(session_id)
    for onion_path in onion_paths:
        session_id = query_sumo_dataset.get_session_id_from_path(onion_path)
        if session_id not in failed_session_ids:
            onion_paths_without_failed[session_id] = onion_path
    onion_paths_mapping = onion_paths_without_failed

    print("===== len(client_paths) after removing failed sessions", len(client_paths))
    print("===== len(onion_paths_mapping) after removing failed sessions", len(onion_paths_mapping))

    ips = get_ips_from_inventory()

    print("\n0:", client_paths[:10])
    print("\n1:", list(onion_paths_mapping.keys())[:10])


    start = 0
    if os.path.isfile(RESULTS_FILE):
        guard_nodes = joblib.load(RESULTS_FILE)
        start = len(guard_nodes)
    else:
        guard_nodes = []
    
    print(f"\n---- Starting at client path {start}")

    for client_path_index in tqdm.tqdm(range(start, len(client_paths))):
        client_capture_path = client_paths[client_path_index]
        #print("client_capture_path", client_capture_path)
        client_session_id = query_sumo_dataset.get_session_id_from_path(client_capture_path)
        #print("client_session_id", client_session_id)
        if client_session_id not in onion_paths_mapping:
            continue
        onion_capture_path = onion_paths_mapping[client_session_id]
        #print("onion_capture_path", onion_capture_path)
        try:
            session = Session(client_session_id, client_capture_path, onion_capture_path, ips)
            guard_nodes.append(session)
        except Exception as e:
            # Could not find guard nodes
            traceback.print_exc()
            continue
        
        joblib.dump(guard_nodes, RESULTS_FILE)

    #guard_nodes = joblib.load(RESULTS_FILE)

if __name__ == "__main__":
    main()



