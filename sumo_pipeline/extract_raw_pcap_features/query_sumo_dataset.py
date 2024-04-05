#!/bin/env pyhton
import os
import sys
import re
import traceback
import glob


class MissingFailedRequestsLogException(Exception):
    "The dataset does not contain files failed_requests.log"         

class InvalidClientCaptureNameException(Exception):
    def __init__(self, path):  
        message = "The following capture path does not contain a valid client name"         
        super().__init__(f'{message} : {path}')

class InvalidOnionCaptureNameException(Exception):
    def __init__(self, path):  
        message = "The following capture path does not contain a valid onion name"         
        super().__init__(f'{message} : {path}')


class InvalidAlexaCaptureNameException(Exception):
    def __init__(self, path):  
        message = "The following capture path does not contain a valid clearwebsite name"         
        super().__init__(f'{message} : {path}')

        # print("    ", dataset.get_client_request(sys.argv[idx+1], int(sys.argv[idx+2]), int(sys.argv[idx+3])))

class SumoException(Exception):
    "Raise this if some exception related with the dataset happens"
    pass

class PathInvalidException(SumoException):
    pass

class InvalidDatasetException(SumoException):
    pass

def get_client_name(capture_path):
    client = re.search("^(client-.*-client[0-9]+)_", capture_path)
    if client:
        return client.group(1)
    else: 
        raise InvalidClientCaptureNameException(capture_path)

def get_onion_name(capture_path):
    onion = re.search("_(os[0-9]+-os.*)_(.*)_session.*", capture_path)
    if onion:
        return onion.group(1)
    else: 
        raise InvalidOnionCaptureNameException(capture_path)

def get_alexa_name(capture_path):
    alexa = re.search("_alexa_(.*)_session.*\\.pcap", capture_path)
    if alexa:
        return alexa.group(1)
    else: 
        raise InvalidAlexaCaptureNameException(capture_path)

def get_session_id_from_path(capture_path):
    pcap_name = capture_path.split('/')[-1]
    client_str = '_client.pcap'
    onion_str = '_hs.pcap'
    if client_str in pcap_name:
        return pcap_name.split(client_str)[0]
    else:
        return pcap_name.split(onion_str)[0]

class DatasetPcaps:
    name : str
    path : str
    sessions : dict[int, object] # TODO: break in modules and then import
    def __init__(self, name : str, path : str):
        self.name = name
        self.path = path
        self.sessions = {}

    def add_session(self, sID : str, session):
        self.sessions[sID] = session

    def get_session(self, sID : str):
        return self.sessions[sID] if sID in self.sessions else None


class OnionDataset(DatasetPcaps):
    clients : dict[str, DatasetPcaps]
    full_onion : str
    
    def add_client(self, interaction : DatasetPcaps):
        self.clients[interaction.name] = interaction

    def get_client(self, interation : DatasetPcaps):
        return self.clients[interation.name]

    def set_full_onion(self, path : str):
        self.full_onion = path


class ClientDataset(DatasetPcaps):
    onions : dict[str, DatasetPcaps]
    alexas : dict[str, DatasetPcaps]

    def add_onion(self, interation : DatasetPcaps):
        self.onions[interation.name] = interation

    def get_onion(self, interation : DatasetPcaps):
        return self.onions[interation.name]

    def add_alexa(self, interation : DatasetPcaps):
        self.alexas[interation.name] = interation

    def get_alexa(self, interation : DatasetPcaps):
        return self.alexas[interation.name]

class Request:
    rID : int
    pcap_path : str
    def __init__(self, session, rID : int, path : str):
        self.session = session
        self.rID = rID
        self.pcap_path = path
        session.add_request(rID, self)

class Session:
    dataset : DatasetPcaps
    sID : str
    pcap_path : str
    reqs : dict[int, Request]
    def __init__(self, dataset : DatasetPcaps, sID : str, path : str):
        self.dataset = dataset
        self.sID = sID
        self.pcap_path = path
        self.reqs = {}
        dataset.add_session(sID, self)

    def add_request(self, rID : int, req : Request):
        self.reqs[rID] = req

class SumoDataset:
    path    : str
    alexas  : dict[str, DatasetPcaps]
    onions  : dict[str, OnionDataset]
    clients : dict[str, ClientDataset]
    full_onions : dict[str, str]
    onion_pages : dict[str, str]

    def aux_add_session_request(self, session, request, dic, key, f_path):
        dataset = dic[key]
        #sID = re.split('(_client|_hs)\.pcap$', f_path.split('/')[-1])[0]
        sID = re.split('(_request_[0-9]+_client|_request_[0-9]+_hs|_client|_hs)\.pcap$', f_path.split('/')[-1])[0]
        if session: # TODO: duplicated code
            s = dataset.get_session(sID)
            if not s:
                s = Session(dataset, sID, f_path)
            else:
                s.pcap_path = f_path
        elif request:
            rID = int(request.group(2))
            s = dataset.get_session(sID)
            if not s:
                s = Session(dataset, sID, "")
            r = Request(s, rID, f_path)

    def __init__(self, path = "."):
        self.path = path
        self.caps_client = "TrafficCapturesClient"
        self.caps_onion = "TrafficCapturesOnion"
        self.alexas = {}
        self.onions = {}
        self.clients = {}
        self.full_onions = {}
        self.onion_pages = {}
        if not os.path.isdir(path):
            raise PathInvalidException("path must be a directory")
        for root, dirs, files in os.walk(path):
            if root == path:
                for f in [self.caps_client, self.caps_onion]:
                    if f not in dirs:
                        raise InvalidDatasetException(f"dataset path must have a {f} folder")
            if len(files) == 0:
                continue
            if self.caps_client in root or self.caps_onion in root:
                for f in files:
                    if not f.endswith(".pcap"):
                        continue
                    f_path = root + "/" + f
                    try:
                        client = get_client_name(f)
                    except InvalidClientCaptureNameException:
                        client = None
                    except Exception as e:
                        traceback.print_exc()
                    try:
                        onion = get_onion_name(f)
                    except InvalidOnionCaptureNameException:
                        onion = None
                    except Exception as e:
                        traceback.print_exc()
                    
                    onion_page = re.search("_(os[0-9]+-os.*)_(.*)_session.*", f)
                    session = re.search("_session_([0-9]+)_(client|hs)?\\.pcap", f)
                    request = re.search("_session_([0-9]+)_request_([0-9]+)_(client|hs)?\\.pcap", f)
                    
                    if client and client not in self.clients:
                        self.clients[client] = ClientDataset(client, "")
                    
                    if onion and onion not in self.onions:
                        self.onions[onion] = OnionDataset(onion, "")
                        
                    if onion_page:
                        if onion_page.group(2) not in self.onion_pages:
                            # print("onion_page:", onion_page.group(2), "-->", onion_page.group(1))
                            self.onion_pages[onion_page.group(2)] = onion_page.group(1)

                    if self.caps_client in root:
                        try:
                            alexa = get_alexa_name(f)
                        except InvalidAlexaCaptureNameException:
                            alexa = None
                        except Exception as e:
                            traceback.print_exc()
                        if alexa:
                            if client:
                                alexa = client + "_" + alexa
                            if alexa not in self.alexas:
                                self.alexas[alexa] = DatasetPcaps(alexa, "")
                            self.aux_add_session_request(session, request, self.alexas, alexa, f_path)
                        if client:
                            self.aux_add_session_request(session, request, self.clients, client, f_path)

                        if f == "/mnt/nas-shared/torpedo/datasets_20230521/small-ostest/experiment_results/TrafficCapturesClient/client-small-test-1/client1/captures-client-small-ostest-1-client1/client-small-ostest-1-client1_alexa_albayan.ae_session_1168_client.pcap":
                            print("\n======= client", client)
                            print("======= alexa", alexa)

                    elif self.caps_onion in root:
                        if "full-onion" in root:
                            full_onion = re.search(".*_(.*)_hs.pcap", f)
                            if full_onion:
                                self.full_onions[full_onion.group(1)] = f_path
                        if onion:
                            self.aux_add_session_request(session, request, self.onions, onion, f_path)

                # end for files in dataset
        # end for OS path walk
        for p in self.onion_pages:
            f = self.full_onions[p]
            self.onions[self.onion_pages[p]].set_full_onion(f)

    
    def filter_sessions_with_failed_requests(self):
        # TODO: should remove both at client and OS sides
        file_paths = glob.glob(os.path.join(self.path, '**/failed_requests.log'), recursive=True)
        if len(file_paths) == 0:
            raise MissingFailedRequestsLogException
        session_paths_to_remove = []
        # Loop through each file and open it for reading
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                # Do something with the file contents
                file_contents = f.readlines()
                for session in file_contents:
                    # Avoid error logs
                    if "_request_" in session:
                        session_path_to_remove = file_path.split("failed_requests.log")[0] + 'captures-' + get_client_name(session) + '/' + session.rstrip('\n').split("_request_")[0] + '_client.pcap'
                        session_paths_to_remove.append(session_path_to_remove)
        return list(set(session_paths_to_remove))


    def list_alexas(self):
        return self.alexas.keys()

    def list_clients(self):
        return self.clients.keys()

    def list_onions(self):
        return self.onions.keys()
    
    def alexa_sessions(self, alexa : str):
        return self.alexas[alexa].sessions.keys()
    
    def alexa_sessions_paths_all(self):
        return [session.pcap_path for alexa in self.alexas.keys() for session in self.alexas[alexa].sessions.values()]
    
    def client_sessions(self, client : str):
        return self.clients[client].sessions.keys()
    
    def onion_sessions(self, onion : str):
        return self.onions[onion].sessions.keys()
    
    def get_full_onion(self, onion : str):
        return self.onions[onion].full_onion
    
    def get_alexa_session(self, alexa : str, sID : str):
        return self.alexas[alexa].sessions[sID].pcap_path
    
    def get_client_session(self, client : str, sID : str):
        return self.clients[client].sessions[sID].pcap_path
    
    def client_sessions_paths_all(self):
        return [session.pcap_path for client in self.clients.keys() for session in self.clients[client].sessions.values()]
    
    def client_sessions_paths_to_oses_all(self):
        return [session.pcap_path for client in self.clients.keys() for session in self.clients[client].sessions.values() if 'alexa' not in session.pcap_path]

    def client_sessions_paths_to_alexa_all(self):
        return [session.pcap_path for client in self.clients.keys() for session in self.clients[client].sessions.values() if 'alexa' in session.pcap_path]

    def get_onion_session(self, onion : str, sID : str):
        return self.onions[onion].sessions[sID].pcap_path

    def onion_sessions_paths_all(self):
        return [session.pcap_path for onion in self.onions.keys() for session in self.onions[onion].sessions.values()]

    def get_alexa_session_nb_requests(self, alexa : str, sID : str):
        return len(self.alexas[alexa].sessions[sID].reqs)
    
    def get_client_session_nb_requests(self, client : str, sID : str):
        return len(self.clients[client].sessions[sID].reqs)
    
    def get_onion_session_nb_requests(self, onion : str, sID : str):
        return len(self.onions[onion].sessions[sID].reqs)
    
    def get_alexa_request(self, alexa : str, sID : str, rID : int):
        return self.alexas[alexa].sessions[sID].reqs[rID].pcap_path
    
    def get_client_request(self, client : str, sID : str, rID : int):
        return self.clients[client].sessions[sID].reqs[rID].pcap_path
    
    
    def get_onion_request(self, onion : str, sID : str, rID : int):
        return self.onions[onion].sessions[sID].reqs[rID].pcap_path

def parse_client_onion_from_path(path : str):
    session = re.search("/(client-.*-client[0-9]+)_(os.*)_.*_session_([0-9]+)_(client|hs)?\\.pcap", path)
    request = re.search("/(client-.*-client[0-9]+)_(os.*)_.*_session_([0-9]+)_session_([0-9]+)_request_([0-9]+)_(client|hs)?\\.pcap", path)
    if session:
        return (session.group(1), session.group(2))
    if request:
        return (request.group(1), request.group(2))
    return (None, None)

if __name__ == "__main__":
    dataset : SumoDataset = None
    # only for testing does not check for valid arguments
    for idx,a in enumerate(sys.argv):
        if "--path" in a:
            del dataset
            dataset = SumoDataset(sys.argv[idx+1])
        if "--list-alexas" in a:
            print(" --list-alexas")
            for k in dataset.list_alexas():
                print("   ", k)
        if "--list-clients" in a:
            print(" --list-clients")
            for k in dataset.list_clients():
                print("   ", k)
        if "--list-onions" in a:
            print(" --list-onions")
            for k in dataset.list_onions():
                print("   ", k)
        if "--sessions-in-alexa" in a:
            print(" --sessions-in-alexa", sys.argv[idx+1])
            print("    ", dataset.alexa_sessions(sys.argv[idx+1]))
        if "--sessions-in-client" in a:
            print(" --sessions-in-client", sys.argv[idx+1])
            print("    ", dataset.client_sessions(sys.argv[idx+1]))
        if "--sessions-in-onion" in a:
            print(" --sessions-in-onion", sys.argv[idx+1])
            print("    ", dataset.onion_sessions(sys.argv[idx+1]))
        if "--get-full-onion" in a:
            print(" --get-full-onion", sys.argv[idx+1])
            print("    ", dataset.get_full_onion(sys.argv[idx+1]))
        if "--get-onion-session" in a:
            print(" --get-onion-session", sys.argv[idx+1], sys.argv[idx+2])
            print("    ", dataset.get_onion_session(sys.argv[idx+1], sys.argv[idx+2]))
        if "--get-client-session" in a:
            print(" --get-client-session", sys.argv[idx+1], sys.argv[idx+2])
            print("    ", dataset.get_client_session(sys.argv[idx+1], sys.argv[idx+2]))
        if "--nb-requests-in-client-session" in a:
            print(" --nb-request-in-client-session", sys.argv[idx+1], sys.argv[idx+2])
            print("    ", dataset.get_client_session_nb_requests(sys.argv[idx+1], sys.argv[idx+2]))
        if "--nb-requests-in-onion-session" in a:
            print(" --nb-request-in-onion-session", sys.argv[idx+1], sys.argv[idx+2])
            print("    ", dataset.get_onion_session_nb_requests(sys.argv[idx+1], sys.argv[idx+2]))
        if "--get-onion-request" in a:
            print(" --get-onion-request", sys.argv[idx+1], sys.argv[idx+2], sys.argv[idx+3])
            print("    ", dataset.get_onion_request(sys.argv[idx+1], sys.argv[idx+2], int(sys.argv[idx+3])))
        if "--get-client-request" in a:
            print(" --get-client-request", sys.argv[idx+1], sys.argv[idx+2], sys.argv[idx+3])
            print("    ", dataset.get_client_request(sys.argv[idx+1], sys.argv[idx+2], int(sys.argv[idx+3])))

