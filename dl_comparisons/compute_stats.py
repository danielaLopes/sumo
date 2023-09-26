import json
import pickle
import pathlib

import numpy as np
from tqdm import tqdm


def print_quartiles(stats, data_root):
    clientdownnpkts, clientupnpkts, \
    serverdownnpkts, serverupnpkts, \
    clientdurs, serverdurs, \
    clientdownbytes, clientupbytes, \
    serverdownbytes, serverupbytes, n_flows = stats
    
    if not isinstance(data_root, pathlib.Path):
        data_root = pathlib.Path(data_root)

    print(f"{data_root} has {n_flows} flow pairs")

    q1, median, q3 = np.percentile(clientdownnpkts, [25, 50, 75])
    print(f"Client received number of packets:\n\tmin: {np.min(clientdownnpkts)}, max: {np.max(clientdownnpkts)}\tq1: {q1}, median: {median}, q3: {q3}")
    
    q1, median, q3 = np.percentile(clientupnpkts, [25, 50, 75])
    print(f"Client sent number of packets:\n\tmin: {np.min(clientupnpkts)}, max: {np.max(clientupnpkts)}\tq1: {q1}, median: {median}, q3: {q3}")

    q1, median, q3 = np.percentile(serverdownnpkts, [25, 50, 75])
    print(f"Server received number of packets:\n\tmin: {np.min(serverdownnpkts)}, max: {np.max(serverdownnpkts)}\tq1: {q1}, median: {median}, q3: {q3}")

    q1, median, q3 = np.percentile(serverupnpkts, [25, 50, 75])
    print(f"Server sent number of packets:\n\tmin: {np.min(serverupnpkts)}, max: {np.max(serverupnpkts)}\tq1: {q1}, median: {median}, q3: {q3}")

    q1, median, q3 = np.percentile(clientdurs, [25, 50, 75])
    print(f"Client durations:\n\tmin: {np.min(clientdurs):.3f}, max: {np.max(clientdurs):.3f}\tq1: {q1:.3f}, median: {median:.3f}, q3: {q3:.3f}")

    q1, median, q3 = np.percentile(serverdurs, [25, 50, 75])
    print(f"Server durations:\n\tmin: {np.min(serverdurs):.3f}, max: {np.max(serverdurs):.3f}\tq1: {q1:.3f}, median: {median:.3f}, q3: {q3:.3f}")

    q1, median, q3 = np.percentile(clientdownbytes, [25, 50, 75])
    print(f"Client received bytes:\n\tmin: {np.min(clientdownbytes)}, max: {np.max(clientdownbytes)}\tq1: {q1}, median: {median}, q3: {q3}")

    q1, median, q3 = np.percentile(clientupbytes, [25, 50, 75])
    print(f"Client sent bytes:\n\tmin: {np.min(clientupbytes)}, max: {np.max(clientupbytes)}\tq1: {q1}, median: {median}, q3: {q3}")

    q1, median, q3 = np.percentile(serverdownbytes, [25, 50, 75])
    print(f"Server received bytes:\n\tmin: {np.min(serverdownbytes)}, max: {np.max(serverdownbytes)}\tq1: {q1}, median: {median}, q3: {q3}")

    q1, median, q3 = np.percentile(serverupbytes, [25, 50, 75])
    print(f"Server sent bytes:\n\tmin: {np.min(serverupbytes)}, max: {np.max(serverupbytes)}\tq1: {q1}, median: {median}, q3: {q3}")

    clientnpkts = clientupnpkts + clientdownnpkts
    q1, median, q3 = np.percentile(clientnpkts, [25, 50, 75])
    print(f"Client number of packets:\n\tmin: {np.min(clientnpkts)}, max: {np.max(clientnpkts)}\tq1: {q1}, median: {median}, q3: {q3}")

    servernpkts = serverupnpkts + serverdownnpkts
    q1, median, q3 = np.percentile(servernpkts, [25, 50, 75])
    print(f"Server number of packets:\n\tmin: {np.min(servernpkts)}, max: {np.max(servernpkts)}\tq1: {q1}, median: {median}, q3: {q3}")

    clientbytes = clientupbytes + clientdownbytes
    q1, median, q3 = np.percentile(clientbytes, [25, 50, 75])
    print(f"Client bytes:\n\tmin: {np.min(clientbytes)}, max: {np.max(clientbytes)}\tq1: {q1}, median: {median}, q3: {q3}")

    serverbytes = serverupbytes + serverdownbytes
    q1, median, q3 = np.percentile(serverbytes, [25, 50, 75])
    print(f"Server bytes:\n\tmin: {np.min(serverbytes)}, max: {np.max(serverbytes)}\tq1: {q1}, median: {median}, q3: {q3}")


def compute_deepcorr_stats(data_root):

    data_root = pathlib.Path(data_root)
    
    clientdownnpkts = []
    clientupnpkts = []
    clientdurs = []
    clientdownbytes = []
    clientupbytes = []

    serverdownnpkts = []
    serverupnpkts = []
    serverdurs = []
    serverdownbytes = []
    serverupbytes = []

    if "deepcorr_tar_bz2" in str(data_root):
        data = []
        for f in sorted(data_root.glob("*_tordata300.pickle")):
            if "8812" in f.name or "8813" in f.name or "8852" in f.name:
                continue
            with open(f, "rb") as fp:
                data.extend(pickle.load(fp))

    else:
        data = []
        for f in sorted(data_root.glob("*.json")):
            with open(f) as fp:
                data.append(json.load(fp))

    n_flows = len(data)
    
    for _, flow in enumerate(tqdm(data, ascii=True, ncols=120)):
                
        # get client downstream number of packets
        clientdownnpkts.append(len(flow["here"][1]["<-"]))
        # get client upstream number of packets
        clientupnpkts.append(len(flow["here"][1]["->"]))
        # get client durations
        clientdurs.append(sum(flow["here"][0]["<-"]) + sum(flow["here"][0]["->"]))
        # get client downstream transmitted number of bytes
        clientdownbytes.append(sum(flow["here"][1]["<-"]))
        # get client upstream transmitted number of bytes
        clientupbytes.append(sum(flow["here"][1]["->"]))

        # get server downstream number of packets
        serverdownnpkts.append(len(flow["there"][1]["<-"]))
        # get server upstream number of packets
        serverupnpkts.append(len(flow["there"][1]["->"]))
        # get server durations
        serverdurs.append(sum(flow["there"][0]["<-"]) + sum(flow["there"][0]["->"]))
        # get server downstream transmitted number of bytes
        serverdownbytes.append(sum(flow["there"][1]["<-"]))
        # get server upstream transmitted number of bytes
        serverupbytes.append(sum(flow["there"][1]["->"]))

    clientdownnpkts = np.array(clientdownnpkts)
    clientupnpkts = np.array(clientupnpkts)
    serverdownnpkts = np.array(serverdownnpkts)
    serverupnpkts = np.array(serverupnpkts)
    clientdurs = np.array(clientdurs)
    serverdurs = np.array(serverdurs)
    clientdownbytes = np.array(clientdownbytes)
    clientupbytes = np.array(clientupbytes)
    serverdownbytes = np.array(serverdownbytes)
    serverupbytes = np.array(serverupbytes)

    stats = [
        clientdownnpkts, clientupnpkts, \
        serverdownnpkts, serverupnpkts, \
        clientdurs, serverdurs, \
        clientdownbytes, clientupbytes, \
        serverdownbytes, serverupbytes, n_flows
    ]
    return stats


def compute_deepcoffea_stats(data_root):

    def compute(flow_fs):
        
        down_npkts = []
        up_npkts = []
        durs = []
        down_bytes = []
        up_bytes = []

        for flow_f in tqdm(flow_fs, ascii=True, ncols=120):
            with open(flow_f) as fp:
                content = fp.read()

            flow = content.split("\n")
            if len(flow) == 0:
                continue

            up_npkt = 0
            down_npkt = 0
            up_byte = 0
            down_byte = 0
            for pkt in flow:
                pkt = pkt.strip()
                if pkt == '':
                    continue

                ts = float(pkt.split("\t")[0])
                psize = int(pkt.split("\t")[1])
                
                if psize < 0:
                    down_npkt += 1
                    down_byte += abs(psize)
                else:
                    up_npkt += 1
                    up_byte += abs(psize)

            down_npkts.append(down_npkt)
            up_npkts.append(up_npkt)
            durs.append(ts)
            down_bytes.append(down_byte)
            up_bytes.append(up_byte)
            
        return down_npkts, up_npkts, durs, down_bytes, up_bytes

    data_root = pathlib.Path(data_root)
    inflows = sorted((data_root / "inflow").glob("*"))
    outflows = sorted((data_root / "outflow").glob("*"))

    assert len(inflows) == len(outflows), "len(inflows) != len(outflows), weird"
    clientdownnpkts, clientupnpkts, clientdurs, clientdownbytes, clientupbytes = compute(inflows)
    serverdownnpkts, serverupnpkts, serverdurs, serverdownbytes, serverupbytes = compute(outflows)

    clientdownnpkts = np.array(clientdownnpkts)
    clientupnpkts = np.array(clientupnpkts)
    serverdownnpkts = np.array(serverdownnpkts)
    serverupnpkts = np.array(serverupnpkts)
    clientdurs = np.array(clientdurs)
    serverdurs = np.array(serverdurs)
    clientdownbytes = np.array(clientdownbytes)
    clientupbytes = np.array(clientupbytes)
    serverdownbytes = np.array(serverdownbytes)
    serverupbytes = np.array(serverupbytes)

    stats = [
        clientdownnpkts, clientupnpkts, \
        serverdownnpkts, serverupnpkts, \
        clientdurs, serverdurs, \
        clientdownbytes, clientupbytes, \
        serverdownbytes, serverupbytes, len(inflows)
    ]
    return stats