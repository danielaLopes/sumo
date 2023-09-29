import pdb
import json
import math
import pickle
import socket
import pathlib
import logging
import logging.handlers
import argparse
from ansible.inventory.manager import InventoryManager
import re
import traceback
import glob
import os
import shutil

import dpkt

from scapy.all import PcapReader, IP, TCP
from tqdm import tqdm


def get_client_name(capture_path):
    client = re.search("(client-(train|test|validate)-[0-9]+-client[0-9]+)", capture_path)
    if client:
        return client.group()
    else:
        return None

def get_onion_name(capture_path):
    onion = re.search("os-(train|test|validate)-[0-9]+", capture_path)
    if onion:
        return onion.group()
    else:
        return None


def filter_empty_ack(pkt):
    if pkt[TCP].flags & 0x10 and len(pkt[TCP].payload) == 0:
        ip_len = pkt[IP].len
        ip_hdr_len = pkt[IP].ihl * 4
        tcp_hdr_len = pkt[TCP].dataofs * 4
        payload_len = ip_len - ip_hdr_len - tcp_hdr_len
        if payload_len == 0:
            return True

    return False


def setLogger(name):
    """A good logger that simultaneously logs to a file and the console output.
    Args:
        filename (str): for the output log file.
    Returns:
        logger (logging.Logger): the logger.
    Output:
        A log file name <filename>.
    """
    logger = logging.getLogger(name)
    infoformatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%m-%dT%H:%M:%S"
    )
    debugformatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(infoformatter)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"{name}.log", when='midnight', backupCount=7)
    file_handler.setFormatter(debugformatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def deepcoffea_to_deepcorr(data_path, n_chunks=10):
    data_path = pathlib.Path(data_path)
    save_dir = data_path.parent / (data_path.name + "_for_DC")
    if not save_dir.exists():
        save_dir.mkdir()

    processed_flows = []

    inflows = sorted((data_path / "inflow").glob("*"))
    chunk_size = math.ceil(len(inflows) / n_chunks)
    storefile_i = 0

    for i, inflow_path in enumerate(tqdm(inflows, ascii=True, ncols=120)):
        outflow_path = data_path / "outflow" / inflow_path.name
        if not outflow_path.exists():
            print(f"inflow: {inflow_path.name} does not have a corresponding outflow, this should not happen.")

        data = {
            "here": [
                {"<-": [], "->": []},   # ipd
                {"<-": [], "->": []},   # size
            ],
            "there": [
                {"<-": [], "->": []},
                {"<-": [], "->": []},
            ]
        }

        # process inflow first
        with open(inflow_path) as fp:
            content = fp.read()
        flow = content.split("\n")
        if len(flow) == 0:
            continue
        ts_prev = 0.0
        for pkt in flow:
            pkt = pkt.strip()
            if pkt == '':
                continue

            ts = float(pkt.split("\t")[0])
            psize = int(pkt.split("\t")[1])

            if psize < 0:
                # client-side downstream (here, <-)
                data["here"][0]["<-"].append(ts - ts_prev)
                data["here"][1]["<-"].append(abs(psize))
            else:
                # client-side upstream (here, ->)
                data["here"][0]["->"].append(ts - ts_prev)
                data["here"][1]["->"].append(abs(psize))

            ts_prev = ts

        # process outflow then
        with open(outflow_path) as fp:
            content = fp.read()
        flow = content.split("\n")
        if len(flow) == 0:
            continue
        ts_prev = 0.0
        for pkt in flow:
            pkt = pkt.strip()
            if pkt == '':
                continue

            ts = float(pkt.split("\t")[0])
            psize = int(pkt.split("\t")[1])

            if psize < 0:
                # client-side downstream (here, <-)
                data["there"][0]["<-"].append(ts - ts_prev)
                data["there"][1]["<-"].append(abs(psize))
            else:
                # client-side upstream (here, ->)
                data["there"][0]["->"].append(ts - ts_prev)
                data["there"][1]["->"].append(abs(psize))

            ts_prev = ts

        processed_flows.append(data)
        if len(processed_flows) == chunk_size or i == len(inflows) - 1:
            with open(save_dir / f"flow_chunk_{storefile_i}.pickle", "wb") as fp:
                pickle.dump(processed_flows, fp)
            processed_flows = []
            storefile_i += 1


def deepcoffea_map(train_path, val_path, output_path=None):
    """Map the file names from the converted dataset to Deep Coffea's default names.
    Notice:
        This function is automatically called in convert
    Args:
        train_path (str/pathlib.Path): the directory that stores the parsed files
    Returns:
        None
    Outputs:
        <output_path>/inflow/
        <output_path>/outflow/
        and the corresponding symlinks
    """
    if output_path == None:
        output_path = train_path

    train_path = pathlib.Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} does not exists.")
    
    osflows = list((train_path / "captures_os").glob("*"))
    clientflows = list((train_path / "captures_client").glob("*"))

    if val_path is not None:
        val_path = pathlib.Path(val_path)
        if val_path.exists():
            osflows.extend(list((val_path / "captures_os").glob("*")))
            clientflows.extend(list((val_path / "captures_client").glob("*")))
        else:
            raise RuntimeWarning(f"{val_path} does not exists. skipping")

    osflows.sort()
    clientflows.sort()

    output_path = pathlib.Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # do a sanity check to confirm that every file in captures_os is also in captures_client
    assert sorted([flow.name for flow in osflows]) == sorted([flow.name for flow in clientflows]), \
        "captures_os/ and captures_client/ differ"

    # construct a map from str to index
    clhosts = set([])
    oshosts = set([])
    osaddrs = set([])
    for osflow in osflows:
        fields = osflow.name.split("_")
        clhosts.add(fields[0])
        oshosts.add(fields[1])
        osaddrs.add(fields[2])

    clhost_map = {clhost: i for i, clhost in enumerate(sorted(clhosts))}
    oshost_map = {oshost: i for i, oshost in enumerate(sorted(oshosts))}
    osaddr_map = {osaddr: i for i, osaddr in enumerate(sorted(osaddrs))}

    # build the filename map
    path_map = dict()
    for osflow in osflows:
        fname = osflow.name
        fields = fname.split("_")
        clhost = fields[0]
        oshost = fields[1]
        osaddr = fields[2]
        #pdb.set_trace()

        if "dataset_21march_2022_3" in train_path.name:
            suffix = fields[3] + fields[4] + fields[6] + fields[8]
        else:
            #suffix = f"{int(fields[4]):03d}{int(fields[6]):02d}"
            suffix = fields[4]

        new_fname = f"{clhost_map[clhost]}{oshost_map[oshost]}_{osaddr_map[osaddr]}{suffix}"

        path_map[f"{osflow.parent.parent}/captures_os/{fname}"] = str(output_path / "outflow" / new_fname)
        path_map[f"{osflow.parent.parent}/captures_client/{fname}"] = str(output_path / "inflow" / new_fname)

    # construct the symlinks
    inflow_dir = output_path / "inflow"
    if not inflow_dir.exists():
        inflow_dir.mkdir()
    outflow_dir = output_path / "outflow"
    if not outflow_dir.exists():
        outflow_dir.mkdir()
    for k, v in path_map.items():
        #print(f"{k} ---> {v}")
        shutil.copyfile(k, v)
        #pathlib.Path(v).symlink_to(k)


def inet_to_str(inet):
    """Convert inet object to a string (From dpkt.examples)
    Args:
        inet (inet struct): inet network address
    Returns:
        str: Printable/readable IP address
    """
    try:
        # try ipv4 first
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        # ipv4 trial failed, try ipv6
        return socket.inet_ntop(socket.AF_INET6, inet)


def parse_pcap(f, data, host, toformat, ip_map_str, scapy=False):
    """Parse the timestamp and the size of the packets from the .pcap files.

    We use the following filters:
    1) Only care about TCP packets.
    2) TCP retransmission followed by ICMP can be detected but can't detect other types of TCP packet errors
       (see https://www.wireshark.org/docs/wsug_html_chunked/ChAdvTCPAnalysis.html).
    3) TCP packets through port 9050, 5005, and 80 are ignored.

    Args:
        f (str/pathlib.Path): the path to the .pcap file.
        data (list): the variable that stores the parsed information.
        host (str): the host (os, client-amsterdam-1-new, etc) for the direction of the packet.
        toformat (str): "deepcorr" or "deepcoffea".
        ip_map_str (dict): a dictionary from host name (ex: client-amsterdam-new-1) to IP in str.
        scapy (bool): if true, use scapy, if false, use dpkt
    Returns:
        data (list): the variable that stores the parsed information.
    Outputs:
        None
    """
    logger = logging.getLogger("format_converter")

    if scapy:
        try:
            pcap = PcapReader(str(f))
        except Exception as e:
            logger.error(f"{f}\t{e}")
            return

    else:
        fp = open(f, "rb")
        try:
            pcap = dpkt.pcap.Reader(fp)
        except Exception as e:
            logger.error(f"{f}\t{e}")
            return

    tcp_prev = None # for a really simple retransmission tracking
    if scapy:
        attrs = ["sport", "dport", "seq", "ack", "dataofs", "flags", "window", "chksum"]
    else:
        attrs = ["sport", "dport", "seq", "ack", "off", "flags", "win", "sum"]

    for i, pkt in enumerate(pcap):

        if scapy:
            ts = float(pkt.time)
        else:
            ts, _ = pkt

        if toformat == "deepcorr":
            if i == 0:
                ts_prev = ts
                ts_prevprev = ts_prev
        else:
            if i == 0:
                ts_off = ts

        if scapy:
            fullsize = pkt.wirelen
            #print("\n\n\n===== packet {}; fullsize {}".format(i, fullsize))
            if not pkt.haslayer(TCP):
                logger.debug(f"frame {i+1:04d}: this packet doesn't have a TCP layer, skipping...")
                continue

            ip = pkt[IP]
            tcp = pkt[TCP]

            if ip.version == 1:
                logger.debug(f"frame {i+1:04d}: ip.version == 1 (ICMP), skipping...")
                continue

        else:
            _, _buf = pkt
            fullsize = len(_buf)
            _eth = dpkt.ethernet.Ethernet(_buf)
            ip = _eth.data
            tcp = ip.data


            if not isinstance(tcp, dpkt.tcp.TCP):
                logger.debug(f"frame {i+1:04d}: not TCP instance (got {type(tcp)}), skipping...")
                continue

            if ip.p == 1:
                logger.debug(f"frame {i+1:04d}: ip.p == 1 (ICMP), skipping...")

        # let's say it's a retransmission when this packet has the same attributes as the previous one
        if tcp_prev != None:
            for attr in attrs:
                if getattr(tcp, attr) != getattr(tcp_prev, attr):
                    break
            else:
                logger.debug(f"frame {i+1:04d}: (suspected) retransmission, skipping...")
                continue

        tcp_prev = tcp

        # we don't want packets through these ports
        if tcp.sport == 9050 or tcp.sport == 5005 or tcp.sport == 80 or tcp.sport == 8080:
            logger.debug(f"frame {i+1:04d}: tcp.sport == {tcp.sport}!")
            continue

        if tcp.dport == 9050 or tcp.dport == 5005 or tcp.dport == 80 or tcp.dport == 8080:
            logger.debug(f"frame {i+1:04d}: tcp.dport == {tcp.dport}!")
            continue

        # figure out the direction of this packet
        if scapy:
            src_ip_str = pkt[IP].src
            dst_ip_str = pkt[IP].dst
        else:
            src_ip_str = inet_to_str(ip.src)
            dst_ip_str = inet_to_str(ip.dst)

        if ip_map_str[host] in src_ip_str:
            upstream = True
        elif ip_map_str[host] in dst_ip_str:
            upstream = False
        else:
            logger.error(f"frame {i+1:04d}: both src and dst are not the host, impossible...")
            continue

        # actual conversion part
        if toformat == "deepcorr":

            # Filter packets with empty TCP ACK payload
            if scapy:
                if filter_empty_ack(pkt):
                    continue
            else:
                if len(tcp) == 0:
                    continue

            if ts == ts_prev:
                tdur = ts - ts_prevprev
            else:
                tdur = ts - ts_prev

            #print("\n--- len(tcp)", len(tcp))
            #print("--- fullsize", fullsize)

            #if host == "os":
            if "os" in host:
                if upstream:
                    # server-side upstream - from exit - there/->
                    data["there"][0]["->"].append(tdur)
                    #data["there"][1]["->"].append(len(tcp))
                    data["there"][1]["->"].append(fullsize)
                else:
                    # server-side downstream - to exit - there/<-
                    data["there"][0]["<-"].append(tdur)
                    #data["there"][1]["<-"].append(len(tcp))
                    data["there"][1]["<-"].append(fullsize)
            else:
                if upstream:
                    # client-side upstream - from entry - here/->
                    data["here"][0]["->"].append(tdur)
                    #data["here"][1]["->"].append(len(tcp))
                    data["here"][1]["->"].append(fullsize)
                else:
                    # client-side downstream - to entry - here/<-
                    data["here"][0]["<-"].append(tdur)
                    #data["here"][1]["<-"].append(len(tcp))
                    data["here"][1]["<-"].append(fullsize)

            if ts_prevprev != ts_prev:
                ts_prevprev = ts_prev
            ts_prev = ts


        elif toformat == "deepcoffea":
            # upstream positive, downstream negative
            sizedcf = fullsize if upstream else -fullsize
            data.append(f"{ts - ts_off:.06f}\t{sizedcf}")

        else:
            raise ValueError(f"toformat: {toformat} is not supported.")

    if not scapy:
        fp.close()
    return data


def get_ip_map(cfg_path):
    ip_map_str = {}
    inventory_manager = InventoryManager(loader=None, sources=[str(cfg_path)])
    hosts = inventory_manager.get_hosts()
    for host in hosts:
        if host.get_name().startswith('client-') or host.get_name().startswith('os-'):
            ip_map_str[host.get_name()] = "172." + str(host.vars['static_docker_ip'])

    return ip_map_str


def convert(data_path, toformat, sess=True):
    """Convert dataset_21march_2022_3 to a target format
    Args:
        data_path (str): path to "dataset_21march_2022_3", "experiment_results_20x15_29november_2022_v1", etc
        toformat (str): "deepcorr" or "deepcoffea"
        sess (bool): parse separate requests (false) or the entire session (sess), only supported for 29november and 30november.
    Returns:
        None
    Outputs:
        <data_path.parent>/<dataset>_for_<toformat>
    """
    if toformat != "deepcorr" and toformat != "deepcoffea":
        raise ValueError(f"toformat: {toformat} is not supported.")

    logger = setLogger("format_converter")

    data_path = pathlib.Path(data_path)

    if "dataset_21march_2022_3" in str(data_path):
        use_scapy = False
        # TODO: change use_scapy to False
        traffic_captures_onion = data_path / "experiment_results" / "TrafficCapturesOnion"
        traffic_captures_client = data_path / "experiment_results" / "TrafficCapturesClient"
    else:
        use_scapy = True
        traffic_captures_onion = data_path / "TrafficCapturesOnion"
        traffic_captures_client = data_path / "TrafficCapturesClient"

    if not traffic_captures_onion.exists():
        raise FileNotFoundError(f"{traffic_captures_onion} does not exists.")

    # paths for the parsed results
    if sess:
        extracted_dir = data_path.parent / f"{data_path.name}_for_{toformat}_req"
    else:
        extracted_dir = data_path.parent / f"{data_path.name}_for_{toformat}_sess"
    if not extracted_dir.exists():
        extracted_dir.mkdir()

    ip_map_str = get_ip_map(data_path / "../inventory.cfg")
    print("ip_map_str", ip_map_str)
    print("use_scapy", use_scapy)


    # iterate through os and get the corresponding client
    #captures_oses = sorted(traffic_captures_onion.glob("*"))
    #if sess and "dataset_21march_2022_3" not in str(data_path):
    # TODO: How do I get out of this problem???? Now I have a single capture and not a list of captures, should change line 494???
    #    os_pcap_paths = sorted(glob.glob(str(traffic_captures_onion) + ("/**/*_session_*_hs.pcap"), recursive=True))
    #    os_pcap_paths = [pcap for pcap in os_pcap_paths if "request" not in p.name]
    #else:
    #    os_pcap_paths = sorted(glob.glob(str(traffic_captures_onion) + ("/**/*_session_*_request_*_hs.pcap"), recursive=True))
    #captures_oses = sorted(glob.glob(str(traffic_captures_onion) + '/**/*.pcap', recursive=True))


    captures_oses = sorted(traffic_captures_onion.glob("*"))
    #captures_oses = set(os.path.dirname(f) for f in glob.glob(str(traffic_captures_onion) + '/**/*.pcap', recursive=True))
    print(captures_oses)
    for i, captures_os in enumerate(captures_oses):
        os_posix_path = pathlib.Path(captures_os)

        print("====== captures_os", captures_os)
        #print("====== client_idx_path", client_idx_path)
        #os_name = captures_os.name.split("captures-")[-1]   # os-amsterdam-1-new
        os_name = get_onion_name(str(captures_os))
        #os_name = get_onion_name(captures_os)
        print("====== os_name", os_name)
        if os_name is None:
            continue

        captures_oses_client = sorted(captures_os.glob("*"))
        #for i, captures_os in enumerate(os_pcap_paths):
        for client_idx, client_idx_path in enumerate(captures_oses_client):
            if 'full-onion' in str(client_idx_path):
                continue

            print("---- client_idx_path", client_idx_path)

            inner_folders = sorted(client_idx_path.glob("*"))
            for inner_folder_idx, inner_folder in enumerate(inner_folders):
                print("---- inner_folder", inner_folder)


                if sess and "dataset_21march_2022_3" not in str(data_path):
                    # TODO: How do I get out of this problem???? Now I have a single capture and not a list of captures, should change line 494???
                    #os_pcap_paths = sorted([p for p in os_posix_path.glob("*_session_*_hs.pcap") if "request" not in p.name])
                    os_pcap_paths = sorted([p for p in inner_folder.glob("*_session_*_hs.pcap") if "request" not in p.name])
                    #if len(os_pcap_paths) > 0:
                    #    print("\n==== os_pcap_paths", os_pcap_paths)
                    #    exit()
                    print("$$$$$ os_pcap_paths", len(os_pcap_paths))
                else:
                    #os_pcap_paths = sorted(client_os_dir.glob("*_session_*_request_*_hs.pcap"))
                    os_pcap_paths = sorted(inner_folder.glob("*_session_*_request_*_hs.pcap"))
                print("====== os_pcap_paths", len(os_pcap_paths))
                for k, os_pcap_path in enumerate(os_pcap_paths):
                    print("------ os_pcap_path", os_pcap_path)
                    client_name = get_client_name(str(os_pcap_path))
                    print("------ client_name", client_name)
                    if client_name is None:
                        continue

                    fname = os_pcap_path.name.split("_hs")[0]   # <os_name>_<client_name>_<osaddr>_<params>
                    print("fname", fname)
                    logger.info(
                        f"Processing os[{i+1:02d}/{len(captures_oses):02d}], " \
                        #f"client[{j+1:02d}/{len(client_os_dirs):02d}], " \
                        f"file[{k+1:03d}/{len(os_pcap_paths):03d}]: " \
                        f"{fname}"
                    )

                    client_name_split = client_name.split('-client')
                    client_machine_name = client_name_split[0]
                    inner_client_id = f'client{client_name_split[1]}'
                    client_pcap_path = traffic_captures_client / client_machine_name / inner_client_id / f"captures-{client_name}" / f"{fname}_client.pcap"

                    if not client_pcap_path.exists():
                        logger.warning(f"Corresponding {client_pcap_path} does not exist, skipping...")
                        continue

                    print("client_machine_name", client_machine_name)
                    if toformat == "deepcorr":

                        data = {
                            "here": [
                                {"<-": [], "->": []},   # ipd
                                {"<-": [], "->": []},   # size
                            ],
                            "there": [
                                {"<-": [], "->": []},
                                {"<-": [], "->": []},
                            ]
                        }

                        try:
                            data = parse_pcap(os_pcap_path, data, os_name, toformat, ip_map_str, scapy=use_scapy)
                            data = parse_pcap(client_pcap_path, data, client_machine_name, toformat, ip_map_str, scapy=use_scapy)
                        except Exception as e:
                            traceback.print_exc()
                            logger.warning(f"Parsing {fname} failed...{e}")

                        with open(extracted_dir / (fname + ".json"), "w") as fp:
                            json.dump(data, fp)

                    else:

                        data_os = []
                        data_cl = []

                        try:
                            data_os = parse_pcap(os_pcap_path, data_os, os_name, toformat, ip_map_str, scapy=use_scapy)
                            data_cl = parse_pcap(client_pcap_path, data_cl, client_machine_name, toformat, ip_map_str, scapy=use_scapy)
                        except Exception as e:
                            traceback.print_exc()
                            logger.warning(f"Parsing {fname} failed...{e}")

                        extracted_os = extracted_dir / "captures_os"
                        if not extracted_os.exists():
                            extracted_os.mkdir()
                        with open(extracted_os / fname, "w") as fp:
                            fp.write("\n".join(data_os))

                        extracted_client = extracted_dir / "captures_client"
                        if not extracted_client.exists():
                            extracted_client.mkdir()
                        with open(extracted_client / fname, "w") as fp:
                            fp.write("\n".join(data_cl))


    # post processing
    if toformat == "deepcoffea":
        deepcoffea_map(extracted_dir)


if __name__ == "__main__":
    # python3 format_converter.py --data_path "/mnt/nas-shared/torpedo/new_datasets/OSTest_filtered/experiment_results_filtered/" --to deepcorr
    # python3 format_converter.py --data_path "/mnt/nas-shared/torpedo/new_datasets/OSTest_filtered/experiment_results_filtered/" --to deepcoffea
    # python3 format_converter.py --data_path "/mnt/nas-shared/torpedo/datasets_20220503/dataset_test/experiment_results_filtered/" --to deepcorr
    # python3 format_converter.py --data_path "/mnt/nas-shared/torpedo/datasets_20220503/dataset_test/experiment_results_filtered/" --to deepcoffea
    parser = argparse.ArgumentParser(description="Format converter.")
    parser.add_argument("--data_path", required=True, type=str,
        help="The path to the dataset. The converted files will be stored at the same level.")
    parser.add_argument("--to", dest="toformat", required=True, type=str,
        help="convert to what format. Options: deepcoffea, deepcorr.")
    args = parser.parse_args()

    if args.toformat == "deepcorr" or args.toformat == "deepcoffea":
        if "CrawlE_Proc" in args.data_path:
            deepcoffea_to_deepcorr(args.data_path, n_chunks=10)
        else:
            convert(args.data_path, args.toformat)
    else:
        raise ValueError(f"--to {args.toformat} is not supported.")
