"""All things related to Deep Coffea data.

- To parse the raw Deep Coffea data, i.e., CrawlE_Proc/ folder which contains the inflow/ and outflow/
  folders, do the following:
  1. filter_windows()
  2. partition_windows()
  This will generate a number (n_wins) of .pkl files that are ready to be preprocessed for the deep model
"""
import pickle
import pathlib
import collections

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm


def filter_windows(delta, win_size, n_wins, threshold, data_root):
    """Make sure there are enough (> threshold) packets in each window.

    Some maths:
    - total durations = (n_wins - 1)*offset + win_size, ex: (11-1)*(5-3) + 5 = 25s
    - For delta = 3, win_size = 5, n_wins = 11, the intervals are:
      - [
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
            (12, 17),
            (14, 19),
            (16, 21),
            (18, 23),
            (20, 25)
        ]

    Prerequisites:
      - The raw data, e.g., CrawlE_Proc/inflow, CrawlE_Proc/outflow

    Params:
      - delta (int): the number of seconds overlapped between each window, default: 3.
      - win_size (int): the number of seconds in each window, default: 5.
      - n_wins (int): the number of window partitions, default: 11.
      - threshold (int): each window must have more than this number of packets to be considered valid, default: 20.
      - data_root (str/pathlib.Path): the path to the data, e.g., "./datasets/CrawlE_Proc".

    Returns:
      - valid_files (list): the valid file names.

    Outputs:
      - {data_root}/d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_valid_files.txt
    """
    def check_threshold(f_path, threshold, intervals):
        intervals_ctr = {interval: 0 for interval in intervals}

        with open(f_path) as fp:
            for line in fp:
                timestamp, size = line.strip().split("\t")
                timestamp, size = float(timestamp), int(size)

                # the windows don't reach this timestamp
                if timestamp > intervals[-1][1]:
                    break

                # check the corresponding window
                for interval in intervals:
                    if timestamp >= interval[0] and timestamp < interval[1]:
                        intervals_ctr[interval] += 1

                    # the upcoming windows are irrelevant
                    if timestamp < interval[0]:
                        break

        for _, cnt in intervals_ctr.items():
            if cnt < threshold:
                return False

        if len(intervals_ctr) < n_wins:
            return False

        return True

    data_root = pathlib.Path(data_root)
    out_path = data_root / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_valid_files.txt"
    if out_path.exists():
        print(f"{out_path} exists, loading...")
        with open(out_path) as fp:
            return fp.read().split("\n")

    print(f"{out_path} does not exist, filtering...")
    here_path = data_root / "inflow"
    there_path = data_root / "outflow"
    files = sorted([f.name for f in here_path.glob("*")])

    offset = win_size - delta
    intervals = [(wi*offset, wi*offset + win_size) for wi in range(n_wins)]
    valid_files = []
    for fn in tqdm(files, ascii=True, ncols=120):
        here_result = check_threshold(here_path / fn, threshold, intervals)

        if here_result:
            there_result = check_threshold(there_path / fn, threshold, intervals)

            if there_result:
                valid_files.append(fn)

    with open(out_path, "w") as fp:
        fp.write("\n".join(sorted(valid_files)))

    return sorted(valid_files)


def partition_windows(delta, win_size, n_wins, threshold, data_root):
    """Partition the windows from raw files. See filter_windows for how the windows are derived.
    
    Prerequisites:
      - The raw data, e.g., CrawlE_Proc/inflow, CrawlE_Proc/outflow

    Params:
      - delta (int): the number of seconds overlapped between each window, default: 3.
      - win_size (int): the number of seconds in each window, default: 5.
      - n_wins (int): the number of window partitions, default: 11.
      - threshold (int): each window must have more than this number of packets to be considered valid, default: 20.
      - data_root (str/pathlib.Path): the path to the data, e.g., "./datasets/CrawlE_Proc".

    Returns:
      - None

    Outputs:
      - {data_root}/filtered_and_partitioned/d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_wi00.pkl
      - {data_root}/filtered_and_partitioned/d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_wi01.pkl
      - ...(a total of n_wins pickles)
    """
    def parse_file(f_path, interval):
        prev_time = 0.0
        big_pkt = []
        seq = []
        with open(f_path) as fp:
            for line in fp:
                timestamp, size = line.strip().split("\t")
                timestamp, size = float(timestamp), int(size)

                if timestamp < interval[0]:
                    continue
                elif timestamp >= interval[1]:
                    break

                if size > 0:
                    ipd = timestamp - prev_time
                else:
                    ipd = prev_time - timestamp

                if abs(size) > 66:  # ignore ack packet
                    if prev_time != 0 and ipd == 0:
                        big_pkt.append(size)
                        continue

                    if len(big_pkt) != 0:
                        last_pkt = seq.pop()
                        seq.append({"ipd": last_pkt["ipd"], "size": sum(big_pkt) + big_pkt[0]})
                        big_pkt = []

                    seq.append({"ipd": ipd, "size": size})
                    prev_time = timestamp

        return seq

    valid_files = filter_windows(delta, win_size, n_wins, threshold, data_root)

    data_root = pathlib.Path(data_root)
    here_path = data_root / "inflow"
    there_path = data_root / "outflow"
    out_dir = data_root / "filtered_and_partitioned"
    if not out_dir.exists():
        out_dir.mkdir()

    for wi in range(n_wins):
        print(f"Partitioning window [{wi+1:02d}/{n_wins:02d}]...")
        offset = win_size - delta
        interval = [wi*offset, wi*offset + win_size]

        here = []
        there = []
        labels = []
        for fn in tqdm(valid_files, ascii=True, ncols=120):
            here_seq = parse_file(here_path / fn, interval)

            if len(here_seq) != 0:
                there_seq = parse_file(there_path / fn, interval)

                if len(there_seq) != 0:
                    here.append(here_seq)
                    there.append(there_seq)
                    labels.append(fn)

        here, there, labels = np.array(here, dtype=object), np.array(there, dtype=object), np.array(labels, dtype=object)

        out_path = out_dir / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_wi{wi:02d}.pkl"
        with open(out_path, "wb") as fp:
            pickle.dump({'tor': here, 'exit': there, 'label': labels}, fp)


def preprocess_dcf(delta, win_size, n_wins, threshold, tor_len, exit_len, n_test, data_root, seed=114):
    """Preprocessing .pkl files for the deep model:
      1) Make sure circuits do not overlap between training and testing sets,
      2) Scale the packet sizes and ipds,
      3) Make sure the initial ipd is always 0 in each window,
      4) Pad/truncate the windows to a fixed length,
      5) Split the training and testing sets.

    For dataset_21march_2022_3 (4445): try 1, 1.6, 5, 10 -> 2932
    
    Prerequisites:
      - The .pkl files generated from partition_windows()

    Params:
      - delta (float): the number of seconds overlapped between each window, default: 3.
      - win_size (float): the number of seconds in each window, default: 5.
      - n_wins (int): the number of window partitions, default: 11.
      - threshold (int): each window must have more than this number of packets to be considered valid, default: 20.
      - tor_len (int): maximum (or pad to) length of the tor flow (inflow, client-side captures), default: 500.
      - exit_len (int): maximum (or pad to) length of the exit flow (outflow, server-side captures), default: 800.
      - n_test (int): number of examples for testing, the actual number will be <= n_test

    Returns:
      - None

    Outputs:
      - two .npz files.
    """
    data_root = pathlib.Path(data_root)
    save_dir = data_root / "filtered_and_partitioned"
    if not save_dir.exists():
        save_dir.mkdir()
    _pkl_file = save_dir / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_wi00.pkl"
    if not _pkl_file.exists():
        partition_windows(delta, win_size, n_wins, threshold, data_root)

    rng = np.random.default_rng(seed)

    train_tor = []
    train_exit = []
    train_label = []
    test_tor = []
    test_exit = []
    test_label = []

    for wi in range(n_wins):
        pkl_path = save_dir / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_wi{wi:02d}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"{pkl_path} does not exist, have we run partition_windows() yet?")

        with open(pkl_path, "rb") as fp:
            traces = pickle.load(fp)
            tor_seq = traces["tor"]
            exit_seq = traces["exit"]
            labels = traces["label"]

        n_flows = len(labels)

        if wi == 0:
            print(f"Window: {wi:02d}. (Only the first window) Shuffling indices for train/test sets")
            circuit_nflows = collections.Counter([label.split('_')[0] for label in labels])
            circuits = sorted(list(circuit_nflows.keys()))
            rng.shuffle(circuits)

            _n_flows = 0
            for i, c in enumerate(circuits):
                _n_flows += circuit_nflows[c]
                if _n_flows >= n_flows - n_test:
                    break

            train_circuits_set = set(circuits[:i+1])

        print(f"Window: {wi:02d}. Loading the whole sequences")
        window_tor_size = []
        window_tor_ipd = []
        window_exit_size = []
        window_exit_ipd = []
        for i in range(n_flows):
            window_tor_size.append([feat["size"] / 1000 for feat in tor_seq[i]])
            window_tor_ipd.append([feat["ipd"] * 1000 for feat in tor_seq[i]])
            window_exit_size.append([feat["size"] / 1000 for feat in exit_seq[i]])
            window_exit_ipd.append([feat["ipd"] * 1000 for feat in exit_seq[i]])

        print(f"Window: {wi:02d}. Making the initial ipd 0")
        window_tor_ipd_iz = []
        for flow in window_tor_ipd:
            window_tor_ipd_iz.append([0.0] + flow[1:])
        window_exit_ipd_iz = []
        for flow in window_exit_ipd:
            window_exit_ipd_iz.append([0.0] + flow[1:])

        print(f"Window: {wi:02d}. Concatenating ipds and sizes")
        window_tor = []
        window_exit = []
        for i in range(n_flows):
            window_tor.append(np.concatenate((window_tor_ipd_iz[i], window_tor_size[i])))
            window_exit.append(np.concatenate((window_exit_ipd_iz[i], window_exit_size[i])))

        print(f"Window: {wi:02d}. Padding/truncating the windows to tor_len*2: {tor_len*2}, exit_len*2: {exit_len*2}")
        window_tor_fixedlen = []
        window_exit_fixedlen = []
        for i in range(n_flows):
            window_tor_fixedlen.append(np.pad(window_tor[i][:tor_len*2], (0, tor_len*2 - len(window_tor[i][:tor_len*2]))).astype(np.float32))
            window_exit_fixedlen.append(np.pad(window_exit[i][:exit_len*2], (0, exit_len*2 - len(window_exit[i][:exit_len*2]))).astype(np.float32))

        assert len(window_tor_fixedlen) == len(window_exit_fixedlen) == len(labels), "window_tor_fixedlen, window_exit_fixedlen, and labels have different lengths"

        print(f"Window: {wi:02d}. Splitting the training/testing sets...")
        train_window_tor = []
        train_window_exit = []
        train_window_label = []
        test_window_tor = []
        test_window_exit = []
        test_window_label = []
        for win_tor, win_exit, label in zip(window_tor_fixedlen, window_exit_fixedlen, labels):
            circuit = label.split("_")[0]
            if circuit in train_circuits_set:
                train_window_tor.append(win_tor)
                train_window_exit.append(win_exit)
                train_window_label.append(label + f"_wi{wi:02d}")
            else:
                test_window_tor.append(win_tor)
                test_window_exit.append(win_exit)
                test_window_label.append(label + f"_wi{wi:02d}")

        train_tor.append(train_window_tor)
        train_exit.append(train_window_exit)
        train_label.append(train_window_label)
        test_tor.append(test_window_tor)
        test_exit.append(test_window_exit)
        test_label.append(test_window_label)
        
        print(f"Window: {wi:02d}. Done.")

    train_tor = np.array(train_tor).astype(np.float32)
    train_exit = np.array(train_exit).astype(np.float32)
    np.savez_compressed(save_dir / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_train", train_tor=train_tor, train_exit=train_exit, train_label=train_label)

    if n_test != 0:
        test_tor = np.array(test_tor).astype(np.float32)
        test_exit = np.array(test_exit).astype(np.float32)
        np.savez_compressed(save_dir / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_test", test_tor=test_tor, test_exit=test_exit, test_label=test_label)

    print("Preprocessing done.")


def get_train_test_filepaths(train_npz_path):
    train_out_path = train_npz_path[:-4] + "_files.txt"
    test_npz_path = train_npz_path[:-9] + "test.npz"
    test_out_path = test_npz_path[:-4] + "_files.txt"

    train_npz_path = pathlib.Path(train_npz_path)
    test_npz_path = pathlib.Path(test_npz_path)

    loaded_train = np.load(train_npz_path)
    loaded_test = np.load(test_npz_path)
    train_labels = loaded_train["train_label"]
    test_labels = loaded_test["test_label"]

    train_paths = []
    for label_wi00 in train_labels[0]:
        label = label_wi00.split("_wi00")[0]
        ln_path = train_npz_path.parents[1] / "inflow" / label
        file_path = ln_path.resolve()
        relfile_path = pathlib.Path(file_path.parts[-3]) / file_path.parts[-2] / file_path.parts[-1]
        train_paths.append(str(relfile_path))

    test_paths = []
    for label_wi00 in test_labels[0]:
        label = label_wi00.split("_wi00")[0]
        ln_path = test_npz_path.parents[1] / "inflow" / label
        file_path = ln_path.resolve()
        relfile_path = pathlib.Path(file_path.parts[-3]) / file_path.parts[-2] / file_path.parts[-1]
        test_paths.append(str(relfile_path))

    assert set(train_paths) & set(test_paths) == set(), "train set and test set not disjoint!"

    with open(train_out_path, "w") as fp:
        fp.write("\n".join(train_paths))

    with open(test_out_path, "w") as fp:
        fp.write("\n".join(test_paths))

    print(f"training and testing file paths are written to:\n{train_out_path}\n{test_out_path}")


def torch_pairwise_cosine_similarity(a, b, eps=1e-8):
    # torch.nn.functional.cosine_similarity is not pairwise cosine similarity
    a_n = a.norm(dim=1)[:, None]
    b_n = b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_table = torch.mm(a_norm, b_norm.T) # (N, N)
    return sim_table


class DeepCoffeaDataset(Dataset):

    def __init__(self, data_root, delta, win_size, n_wins, threshold, tor_len, exit_len, n_test, train=True):
        self.train = train
        data_root = pathlib.Path(data_root)

        if self.train:
            npz_path = data_root / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_train.npz"
            mode = "train"
        else:
            npz_path = data_root / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_test.npz"
            mode = "test"

        if not npz_path.exists():
            print(npz_path)
            raise FileNotFoundError("Make sure the .npz files exit by running preprocess_dcf() with appropriate parameters.")

        loaded = np.load(npz_path)
        if n_test == 0:
            load_key = "train"
        else:
            load_key = mode
        v_tor, v_exit, v_label = loaded[f"{load_key}_tor"], loaded[f"{load_key}_exit"], loaded[f"{load_key}_label"]

        print(f"Number of examples to {mode}: {v_tor.shape[1]}, number of windows: {v_tor.shape[0]}")

        self.tor_flows = np.reshape(v_tor, [-1, tor_len*2]).astype('float32')     # xa_all
        self.exit_flows = np.reshape(v_exit, [-1, exit_len*2]).astype('float32')  # xp_all
        self.labels = v_label

        self.sim_table = None

    def __len__(self):
        return self.tor_flows.shape[0]

    def __getitem__(self, idx):
        if self.train:
            idx, idx_n = idx
            return self.tor_flows[idx], self.exit_flows[idx], self.exit_flows[idx_n]
        else:
            return self.tor_flows[idx], self.exit_flows[idx]


def random_derangement(n):
    """
    An analysis of a simple algorithm for random derangements
    D Merlini, R Sprugnoli, MC Verri - Theoretical Computer Science, 2007 - World Scientific
    """
    while True:
        v = list(range(n))
        for j in range(n - 1, -1, -1):
            p = np.random.randint(0, j+1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return list(v)


class TripletSampler(Sampler):

    def __init__(self, dataset, alpha, rng, train=True, shuffle=True, max_retries=10):
        self.dataset = dataset
        self.alpha = alpha
        self.rng = rng
        self.train = train
        self.shuffle = shuffle
        self.max_retries = max_retries

    def __iter__(self):

        if self.train:
            print("Sampler is in training mode")
            self.dataset.train = True

            if self.dataset.sim_table is None:
                # building the sampler for random negative samples
                derangement_idxs = random_derangement(len(self.dataset))
                idxs = [(i, di) for i, di in enumerate(derangement_idxs)]
            else:
                # building the sampler for semi-hard samples
                # we did not implement the pool separation technique as the model still managed to converge without it
                idxs = []
                for idx in tqdm(range(len(self.dataset)), ascii=True, ncols=120):
                    semihard_idxs = np.asarray(self.dataset.sim_table[idx] < self.dataset.sim_table[idx, idx] + self.alpha).nonzero()[0]
                    for _ in range(min(len(semihard_idxs), self.max_retries)):
                        idx_n = self.rng.choice(semihard_idxs)
                        if idx_n != idx:
                            break
                    else:
                        idx_n = self.rng.choice(list(range(0, idx)) + list(range(idx + 1, len(self.dataset))))

                    idxs.append((idx, idx_n))

        else:
            print("Sampler is in testing mode")
            self.dataset.train = False
            idxs = [idx for idx in range(len(self.dataset))]

        if self.shuffle:
            self.rng.shuffle(idxs)

        return iter(idxs)        

    def __len__(self):
        return len(self.dataset)
