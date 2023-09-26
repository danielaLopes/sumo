import pdb
import time
import datetime

from math import sqrt
import numpy as np
import sys

# pytorch version 1.13
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class FakeData(Dataset):

    def __init__(self, size, mimic="deepcorr"):
        self.mimic = mimic
        if mimic == "deepcorr":
            self.examples = np.random.rand(*size).astype(np.float32)
            self.labels = np.random.randint(2, size=size[0])
        elif mimic == "deepcoffea":
            self.examples_a = np.random.rand(size[0], size[1]).astype(np.float32)
            self.examples_p = np.random.rand(size[0], size[2]).astype(np.float32)
            self.examples_n = np.random.rand(size[0], size[2]).astype(np.float32)
        else:
            raise ValueError(f"mimic = {mimic} not supported.")


    def __len__(self):
        if self.mimic == "deepcorr":
            return self.examples.shape[0]
        else:
            return self.examples_a.shape[0]

    def __getitem__(self, idx):
        if self.mimic == "deepcorr":
            return self.examples[idx], self.labels[idx]
        else:
            return self.examples_a[idx], self.examples_p[idx], self.examples_n[idx]


class DeepCorr(nn.Module):

    def __init__(self, flow_size=300, ruinit=False):
        super().__init__()
        """
        Author's implementation (https://github.com/SPIN-UMass/DeepCorr/blob/master/single_model-Tor-300.py)
        only provides model with flow_size = 300.
        """
        if flow_size == 300:
            params = [20, 10, 49600]
        elif flow_size == 200:
            params = [13, 7, 32000]
        elif flow_size == 150:
            params = [10, 5, 24000]
        elif flow_size == 100:
            params = [7, 4, 14400]
        else:
            raise ValueError(f"flow_size: {flow_size} is not supported.")

        self.conv1 = nn.Conv2d(1, 2000, (2, params[0]), stride=2)
        self.conv2 = nn.Conv2d(2000, 800, (4, params[1]), stride=2)
        self.fc1 = nn.Linear(params[2], 3000)
        self.fc2 = nn.Linear(3000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.classifier = nn.Linear(100, 1)

        self.dropout = nn.Dropout(p=0.4)

        if ruinit:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (1, 5), stride=1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (1, 3), stride=1)

        x = torch.reshape(x, (x.size(0), -1))

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return torch.squeeze(self.classifier(x))


class FeatureEmbeddingNetwork(nn.Module):

    def __init__(self, emb_size=64, input_size=1000):
        super().__init__()

        if input_size == 1000:
            emb_in = 1024
        elif input_size == 1600:
            emb_in = 1536
        else:
            raise ValueError(f"input_size: {input_size} is not supported")

        self.dropout = nn.Dropout(p=0.1)

        self.conv11 = nn.Conv1d(1, 32, 8, padding="same")
        self.conv12 = nn.Conv1d(32, 32, 8, padding="same")

        self.conv21 = nn.Conv1d(32, 64, 8, padding="same")
        self.conv22 = nn.Conv1d(64, 64, 8, padding="same")

        self.conv31 = nn.Conv1d(64, 128, 8, padding="same")
        self.conv32 = nn.Conv1d(128, 128, 8, padding="same")

        self.conv41 = nn.Conv1d(128, 256, 8, padding="same")
        self.conv42 = nn.Conv1d(256, 256, 8, padding="same")

        self.emb = nn.Linear(emb_in, emb_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = F.elu(self.conv11(x))
        x = F.elu(self.conv12(x))
        x = F.max_pool1d(x, 8, 4, padding=2)
        x = self.dropout(x)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool1d(x, 8, 4, padding=3)
        x = self.dropout(x)

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.max_pool1d(x, 8, 4, padding=3)
        x = self.dropout(x)

        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = F.max_pool1d(x, 8, 4, padding=3)

        x = torch.reshape(x, (x.shape[0], -1))
        x = torch.squeeze(self.emb(x))
        return x


def deepcorr_performance(gpu=True, flow_size=300, batch_size=256, n_test=300):

    # print("Simulating Deep Corr.")

    if torch.cuda.is_available() and gpu:
        dev = torch.device("cuda")
        begin_mem = torch.cuda.memory_allocated()
    else:
        dev = torch.device("cpu")

    model = DeepCorr(flow_size=flow_size, ruinit=True)
    criterion = nn.BCEWithLogitsLoss().to(dev)
    optimizer = optim.Adam(model.parameters())

    model.to(dev)
    if torch.cuda.is_available() and gpu:
        model_mem = torch.cuda.memory_allocated() - begin_mem
        # print(f"Model occupies {model_mem/1e6:.3f} MB GPU memory")

    # Measuing training
    train_set = FakeData((batch_size*12, 8, flow_size), mimic="deepcorr")
    train_loader = DataLoader(train_set, batch_size=batch_size)

    tdurs = []
    model.train()
    for i, (x_batch, y_batch) in enumerate(tqdm(train_loader, ascii=True, ncols=120)):
        tst = time.time()
        x_batch, y_batch = x_batch.type(torch.float).to(dev), y_batch.type(torch.float).to(dev)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        tdurs.append(time.time() - tst)

    tdurs = np.array(tdurs)
    # print("Disregard the total time shown in tqdm. The purpose of this loop is to get the average only.")
    # print(f"Each training iteration (after warm up) takes on average {np.mean(tdurs[2:]):.6f} seconds including moving data between devices, excluding recording the loss or evaluating training accuracy.")

    # Measuring testing
    model.eval()
    x_batch = []
    tdurs = []
    for i in tqdm(range(n_test), ascii=True, ncols=120):
        for j in range(n_test):
            x_batch.append(np.random.rand(8, 300).astype(np.float32))

            tst = time.time()

            if len(x_batch) >= batch_size or (i == n_test - 1 and j == n_test - 1):
                x_batch = np.stack(x_batch)     # (batch_size, 8, flow_size)
                x_batch = torch.from_numpy(x_batch).type(torch.float).to(dev)
                yprob_batch = torch.sigmoid(model(x_batch))
                yprob_batch = yprob_batch.numpy(force=True)
                x_batch = []
            
            tdurs.append(time.time() - tst)

    assert len(x_batch) == 0, "x_batch should be empty by now"

    # print(f"Testing {n_test} flow pairs takes {str(datetime.timedelta(seconds=sum(tdurs)))} including moving data between devices, inference and moving results between devices, excluding data preprocessing and result recording.")
    # totalsecfor1000pairs = sum(tdurs) / n_test * 1000 / n_test * 1000
    # print(f"So, for 1000 flow pairs, it will take {str(datetime.timedelta(seconds=totalsecfor1000pairs))}")
    # print(f"So, a single flow pair takes {sum(tdurs)/(n_test*n_test):.6f} seconds")
    print("pairs:", n_test)
    print("time:", sum(tdurs))


def deepcoffea_performance(gpu=True, batch_size=256, n_test=1000, n_wins=11, nonbatch=True):

    def triplet_loss(a_out, p_out, n_out, dev, alpha=0.1):
        pos_sim = F.cosine_similarity(a_out, p_out)
        neg_sim = F.cosine_similarity(a_out, n_out)
        zeros = torch.zeros(pos_sim.shape[0]).to(dev)               # (batch_size, )
        losses = torch.maximum(zeros, neg_sim - pos_sim + alpha)    # (batch_size, )
        return losses.mean()

    # print("Simulating Deep Coffea, flow length: 500, 800.")

    if torch.cuda.is_available() and gpu:
        dev = torch.device("cuda")
        begin_mem = torch.cuda.memory_allocated()
    else:
        dev = torch.device("cpu")

    anchor = FeatureEmbeddingNetwork(emb_size=64, input_size=500*2).to(dev)
    # if torch.cuda.is_available() and gpu:
    #     now_mem = torch.cuda.memory_allocated()
    #     print(f"Model anchor occupies {(now_mem - begin_mem)/1e6:.3f} MB GPU memory")

    pandn = FeatureEmbeddingNetwork(emb_size=64, input_size=800*2).to(dev)
    # if torch.cuda.is_available() and gpu:
    #     print(f"Model pandn occupies {(torch.cuda.memory_allocated() - now_mem)/1e6:.3f} MB GPU memory")

    
    trainable_params = list(anchor.parameters()) + list(pandn.parameters())
    optimizer = optim.SGD(trainable_params, lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)

    train_set = FakeData((batch_size*2, 500*2, 800*2), mimic="deepcoffea")
    train_loader = DataLoader(train_set, batch_size=batch_size)

    tdurs_train = []
    anchor.train()
    pandn.train()
    for i, (xa_batch, xp_batch, xn_batch) in enumerate(train_loader):
        tst = time.time()

        xa_batch, xp_batch, xn_batch = xa_batch.to(dev), xp_batch.to(dev), xn_batch.to(dev)

        optimizer.zero_grad()
        a_out = anchor(xa_batch)
        p_out = pandn(xp_batch)
        n_out = pandn(xn_batch)

        loss = triplet_loss(a_out, p_out, n_out, dev, 0.1)
        loss.backward()
        optimizer.step()

        tdurs_train.append(time.time() - tst)

    tdurs_train = np.array(tdurs_train)
    
    # print(f"Each training iteration (batch size 256, after warm up) takes on average {np.mean(tdurs_train[2:]):.6f} seconds including moving data between devices, excluding recording the loss or evaluating training accuracy.")

    # Measuring testing
    tdurs_inference = []
    tdurs_cosine_sim = []
    anchor.eval()
    pandn.eval()
    for _ in range(n_wins):

        if nonbatch:
            # This is basically identical to author's implementation where
            # they inference the entire test set at once.
            test_tor = np.random.rand(n_test, 500*2).astype(np.float32)
            test_exit = np.random.rand(n_test, 800*2).astype(np.float32)

            tst = time.time()
            xa_batch = torch.from_numpy(test_tor).type(torch.float).to(dev)
            xp_batch = torch.from_numpy(test_exit).type(torch.float).to(dev)
            a_out = anchor(xa_batch)
            p_out = pandn(xp_batch)
            tor_embs = a_out.numpy(force=True)
            exit_embs = p_out.numpy(force=True)
            tdurs_inference.append(time.time() - tst)

            ttst = time.time()
            _ = cosine_similarity(tor_embs, exit_embs)
            tdurs_cosine_sim.append(time.time() - ttst)

        else:
            # However if we have a lot more testing examples then the above will
            # quickly run out of GPU memory, so we batch the test set to make sure we always have
            # enough space and compute the cosine similarity table when the inference is done
            tor_embs, exit_embs = [], []
            test_set = FakeData((n_test, 500*2, 800*2), mimic="deepcoffea")
            test_loader = DataLoader(test_set, batch_size=batch_size)
            tst = time.time()
            for _, (xa_batch, xp_batch, _) in enumerate(test_loader):
                
                xa_batch, xp_batch = xa_batch.to(dev), xp_batch.to(dev)
                a_out = anchor(xa_batch)
                p_out = pandn(xp_batch)
                tor_embs.append(a_out.numpy(force=True))
                exit_embs.append(p_out.numpy(force=True))

            tor_embs = np.concatenate(tor_embs)
            exit_embs = np.concatenate(exit_embs)
            tdurs_inference.append(time.time() - tst)

            ttst = time.time()
            _ = cosine_similarity(tor_embs, exit_embs)
            tdurs_cosine_sim.append(time.time() - ttst)

    # print(f"Inference loop - each iteration takes on average {np.mean(ttdurs):.6f} seconds.")
    # print(f"Testing {n_test} flow pairs takes {str(datetime.timedelta(seconds=tdur))} including moving data between devices, inference, moving results between devices, and compute cosine similarity, excluding data preprocessing and result recording.")
    print(f"pairs: {n_test}, n wins: {n_wins}")
    print(f"time: {sum(tdurs_inference) + (sum(tdurs_cosine_sim)):.4f}s")
    print(f"(each window) tdurs_inference: {np.mean(tdurs_inference):.4f}s\n(each window) tdurs_cosine_sim: {np.mean(tdurs_cosine_sim):.4f}s")

if __name__ == "__main__":
    sol = "deepcoffea"
    if len(sys.args) > 1:
        sol = sys.args[1]

    if sol == "deepcoffea":
        n_pairs = [100, 1000, 10000, 20000, 30000, 40000, 50000, 60000]
        for i in n_pairs:
            print("n_test:", i)
            deepcoffea_performance(gpu=True, batch_size=128, n_test=i, nonbatch=False)
    else: # sol == "deepcorr"
        n_pairs = [2, 4, 8, 16, 32, 64, 128, 256]
        for i in n_pairs:
            print("n_test:", i)
            deepcorr_performance(gpu=True, flow_size=128, batch_size=128, n_test=i)

    # deepcorr_performance(gpu=True, flow_size=300, batch_size=128, n_test=300)
    # deepcoffea_performance(gpu=True, batch_size=128, n_test=1000)
    # n_pairs = [100, 1000, 10000, 20000, 30000, 40000, 50000, 60000]
    # for i in n_pairs:
    #     print("n_test:", i)
    #     deepcorr_performance(gpu=True, flow_size=300, batch_size=128, n_test=int(sqrt(i)))
    # n_pairs = [2294, 4589, 9179, 18359, 36718, 73437, 146874, 220311, 293748, 367185, 440622, 514059, 587496, 660933, 734370, 807807, 881244, 954681, 1028118, 1101555]

    # n_pairs = [100, 200, 300, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000]
    # n_pairs = [5000]

    # for i in n_pairs:
        # print("n_test:", i)
        #deepcoffea_performance(gpu=True, batch_size=128, n_test=i, nonbatch=True)
        # deepcoffea_performance(gpu=True, batch_size=128, n_test=i, nonbatch=False)
