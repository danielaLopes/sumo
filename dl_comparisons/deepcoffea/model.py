import time
import pathlib
import datetime
import argparse

import numpy as np

# pytorch version 1.13
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import DeepCoffeaDataset, TripletSampler


class FeatureEmbeddingNetwork(nn.Module):

    def __init__(self, emb_size=64, input_size=1000):
        super().__init__()

        if input_size == 1000:
            emb_in = 1024
            self.last_pool = [8, 4, 3]
        elif input_size == 1600:
            emb_in = 1536
            self.last_pool = [8, 4, 3]
        elif input_size == 400:
            emb_in = 1024
            self.last_pool = [4, 2, 2]
        elif input_size == 600:
            emb_in = 1536
            self.last_pool = [4, 2, 2]
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
        if len(x.size()) == 2:
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
        x = F.max_pool1d(x, self.last_pool[0], self.last_pool[1], padding=self.last_pool[2])
        x = self.dropout(x)

        x = torch.reshape(x, (x.size(0), -1))
        x = torch.squeeze(self.emb(x))
        return x


def triplet_loss(a_out, p_out, n_out, dev, alpha=0.1):
    # cosine similarity is the default correlation function used in the paper
    #  other distance function such as softmax, kNN clustering can be used.
    # a_out.size() (or p_out/n_out): (batch_size, emb_size)
    pos_sim = F.cosine_similarity(a_out, p_out)     # (batch_size,)
    neg_sim = F.cosine_similarity(a_out, n_out)     # (batch_size,)

    zeros = torch.zeros(pos_sim.size(0), device=dev)           # (batch_size,)
    losses = torch.maximum(zeros, neg_sim - pos_sim + alpha)    # (batch_size,)
    return losses.mean()


@torch.no_grad()
def inference(anchor, pandn, loader, dev):
    loader.sampler.train = False
    anchor.eval()
    pandn.eval()

    tor_embs = []
    exit_embs = []
    for _, (xa_batch, xp_batch) in enumerate(loader):
        xa_batch, xp_batch = xa_batch.to(dev), xp_batch.to(dev)

        a_out = anchor(xa_batch)
        p_out = pandn(xp_batch)

        tor_embs.append(a_out.numpy(force=True))
        exit_embs.append(p_out.numpy(force=True))

    tor_embs = np.concatenate(tor_embs)     # (N, emb_size)
    exit_embs = np.concatenate(exit_embs)   # (N, emb_size)
    print(f"Inference {len(loader.dataset)} pairs done.")
    return tor_embs, exit_embs


def main(mode: str,
         delta: int,
         win_size: int,
         n_wins: int,
         threshold: int,
         tor_len: int,
         exit_len: int,
         n_test: int,
         alpha: float,
         emb_size: int,
         lr: float,
         max_ep: int,
         batch_size: int,
         data_root: str,
         ckpt: str):
    assert mode in set(["train", "test"]), f"mode: {mode} is not supported"

    # To ensure device-agnostic reproducibility
    torch.manual_seed(114)
    rng = np.random.default_rng(114)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)   # If set to True, will throw error when encountering non-deterministic ops
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    data_root = pathlib.Path(data_root)

    if mode == "train":

        anchor = FeatureEmbeddingNetwork(emb_size=emb_size, input_size=tor_len*2).to(dev)
        pandn = FeatureEmbeddingNetwork(emb_size=emb_size, input_size=exit_len*2).to(dev)

        if "dataset_21march_2022_3" in data_root.name:
            data_name = "21march"
        elif "20x15_29november_2022_v1" in data_root.name:
            data_name = "29november"
        elif "30november_2022_v1" in data_root.name:
            data_name = "30november"
        elif "CrawlE_Proc" in data_root.name:
            data_name = "deepcoffea"
        elif "datasets_20230521" in data_root.name:
            data_name = "data230521"
        else:
            raise ValueError(f"data: {data_root.name} is not supported.")

        save_dir = pathlib.Path("./experiments") / f"deepcoffea_{data_name}_d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_ap{alpha:.0e}_es{emb_size}_lr{lr:.0e}_mep{max_ep}_bs{batch_size}"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        trainable_params = list(anchor.parameters()) + list(pandn.parameters())
        #optimizer = optim.SGD(trainable_params, lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = optim.Adam(trainable_params, lr=lr)

        train_set = DeepCoffeaDataset(data_root, delta, win_size, n_wins, threshold, tor_len, exit_len, n_test, True)
        train_sampler = TripletSampler(train_set, alpha, rng, True)
        train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size)

        test_set = DeepCoffeaDataset(data_root, delta, win_size, n_wins, threshold, tor_len, exit_len, n_test, False)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        best_loss_mean = 0.006

        for ep in range(max_ep):
            if ep != 0:
                # compute the cosine similarity table
                tor_embs, exit_embs = inference(anchor, pandn, train_loader, dev)
                train_set.sim_table = cosine_similarity(tor_embs, exit_embs)    # this is going to take a while

            tst = time.time()
            
            train_sampler.train = True
            losses = []
            anchor.train()
            pandn.train()
            for i, (xa_batch, xp_batch, xn_batch) in enumerate(train_loader):
                xa_batch, xp_batch, xn_batch = xa_batch.to(dev), xp_batch.to(dev), xn_batch.to(dev)          

                optimizer.zero_grad()
                a_out = anchor(xa_batch)
                p_out = pandn(xp_batch)
                n_out = pandn(xn_batch)

                loss = triplet_loss(a_out, p_out, n_out, dev, alpha)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())

                if i % 100 == 0:
                    print(f"[{ep+1:03d}/{max_ep}] [{i:04d}/{len(train_loader):04d}] Loss: {loss.item():.4f}")

            tdur = time.time() - tst
            print(f"[{ep+1:03d}/{max_ep}] (Training) Loss μ: {np.mean(losses):.4f}, σ: {np.std(losses):.4f}, dur: {str(datetime.timedelta(seconds=tdur))}")

            # generate the corr_matrix and save it, evaluate it while generating the plots
            tor_embs, exit_embs = inference(anchor, pandn, test_loader, dev)
            corr_matrix = cosine_similarity(tor_embs, exit_embs)

            if np.mean(losses) < best_loss_mean:
                best_loss_mean = np.mean(losses)
                print("Best training loss (avg) so far.\n")

                # save the metrics
                np.savez_compressed(save_dir / f"ep-{ep+1:03d}_loss{best_loss_mean:.5f}_metrics", corr_matrix=corr_matrix, loss_mean=best_loss_mean)

                # save the model snapshot
                torch.save({
                    "ep": ep+1,
                    "anchor_state_dict": anchor.state_dict(),
                    "pandn_state_dict": pandn.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }, save_dir / f"best_loss.pth")

            if np.mean(losses) < 0.001:
                break

    else:
        if ckpt is None:
            raise ValueError("ckpt is not set!")
        
        ckpt = pathlib.Path(ckpt).resolve()
        fields = ckpt.parent.name.split("_")

        delta = int(fields[-12].split("d")[-1])
        win_size = int(fields[-11].split("ws")[-1])
        n_wins = int(fields[-10].split("nw")[-1])
        threshold = int(fields[-9].split("thr")[-1])
        tor_len = int(fields[-8].split("tl")[-1])
        exit_len = int(fields[-7].split("el")[-1])
        n_test = int(fields[-6].split("nt")[-1])
        emb_size = int(fields[-4].split("es")[-1])
        batch_size = int(fields[-1].split("bs")[-1])
        #ep = int(ckpt.name.split("_")[0].split("-")[1])

        pth_content = torch.load(ckpt)

        anchor = FeatureEmbeddingNetwork(emb_size=emb_size, input_size=tor_len*2).to(dev)
        anchor.load_state_dict(pth_content["anchor_state_dict"])
        pandn = FeatureEmbeddingNetwork(emb_size=emb_size, input_size=exit_len*2).to(dev)
        pandn.load_state_dict(pth_content["pandn_state_dict"])
        print(f"Snapshot: '{ckpt}' loaded")

        test_set = DeepCoffeaDataset(data_root, delta, win_size, n_wins, threshold, tor_len, exit_len, n_test, False)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        tor_embs, exit_embs = inference(anchor, pandn, test_loader, dev)
        corr_matrix = cosine_similarity(tor_embs, exit_embs)

        np.savez_compressed(ckpt.parent / f"best_loss_corrmatrix", corr_matrix=corr_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Coffea.")
    parser.add_argument("--mode", default="train", type=str, help="To train or test.")
    parser.add_argument("--delta", default=3, type=float, help="For window partition (see data_utils.py).")
    parser.add_argument("--win_size", default=5, type=float, help="For window partition (see data_utils.py).")
    parser.add_argument("--n_wins", default=11, type=int, help="For window partition (see data_utils.py).")
    parser.add_argument("--threshold", default=20, type=int, help="For window partition (see data_utils.py).")
    parser.add_argument("--tor_len", default=500, type=int, help="Flow size for the tor pairs.")
    parser.add_argument("--exit_len", default=800, type=int, help="Flow size for the exit pairs.")
    parser.add_argument("--n_test", default=1000, type=int, help="Number of testing flow pairs.")
    parser.add_argument("--alpha", default=0.1, type=float, help="For triplet loss.")
    parser.add_argument("--emb_size", default=64, type=int, help="Feature embedding size.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--ep", default=100000, type=int, help="Epochs to train.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
    
    
    parser.add_argument("--data_root", required=True, type=str, help="Path to preprocessed .npz.")
    parser.add_argument("--ckpt", default=None, type=str, help="Load path for the checkpoint model.")
    args = parser.parse_args()
    main(args.mode, args.delta, args.win_size, args.n_wins, args.threshold, args.tor_len, args.exit_len, args.n_test, args.alpha, args.emb_size, args.lr, args.ep, args.batch_size, args.data_root, args.ckpt)
