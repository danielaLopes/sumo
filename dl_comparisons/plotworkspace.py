#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# # Plot workspace
# This workspace let us test the proper style, font size, etc.

# In[ ]:


import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description="A program to test DeepCorr and DeepCoFFEA with SUMo.")
parser.add_argument("--deepcorr", help="Indicates if wants to test with DeepCorr. Requires having DeepCorr's data.", action="store_true")
parser.add_argument("--deepcoffea", help="Indicates if wants to test with DeepCoFFEA. Requires having DeepCoFFEA's data.", action="store_true")
args = parser.parse_args()
deepcorr = args.deepcorr
deepcoffea = args.deepcoffea

# 
# ## Compute stats
# Compare the number of packets, bytes transferred, durations of the flows from different datasets.

# In[ ]:

from compute_stats import compute_deepcorr_stats, compute_deepcoffea_stats, print_quartiles
data = {}
# get deepcorr
if deepcorr:
    stats = compute_deepcorr_stats("datasets/deepcorr_tar_bz2")
    data['deepcorr'] = {
        "clientdownnpkts": stats[0], "clientupnpkts": stats[1], "serverdownnpkts": stats[2], "serverupnpkts": stats[3], "clientdurs": stats[4], "serverdurs": stats[5], "clientdownbytes": stats[6], "clientupbytes": stats[7], "serverdownbytes": stats[8], "serverupbytes": stats[9], "n_flows": stats[10]
    }
    print_quartiles(stats, "datasets/deepcorr_tar_bz2")


# In[ ]:


# get deepcoffea
if deepcoffea:
    stats = compute_deepcoffea_stats("datasets/CrawlE_Proc")
    data['deepcoffea'] = {
        "clientdownnpkts": stats[0], "clientupnpkts": stats[1], "serverdownnpkts": stats[2], "serverupnpkts": stats[3], "clientdurs": stats[4], "serverdurs": stats[5], "clientdownbytes": stats[6], "clientupbytes": stats[7], "serverdownbytes": stats[8], "serverupbytes": stats[9], "n_flows": stats[10]
    }
    print_quartiles(stats, "datasets/CrawlE_Proc")


# In[ ]:

if deepcoffea:
    # get datasets_20230521_train
    stats = compute_deepcoffea_stats("datasets/datasets_20230521_train_deepcoffea")
    data['20230521_train_dcf'] = {
        "clientdownnpkts": stats[0], "clientupnpkts": stats[1], "serverdownnpkts": stats[2], "serverupnpkts": stats[3], "clientdurs": stats[4], "serverdurs": stats[5], "clientdownbytes": stats[6], "clientupbytes": stats[7], "serverdownbytes": stats[8], "serverupbytes": stats[9], "n_flows": stats[10]
    }
    print_quartiles(stats, "datasets/datasets_20230521_train_deepcoffea")

    # get datasets_20230521_test
    stats = compute_deepcoffea_stats("datasets/datasets_20230521_test_deepcoffea")
    data['20230521_test_dcf'] = {
        "clientdownnpkts": stats[0], "clientupnpkts": stats[1], "serverdownnpkts": stats[2], "serverupnpkts": stats[3], "clientdurs": stats[4], "serverdurs": stats[5], "clientdownbytes": stats[6], "clientupbytes": stats[7], "serverdownbytes": stats[8], "serverupbytes": stats[9], "n_flows": stats[10]
    }
    print_quartiles(stats, "datasets/datasets_20230521_test_deepcoffea")


# In[ ]:


plt.style.use('seaborn-v0_8-paper')
params = {
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False
}
plt.rcParams.update(params)


# In[ ]:


# plot the number of packets
keys = ["deepcoffea", "20230521_train_dcf", "20230521_test_dcf"]
labels = ["DeepCoFFEA", "OSTr+OSVal", "OSTest"]

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
positions = np.arange(0.74, 1.5*len(labels), 1.5)
xticks = np.arange(1, 1.5*len(labels), 1.5)

clientdownnpkts = [data[key]['clientdownnpkts'] for key in keys]
cldbplot = axs[0].boxplot(clientdownnpkts, positions=positions, widths=0.5, sym='', patch_artist=True)
for patch in cldbplot['boxes']:
    patch.set_facecolor("#FF7F0E")

clientupnpkts = [data[key]['clientupnpkts'] for key in keys]
clubplot = axs[0].boxplot(clientupnpkts, positions=positions + 0.52, widths=0.5, sym='', patch_artist=True)
for patch in clubplot['boxes']:
    patch.set_facecolor("#FED8B1")

axs[0].set_xticks(xticks, labels)
axs[0].set_ylim(top=6000)
axs[0].legend([cldbplot['boxes'][0], clubplot['boxes'][0]], ["Client received", "Client sent"], fontsize=18)
axs[0].grid(axis='y')

serverdownnpkts = [data[key]['serverdownnpkts'] for key in keys]
svdbplot = axs[1].boxplot(serverdownnpkts, positions=positions, widths=0.5, sym='', patch_artist=True)
for patch in svdbplot['boxes']:
    patch.set_facecolor("#1E77B4")

serverupnpkts = [data[key]['serverupnpkts'] for key in keys]
svubplot = axs[1].boxplot(serverupnpkts, positions=positions + 0.52, widths=0.5, sym='', patch_artist=True)
for patch in svubplot['boxes']:
    patch.set_facecolor("#00BFFF")

axs[1].set_xticks(xticks, labels)
axs[1].legend([svdbplot['boxes'][0], svubplot['boxes'][0]], ["Server received", "Server sent"], fontsize=18)
axs[1].set_ylim(top=9000)
axs[1].grid(axis='y')

fig.tight_layout()
results_dir = "datasets/stats"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
fig.savefig(f"datasets/stats/all_npkts_wdc.pdf")
fig.savefig(f"datasets/stats/all_npkts_wdc.png")


# In[ ]:


keys = ["deepcoffea", "20230521_train_dcf", "20230521_test_dcf"]
labels = ["DeepCoFFEA", "OSTr+OSVal", "OSTest"]

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
positions = np.arange(0.74, 1.5*len(labels), 1.5)
xticks = np.arange(1, 1.5*len(labels), 1.5)

clientdurs = [data[key]['clientdurs'] for key in keys]
clbplot = axs[0].boxplot(clientdurs, positions=positions, widths=0.5, sym='', patch_artist=True)
for patch in clbplot['boxes']:
    patch.set_facecolor("#FF7F0E")

serverdurs = [data[key]['serverdurs'] for key in keys]
svbplot = axs[0].boxplot(serverdurs, positions=positions + 0.52, widths=0.5, sym='', patch_artist=True)
for patch in svbplot['boxes']:
    patch.set_facecolor("#1E77B4")

axs[0].set_xticks(xticks, labels)
axs[0].set_ylabel("Durations (s)", fontsize=18)
axs[0].legend([clbplot['boxes'][0], svbplot['boxes'][0]], ["Client", "Server"], fontsize=18)
axs[0].grid(axis='y')


clnpktsps = []
svnpktsps = []
for key in keys:
    key_clnpkts = data[key]['clientdownnpkts'] + data[key]['clientupnpkts']
    zero_cldur_idxs = []
    for i, dur in enumerate(data[key]['clientdurs']):
        if dur == 0:
            zero_cldur_idxs.append(i)
    key_clnpkts = np.delete(key_clnpkts, zero_cldur_idxs)
    key_cldurs = np.delete(data[key]['clientdurs'], zero_cldur_idxs)
    clqs = np.percentile(key_clnpkts / key_cldurs, [25, 50, 75])
    clnpktsps.append(key_clnpkts / key_cldurs)

    key_svnpkts = data[key]['serverdownnpkts'] + data[key]['serverupnpkts']
    zero_svdur_idxs = []
    for i, dur in enumerate(data[key]['serverdurs']):
        if dur == 0:
            zero_svdur_idxs.append(i)
    key_svnpkts = np.delete(key_svnpkts, zero_svdur_idxs)
    key_svdurs = np.delete(data[key]['serverdurs'], zero_svdur_idxs)
    svqs = np.percentile(key_svnpkts / key_svdurs, [25, 50, 75])

    #print(f"{key}, cl qs: {clqs}, sv qs: {svqs}, avg qs: {(clqs+svqs)/2}")
    svnpktsps.append(key_svnpkts / key_svdurs)

clbplot = axs[1].boxplot(clnpktsps, positions=positions, widths=0.5, sym='', patch_artist=True)
for patch in clbplot['boxes']:
    patch.set_facecolor("#FF7F0E")

svbplot = axs[1].boxplot(svnpktsps, positions=positions + 0.52, widths=0.5, sym='', patch_artist=True)
for patch in svbplot['boxes']:
    patch.set_facecolor("#1E77B4")

axs[1].set_xticks(xticks, labels)
axs[1].set_ylim(bottom=-10, top=200)
axs[1].set_ylabel("Packets per second", fontsize=18)
axs[1].legend([clbplot['boxes'][0], svbplot['boxes'][0]], ["Client", "Server"], fontsize=18)
axs[1].grid(axis='y')
fig.tight_layout()

fig.savefig(f"datasets/stats/all_durs_npktps_wdc.pdf")
fig.savefig(f"datasets/stats/all_durs_npktps_wdc.png")


# ## DeepCorr evaluation
# Reproduce DeepCorr's Figure 4 and 5.
# 
# Figure 4: TPR vs ETA, FPR vs ETA. ETA is the detection threshold.
# 
# Figure 5: TPR vs FPR (ROC Curve)

# In[ ]:

if deepcorr:
    corr_matrix_path = "experiments/deepcorr_deepcorr_tar_bz2_us199_fs300_nt1000_ruTrue_lr1e-04_mep60_bs128/ep-003_racc0.835_metrics.npz"
    corr_matrix_path = pathlib.Path(corr_matrix_path)

    loaded = np.load(corr_matrix_path)
    corr_matrix = loaded["corr_matrix"]
    assert corr_matrix.shape[0] == corr_matrix.shape[1], "not a square matrix"

    labels = np.eye(corr_matrix.shape[0])


    # In[ ]:


    etas = [i/100 for i in range(0, 100, 2)]
    tprs = []
    fprs = []
    for eta in etas:
        eta_yhats = (corr_matrix >= eta).astype(np.int64)

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if eta_yhats[i,j] == 1 and labels[i,j] == 1:
                    tp += 1
                elif eta_yhats[i,j] == 1 and labels[i,j] == 0:
                    fp += 1
                elif eta_yhats[i,j] == 0 and labels[i,j] == 1:
                    fn += 1
                else:
                    tn += 1

        tprs.append(tp / (tp + fn))
        fprs.append(fp / (fp + tn))


# In[ ]:


    plt.style.use('seaborn-v0_8-paper')
    params = {
        'axes.titlesize': 20,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
    }
    plt.rcParams.update(params)


    # In[ ]:


    plt.clf()
    fig, ax = plt.subplots(1, 1)
    ax.plot(etas, tprs, linestyle="solid", marker="o", markersize=6)
    ax.set_xlabel("Threshold (eta).")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("TPR with different etas.")
    ax.grid()
    fig.tight_layout()


    # In[ ]:


    plt.clf()
    fig, ax = plt.subplots(1, 1)
    ax.plot(etas, fprs, linestyle="solid", marker="o")
    ax.set_xlabel("Threshold (eta).")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR with different etas.")
    ax.grid()
    fig.tight_layout()


# In[ ]:


    plt.clf()
    fig, ax = plt.subplots(1, 1)
    ax.plot(fprs, tprs, linestyle="solid", marker="o")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve.")
    ax.grid()
    fig.tight_layout()


# ## DeepCoffea evaluation
# Reproduce DeepCoffea's Figure 6 and 7(b).
# 
# Figure 6: ROC curve for global vs local threshold
# 
# Figure 7(b): ROC at various loss values

# In[ ]:


def get_tprsfprsprs_localthr(corr_matrix, n_wins, vote_thr):
    n_test = corr_matrix.shape[0] // n_wins
    tprs = []
    fprs = []
    prs = []
    for kappa in tqdm(range(0, n_test, 10), ascii=True, ncols=120):
        votes = np.zeros((n_test, n_test), dtype=np.int64)
        for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            thresholds = []
            for i in range(n_test):
                corr_v = corr_matrix_win[i] # (n_test,)
                corr_v_sorted = np.sort(corr_v)[::-1]
                thresholds.append(corr_v_sorted[kappa])

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= thresholds[i]:
                        votes[i,j] += 1

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(n_test):
            for j in range(n_test):
                if votes[i,j] >= vote_thr and i == j:
                    tp += 1
                elif votes[i,j] >= vote_thr and i != j:
                    fp += 1
                elif votes[i,j] < vote_thr and i == j:
                    fn += 1
                else:   # votes[i,j] < vote_thr and i != j
                    tn += 1

        if tp + fn == 0:
            tprs.append(0.0)
        else:
            tprs.append(tp / (tp + fn))
        
        if fp + tn == 0:
            fprs.append(0.0)
        else:
            fprs.append(fp / (fp + tn))

        if tp + fp == 0:
            prs.append(0.0)
        else:
            prs.append(tp / (tp + fp))

    return tprs, fprs, prs

def get_tprsfprsprs_globalthr(corr_matrix, n_wins, vote_thr):
    n_test = corr_matrix.shape[0] // n_wins
    tprs = []   # true positive rate, recall, sensitivity
    fprs = []   # false positive rate
    prs = []    # precision, positive predictive value
    for eta in tqdm(np.arange(-0.35, 0.65, 0.025), ascii=True, ncols=120):
        votes = np.zeros((n_test, n_test), dtype=np.int64)
        for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= eta:
                        votes[i,j] += 1

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(n_test):
            for j in range(n_test):
                if votes[i,j] >= vote_thr and i == j:
                    tp += 1
                elif votes[i,j] >= vote_thr and i != j:
                    fp += 1
                elif votes[i,j] < vote_thr and i == j:
                    fn += 1
                else:   # votes[i,j] < vote_thr and i != j
                    tn += 1

        if tp + fn == 0:
            tprs.append(0.0)
        else:
            tprs.append(tp / (tp + fn))
        
        if fp + tn == 0:
            fprs.append(0.0)
        else:
            fprs.append(fp / (fp + tn))

        if tp + fp == 0:
            prs.append(0.0)
        else:
            prs.append(tp / (tp + fp))

    return tprs, fprs, prs


if deepcoffea:
    best_npz_path = pathlib.Path("experiments/deepcoffea_data230521_d3.0_ws5.0_nw5_thr20_tl300_el500_nt0_ap1e-01_es64_lr1e-03_mep100000_bs256/ep-975_loss0.00199_metrics.npz")
    lsetup = "_".join(best_npz_path.parent.name.split("_")[-12:-9])

    result_fpath = pathlib.Path("./datasets/stats/lthr.p")
    if result_fpath.exists():
        with open(result_fpath, "rb") as fp:
            ltprs, lfprs, lprs = pickle.load(fp)
    else:
        loaded = np.load(best_npz_path)
        ltprs, lfprs, lprs = get_tprsfprsprs_localthr(loaded['corr_matrix'], 5, 3)
        with open(result_fpath, "wb") as fp:
            pickle.dump((ltprs, lfprs, lprs), fp)


# In[ ]:


    ltprs, lfprs, lprs = {}, {}, {}
    with open("./datasets/stats/lthr.p", "rb") as fp:
        ltprs['d3.0_ws5.0_nw5'], lfprs['d3.0_ws5.0_nw5'], lprs['d3.0_ws5.0_nw5'] = pickle.load(fp)

    with open("./datasets/stats/lthr_d2.p", "rb") as fp:
        ltprs['d2.0_ws3.0_nw7'], lfprs['d2.0_ws3.0_nw7'], lprs['d2.0_ws3.0_nw7'] = pickle.load(fp)


# In[ ]:


    npz_paths = [
        "experiments/deepcoffea_data230521_d2.0_ws3.0_nw7_thr15_tl200_el300_nt0_ap1e-01_es64_lr1e-03_mep100000_bs256/ep-1441_loss0.00292_metrics.npz",
        "experiments/deepcoffea_data230521_d3.0_ws5.0_nw5_thr20_tl300_el500_nt0_ap1e-01_es64_lr1e-03_mep100000_bs256/ep-975_loss0.00199_metrics.npz"
    ]

    result_fpath = pathlib.Path("./datasets/stats/gthr.p")
    if result_fpath.exists():
        with open(result_fpath, "rb") as fp:
            gtprs, gfprs, gprs = pickle.load(fp)
    else:
        gtprs, gfprs, gprs = {}, {}, {}
        for npz_path in npz_paths:
            npz_path = pathlib.Path(npz_path)
            fields = npz_path.parent.name.split("_")
            n_wins = int(fields[-10].split("nw")[1])
            setup = "_".join(fields[-12:-9])
            
            loaded = np.load(npz_path)
            corr_matrix = loaded['corr_matrix']
            loss_mean = loaded['loss_mean']

            if n_wins == 5:
                vote_thr = 3
            elif n_wins == 7:
                vote_thr = 4
            elif n_wins == 9:
                vote_thr = 5
            elif n_wins == 11:
                vote_thr = 9    # the number dcf authors used

            tprs, fprs, prs = get_tprsfprsprs_globalthr(corr_matrix, n_wins, vote_thr)
            gtprs[setup] = tprs
            gfprs[setup] = fprs
            gprs[setup] = prs

        with open(result_fpath, "wb") as fp:
            pickle.dump((gtprs, gfprs, gprs), fp)


# In[ ]:


    plt.style.use('seaborn-v0_8-paper')
    params = {
        'axes.titlesize': 18,
        'axes.labelsize': 20,
        'font.size': 10,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'text.usetex': False,
        'axes.spines.top': False,
        'axes.spines.right': False
    }
    plt.rcParams.update(params)


    # In[ ]:


    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

    setups = []
    for key in gtprs.keys():
        setups.append("_".join(key.split("_")))
    setups = sorted(set(setups))

    for key, gtprs_data in gtprs.items():
        gfprs_data = gfprs[key]
        ltprs_data = ltprs[key]
        lfprs_data = lfprs[key]
        
        setupi = setups.index(key)
        if setupi == 0:
            linestyle = "solid"
            label = "setup-1"
        elif setupi == 1:
            linestyle = "dashed"
            label = "setup-2"
        else:
            raise ValueError(f"setupi: {setupi} not supported now.")
        
        ax.plot(gfprs_data, gtprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_g")
        ax.plot(lfprs_data, ltprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_l")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"datasets/stats/dcf_eval_roc.pdf")
    fig.savefig(f"datasets/stats/dcf_eval_roc.png")

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    for key, gtprs_data in gtprs.items():
        gprs_data = gprs[key]
        ltprs_data = ltprs[key]
        lprs_data = lprs[key]
        
        setupi = setups.index(key)
        if setupi == 0:
            linestyle = "solid"
            label = "setup-1"
        elif setupi == 1:
            linestyle = "dashed"
            label = "setup-2"
        else:
            raise ValueError(f"setupi: {setupi} not supported now.")
        
        ax.plot(gtprs_data, gprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_g")
        ax.plot(ltprs_data, lprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_l")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"datasets/stats/dcf_eval_pr.pdf")
    fig.savefig(f"datasets/stats/dcf_eval_pr.png")









