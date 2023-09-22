import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from mycolorpy import colorlist as mcp
from typing import List, Dict, Tuple, Literal
import pickle
import sys
import math

import query_sumo_dataset


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


RED = '#d7261b'
LIGHT_ORANGE = '#fdae60'
ORANGE = '#f46d43'
YELLOW = '#fee091'
VERY_LIGHT_BLUE = '#e0f3f9'
LIGHT_BLUE = '#abd9e9'
MEDIUM_BLUE = '#91bfdb'
EMERALD = '#69B578'


#plt.rc('font', size=28)          # controls default text sizes
plt.rc('font', size=36)          # controls default text sizes
plt.rc('axes', titlesize=52)     # fontsize of the axes title
#plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
#plt.rc('axes', labelsize=46)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=42)    # fontsize of the tick labels
plt.rc('ytick', labelsize=42)    # fontsize of the tick labels
#plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('legend', fontsize=42)    # legend fontsize
plt.rc('figure', titlesize=56)  # fontsize of the figure title

plt.rcParams['figure.figsize'] = (24,14)
#plt.rcParams['figure.figsize'] = (24,9)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


def plot_triple_bars(FIGURES_RESULTS_FOLDER, metrics_map_per_client: dict[float, dict], dataset_name: str, threshold=-0.05) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(50, 26))
    
    ticks_font_size = 42
    font_size = 34
    label_font_size = 46

    clients = list(metrics_map_per_client[threshold].keys())

    #print(metrics_map_per_client)

    total_sessions_per_client = []
    for client, metrics in metrics_map_per_client[threshold].items():
        total_sessions_per_client.append(metrics.tp + metrics.fn)
    pps1 = ax1.bar(clients, total_sessions_per_client, color=MEDIUM_BLUE, width=0.8)

    for i, p in enumerate(pps1):
        height = total_sessions_per_client[i]
        ax1.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_sessions_per_client[i]), ha='center', fontsize=font_size)


    tps_per_client = []
    for i, (client, metrics) in enumerate(metrics_map_per_client[threshold].items()):
        tps_per_client.append(metrics.tp / total_sessions_per_client[i])
    pps2 = ax2.bar(clients, tps_per_client, color=EMERALD, width=0.8)

    for i, p in enumerate(pps2):
        height = tps_per_client[i]
        ax2.text(x=p.get_x() + p.get_width() / 2, y=height+.01, s="{:.2f}%".format(tps_per_client[i]), ha='center', fontsize=font_size)
        

    fps_per_client = []
    for i, (client, metrics) in enumerate(metrics_map_per_client[threshold].items()):
        fps_per_client.append(metrics.fp / total_sessions_per_client[i])
    pps3 = ax3.bar(clients, fps_per_client, color=EMERALD, width=0.8)

    for i, p in enumerate(pps3):
        height = fps_per_client[i]
        ax3.text(x=p.get_x() + p.get_width() / 2, y=height+.001, s="{:.2f}%".format(fps_per_client[i]), ha='center', fontsize=font_size)

    ax3.set_xticks(clients)
    #ax3.set_xticklabels(oses, rotation=-45)
    ax3.set_xticklabels(clients, rotation=90)
    ax3.set_xlabel("Client name", fontsize=label_font_size)
    ax1.set_ylabel("Number of\nsessions", fontsize=label_font_size)
    ax2.set_ylabel("Percentage of\ntrue positives", fontsize=label_font_size)
    ax3.set_ylabel("Percentage of\nfalse positives", fontsize=label_font_size)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(FIGURES_RESULTS_FOLDER + "clients_{}_tps_fps_percentage_thr_{}.pdf".format(dataset_name, threshold), bbox_inches='tight')
    plt.savefig(FIGURES_RESULTS_FOLDER + "clients_{}_tps_fps_percentage_thr_{}.png".format(dataset_name, threshold), bbox_inches='tight')
    plt.clf()