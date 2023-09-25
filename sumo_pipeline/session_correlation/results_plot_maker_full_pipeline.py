import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from mycolorpy import colorlist as mcp
import pickle
import sys

from constants import *


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


RED = '#d7261b'
ORANGE = '#fdae60'
YELLOW = '#fee091'
LIGHT_BLUE = '#abd9e9'


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


def precision_recall_variation_with_duration_full_pipeline(min_session_durations, dataset_name) -> None:  
    plt.rcParams['figure.figsize'] = (24,7)

    precision_color = 'tab:blue'
    recall_color = 'tab:orange'
    x_axis = np.array(list(min_session_durations.keys())) / 60 # minutes to seconds

    y_axis_precision = []
    y_axis_recall = []

    for min_session_duration in min_session_durations.keys():
        y_axis_precision.append(min_session_durations[min_session_duration].precision)
        y_axis_recall.append(min_session_durations[min_session_duration].recall)

    fig, ax1 = plt.subplots()
    ax1.plot(x_axis, y_axis_precision, color=precision_color, linewidth=4, label="Precision")
    ax1.plot(x_axis, y_axis_recall, color=recall_color, linewidth=4, label="Recall")

    #ax1.set_ylim(0, 1.05)
    #ax1.set_ylim(0.5, 1.05)
    ax1.set_ylim(0.7, 1.05)
    ax1.set_xlim(0, 20)
    ax1.set_ylabel('Metrics', fontsize=50)
    ax1.set_xlabel('Minimum session duration (minutes)', fontsize=50)
    #ylabels = np.arange(0, 120, 20)
    #ylabels = np.arange(50, 110, 10)
    ylabels = np.arange(70, 110, 10)
    xlabels = np.arange(0, 21, 2)
    ylabels_final = []
    for i, label in enumerate(ylabels):
        ylabels_final.append(label / 100)
    ax1.set_yticks(ylabels_final, ylabels_final)
    ax1.set_xticks(xlabels, xlabels)

    for y_label in ylabels_final:
        ax1.axhline(y=y_label, color='gray', linestyle='-', alpha=0.3)

    #plt.legend(loc="upper right", bbox_to_anchor=(0.99, 0.9))
    plt.legend(loc="lower right")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig('{}precision_recall_variation_with_duration_full_pipeline_{}.png'.format(FIGURES_FULL_PIPELINE_RESULTS_FOLDER, dataset_name))
    plt.savefig('{}precision_recall_variation_with_duration_full_pipeline_{}.pdf'.format(FIGURES_FULL_PIPELINE_RESULTS_FOLDER, dataset_name))
    plt.clf()


def precision_recall_variation_with_duration_full_pipeline_table(min_session_durations, dataset_name) -> None:  
    print("\n\n\\begin{table}[t] \n    \
            \\footnotesize \n   \
            \\addtolength{\\tabcolsep}{0.18em} \n    \
            \\vspace{0.1cm} \n     \
            \\resizebox{\\linewidth}{!}{%  \n  \
            \\begin{tabular}{ccc} \n   \
            \\Xhline{1pt} \n   \
            \\textbf{Min duration} & \\textbf{Precision} & \\textbf{Recall} \\\\ \\hline \n")
    
    for min_session_duration in min_session_durations.keys():
        print(f"{int(min_session_duration / 60)} & {round(min_session_durations[min_session_duration].precision, 4):%.4f} & {round(min_session_durations[min_session_duration].recall, 4):%.4f} \\\\")

    print("\n\\Xhline{1pt}  \n     \
            \\end{tabular} \n    \
            } \n    \
            \\vspace{-0.3cm} \n     \
            \\caption{\\label{tab:full-pipeline-results}Precision and recall when varying minimum session duration analysis of full pipeline execution.} \n   \
            \\vspace{-0.5cm} \n     \
            \\end{table}")
