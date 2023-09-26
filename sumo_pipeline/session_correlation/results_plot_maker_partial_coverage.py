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


def precision_recall_curve_with_threshold_excluding_zones(results_by_zone, dataset_name):
    zones = list(results_by_zone.keys())
    thresholds = list(results_by_zone[zones[0]].keys())
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']
    line_styles = ['-', '--', '-.', ':', '-']
    
    fig, ax = plt.subplots()
    for i, zone in enumerate(zones):
        x_axis_recall = []
        y_axis_precision = []
        for threshold in thresholds:
            x_axis_recall.append(results_by_zone[zone][threshold].recall)
            y_axis_precision.append(results_by_zone[zone][threshold].precision)

        total_flows = results_by_zone[zone][thresholds[0]].tp + results_by_zone[zone][thresholds[0]].fn

        if i == 0:
            legend = "All regions, {} pairs".format(total_flows)
        else:
            legend = "Without {}, {} pairs".format(zone, total_flows)

        if i == len(zones) - 1:
            ax.plot(x_axis_recall, y_axis_precision, color=colors[i], marker='s', zorder=0, linestyle=line_styles[i], linewidth=8, label=legend)
        else:
            ax.plot(x_axis_recall, y_axis_precision, color=colors[i], zorder=0, linestyle=line_styles[i], linewidth=8, label=legend)

    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.tick_params(width=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.set_ylabel('Precision', fontsize=52)
    ax.set_xlabel('Recall', fontsize=52)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower right', labelspacing=0.3)

    plt.xticks(rotation=45)
    #fig.tight_layout()


    plt.savefig('{}precision_recall_curve_with_threshold_excluding_zones_{}.png'.format(FIGURES_RESULTS_FOLDER, dataset_name), bbox_inches='tight')
    plt.savefig('{}precision_recall_curve_with_threshold_excluding_zones_{}.pdf'.format(FIGURES_RESULTS_FOLDER, dataset_name), bbox_inches='tight')
    plt.clf()


def precision_recall_curve_with_threshold_by_eu_country(results_by_zone, coverage_percentages, dataset_name):
    zones = list(results_by_zone.keys())
    thresholds = list(results_by_zone[zones[0]].keys())
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']
    line_styles = ['-', '--', '-.', ':', '-']
    
    fig, ax = plt.subplots()
    for i, (coverage, results) in enumerate(results_by_zone.items()):
        x_axis_recall = []
        y_axis_precision = []
        for threshold in thresholds:
            x_axis_recall.append(results[threshold].recall)
            y_axis_precision.append(results[threshold].precision)

        #total_flows = results_by_zone[zone][thresholds[0]].tp + results_by_zone[zone][thresholds[0]].fp + results_by_zone[zone][thresholds[0]].tn + results_by_zone[zone][thresholds[0]].fn
        total_flows = results[thresholds[0]].tp + results[thresholds[0]].fn

        if i == 0:
            legend = "{:.0f}% coverage, {} pairs".format(coverage * 100, total_flows)
        else:
            legend = "{:.2f}% coverage, {} pairs".format(coverage * 100, total_flows)
        if i == len(zones) - 1:
            ax.plot(x_axis_recall, y_axis_precision, color=colors[i], marker='s', zorder=0, linestyle=line_styles[i], linewidth=8, label=legend)
        else:
            ax.plot(x_axis_recall, y_axis_precision, color=colors[i], zorder=0, linestyle=line_styles[i], linewidth=8, label=legend)

    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.tick_params(width=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1)
    ax.set_ylabel('Precision', fontsize=52)
    ax.set_xlabel('Recall', fontsize=52)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower right', labelspacing=0.3, fontsize=40)

    plt.xticks(rotation=45)
    #fig.tight_layout()


    plt.savefig('{}precision_recall_curve_with_threshold_by_eu_country_{}.png'.format(FIGURES_RESULTS_FOLDER, dataset_name), bbox_inches='tight')
    plt.savefig('{}precision_recall_curve_with_threshold_by_eu_country_{}.pdf'.format(FIGURES_RESULTS_FOLDER, dataset_name), bbox_inches='tight')
    plt.clf()

