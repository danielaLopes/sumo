import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from mycolorpy import colorlist as mcp
import pickle
import sys
import math
import glob
import os

import query_sumo_dataset


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


# https://stackoverflow.com/questions/69520456/how-can-i-plot-a-cdf-in-matplotlib-without-binning-my-data
def ecdf4plot(seq, assumeSorted = False):
    """
    In:
    seq - sorted-able object containing values
    assumeSorted - specifies whether seq is sorted or not
    Out:
    0. values of support at both points of jump discontinuities
    1. values of ECDF at both points of jump discontinuities
       ECDF's true value at a jump discontinuity is the higher one    
    """
    if len(seq) == 0:
        return [], []
    if not assumeSorted:
        seq = sorted(seq)
    prev = seq[0]
    n = len(seq)
    support = [prev]
    ECDF = [0.]
    for i in range(1, n):
        seqi = seq[i]
        if seqi != prev:
            preP = i/n
            support.append(prev)
            ECDF.append(preP)
            support.append(seqi)
            ECDF.append(preP)
            prev = seqi
    support.append(prev)
    ECDF.append(1.)
    return support, ECDF


def normalize(data):
    min_val = min(data)
    max_val = max(data)

    normalized_data = {}
    for el in data:
        normalized_data[el] = (el - min_val) / (max_val - min_val)

    return normalized_data


"""
# ---------------------- Plots ----------------------
def precision_recall_variation_with_threshold(figures_results_folder, metricsMapFinalScores, dataset_name):

    precision_color = 'tab:blue'
    recall_color = 'tab:orange'
    x_axis = list(metricsMapFinalScores.keys())

    y_axis_precision = []
    y_axis_recall = []

    for threshold in metricsMapFinalScores.keys():
        y_axis_precision.append(metricsMapFinalScores[threshold]['precision'])
        y_axis_recall.append(metricsMapFinalScores[threshold]['recall'])

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(x_axis, y_axis_precision, color=precision_color, linewidth=4)
    ax2.plot(x_axis, y_axis_recall, color=recall_color, linewidth=4)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(min(x_axis), max(x_axis))
    ax1.set_ylabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_xlabel('Threshold')
    ax1.set_title('Precision and recall variation with multiple thresholds')

    plt.savefig('{}precision_recall_variation_with_threshold_{}.png'.format(figures_results_folder, dataset_name))
    plt.savefig('{}precision_recall_variation_with_threshold_{}.pdf'.format(figures_results_folder, dataset_name))
    plt.clf()
"""


def precision_recall_curve(figures_results_folder, metrics_map_final_scores, dataset_name):
    thresholds = list(metrics_map_final_scores.keys())
    
    x_axis_recall = []
    y_axis_precision = []

    for threshold in thresholds:
        x_axis_recall.append(metrics_map_final_scores[threshold].recall)
        y_axis_precision.append(metrics_map_final_scores[threshold].precision)

    fig, ax = plt.subplots(1,1, figsize=(6, 8))
    
    ax.plot(x_axis_recall, y_axis_precision, color='tab:blue', zorder=0, linewidth=4)

    #ax.set_xlim(0, 0.5)
    #ax.set_ylim(0, 1)
    ax.set_xlim(0, 0.01)
    ax.set_ylim(0, 0.01)
    ax.set_ylabel('Precision', fontsize=40)
    ax.set_xlabel('Recall', fontsize=40)
    plt.yticks(fontsize=30)
    #plt.xticks([0, 0.25, 0.5], fontsize=30)
    plt.xticks(fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.xticks(rotation=45)
    fig.tight_layout()

    plt.savefig('{}precision_recall_curve_{}.png'.format(figures_results_folder, dataset_name), bbox_inches='tight')
    plt.savefig('{}precision_recall_curve_{}.pdf'.format(figures_results_folder, dataset_name), bbox_inches='tight')
    plt.clf()


def precision_recall_curve_with_threshold(figures_results_folder, metrics_map_final_scores, dataset_name):
    """

    """
    #cmap.set_under('red')

    thresholds = list(metrics_map_final_scores.keys())

    num_colors = len(thresholds)
    #cmap = plt.get_cmap('plasma', num_colors)
    #cmap = LinearSegmentedColormap.from_list('name', ['red', 'blue'])
    cmap = LinearSegmentedColormap.from_list('name', [RED, 'tab:blue'])

    x_axis_recall = []
    y_axis_precision = []

    fig, ax = plt.subplots()
    for threshold in thresholds:
        x_axis_recall.append(metrics_map_final_scores[threshold].recall)
        y_axis_precision.append(metrics_map_final_scores[threshold].precision)

    normalized = normalize(thresholds)
    #print("\n ---- normalized:", normalized)
    thresholds = list(normalized.values())
    plot = ax.scatter(x_axis_recall, y_axis_precision, c=thresholds, cmap=cmap, vmin=min(thresholds), vmax=max(thresholds))
    #ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    #ax.set_xticklabels(np.arange(0.1, 1.1, 0.1), rotation=45)
    #fig.colorbar(plot, extend='min')
    fig.colorbar(plot)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title('Precision and recall curve with multiple thresholds')

    plt.savefig('{}precision_recall_curve_with_threshold.png'.format(figures_results_folder))
    plt.savefig('{}precision_recall_curve_with_threshold.pdf'.format(figures_results_folder))
    plt.clf()

    fig, ax = plt.subplots()
    thresholds_without_outliers = []
    x_axis_recall_without_outliers = []
    y_axis_precision_without_outliers = []
    for i, threshold in enumerate(thresholds):
        if x_axis_recall[i] > 0.1:
            thresholds_without_outliers.append(threshold)
            x_axis_recall_without_outliers.append(x_axis_recall[i])
            y_axis_precision_without_outliers.append(y_axis_precision[i])
    
    ax.plot(x_axis_recall_without_outliers, y_axis_precision_without_outliers, color='blue', zorder=0, linewidth=4)
    plot = ax.scatter(x_axis_recall_without_outliers, y_axis_precision_without_outliers, c=thresholds_without_outliers, cmap=cmap, \
                    vmin=min(thresholds_without_outliers), vmax=max(thresholds_without_outliers), \
                    s=200, marker='o', zorder=10)
    
    #for i in range(len(thresholds_without_outliers) - 1):
    #    rgba = cmap(thresholds_without_outliers[i])
    #    ax.plot(x_axis_recall_without_outliers[i : i + 2], y_axis_precision_without_outliers[i : i + 2], color=rgba)

    ax.set_xlim(0.1, 1)
    ax.set_ylim(0.9, 1)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_title('Precision and recall curve with multiple thresholds')
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #plt.subplots_adjust(right=0.1)
    fig.colorbar(plot, pad=0.01)
    plt.xticks(rotation=45)
    fig.tight_layout()

    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.9-1.png'.format(figures_results_folder), bbox_inches='tight')
    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.9-1.pdf'.format(figures_results_folder), bbox_inches='tight')
    plt.clf()

    
    ax.plot(x_axis_recall_without_outliers, y_axis_precision_without_outliers, color='tab:blue', zorder=0, linewidth=4)
    plot = ax.scatter(x_axis_recall_without_outliers, y_axis_precision_without_outliers, c=thresholds_without_outliers, cmap=cmap, \
                    vmin=min(thresholds_without_outliers), vmax=max(thresholds_without_outliers), \
                    s=200, marker='o', zorder=10)
    
    #for i in range(len(thresholds_without_outliers) - 1):
    #    rgba = cmap(thresholds_without_outliers[i])
    #    ax.plot(x_axis_recall_without_outliers[i : i + 2], y_axis_precision_without_outliers[i : i + 2], color=rgba)
    
    #fig.colorbar(plot)

    ax.set_xlim(0.1, 1)
    ax.set_ylim(0.8, 1)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    #ax.set_title('Precision and recall curve with multiple thresholds')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout()

    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.8-1_{}.png'.format(figures_results_folder, dataset_name))
    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.8-1_{}.pdf'.format(figures_results_folder, dataset_name))
    plt.clf()


def precision_recall_variation_with_threshold(figures_results_folder, metrics_map_final_scores, dataset_name):

    plt.rc('font', size=36)          # controls default text sizes
    plt.rc('axes', titlesize=52)     # fontsize of the axes title
    plt.rc('xtick', labelsize=42)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=42)    # fontsize of the tick labels
    plt.rc('legend', fontsize=42)    # legend fontsize
    plt.rc('figure', titlesize=56)  # fontsize of the figure title

    #plt.rcParams['figure.figsize'] = (24,14)
    plt.rcParams['figure.figsize'] = (24,6)
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0


    thresholds = list(metrics_map_final_scores.keys())

    #print("\n--- 1", metrics_map_final_scores[thresholds[0]])
    #print("\n--- 2", metrics_map_final_scores[thresholds[int(len(thresholds) / 2)]])
    #print("\n--- 3", metrics_map_final_scores[thresholds[int(len(thresholds) / 4)]])
    #print("\n--- 4", metrics_map_final_scores[thresholds[3 * int(len(thresholds) / 4)]])
    #print("\n--- 5", metrics_map_final_scores[thresholds[-1]])

    fig, ax = plt.subplots()
    x_axis = []
    y_axis_precision = []
    y_axis_recall = []
    for threshold in thresholds:
        x_axis.append(threshold)
        y_axis_precision.append(metrics_map_final_scores[threshold].precision)
        y_axis_recall.append(metrics_map_final_scores[threshold].recall)


    ax.plot(x_axis, y_axis_precision, color="tab:blue", zorder=0, linewidth=8, label="Precision")
    ax.plot(x_axis, y_axis_recall, color="tab:orange", zorder=0, linewidth=8, label="Recall")
    
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.tick_params(width=4)

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, 1.005)
    #ax.set_xticks(np.arange(0, 1.005, 0.1))
    ax.set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
    #ax.set_yticks(np.arange(0.98, 1.005, 0.005))
    ax.set_ylabel('Proportion', fontsize=52)
    ax.set_xlabel('Threshold', fontsize=52)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower left', labelspacing=0.3)

    plt.xticks(rotation=45)
    #fig.tight_layout()

    plt.savefig('{}precision_recall_variation_with_threshold_{}.png'.format(figures_results_folder, dataset_name), bbox_inches='tight')
    plt.savefig('{}precision_recall_variation_with_threshold_{}.pdf'.format(figures_results_folder, dataset_name), bbox_inches='tight')

    plt.clf()


def precision_recall_curve_with_threshold_multiple_session_durations(figures_results_folder, results_by_min_duration, flow_count_by_duration, dataset_name):
    """
    """

    plt.rc('font', size=36)          # controls default text sizes
    plt.rc('axes', titlesize=52)     # fontsize of the axes title
    plt.rc('xtick', labelsize=42)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=42)    # fontsize of the tick labels
    plt.rc('legend', fontsize=42)    # legend fontsize
    plt.rc('figure', titlesize=56)  # fontsize of the figure title

    plt.rcParams['figure.figsize'] = (24,14)
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0


    durations_to_analyze = [0, 2, 4, 6, 8]

    # TODO: this is just to check if we can get high precision points around 0.999 and their respective recall, not needed for plots creation
    high_precision_results = {}
    for threshold, results_by_threshold in results_by_min_duration.items():
        high_precision_results[threshold] = {}
        for min_duration, results_by_min_duration_by_threshold in results_by_threshold.items():
            if results_by_min_duration_by_threshold.precision > 0.99:
                high_precision_results[threshold][min_duration] = {}
                high_precision_results[threshold][min_duration]['precision'] = results_by_min_duration_by_threshold.precision
                high_precision_results[threshold][min_duration]['recall'] = results_by_min_duration_by_threshold.recall
    #print("\n\nXXX high_precision_results", high_precision_results)

    thresholds = list(results_by_min_duration.keys())
    min_durations_all = np.array(list(results_by_min_duration[thresholds[0]].keys())) # minutes to seconds
    #min_durations = np.arange(min(min_durations_all), max(min_durations_all) + 1, 4 * 60)
    min_durations = np.array(durations_to_analyze) * 60
    #min_duration_flows = list(flow_count_by_duration.values())
    min_duration_flows = []
    for min_duration in min_durations:
        min_duration_flows.append(flow_count_by_duration[min_duration])

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']
    line_styles = ['-', '--', '-.', ':']
    markers = ['x', 'o', 'v', 's', '*', 'D']

    max_line_styles = len(line_styles)
    fig, ax = plt.subplots()
    for i, min_duration in enumerate(min_durations):
        x_axis_recall = []
        y_axis_precision = []
        for threshold in thresholds:
            x_axis_recall.append(results_by_min_duration[threshold][min_duration].recall)
            y_axis_precision.append(results_by_min_duration[threshold][min_duration].precision)

        for j, (x, y) in enumerate(zip(x_axis_recall, y_axis_precision)):
            if x == 0 and y == 0:
                x_axis_recall[j] = float('nan')
                y_axis_precision[j] = float('nan')

        if i < max_line_styles:
            ax.plot(x_axis_recall, y_axis_precision, color=colors[i], zorder=0, linestyle=line_styles[i], marker=markers[i+1], linewidth=8, markersize=28, label="Min duration={}, {} pairs".format(int(min_duration / 60), min_duration_flows[i]))
        else:
            ax.plot(x_axis_recall, y_axis_precision, color=colors[i], zorder=0, marker=markers[i-max_line_styles], linewidth=8, markersize=28, label="Min duration={}, {} pairs".format(int(min_duration / 60), min_duration_flows[i]))
        #ax.plot(x_axis_recall, y_axis_precision, color=colors[i], zorder=0, marker=markers[i], linewidth=4, markersize=20, label="Min duration={}, {} pairs".format(int(min_duration / 60), min_duration_flows[i]))

    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.tick_params(width=4)

    ax.set_xlim(0, 1.005)
    #ax.set_ylim(0.98, 1.001)
    ax.set_ylim(0.9, 1.001)
    ax.set_xticks(np.arange(0, 1.005, 0.1))
    #ax.set_yticks(np.arange(0.98, 1.005, 0.005))
    ax.set_yticks(np.arange(0.9, 1.005, 0.01))
    #ax.set_xlim(0, 1.005)
    #ax.set_ylim(0, 1.005)
    ax.set_ylabel('Precision', fontsize=52)
    ax.set_xlabel('Recall', fontsize=52)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower right', labelspacing=0.3)

    plt.xticks(rotation=45)
    #fig.tight_layout()

    #plt.savefig('{}precision_recall_curve_with_threshold_multiple_session_durations_{}.png'.format(figures_results_folder, dataset_name), bbox_inches='tight')
    #plt.savefig('{}precision_recall_curve_with_threshold_multiple_session_durations_{}.pdf'.format(figures_results_folder, dataset_name), bbox_inches='tight')
    plt.savefig('{}precision_recall_curve_with_threshold_multiple_session_durations_metricsMap_{}.png'.format(figures_results_folder, dataset_name), bbox_inches='tight')
    plt.savefig('{}precision_recall_curve_with_threshold_multiple_session_durations_metricsMap_{}.pdf'.format(figures_results_folder, dataset_name), bbox_inches='tight')


    plt.clf()


def get_session_durations(dataset_name):
    topPath = f"/mnt/nas-shared/torpedo/extracted_features_{dataset_name}"
    client_file_paths = list(glob.iglob(os.path.join(topPath+'/client', '**/folderDict.pickle'), recursive=True))

    session_durations = []
    #for testPair in testPairs['correlated']['samples']:
    for test_idx, client_file_path in enumerate(client_file_paths):
        clientFolderDict = pickle.load(open(client_file_path, 'rb'))
        #allAbsTimes = testPair['clientFlow']['timesOutAbs'] + testPair['clientFlow']['timesInAbs']
        allAbsTimes = clientFolderDict['clientFlow']['timesOutAbs'] + clientFolderDict['clientFlow']['timesInAbs']
        absoluteInitialTime = min(allAbsTimes)
        maxAbsoluteTime = max(allAbsTimes)
        session_duration = maxAbsoluteTime - absoluteInitialTime
        session_durations.append(session_duration)

    avg_duration = sum(session_durations) / len(session_durations)
    max_duration = np.max(session_durations)
    min_duration = np.min(session_durations)
    mean_duration = np.mean(session_durations)
    var_duration = np.var(session_durations) # variance
    std_duration = np.std(session_durations) # standard deviation
    median_duration = np.median(session_durations)
    percentile_duration_25 = np.percentile(session_durations, 25)
    percentile_duration_75 = np.percentile(session_durations, 75)

    session_durations_minutes = []
    for duration in session_durations:
        session_durations_minutes.append(duration / 60)

    return ecdf4plot(session_durations_minutes)


def get_session_durations_results(results, dataset_name):
    topPath = f"/mnt/nas-shared/torpedo/extracted_features_{dataset_name}"
    client_file_paths = list(glob.iglob(os.path.join(topPath+'/client', '**/folderDict.pickle'), recursive=True))

    #print("results", results)
    session_durations = []
    for test_idx, client_file_path in enumerate(client_file_paths):
        clientFolderDict = pickle.load(open(client_file_path, 'rb'))
        sessionId = clientFolderDict['clientSessionId'].split('_client')[0]
        #print("=== sessionId", sessionId)

        if sessionId in results:
            fps = results[sessionId].fp
            allAbsTimes = clientFolderDict['clientFlow']['timesOutAbs'] + clientFolderDict['clientFlow']['timesInAbs']
            absoluteInitialTime = min(allAbsTimes)
            maxAbsoluteTime = max(allAbsTimes)
            session_duration = maxAbsoluteTime - absoluteInitialTime
            #print("---session_duration", session_duration)
            #print("---fps", fps)
            
            for _ in range(fps):
                session_durations.append(session_duration)

    session_durations_minutes = []
    for duration in session_durations:
        session_durations_minutes.append(duration / 60)

    return ecdf4plot(session_durations_minutes)


def get_requests_per_session(data_folder, dataset_name):
    dataset = query_sumo_dataset.SumoDataset(data_folder)
    requests_per_session = {}
    topPath = f"/mnt/nas-shared/torpedo/extracted_features_{dataset_name}"
    client_file_paths = list(glob.iglob(os.path.join(topPath+'/client', '**/folderDict.pickle'), recursive=True))

    for test_idx, client_file_path in enumerate(client_file_paths):
        clientFolderDict = pickle.load(open(client_file_path, 'rb'))
        session_id = query_sumo_dataset.get_session_id_from_path(clientFolderDict['clientSessionId'])
        client_name = query_sumo_dataset.get_client_name(session_id)
        requests_per_session[session_id] = dataset.get_client_session_nb_requests(client_name, session_id)

    requests_per_session_x = list(requests_per_session.values())

    return ecdf4plot(requests_per_session_x)


def get_requests_per_session_results(results, data_folder):
    dataset = query_sumo_dataset.SumoDataset(data_folder)
    requests_per_session = {}
    client_session_ids = list(results.keys())
    for session_id in client_session_ids:
        client_name = query_sumo_dataset.get_client_name(session_id)
        requests_per_session[session_id] = dataset.get_client_session_nb_requests(client_name, session_id)

    requests_per_session_x = []

    for sessionId, value in requests_per_session.items():
        if sessionId in results:
            fps = results[sessionId].fp
            for _ in range(fps):
                requests_per_session_x.append(requests_per_session[sessionId])

    return ecdf4plot(requests_per_session_x)


def get_sessions_per_os(dataset_name):
    topPath = f"/mnt/nas-shared/torpedo/extracted_features_{dataset_name}"
    client_file_paths = list(glob.iglob(os.path.join(topPath+'/client', '**/folderDict.pickle'), recursive=True))
    sessions_per_os = {}
    
    for test_idx, client_file_path in enumerate(client_file_paths):
        if 'alexa' in client_file_path: 
            continue
        clientFolderDict = pickle.load(open(client_file_path, 'rb'))
        session_id = query_sumo_dataset.get_session_id_from_path(clientFolderDict['clientSessionId'])
        os_name = query_sumo_dataset.get_onion_name(session_id)

        if os_name not in sessions_per_os:
            sessions_per_os[os_name] = 0
        sessions_per_os[os_name] += 1
    
    sessions_per_os_x = list(sessions_per_os.values())

    return ecdf4plot(sessions_per_os_x)


def session_dataset_statistics(figures_results_folder, data_folder, dataset_name):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)
    axis_labels_size = 58
    axis_ticks_size = 54
    linewidth = 8
    ax1.set_ylabel('CDF', fontsize=axis_labels_size)
    ax1.set_ylim(0, 1.01)
    ax2.set_ylim(0, 1.01)
    ax3.set_ylim(0, 1.01)

    session_durations_minutes, session_durations_y = get_session_durations(dataset_name)
    ax1.set_xlabel('Session duration\n(minutes)', fontsize=axis_labels_size)
    ax1.xaxis.labelpad = 45
    max_x_ticks_session_durations = math.ceil((max(session_durations_minutes) + 0.05)/100) * 100
    max_x_ticks_session_durations_step = int(max_x_ticks_session_durations / 4)
    # TODO: check if this is needed
    #ax1.set_xlim(0, max_x_ticks_session_durations)
    if "small_OSTest" in dataset_name:
        x_ticks = [0, 20, 40, 60]
    elif "small_OSValidate" in dataset_name:
        x_ticks = [0, 20, 40]
    elif "small_OSTrain" in dataset_name:
        x_ticks = [0, 20, 40, 60, 80]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=45)
    ax1.step(session_durations_minutes, session_durations_y, color='tab:blue', linewidth=linewidth)

    #print("\n=== Average session durations:", sum(session_durations_minutes) / len(session_durations_minutes))
    #print("\n=== Max session durations:", max(session_durations_minutes))

    requests_per_session_x, requests_per_session_y = get_requests_per_session(data_folder, dataset_name)
    ax2.set_xlabel('Requests per\nsession (#)', fontsize=axis_labels_size)
    ax2.xaxis.labelpad = 45
    ax2.set_xlim(0, max(requests_per_session_x) + 0.05)
    ax2.set_xticks(np.arange(0, 50, 10))
    ax2.set_xticklabels(np.arange(0, 50, 10), rotation=45)
    ax2.step(requests_per_session_x, requests_per_session_y, color='tab:blue', linewidth=linewidth)

    sessions_per_os_x, sessions_per_os_y = get_sessions_per_os(dataset_name)
    ax3.set_xlabel('Sessions per\nonion service (#)', fontsize=axis_labels_size)
    max_x_ticks_sessions_per_os = math.ceil((max(sessions_per_os_x) + 10)/100) * 100
    max_x_ticks_sessions_per_os_step = int(max_x_ticks_sessions_per_os / 4)
    ax3.set_xlim(0, max_x_ticks_sessions_per_os)
    ax3.set_xticks(np.arange(0, max_x_ticks_sessions_per_os + 1, max_x_ticks_sessions_per_os_step))
    ax3.set_xticklabels(np.arange(0, max_x_ticks_sessions_per_os + 1, max_x_ticks_sessions_per_os_step), rotation=45)
    ax3.step(sessions_per_os_x, sessions_per_os_y, color='tab:blue', linewidth=linewidth)

    ax1.tick_params(labelsize=axis_ticks_size)
    ax2.tick_params(labelsize=axis_ticks_size)
    ax3.tick_params(labelsize=axis_ticks_size)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    opacity = 0.3
    y_values_hlines = np.arange(0, 1.1, 0.25)
    print(y_values_hlines)
    for y_val in y_values_hlines:
        ax1.axhline(y = y_val, color = 'gray', linestyle = '-', alpha=opacity)
        ax2.axhline(y = y_val, color = 'gray', linestyle = '-', alpha=opacity)
        ax3.axhline(y = y_val, color = 'gray', linestyle = '-', alpha=opacity)
    #ax2.axhline(y = 0.74, color = 'gray', linestyle = '-')
    fig.tight_layout()

    plt.savefig('{}sessions_statistics_cdfs_{}.png'.format(figures_results_folder, dataset_name))
    plt.savefig('{}sessions_statistics_cdfs_{}.pdf'.format(figures_results_folder, dataset_name))



def session_results_statistics(figures_results_folder, metricsMapFinalScoresPerSession, dataset_folder, dataset_name, threshold=-0.05):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(30, 18))
    axis_labels_size = 58
    axis_ticks_size = 54
    linewidth = 8
    ax1.set_ylabel('CDF', fontsize=axis_labels_size)
    ax1.set_ylim(0, 1.01)
    ax2.set_ylim(0, 1.01)

    session_durations_minutes, session_durations_y = get_session_durations_results(metricsMapFinalScoresPerSession[threshold], dataset_name)
    
    if len(session_durations_minutes) == 0:
        ax1.set_xlabel('Session duration (minutes)', fontsize=axis_labels_size)
        ax1.xaxis.labelpad = 10
        x_ticks = np.arange(0, 7, 1)
        ax1.set_xticks(x_ticks)
        #ax1.set_xticklabels(x_ticks, rotation=45)
        ax1.set_xticklabels(x_ticks)
        ax1.step(session_durations_minutes, session_durations_y, color='tab:blue', linewidth=linewidth)

        requests_per_session_x, requests_per_session_y = get_requests_per_session_results(metricsMapFinalScoresPerSession[threshold], dataset_folder)
        ax2.set_xlabel('Requests per session (#)', fontsize=axis_labels_size)
        ax2.xaxis.labelpad = 10
        x_ticks = np.arange(0, 10, 1)
        ax2.set_xticks(x_ticks)
        #ax2.set_xticklabels(x_ticks, rotation=45)
        ax2.set_xticklabels(x_ticks)
        ax2.step(requests_per_session_x, requests_per_session_y, color='tab:blue', linewidth=linewidth)
    
    else:
        ax1.set_xlabel('Session duration (minutes)', fontsize=axis_labels_size)
        ax1.xaxis.labelpad = 10
        ax1.set_xlim(0, max(session_durations_minutes) + 0.05)
        x_ticks = np.arange(0, 7, 1)
        ax1.set_xticks(x_ticks)
        #ax1.set_xticklabels(x_ticks, rotation=45)
        ax1.set_xticklabels(x_ticks)
        ax1.step(session_durations_minutes, session_durations_y, color='tab:blue', linewidth=linewidth)

        requests_per_session_x, requests_per_session_y = get_requests_per_session_results(metricsMapFinalScoresPerSession[threshold], dataset_folder)
        ax2.set_xlabel('Requests per session (#)', fontsize=axis_labels_size)
        ax2.xaxis.labelpad = 10
        ax2.set_xlim(0, max(requests_per_session_x) + 0.05)
        x_ticks = np.arange(0, 10, 1)
        ax2.set_xticks(x_ticks)
        #ax2.set_xticklabels(x_ticks, rotation=45)
        ax2.set_xticklabels(x_ticks)
        ax2.step(requests_per_session_x, requests_per_session_y, color='tab:blue', linewidth=linewidth)

    ax1.tick_params(labelsize=axis_ticks_size)
    ax2.tick_params(labelsize=axis_ticks_size)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    opacity = 0.3
    y_values_hlines = np.arange(0, 1.1, 0.25)
    #print(y_values_hlines)
    for y_val in y_values_hlines:
        ax1.axhline(y = y_val, color = 'gray', linestyle = '-', alpha=opacity)
        ax2.axhline(y = y_val, color = 'gray', linestyle = '-', alpha=opacity)
        
    fig.tight_layout()

    plt.savefig('{}sessions_results_statistics_cdfs_{}_thr_{}.png'.format(figures_results_folder, dataset_name, threshold))
    plt.savefig('{}sessions_results_statistics_cdfs_{}_thr_{}.pdf'.format(figures_results_folder, dataset_name, threshold))
    plt.clf()


def cdf_pair_scores(figures_results_folder, predictions, dataset_name):
    
    scores_correlated = []
    scores_non_correlated = []
    
    for pair_preds in predictions.values():
        for (clientSessionId, osSessionId), label_score in pair_preds.items():
            if clientSessionId == osSessionId:
                scores_correlated.append(label_score['score'])
            else:
                scores_non_correlated.append(label_score['score'])

    fig, ax1 = plt.subplots(ncols=1, sharey=True)

    scores_correlated_x, scores_correlated_y = ecdf4plot(scores_correlated)
    scores_non_correlated_x, scores_non_correlated_y = ecdf4plot(scores_non_correlated)
    ax1.set_xlabel('Session scores')
    ax1.set_ylabel('CDF')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.step(scores_correlated_x, scores_correlated_y, color='tab:blue', linewidth=8, label="Correlated")
    ax2.step(scores_non_correlated_x, scores_non_correlated_y, color='tab:orange', linewidth=8,  label="Non correlated")
    
    fig.tight_layout()

    plt.legend()

    plt.savefig('{}score_cdfs_{}.png'.format(figures_results_folder, dataset_name))
    plt.savefig('{}score_cdfs_{}.pdf'.format(figures_results_folder, dataset_name))