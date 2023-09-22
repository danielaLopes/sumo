import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullFormatter
from mycolorpy import colorlist as mcp
import pickle
import sys
import pandas as pd
import seaborn as sns


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


os_aliases = {'os1-os-small-ostest-1': 'onion1',
               'os1-os-small-ostest-2': 'onion2',
               'os1-os-small-ostest-4': 'onion3',    
               'os1-os-small-ostest-3': 'onion4',
               'os4-os-small-ostest-4': 'onion5',
               'os3-os-small-ostest-4': 'onion6',
               'os2-os-small-ostest-4': 'onion7',
               'os4-os-small-ostest-3': 'onion8',
               'os3-os-small-ostest-3': 'onion9',
               'os4-os-small-ostest-2': 'onion10',
               'os2-os-small-ostest-3': 'onion11',
               'os4-os-small-ostest-1': 'onion12',
               'os3-os-small-ostest-2': 'onion13',
               'os2-os-small-ostest-2': 'onion14',
               'os3-os-small-ostest-1': 'onion15',
               'os2-os-small-ostest-1': 'onion16'
            }


plt.rc('font', size=36)          # controls default text sizes
plt.rc('axes', titlesize=52)     # fontsize of the axes title
plt.rc('xtick', labelsize=42)    # fontsize of the tick labels
plt.rc('ytick', labelsize=42)    # fontsize of the tick labels
plt.rc('legend', fontsize=42)    # legend fontsize
plt.rc('figure', titlesize=56)  # fontsize of the figure title

plt.rcParams['figure.figsize'] = (24,14)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


# ---------------------- Plots ----------------------
def plot_heatmap_fps_per_oses(figures_results_folder, results, fps, dataset_name, threshold=-0.05):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)

    tps = {}
    for os_name, metrics in results[threshold].items():
        os_alias = os_aliases[os_name]
        if os_name not in tps:
            tps[os_alias] = 0
        tps[os_alias] += metrics.tp

    matrix = {}
    matrix_labels = {}
    matrix_mask = {}

    # Fill matrix with tps (positive values for each OS in blue)
    for os1 in tps.keys():
        matrix[os1] = {}
        matrix_labels[os1] = {}
        matrix_mask[os1] = {}

        for os2 in tps.keys():
            if os1 == os2:
                matrix_labels[os1][os2] = tps[os1]
                matrix_mask[os1][os2] = tps[os1]
            else:
                matrix_labels[os1][os2] = 0
                matrix_mask[os1][os2] = 0
            matrix[os1][os2] = 0

     # Fill matrix with fps (positive values that were wrongly confused with another OS)
    for osSessionId, falseSessions in fps[threshold].items():
        os_name = osSessionId.split('_')[1]
        os_alias = os_aliases[os_name]
        for fp in falseSessions:
            falseos_name = fp.split('_')[1]
            false_os_alias = os_aliases[falseos_name]
            matrix[os_alias][false_os_alias] += 1
            matrix_labels[os_alias][false_os_alias] += 1


    df = pd.DataFrame(matrix).T.fillna(0)
    df_labels = pd.DataFrame(matrix_labels).T.fillna(0)
    df_mask = pd.DataFrame(matrix_mask).T.fillna(0)

    # Sort dataframe according to popularity
    row_max_values = df_mask.max(axis=1)
    df = df.loc[row_max_values.sort_values(ascending=False).index]
    df = df[row_max_values.sort_values(ascending=False).index]
    df_labels = df_labels.loc[row_max_values.sort_values(ascending=False).index]
    df_labels = df_labels[row_max_values.sort_values(ascending=False).index]
    df_mask = df_mask.loc[row_max_values.sort_values(ascending=False).index]
    df_mask = df_mask[row_max_values.sort_values(ascending=False).index]


    plt.figure(figsize=(42, 32))
    plt.rc('axes', labelsize=54)   
    plt.rc('xtick', labelsize=54)    
    plt.rc('ytick', labelsize=54)    

    ax = sns.heatmap(df, cmap='magma_r', annot=df_labels, annot_kws={"size": 50}, xticklabels=True, yticklabels=True, cbar_kws=dict(pad=0.01), fmt='.4g')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=52)

    cmap1 = mpl.colors.ListedColormap(['lightskyblue'])
    sns.heatmap(df_mask, mask=df_mask == 0, cmap=cmap1, cbar=False, ax=ax)

    ax.figure.tight_layout()
    #plt.savefig('{}fps_oses_heatmap.png'.format(BASE_DIR))
    #plt.savefig('{}fps_oses_heatmap.pdf'.format(BASE_DIR))
    plt.savefig('{}fps_oses_heatmap_ordered_{}_thr_{}.png'.format(figures_results_folder, dataset_name, threshold))
    plt.savefig('{}fps_oses_heatmap_ordered_{}_thr_{}.pdf'.format(figures_results_folder, dataset_name, threshold))
    plt.clf()


def percent_without_decimal(x, pos):
    #return '{:.0%}'.format(x*100)
    return '{:.0%}'.format(x)


def plot_stacked_bars(concurrency_per_level, oses, concurrency):
    fig, ax1 = plt.subplots(nrows=1, figsize=(44, 20))

    ticks_font_size = 60
    font_size = 58
    label_font_size = 64

    # first subplot
    pps = {}
    bottom = []
    for _ in oses:
        bottom.append(0)
    
    total_per_level = []
    for i, _ in enumerate(oses):
        total_per_level.append(0)

    # creating the bar plot
    for i, (level, values) in enumerate(concurrency_per_level.items()):
        pps[level] = ax1.bar(oses, values, bottom=bottom, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level[i] += value

        for i in range (0, len(values)):  
            bottom[i] += values[i]

    for i, p in enumerate(pps['>10']):
        height = total_per_level[i]
        ax1.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s=f"{total_per_level[i]}", ha='center', fontsize=font_size)

    legend = ax1.legend(ncols=1, labelspacing=0.1, columnspacing=1.1, handletextpad=0.2, handlelength=1.5, borderpad=0.2, loc="upper left", fontsize=52)
    legend.set_title('Number of\nconcurrent\nsessions', prop={'size': 56})
    plt.setp(legend.get_title(), multialignment='center')
    for t in legend.get_texts():
        t.set_ha('left')

    ax1.set_xticks(oses)
    ax1.set_xticklabels(oses, rotation=45)
    ax1.set_xlabel("OS name", fontsize=label_font_size)
    ax1.set_ylabel("Number of sessions", fontsize=label_font_size)

    ax1.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    ax1.tick_params(labelsize=ticks_font_size)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0.2)


def plot_triple_stacked_bars(concurrency_per_level, concurrency_per_level_tps, concurrency_per_level_fps, oses, concurrency):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(50, 26))

    ticks_font_size = 42
    font_size = 34
    label_font_size = 46

    # first subplot
    pps = {}
    bottom = []
    for os in oses:
        bottom.append(0)
    
    total_per_level = []
    for i, os in enumerate(oses):
        total_per_level.append(0)

    # creating the bar plot
    for i, (level, values) in enumerate(concurrency_per_level.items()):
        pps[level] = ax1.bar(oses, values, bottom=bottom, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level[i] += value

        for i in range(0, len(values)):  
            bottom[i] += values[i]

    for i, p in enumerate(pps['>10']):
        height = total_per_level[i]
        ax1.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_per_level[i]), ha='center', fontsize=font_size)
    #print("\n--- TOTAL SESSIONS OS:", sum(total_per_level))
    # second subplot
    bottom_tps = []
    pps_tps = {}
    for os in oses:
        bottom_tps.append(0)
    
    total_per_level_tps = []
    for i, os in enumerate(oses):
        total_per_level_tps.append(0)

    for level, values in concurrency_per_level_tps.items():
        pps_tps[level] = ax2.bar(oses, values, bottom=bottom_tps, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level_tps[i] += value

        for i in range (0, len(values)):  
            bottom_tps[i] += values[i]
    
    for i, p in enumerate(pps_tps['>10']):
        height = total_per_level_tps[i]
        ax2.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_per_level_tps[i]), ha='center', fontsize=font_size)

    # third subplot
    bottom_fps = []
    pps_fps = {}
    for os in oses:
        bottom_fps.append(0)

    total_per_level_fps = []
    for i, os in enumerate(oses):
        total_per_level_fps.append(0)

    for level, values in concurrency_per_level_fps.items():
        pps_fps[level] = ax3.bar(oses, values, bottom=bottom_fps, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level_fps[i] += value

        for i in range (0, len(values)):  
            bottom_fps[i] += values[i]

    for i, p in enumerate(pps_fps['>10']):
        height = total_per_level_fps[i]
        ax3.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_per_level_fps[i]), ha='center', fontsize=font_size)


    ax1.legend(ncols=8, labelspacing=0.1, columnspacing=1.1, handletextpad=0.2, handlelength=1.5, borderpad=0.2, title='Number of concurrent sessions', bbox_to_anchor=(0.7, 1.1), fontsize=30)

    ax3.set_xticks(oses)
    ax3.set_xticklabels(oses, rotation=45)
    #ax3.set_xticklabels(oses, rotation=90)
    ax3.set_xlabel("OS name", fontsize=label_font_size)
    ax1.set_ylabel("Number of\nsessions", fontsize=label_font_size)
    ax2.set_ylabel("Number of\ntrue positives", fontsize=label_font_size)
    ax3.set_ylabel("Number of\nfalse positives", fontsize=label_font_size)


    ax1.tick_params(axis='x', which='major', labelsize=ticks_font_size)
    ax1.tick_params(axis='y', which='major', length=4, width=2, labelsize=ticks_font_size)
    ax1.tick_params(axis='y', which='minor', length=2, width=1)
    ax2.tick_params(axis='x', which='major', labelsize=ticks_font_size)
    ax2.tick_params(axis='y', which='major', length=4, width=2, labelsize=ticks_font_size)
    ax2.tick_params(axis='y', which='minor', length=2, width=1)
    ax3.tick_params(axis='x', which='major', labelsize=ticks_font_size)
    ax3.tick_params(axis='y', which='major', length=4, width=2, labelsize=ticks_font_size)
    ax3.tick_params(axis='y', which='minor', length=2, width=1)

    ax1.set_ylim(0, max(bottom) + 50)
    ax2.set_ylim(0, max(bottom_tps) + 50)
    ax3.set_ylim(0, max(bottom_fps) + 10)

    ax1.set_yticks([0, 1000, 2000, 3000])
    ax2.set_yticks([0, 1000, 2000, 3000])
    ax3.set_yticks([0, 25, 50])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)


def plot_triple_stacked_bars_log(concurrency_per_level, concurrency_per_level_tps, concurrency_per_level_fps, oses, concurrency):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(50, 26))

    ticks_font_size = 42
    font_size = 34
    label_font_size = 46

    # first subplot
    pps = {}
    bottom = []
    for os in oses:
        bottom.append(0)
    
    total_per_level = []
    for i, os in enumerate(oses):
        total_per_level.append(0)

    # creating the bar plot
    for i, (level, values) in enumerate(concurrency_per_level.items()):
        pps[level] = ax1.bar(oses, values, bottom=bottom, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level[i] += value

        for i in range(0, len(values)):  
            bottom[i] += values[i]

    for i, p in enumerate(pps['>10']):
        height = total_per_level[i]
        ax1.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_per_level[i]), ha='center', fontsize=font_size)

    # second subplot
    bottom_tps = []
    pps_tps = {}
    for os in oses:
        bottom_tps.append(0)
    
    total_per_level_tps = []
    for i, os in enumerate(oses):
        total_per_level_tps.append(0)

    for level, values in concurrency_per_level_tps.items():
        pps_tps[level] = ax2.bar(oses, values, bottom=bottom_tps, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level_tps[i] += value

        for i in range (0, len(values)):  
            bottom_tps[i] += values[i]
    
    for i, p in enumerate(pps_tps['>10']):
        height = total_per_level_tps[i]
        ax2.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_per_level_tps[i]), ha='center', fontsize=font_size)

    # third subplot
    bottom_fps = []
    pps_fps = {}
    for os in oses:
        bottom_fps.append(0)

    total_per_level_fps = []
    for i, os in enumerate(oses):
        total_per_level_fps.append(0)

    for level, values in concurrency_per_level_fps.items():
        pps_fps[level] = ax3.bar(oses, values, bottom=bottom_fps, color=concurrency[level],
            width=0.8, label=level)

        for i, value in enumerate(values):
            total_per_level_fps[i] += value

        for i in range (0, len(values)):  
            bottom_fps[i] += values[i]

    for i, p in enumerate(pps_fps['>10']):
        height = total_per_level_fps[i]
        ax3.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(total_per_level_fps[i]), ha='center', fontsize=font_size)


    ax1.legend(ncols=8, labelspacing=0.1, columnspacing=1.1, handletextpad=0.2, handlelength=1.5, borderpad=0.2, title='Number of concurrent sessions', bbox_to_anchor=(0.7, 1.1), fontsize=30)

    ax3.set_xticks(oses)
    ax3.set_xticklabels(oses, rotation=45)
    #ax3.set_xticklabels(oses, rotation=90)
    ax3.set_xlabel("OS name", fontsize=label_font_size)
    ax1.set_ylabel("Number of\nsessions", fontsize=label_font_size)
    ax2.set_ylabel("Number of\ntrue positives", fontsize=label_font_size)
    ax3.set_ylabel("Number of\nfalse positives", fontsize=label_font_size)

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax3.yaxis.set_major_formatter(formatter)

    #locmin = LogLocator(base=10, subs=(0.5, 1), numticks=10)
    locmin = LogLocator(base=10, subs=np.arange(0.1, 1, 0.1), numticks=10)
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax2.yaxis.set_minor_locator(locmin)
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax3.yaxis.set_minor_locator(locmin)
    ax3.yaxis.set_minor_formatter(NullFormatter())

    ax1.tick_params(axis='x', which='major', labelsize=ticks_font_size)
    ax1.tick_params(axis='y', which='major', length=4, width=2, labelsize=ticks_font_size)
    ax1.tick_params(axis='y', which='minor', length=2, width=1)
    ax2.tick_params(axis='x', which='major', labelsize=ticks_font_size)
    ax2.tick_params(axis='y', which='major', length=4, width=2, labelsize=ticks_font_size)
    ax2.tick_params(axis='y', which='minor', length=2, width=1)
    ax3.tick_params(axis='x', which='major', labelsize=ticks_font_size)
    ax3.tick_params(axis='y', which='major', length=4, width=2, labelsize=ticks_font_size)
    ax3.tick_params(axis='y', which='minor', length=2, width=1)

    ax1.set_ylim(0, max(bottom) + 50)
    ax2.set_ylim(0, max(bottom_tps) + 50)
    ax3.set_ylim(0, max(bottom_fps) + 50)

    ax1.set_yticks([1, 10, 100, 1000, 3000])
    ax2.set_yticks([1, 10, 100, 1000, 3000])
    ax3.set_yticks([1, 10, 50])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.subplots_adjust(
                    wspace=0.1,
                    hspace=0.1)


def get_level(value):
    if 0 <= value <= 5:
        return str(value)
    elif 6 <= value <= 10:
        return '6-10'
    elif value > 10:
        return '>10'
    else:
        return None


def plot_session_concurrency_per_os(figures_results_folder, results_per_session, concurrent_requests, dataset_name, threshold=-0.05, plot_triple=True):
    # Initialize data structures
    concurrency = {'0': 'tab:blue', '1': MEDIUM_BLUE, '2': LIGHT_BLUE, '3': VERY_LIGHT_BLUE, '4': YELLOW, '5': LIGHT_ORANGE, '6-10': ORANGE, '>10': RED}
    concurrency_per_onion = {os_name: {level: {'count': 0, 'tps': 0, 'fps': 0, 'tns': 0, 'fns': 0} for level in concurrency.keys()} for os_name in os_aliases.values()}
    concurrency_per_level = {level: [0] * len(os_aliases) for level in concurrency.keys()}
    concurrency_per_level_tps = {level: [0] * len(os_aliases) for level in concurrency.keys()}
    concurrency_per_level_fps = {level: [0] * len(os_aliases) for level in concurrency.keys()}
    total_sessions_per_OS_all_levels = {os_name: 0 for os_name in os_aliases.values()}

    # Iterate over sessions
    for sessionId, session_data in concurrent_requests.items():
        if sessionId not in results_per_session[threshold]:
            continue

        os_name = os_aliases[sessionId.split('_')[1]]

        # Update statistics
        level = get_level(session_data['concurrent'])
        total_sessions_per_OS_all_levels[os_name] += 1
        # 'count' is the amount of flows sessions that were in a given concurrency level
        concurrency_per_onion[os_name][level]['count'] += 1
        concurrency_per_onion[os_name][level]['tps'] += results_per_session[threshold][sessionId].tp
        concurrency_per_onion[os_name][level]['fps'] += results_per_session[threshold][sessionId].fp
        concurrency_per_onion[os_name][level]['tns'] += results_per_session[threshold][sessionId].tn
        concurrency_per_onion[os_name][level]['fns'] += results_per_session[threshold][sessionId].fn

    # Sort onions by crescent order of total number of sessions
    oses = sorted(total_sessions_per_OS_all_levels.items(), key=lambda x: x[1])
    # Extract the keys only
    oses = [item[0] for item in oses]

    # Calculate statistics for each level using NumPy operations
    for i, os_name in enumerate(oses):
        for level in concurrency.keys():
            concurrency_per_level[level][i] = concurrency_per_onion[os_name][level]['count']
            concurrency_per_level_tps[level][i] = concurrency_per_onion[os_name][level]['tps']
            concurrency_per_level_fps[level][i] = concurrency_per_onion[os_name][level]['fps']

    if plot_triple:
        plot_triple_stacked_bars(concurrency_per_level, concurrency_per_level_tps, concurrency_per_level_fps, oses, concurrency)

        plt.savefig(figures_results_folder + "os_session_concurrency_{}_tps_fps_thr_{}.pdf".format(dataset_name, threshold), bbox_inches='tight')
        plt.savefig(figures_results_folder + "os_session_concurrency_{}_tps_fps_thr_{}.png".format(dataset_name, threshold), bbox_inches='tight')
        plt.clf()

        plot_triple_stacked_bars_log(concurrency_per_level, concurrency_per_level_tps, concurrency_per_level_fps, oses, concurrency)

        plt.savefig(figures_results_folder + "os_session_concurrency_{}_tps_fps_log_thr_{}.pdf".format(dataset_name, threshold), bbox_inches='tight')
        plt.savefig(figures_results_folder + "os_session_concurrency_{}_tps_fps_log_thr_{}.png".format(dataset_name, threshold), bbox_inches='tight')
        plt.clf()

    plot_stacked_bars(concurrency_per_level, oses, concurrency)
    plt.savefig(figures_results_folder + "os_session_concurrency_{}_thr_{}.pdf".format(dataset_name, threshold), bbox_inches='tight')
    plt.savefig(figures_results_folder + "os_session_concurrency_{}_thr_{}.png".format(dataset_name, threshold), bbox_inches='tight')
    plt.clf()


def results_per_onion(figures_results_folder, metricsMapPerOS, dataset_name, threshold=-0.05):
    total_sessions_color = 'tab:blue'
    false_positives_color = RED
    true_positives_color = YELLOW
    bar_width = 0.8

    y_axis_total_sessions = []
    y_axis_total_sessions_non_correlated = []
    y_axis_false_positives = []
    y_axis_true_positives = []
    labels = []

    for os_name, results in metricsMapPerOS[threshold].items():
        y_axis_total_sessions.append(results.tp + results.fn)
        y_axis_total_sessions_non_correlated.append(results.fp + results.tn)
        y_axis_false_positives.append(results.fp)
        y_axis_true_positives.append(results.tp)
        labels.append(os_name)

    # order all lists from least popular to more popular
    y_axis_total_sessions = np.array(y_axis_total_sessions)
    y_axis_total_sessions_non_correlated = np.array(y_axis_total_sessions_non_correlated)
    y_axis_false_positives = np.array(y_axis_false_positives)
    y_axis_true_positives = np.array(y_axis_true_positives)
    labels = np.array(labels)
    inds = y_axis_total_sessions.argsort()
    y_axis_total_sessions = y_axis_total_sessions[inds]
    y_axis_total_sessions_non_correlated = y_axis_total_sessions_non_correlated[inds]
    y_axis_false_positives = y_axis_false_positives[inds]
    y_axis_true_positives = y_axis_true_positives[inds]
    labels = labels[inds]

    #print("---y_axis_total_sessions", y_axis_total_sessions)
    #print("---y_axis_false_positives", y_axis_false_positives)
    #print("---y_axis_true_positives", y_axis_true_positives)
    #print("---y_axis_total_sessions_non_correlated", y_axis_total_sessions_non_correlated)
    # TODO
    #for i, label in enumerate(labels):
    #    labels[i] = os_aliases[label]
    x_axis = np.arange(0, len(labels), 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    bar1 = ax1.bar(x_axis, y_axis_total_sessions, color=total_sessions_color, width=bar_width, label='FN')
    ax1.bar(x_axis, y_axis_true_positives, color=true_positives_color, label='TP')
    bar2 = ax2.bar(x_axis, y_axis_false_positives, color=false_positives_color)

    for i, p in enumerate(bar1):
        height = p.get_height()
        ax1.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(y_axis_total_sessions[i]), ha='center')

    for i, p in enumerate(bar2):
        height = p.get_height()
        ax2.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(y_axis_false_positives[i]), ha='center')

    ax1.set_ylabel('Number of sessions')
    ax2.set_ylabel('Number of\nfalse positives')
    #ax2.set_ylabel('Number of\ncorrelations', labelpad=10)
    ax1.set_ylim(0, max(y_axis_total_sessions) + 200)
    ax2.set_xlabel('OS name')
    ax2.set_ylim(0, max(y_axis_false_positives) + 10)
    #ax2.tick_params("y", which="minor", pad=200, left=False)
    #ax1.set_title('Correlated and non correlated sessions and respective true positives and false positives')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_xticks(x_axis)
    #ticks_loc = ax2.get_yticks().tolist()
    #ax2.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax2.set_xticklabels(labels, rotation=45)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.3)
    #plt.legend(loc='upper left', ncol=4)
    #ax1.legend(loc='center', bbox_to_anchor = (0.175, 0.9))
    ax1.legend()

    plt.tight_layout()

    plt.savefig('{}oses_total_sessions_correlated_tp_fn_{}.png'.format(figures_results_folder, dataset_name))
    plt.savefig('{}oses_total_sessions_correlated_tp_fn_{}.pdf'.format(figures_results_folder, dataset_name))
    plt.clf()



