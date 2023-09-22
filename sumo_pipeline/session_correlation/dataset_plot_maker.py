import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import sys


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


BASE_DIR = 'ccs_paper2023/draft/'


plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=32)     # fontsize of the axes title
plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=36)  # fontsize of the figure title

plt.rcParams['figure.figsize'] = (24,9)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


# ---------------------- Plots ----------------------

# This is a good plot to see the influence that concurrency and popularity have on our results
def plot_cdf(metricsMapPerOS, threshold=-0.05):
    """
    This graph was taken using the following parameters:
    epoch_size = 10
    epoch_tolerance = 8
    timeSamplingInterval = 500
    window_size = (2 * timeSamplingInterval) / 1000 
    overlap = 500 / 1000
    min_session_duration = int(30 * 1000 / 500)
    deltas = [10]
    thresholds = [-0.01]
    """
    total_sessions_color = 'tab:blue'
    total_sessions_non_correlated_color = 'tab:brown'
    false_positives_color = 'tab:orange'
    true_positives_color = 'tab:cyan'
    y_axis_total_sessions = []
    y_axis_total_sessions_non_correlated = []
    y_axis_false_positives = []
    y_axis_true_positives = []
    labels = []

    for osName, results in metricsMapPerOS[threshold].items():
        y_axis_total_sessions.append(results['tp'] + results['fn'])
        y_axis_total_sessions_non_correlated.append(results['fp'] + results['tn'])
        y_axis_false_positives.append(results['fp'])
        y_axis_true_positives.append(results['tp'])
        labels.append(osName)

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

    for i in range(0, len(labels)):
        labels[i] = labels[i].split('-new')[0]
    x_axis = np.arange(0, len(labels), 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    bar1 = ax1.bar(x_axis, y_axis_total_sessions, color=total_sessions_color)
    ax1.bar(x_axis, y_axis_true_positives, color=true_positives_color)
    bar2 = ax2.bar(x_axis, y_axis_total_sessions_non_correlated, color=total_sessions_non_correlated_color)
    ax2.bar(x_axis, y_axis_false_positives, color=false_positives_color)

    for i, p in enumerate(bar1):
        height = p.get_height()
        ax1.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{0:.0%}".format(y_axis_true_positives[i] / y_axis_total_sessions[i]), ha='center')

    for i, p in enumerate(bar2):
        height = p.get_height()
        ax2.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{0:.0%}".format(y_axis_false_positives[i] / y_axis_total_sessions_non_correlated[i]), ha='center')

    plt.xticks(x_axis, labels, rotation=45)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.3)

    ax1.set_ylabel('Correlated sessions')
    ax2.set_ylabel('Non correlated sessions')
    ax2.set_xlabel('OS name')
    #ax1.set_title('Correlated and non correlated sessions and respective true positives and false positives')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()

    plt.savefig('{}oses_total_sessions_tp_fp.png'.format(BASE_DIR))
    plt.savefig('{}oses_total_sessions_tp_fp.pdf'.format(BASE_DIR))
    #plt.show()


def precision_recall_variation_with_threshold(metricsMapFinalScores):
    """
    This graph was taken using the following parameters:
    epoch_size = 10
    epoch_tolerance = 8
    timeSamplingInterval = 500
    window_size = (2 * timeSamplingInterval) / 1000 
    overlap = 500 / 1000
    min_session_duration = int(30 * 1000 / 500)
    deltas = [10]
    thresholds = np.arange(-0.1, 0.05, 0.0005)
    """
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

    plt.savefig('{}precision_recall_variation_with_threshold.png'.format(BASE_DIR))
    plt.savefig('{}precision_recall_variation_with_threshold.pdf'.format(BASE_DIR))
    #plt.show()


def precision_recall_curve_with_threshold(metricsMapFinalScores):
    """
    This graph was taken using the following parameters:
    epoch_size = 10
    epoch_tolerance = 8
    timeSamplingInterval = 500
    window_size = (2 * timeSamplingInterval) / 1000 
    overlap = 500 / 1000
    min_session_duration = int(30 * 1000 / 500)
    deltas = [10]
    thresholds = np.arange(-0.1, 0.05, 0.0005)
    """
    #cmap.set_under('red')

    thresholds = list(metricsMapFinalScores.keys())
    num_colors = len(thresholds)
    #cmap = plt.get_cmap('plasma', num_colors)
    cmap = LinearSegmentedColormap.from_list('name', ['red', 'blue'])

    x_axis_recall = []
    y_axis_precision = []

    fig, ax = plt.subplots()
    for threshold in thresholds:
        x_axis_recall.append(metricsMapFinalScores[threshold]['recall'])
        y_axis_precision.append(metricsMapFinalScores[threshold]['precision'])

    plot = ax.scatter(x_axis_recall, y_axis_precision, c=thresholds, cmap=cmap, vmin=min(thresholds), vmax=max(thresholds))
    #fig.colorbar(plot, extend='min')
    fig.colorbar(plot)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title('Precision and recall curve with multiple thresholds')

    plt.savefig('{}precision_recall_curve_with_threshold.png'.format(BASE_DIR))
    plt.savefig('{}precision_recall_curve_with_threshold.pdf'.format(BASE_DIR))

    fig, ax = plt.subplots()
    thresholds_without_outliers = []
    x_axis_recall_without_outliers = []
    y_axis_precision_without_outliers = []
    for threshold in thresholds:
        if metricsMapFinalScores[threshold]['recall'] > 0.1:
            thresholds_without_outliers.append(threshold)
            x_axis_recall_without_outliers.append(metricsMapFinalScores[threshold]['recall'])
            y_axis_precision_without_outliers.append(metricsMapFinalScores[threshold]['precision'])
    
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
    #ax.set_title('Precision and recall curve with multiple thresholds')
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.subplots_adjust(right=0.1)
    fig.colorbar(plot)
    fig.tight_layout()

    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.9-1.png'.format(BASE_DIR))
    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.9-1.pdf'.format(BASE_DIR))

    
    ax.plot(x_axis_recall_without_outliers, y_axis_precision_without_outliers, color='blue', zorder=0, linewidth=4)
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
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout()

    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.8-1.png'.format(BASE_DIR))
    plt.savefig('{}precision_recall_curve_with_threshold_yy_0.8-1.pdf'.format(BASE_DIR))


def execution_time():
    x_axis = [1112, 2223, 3334, 4444, ]
    y_axis = []

    fig, ax1 = plt.subplots()
    ax1.plot(x_axis, y_axis, color='tab:blue', linewidth=4)

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Time in seconds')
    ax1.set_xlabel('# flow pairs')
    ax1.set_title('Pairs of flows processed per time unit')

    plt.savefig('{}execution_time_evolution.png'.format(BASE_DIR))
    plt.savefig('{}execution_time_evolution.pdf'.format(BASE_DIR))
    
    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis, y_axis, color='tab:blue', linewidth=4)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Time per flow pair in seconds')
    ax2.set_xlabel('# flow pairs')
    ax2.set_title('Pairs of flows processed per time unit')

    plt.savefig('{}execution_time_per_flow_evolution.png'.format(BASE_DIR))
    plt.savefig('{}execution_time_per_flow_evolution.pdf'.format(BASE_DIR))