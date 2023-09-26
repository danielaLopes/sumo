import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys

data = []
avg_pairs = []
avg_times = []
avg_throughput = []
stddev_times = []
stddev_throughput = []
sol = ["subsetsum", "subsetsum2d", "deepcorr", "deepcoffea"]
folder = "A100SXM4"
nb_samples = 2

if len(sys.argv) > 1:
    sol = [sys.argv[1]]
if len(sys.argv) > 2:
    folder = sys.argv[2]
if len(sys.argv) > 3:
    nb_samples = int(sys.argv[3])

for s in range(len(sol)):
    s_str = sol[s]
    data += [[]]
    for i in range(1,nb_samples+1):
        filename_subsetsum = "{2}/samples_{0}/parsed_sample_{0}_s{1}".format(s_str, i, folder)
        idx   = np.array([])
        pairs = np.array([])
        time  = np.array([])
        throughput = np.array([])
        d = genfromtxt(filename_subsetsum, delimiter='\t', skip_header=1)
        for t in d:
            idx = np.append(idx, t[0])
            pairs = np.append(pairs, t[1])
            time = np.append(time, t[2])
            throughput = np.append(throughput, t[1]/t[2] / 1000) # NOTE: 1000 pairs / s
        data[s].append([idx, pairs, time, throughput])

    avg_times += [[]]
    avg_throughput += [[]]
    avg_pairs += [[]]
    stddev_times += [[]]
    stddev_throughput += [[]]
    transposed_data = np.transpose(data[s])
    for t in transposed_data:
        avg_pairs[s] += [np.average(t[1])]
        avg_times[s] += [np.average(t[2])]
        avg_throughput[s] += [np.average(t[3])]
        stddev_times[s] += [np.std(t[2])]
        stddev_throughput[s] += [np.std(t[3])]

plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=28)   # fontsize of the figure title

plt.rcParams['figure.figsize'] = (6,3)
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.05

fig, ax = plt.subplots(ncols=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel('Latency of batch (s)')
ax.set_xlabel('Throughput (1000 pairs/s)')
#ax.xaxis.labelpad = 40
#ax.set_xlim(0, max(session_durations_minutes) + 0.05)
#ax.set_xticks(np.arange(0, 50, 10))
#ax.set_xticklabels(np.arange(0, 50, 10), rotation=45)
#ax.step(session_durations_minutes, session_durations_y, color='tab:blue', linewidth=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_title("Torpedo Correlator", fontsize=24)
# ax.errorbar(avg_throughput[0], avg_times[0], xerr=stddev_throughput[0], yerr=stddev_times[0], label="subset-sum", linestyle='-', marker='o', color='b')

# sol = ["DC", "DCF", "SUMo"]
# idx = [2, 3, 1]
# color = ["blue", "green", "red"]
# marker = ["o", "t", "x"]
sol = ["SUMo"]
idx = [0]
color = ["red"]
marker = ["x"]

# first point looks like an outlier
#avg_throughput[1] = avg_throughput[1][1:]
#avg_times[1] = avg_times[1][1:]
#stddev_throughput[1] = stddev_throughput[1][1:]
#stddev_times[1] = stddev_times[1][1:]

for i in range(len(sol)):
    j = idx[i]
    c = color[i]
    n = sol[i]
    maxarg = np.argmax(np.array(avg_throughput[j]))
    print(n, "max throughput", avg_throughput[j][maxarg], "at ", avg_pairs[j][maxarg], "pairs, latency", avg_times[j][maxarg])
    ax.plot(np.array(avg_throughput[j])-np.array(stddev_throughput[j]), avg_times[j], linestyle='dotted', color=c)
    ax.plot(np.array(avg_throughput[j])+np.array(stddev_throughput[j]), avg_times[j], linestyle='dotted', color=c)
    ax.fill_betweenx(avg_times[j], np.array(avg_throughput[j])-np.array(stddev_throughput[j]), np.array(avg_throughput[j])+np.array(stddev_throughput[j]), color=c, alpha=0.2)
    ax.errorbar(avg_throughput[j], avg_times[j], yerr=stddev_times[j], label=n, linestyle='-', marker='x', color=c)


plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
# plt.legend(loc="upper right")
plt.savefig('plot_subsetsum2d.pdf', bbox_inches='tight')




