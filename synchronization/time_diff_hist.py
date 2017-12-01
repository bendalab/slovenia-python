import shelve
import numpy as np
from matplotlib import pyplot as plt


def cm_to_inch(tupl):
    return tuple(i / 2.54 for i in tupl)


plt.rcParams['font.size'] = 20
data = shelve.open('data')
fig, a = plt.subplots(2, 3, num=1)
fig.set_size_inches(cm_to_inch((25.4, 19.01)))
fig.set_dpi(192)
a = a.ravel()
for idx, ax in enumerate(a):
    diff = np.diff(data['{0}. trace'.format(idx)]['CallStart'])
    median = np.median(diff)
    ax.hist(diff, bins=np.arange(0, 15, .5))
    ax.plot([median, median], [0, 150], 'r')
    ax.text(median + 1, 100, '{0}s'.format(np.round(median, 2)), color='r')
    if idx == 3:
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Count')
        ax.set_xticks([0, 5, 10, 15])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 150])
    ax.set_title('Cricket #{0}'.format(idx + 1))
plt.show()
data.close()