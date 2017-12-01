import shelve
import numpy as np
from matplotlib import pyplot as plt

def cm_to_inch(tupl):
    return tuple(i / 2.54 for i in tupl)

with shelve.open('myShelf') as data:
    all_correlations = data['my dictionary']
    data.close()

lags = all_correlations['lags']

plt.rcParams['font.size'] = 12
corr_color = '#1f77b4'
conf_color = '#ff7f0e'

fig_width = 25.4
fig_height = 19.05
fig = plt.figure(figsize=cm_to_inch((fig_width, fig_height)), dpi=192)
for i_1 in range(6):
    first_flag = 1
    for i_2 in range(i_1, 6):
        correlation = all_correlations['{0}. trace'.format(i_1)]['{0}. trace'.format(i_2)]['correlation']
        bootstrapping = all_correlations['{0}. trace'.format(i_1)]['{0}. trace'.format(i_2)]['bootstrapping']

        ax = plt.subplot2grid((6, 6), (i_1, i_2), fig=fig)
        ax.plot(lags, correlation, corr_color)
        ax.plot(lags, np.percentile(bootstrapping, 2.5, axis=0), conf_color)
        ax.plot(lags, np.percentile(bootstrapping, 97.5, axis=0), conf_color)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-.5, 1.1)
        if i_1 == 0 and i_2 == 0:
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-.5, 0, .5, 1])
            ax.set_xlabel(r'$\Delta$t [s]')
            ax.set_ylabel('Pearson')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        first_flag = 0
        if i_1 == 0:
            ax.text(.5, 1.1,
                    'Cricket #{0}'.format(i_2 + 1),
                    ha='center',
                    va='bottom',
                    transform=ax.transAxes)
        if i_2 == 5:
            ax.text(1.1, .5,
                    'Cricket #{0}'.format(i_1 + 1),
                    ha='left',
                    va='center',
                    rotation='vertical',
                    transform=ax.transAxes)
plt.show()