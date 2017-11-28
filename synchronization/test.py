from matplotlib import pyplot as plt
import numpy as np

import shelve

myShelf = shelve.open('myShelf')
all_correlations = myShelf['my dictionary']
myShelf.close()

for cricket_1 in range(6):
    flag = 1
    for cricket_2 in range (cricket_1, 6):
        ax = plt.subplot2grid((6, 6), (cricket_1, cricket_2))
        lags = all_correlations['lags']
        correlation = all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)]['correlation']
        bootstrap_correlations = all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)]['bootstrapping']
        ax.plot(lags, correlation, 'k')
        #ax.plot(lags, np.median(bootstrap_correlations, axis=0,), 'b')
        ax.plot(lags, np.percentile(bootstrap_correlations, 2.5, axis=0), 'r')
        ax.plot(lags, np.percentile(bootstrap_correlations, 97.5, axis=0), 'r')
        #ax.plot(lags, np.percentile(bootstrap_correlations, 25, axis=0), 'c')
        #ax.plot(lags, np.percentile(bootstrap_correlations, 75, axis=0), 'c')
        ax.set_ylim([-.5, 1.1])
        if  not flag:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        if flag:
            ax.set_xlabel(r'$\Delta$t [s]')
            ax.set_ylabel('Pearson correlation')
        flag = 0
        if cricket_1 == 0:
            ax.text(.5, 1.1, 'Cricket #{0}'.format(cricket_2),
                    ha='center',
                    va='bottom',
                    transform=ax.transAxes)
        if cricket_2 == 5:
            ax.text(1.1, .5, 'Cricket #{0}'.format(cricket_1),
                    ha='left',
                    va='center',
                    rotation='vertical',
                    transform=ax.transAxes)
plt.show()