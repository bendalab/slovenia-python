from matplotlib import pyplot as plt
import numpy as np

import shelve

myShelf = shelve.open('myShelf')
all_correlations = myShelf['my dictionary']
myShelf.close()

plt.style.use('ggplot')
for cricket_1 in range(6):
    for cricket_2 in range (cricket_1, 6):
        ax = plt.subplot2grid((6, 6), (cricket_1, cricket_2))
        lags = all_correlations['lags']
        correlation = all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)]['correlation']
        bootstrap_correlations = all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)]['bootstrapping']
        ax.plot(lags, correlation)
        ax.plot(lags, np.median(bootstrap_correlations, axis=0))
        ax.plot(lags, np.percentile(bootstrap_correlations, 2.5, axis=0))
        ax.plot(lags, np.percentile(bootstrap_correlations, 97.5, axis=0))
        ax.plot(lags, np.percentile(bootstrap_correlations, 25, axis=0))
        ax.plot(lags, np.percentile(bootstrap_correlations, 75, axis=0))
plt.show()