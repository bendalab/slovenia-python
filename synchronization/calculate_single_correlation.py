import numpy as np
import shelve

from matplotlib import pyplot as plt

import synch_funcs


with shelve.open('data') as data:
    times_1 = data['2. trace']['CallStart']
    times_2 = data['1. trace']['CallStart']

fs = 100
sigma = .4
gauss_step = 5
max_lag = 10

lags, correlation =  synch_funcs.calculate_cross_correlation(times_1, times_2, fs, sigma, gauss_step, max_lag)
#bootstrap_correlations = synch_funcs.bootstrap_cross_correlation(times_1, times_2, 10, fs, sigma, gauss_step, max_lag)

plt.plot(lags, correlation)
#plt.plot(lags, np.median(bootstrap_correlations, axis=0), 'm')
#plt.plot(lags, np.percentile(bootstrap_correlations, 2.5, axis=0))
#plt.plot(lags, np.percentile(bootstrap_correlations, 97.5, axis=0))
plt.show()