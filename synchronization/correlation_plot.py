import shelve
import numpy as np
from matplotlib import pyplot as plt

def gen_plot(lag, all_corrs):
    plt.style.use('ggplot')
    for c_1 in range(len(all_corrs)):
        for c_2 in range(c_1, len(all_corrs)):
            ax = plt.subplot2grid((len(all_corrs), len(all_corrs)), (c_1, c_2))
            c = all_corrs['{0}. trace'.format(c_1)]['{0}. trace'.format(c_2)]['correlation']
            bs_c = all_corrs['{0}. trace'.format(c_1)]['{0}. trace'.format(c_2)]['bootstrapping']
            ax.plot(lag, c)
            ax.plot(lag, np.median(bs_c, axis=0), 'm')
            ax.plot(lag, np.percentile(bs_c, 2.5, axis=0), 'r')
            ax.plot(lag, np.percentile(bs_c, 97.5, axis=0), 'r')
            ax.plot(lag, np.percentile(bs_c, 25, axis=0), 'g')
            ax.plot(lag, np.percentile(bs_c, 75, axis=0), 'g')
    plt.show()


with shelve.open('all_correlations') as shelf_obj:
    corr_lags = shelf_obj['correlation lags']
    all_correlations = {}
    for correlation in range(len(shelf_obj) - 1):
        all_correlations['{0}. trace'.format(correlation)] = shelf_obj['{0}. trace'.format(correlation)]
    gen_plot(corr_lags, all_correlations)