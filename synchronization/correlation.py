import sys
import shelve
import numpy as np
from matplotlib import pyplot as plt


def cut_times(t_1, t_2, start, stop):
    t_1 = t_1[(start < t_1) & (t_1 < stop)]
    t_2 = t_2[(start < t_2) & (t_2 < stop)]
    return t_1, t_2


def subtract_min(t_1, t_2):
    mini = np.minimum(t_1[0], t_2[0])
    t_1 -= mini
    t_2 -= mini
    return t_1, t_2


def gen_logical(t_1, t_2, fs):
    dur = np.ceil(np.maximum(t_1[-1], t_2[-1]))
    log_1 = np.zeros(dur.astype(int) * fs + 1)
    log_2 = np.copy(log_1)
    return log_1, log_2


def fill_logical(t_1, t_2, log_1, log_2, fs):
    log_1[np.round(t_1 * fs).astype(int)] += 1
    log_2[np.round(t_2 * fs).astype(int)] += 1
    return log_1, log_2


def gaussian(sig, step, fs):
    x = np.arange(-step * sig, step * sig, 1 / fs)
    y = np.exp(-np.power(x, 2) / (2*np.power(sig, 2)))
    return y


def conv_gauss_logical(log_1, log_2, sig, step, fs):
    y = gaussian(sig, step, fs)
    log_1 = np.convolve(y, log_1, mode='same')
    log_2 = np.convolve(y, log_2, mode='same')
    return log_1, log_2


def correlate(d1, d2, mode='full'):
    return np.correlate(d1 - d1.mean(), d2 - d2.mean(), mode=mode) / (d1.std() * d2.std()) / d1.shape[0]


def correlate_calls(t_1, t_2, fs, sig, step, m_l, mode='valid'):
    t_1, t_2 = subtract_min(t_1, t_2)
    log_1, log_2 = gen_logical(t_1, t_2, fs)
    log_1, log_2 = fill_logical(t_1, t_2, log_1, log_2, fs)
    log_1, log_2 = conv_gauss_logical(log_1, log_2, sig, step, fs)
    c = correlate(log_1, log_2[m_l * fs:-m_l * fs], mode=mode)
    lag = np.arange(-m_l, m_l + 1 / fs, 1 / fs)
    return lag, c, log_1, log_2


def bootstrap(t_1, t_2, n, fs, sig, step, m_l, mode='valid'):
    bs_c = np.zeros((n, m_l * 2 * fs + 1), dtype=np.float)
    for i in range(n):
        new_t_1 = gen_new_times(t_1)
        new_t_2 = gen_new_times(t_2)
        _, c, _, _ = correlate_calls(new_t_1, new_t_2, fs, sig, step, m_l, mode=mode)
        bs_c[i, :] = c
        sys.stdout.write('\rProgress: {0} %'.format(str(round((i+1)/n*100, 2))))
        sys.stdout.flush()
    return bs_c


def gen_new_times(t):
    diffs = np.diff(t)
    new_t = np.zeros(t.shape)
    for i in range(new_t.shape[0] - 1):
        new_t[i + 1] = new_t[i] + np.random.choice(diffs)
    return new_t


def gen_all_cross_corrs(c_s, n, fs, sig, step, m_l):
    all_corrs = {}
    for c_1 in range(len(c_s)):
        t_1 = np.copy(c_s['{0}. trace'.format(c_1)])
        all_corrs['{0}. trace'.format(c_1)] = {}
        for c_2 in range(c_1, len(c_s)):
            print('Calculating correlation between trace {0} and trace {1}...'.format(c_1, c_2))
            all_corrs['{0}. trace'.format(c_1)]['{0}. trace'.format(c_2)] = {}
            t_2 = np.copy(c_s['{0}. trace'.format(c_2)])
            lag, c, _, _ = correlate_calls(t_1, t_2, fs, sig, step, m_l)
            all_corrs['{0}. trace'.format(c_1)]['{0}. trace'.format(c_2)]['correlation'] = c
            print('Correlation done!\nBootstrapping trace {0} and trace {1}...'.format(c_1, c_2))
            corrs = bootstrap(t_1, t_2, n, fs, sig, step, m_l)
            all_corrs['{0}. trace'.format(c_1)]['{0}. trace'.format(c_2)]['bootstrapping'] = corrs
            print('\nBootstrapping done!\n\n')
    return lag, all_corrs

# times_1 and times_2 should be numpy arrays with shape (x,) that contain call times in seconds:
with shelve.open(r'.\data') as shelf_obj:
    call_start = {}
    for key in shelf_obj.keys():
        call_start[key] = shelf_obj[key]['CallStart']

# Sampling rate for binning call times in Hz:
Fs = 100
dt = 1 / Fs
# Sigma of gaussian for convolution:
sigma = .4
# Size of x axis of gaussian (-gauss_step * sigma : gauss_step * sigma):
gauss_step = 5
# Maximum lag for cross correlation in both directions in seconds:
max_lag = 10  # seconds
# Number of cross correlations for boot strapping:
n_boot = 1

#times_1, times_2 = cut_times(times_1, times_2, 0, 9999)
# Calculate cross correlation between times_1 and times_2:
print(call_start['1. trace'])
print(call_start['5. trace'])
corr_lags, corr, logical_1, logical_2 = correlate_calls(call_start['1. trace'], call_start['5. trace'], Fs, sigma, gauss_step, max_lag)
# Calculate boot strapping correlations:
#bs_corrs = bootstrap(times_1, times_2, n_boot, Fs, sigma, gauss_step, max_lag)
corr_lags1, all_correlations = gen_all_cross_corrs(call_start, n_boot, Fs, sigma, gauss_step, max_lag)

with shelve.open('test') as shelf_obj:
    shelf_obj['correlation lags'] = corr_lags
    for key in all_correlations.keys():
        shelf_obj[key] = all_correlations[key]

fig, ax = plt.subplots(2)
ax[0].plot(corr_lags, corr)
ax[1].plot(corr_lags1, all_correlations['1. trace']['5. trace']['correlation'])
plt.show()