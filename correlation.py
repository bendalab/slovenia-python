import time
import shelve
import numpy as np

from  matplotlib import pyplot as plt


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


def correlate_calls(t_1, t_2, fs, sig, step, m_lag, mode='valid'):
    t_1, t_2 = subtract_min(t_1, t_2)
    log_1, log_2 = gen_logical(t_1, t_2, fs)
    log_1, log_2 = fill_logical(t_1, t_2, log_1, log_2, fs)
    log_1, log_2 = conv_gauss_logical(log_1, log_2, sig, step, fs)
    c = correlate(log_1, log_2[m_lag * fs:-m_lag * fs], mode=mode)
    lag = np.arange(-m_lag, m_lag + 1/fs, 1/fs)
    return lag, c, log_1, log_2


def bootstrap(t_1, t_2, n, fs, sig, step, m_lag, mode='valid'):
    corrs = np.zeros((n, m_lag*2*fs + 1), dtype=np.float)
    print('Bootstrapping...')
    for i in range(n):
        new_t_1 = gen_new_times(t_1)
        new_t_2 = gen_new_times(t_2)
        _, c, _, _ = correlate_calls(new_t_1, new_t_2, fs, sig, step, m_lag, mode=mode)
        corrs[i, :] = c
        print('Progress: {0} %'.format(str(round((i+1)/n*100, 2))))
    print('Done!')
    median = np.median(corrs, axis=0)
    low_2_5 = np.percentile(corrs, 2.5, axis=0)
    up_97_5 = np.percentile(corrs, 97.5, axis=0)
    low_25 = np.percentile(corrs, 25, axis=0)
    up_75 = np.percentile(corrs, 75, axis=0)
    return median, low_2_5, up_97_5, low_25, up_75


def gen_new_times(t):
    diffs = np.diff(t)
    new_t = np.zeros(t.shape)
    for i in range(new_t.shape[0] - 1):
        new_t[i + 1] = new_t[i] + np.random.choice(diffs)
    return new_t

# times_1 and times_2 should be numpy arrays with shape (x,) that contain call times in seconds:
with shelve.open(r'.\data') as d:
    call_start = {}
    for key in d.keys():
        call_start[key] = d[key]['CallStart']
times_1 = call_start['0. trace']
times_2 = call_start['2. trace']

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
n_boot = 1000

times_1, times_2 = cut_times(times_1, times_2, 0, 9999)
# Calculate cross correlation between times_1 and times_2:
corr_lags, corr, logical_1, logical_2 = correlate_calls(times_1, times_2, Fs, sigma, gauss_step, max_lag)
# Calculate 95% confidence interval for correlation using boot strapping:
bs_med, bs_low_2_5, bs_up_97_5, bs_low_25, bs_up_75 = bootstrap(times_1, times_2, n_boot, Fs, sigma, gauss_step, max_lag)

# Plotting:
plt.plot(corr_lags, corr)
plt.plot(corr_lags, bs_med)
plt.plot(corr_lags, bs_low_2_5)
plt.plot(corr_lags, bs_up_97_5)
plt.plot(corr_lags, bs_low_25)
plt.plot(corr_lags, bs_up_75)
plt.show()