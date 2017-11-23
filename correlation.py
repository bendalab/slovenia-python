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
    log_1 = np.zeros(dur.astype(int) * fs)
    log_2 = np.copy(log_1)
    return log_1, log_2


def fill_logical(t_1, t_2, log_1, log_2, fs):
    log_1[np.round(t_1 * fs).astype(int) - 1] += 1
    log_2[np.round(t_2 * fs).astype(int) - 1] += 1
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
    for i in range(n):
        t_start = time.time()
        new_t_1 = gen_new_times(t_1)
        new_t_2 = gen_new_times(t_2)
        _, c, _, _ = correlate_calls(new_t_1, new_t_2, fs, sig, step, m_lag, mode=mode)
        corrs[i, :] = c
        print(str(time.time()-t_start))
    low = np.percentile(corrs, 2.5, axis=0)
    up = np.percentile(corrs, 97.5, axis=0)
    return low, up


def gen_new_times(t):
    diffs = np.diff(t)
    new_t = np.zeros(t.shape)
    for i in range(new_t.shape[0] - 1):
        new_t[i + 1] = new_t[i] + np.random.choice(diffs)
    return new_t


with shelve.open(r'C:\Users\chris\Documents\calltiming_17_11_21\data') as d:
    times_1 = d['0. trace']['CallStart']
    times_2 = d['2. trace']['CallStart']


Fs = 100  # Hz
dt = 1 / Fs  # seconds
sigma = .4
gauss_step = 5
max_lag = 10  # seconds
n_boot = 100
times_1, times_2 = cut_times(times_1, times_2, 0, 9999)
corr_lags, corr, logical_1, logical_2 = correlate_calls(times_1, times_2, Fs, sigma, gauss_step, max_lag)
low_conf, up_conf = bootstrap(times_1, times_2, n_boot, Fs, sigma, gauss_step, max_lag)

fig, ax = plt.subplots(2)
ax[0].plot(logical_1)
ax[0].plot(logical_2)
ax[1].plot(corr_lags, corr)
ax[1].plot(corr_lags, low_conf)
ax[1].plot(corr_lags, up_conf)
plt.show()
