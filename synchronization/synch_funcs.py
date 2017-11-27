import sys
import numpy as np


def calculate_cross_correlation(times_1, times_2, fs, sigma, gauss_step, max_lag):
    times_1 = np.copy(times_1)
    times_2 = np.copy(times_2)
    # Subtract smallest time from all times:
    min_time = np.minimum(times_1[0], times_2[0])
    times_1 -= min_time
    times_2 -= min_time
    # Generate logical vector for events:
    duration = np.ceil(np.amax(np.hstack((times_1, times_2)))).astype(int)
    logical_1 = np.zeros(duration * fs + 1)
    logical_2 = np.copy(logical_1)
    # Fill logical vector with events:
    for number in times_1:
        logical_1[np.round(number * fs).astype(int)] += 1
    for number in times_2:
        logical_2[np.round(number * fs).astype(int)] += 1
    # Generate gaussian for convolution with logical vector:
    x_gauss = np.arange(-gauss_step * sigma, gauss_step * sigma, 1 / fs)
    y_gauss = np.exp(-np.power(x_gauss, 2) / (2 * np.power(sigma, 2)))
    # Convolve gaussian with logical vector:
    convolved_1 = np.convolve(y_gauss, logical_1, mode='full')
    convolved_2 = np.convolve(y_gauss, logical_2, mode='full')
    # Zero pad convolved vectors to avoid cutting off events later:
    convolved_1_pad = np.pad(convolved_1, int(max_lag * fs), mode='constant')
    convolved_2_pad = np.pad(convolved_2, int(max_lag * fs), mode='constant')
    # Subtract means of convolved vectors:
    convolved_1_pad -= convolved_1_pad.mean()
    convolved_2_pad -= convolved_2_pad.mean()
    # Calculate cross correlation between convolved vectors:
    convolved_2_part = convolved_2_pad[max_lag * fs:-max_lag * fs]
    correlation = np.correlate(convolved_1_pad, convolved_2_part, mode='valid')
    # Divide correlation by standard deviations and length to get pearson:
    correlation = correlation / (convolved_1.std() * convolved_2.std() * convolved_1.shape[0])
    # Calculate lags for correlation:
    lags = np.arange(-max_lag, max_lag + 1 / fs, 1 / fs)
    return (lags, correlation)


def bootstrap_cross_correlation(times_1, times_2, n, fs, sigma, gauss_step, max_lag):
    # Find time differences between events in vectors:
    time_diffs_1 = np.diff(times_1)
    time_diffs_2 = np.diff(times_2)
    # Initialize array for cross correlations:
    bootstrap_correlations = np.zeros((n, max_lag * 2 * fs + 1))
    for repetition in range(n):
        # Generate new time vectors:
        new_times_1 = generate_new_times(times_1, time_diffs_1)
        new_times_2 = generate_new_times(times_2, time_diffs_2)
        # Calculate cross correlation with new time vectors:
        _, correlation = calculate_cross_correlation(new_times_1, new_times_2, fs, sigma, gauss_step, max_lag)
        bootstrap_correlations[repetition, :] = correlation
        sys.stdout.write('\rProgress: {0} %'.format(str(round((repetition + 1) / n * 100, 2))))
        sys.stdout.flush()
    return bootstrap_correlations


def generate_new_times(old_times, old_time_diffs):
    # Initialize new time vector:
    new_times = np.zeros(old_times.shape)
    for time in range(new_times.size - 1):
        new_times[time + 1] = new_times[time] + np.random.choice(old_time_diffs)
    return new_times