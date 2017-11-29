import numpy as np

from matplotlib import pyplot as plt

import synch_funcs


def make_figure(name, size=(25.4, 19.01)):
    return plt.figure(name, figsize=cm_to_inch(size), dpi=192)


def cm_to_inch(tupl):
    return tuple(i / 2.54 for i in tupl)


def plot_times(ts, y, color):
    for t in ts:
        plt.plot([t, t], [y - .5, y + .5], color)


def set_plot_stuff_times(ax):
    ax.set_xlabel('Time [s]')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 2])
    ax.set_yticks([.5, 1.5])
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels(['#1', '#2'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)


def set_plot_stuff_corr(ax):
    ax.set_xlabel(r'$\Delta$t [s]')
    ax.set_ylabel('Pearson')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-1, 1])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-1, -.5, 0, .5, 1])


def put_in_diff(x, y, color, choices=None):
    text_positions = x[:-1:] + np.diff(x) / 2
    for idx, text_position in enumerate(text_positions):
        if choices:
            plt.text(text_position, y, r'$isi_{0}$'.format(choices[idx]),
                     va='center',
                     ha='center',
                     color=color)
        else:
            plt.text(text_position, y, r'$isi_{0}$'.format(idx),
                     va='center',
                     ha='center',
                     color=color)

def make_new_times(old_ts):
    old_ts_diffs = np.diff(old_ts)
    new_ts = np.zeros(old_ts.shape)
    choices = []
    for t in range(new_ts.size - 1):
        choices.append(np.random.choice(np.arange(0, old_ts_diffs.size)))
        new_ts[t + 1] = new_ts[t] + old_ts_diffs[choices[-1]]
    return new_ts, choices


def set_plot_stuff_hist(ax):
    ax.set_xlabel('Pearson')
    ax.set_ylabel('Count')
    ax.set_xlim([-.5, .5])
    ax.set_ylim([0, 500])
    ax.set_xticks([-.5, 0, .5])
    ax.set_yticks([0, 200, 400])


plt.rcParams['font.size'] = 20

times_1 = np.arange(2, 50, 5, dtype=float)
times_2 = np.arange(1, 50, 5, dtype=float)
times_1 += np.random.random(times_1.shape) - 1
times_2 += np.random.random(times_2.shape) - 1
color_1 = '#1f77b4'
color_2 = '#ff7f0e'
corr_color = '#d62728'
conf_color = '#2ca02c'

fig_1 = make_figure('times')
plot_times(times_1, .5, color_1)
plot_times(times_2, 1.5, color_2)
axes_1 = plt.gca()
axes_1.set_title('Cricket Songs')
set_plot_stuff_times(axes_1)

fig_2 = make_figure('orig_corr')
lags, orig_correlation = synch_funcs.calculate_cross_correlation(times_1, times_2, 100, 0.25, 5, 10)
plt.plot(lags, orig_correlation, corr_color)
axes_2 = plt.gca()
axes_2.set_title('Cross-Correlation')
set_plot_stuff_corr(axes_2)

fig_3 = make_figure('times_with_diff')
plot_times(times_1, .5, color_1)
plot_times(times_2, 1.5, color_2)
put_in_diff(times_1, .5, color_1)
put_in_diff(times_2, 1.5, color_2)
axes_3 = plt.gca()
axes_3.set_title('Cricket Songs & Inter-Song Intervals')
set_plot_stuff_times(axes_3)

fig_4 = make_figure('new_times')
new_times_1, choices_1 = make_new_times(times_1)
new_times_2, choices_2 = make_new_times(times_2)
new_times_1 += 2
new_times_2 += 1
plot_times(new_times_1, .5, color_1)
plot_times(new_times_2, 1.5, color_2)
put_in_diff(new_times_1, .5, color_1, choices_1)
put_in_diff(new_times_2, 1.5, color_2, choices_2)
axes_4 = plt.gca()
axes_4.set_title('Resampled Cricket Songs')
set_plot_stuff_times(axes_4)

fig_5 = make_figure('first_bootstrap_corr')
bs_correlations = synch_funcs.bootstrap_cross_correlation(new_times_1, new_times_2, 1000, 100, 0.25, 5, 10)
plt.plot(lags, bs_correlations[0, :], corr_color)
axes_5 = plt.gca()
axes_5.set_title('Cross-Correlation of Resampled Cricket Songs')
set_plot_stuff_corr(axes_5)

fig_6 = make_figure('first_10_bootstrap_corrs')
for idx in range(5):
    plt.plot(lags, bs_correlations[idx, :], corr_color)
axes_6 = plt.gca()
axes_6.set_title('Cross-Correlation of Resampled Cricket Songs')
set_plot_stuff_corr(axes_6)

fig_7 = make_figure('one_time_step')
for idx in range(5):
    plt.plot(lags, bs_correlations[idx, :], corr_color)
#plt.plot([2, 2], [-1, 1], 'k')
plt.arrow(2, -.8, 0, -.2,
          width=.2,
          head_width=.6,
          head_length=.06,
          fc='k',
          ec='k',
          length_includes_head=True)
axes_7 = plt.gca()
axes_7.set_title('Cross-Correlation of Resampled Cricket Songs')
set_plot_stuff_corr(axes_7)

fig_8 = make_figure('bs_corrs_hist')
hist_data = bs_correlations[:, 1200]
plt.hist(hist_data, bins=np.arange(-1, 1, 0.01))
axes_8 = plt.gca()
axes_8.set_title(r'Histogram of Cross-Correlations at $\Delta$t = 2s')
set_plot_stuff_hist(axes_8)

fig_9 = make_figure('bs_corrs_hist_limits')
plt.hist(hist_data, bins=np.arange(-1, 1, 0.01))
lower = np.percentile(hist_data, 2.5)
upper = np.percentile(hist_data, 97.5)
plt.plot([lower, lower], [0, 1000], conf_color)
plt.plot([upper, upper], [0, 1000], conf_color)
axes_9 = plt.gca()
axes_9.set_title(r'Histogram of Cross-Correlations at $\Delta$t = 2s')
set_plot_stuff_hist(axes_9)

fig_10 = make_figure('first_corr_with_one')
plt.plot(lags, orig_correlation, corr_color)
plt.plot(2, lower,
         marker='o',
         markerfacecolor=conf_color,
         markeredgecolor=conf_color)
plt.plot(2, upper,
         marker='o',
         markerfacecolor=conf_color,
         markeredgecolor=conf_color)
axes_10 = plt.gca()
axes_10.set_title('Cross-Correlation')
set_plot_stuff_corr(axes_10)

fig_11 = make_figure('first_corr_with confidence')
plt.plot(lags, orig_correlation, corr_color)
plt.plot(lags, np.percentile(bs_correlations, 2.5, axis=0), conf_color)
plt.plot(lags, np.percentile(bs_correlations, 97.5, axis=0), conf_color)
axes_11 = plt.gca()
axes_11.set_title('Cross-Correlation with Confidence Intervals')
set_plot_stuff_corr(axes_11)
plt.show()
