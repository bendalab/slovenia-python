from distance_attenuation import *
from IPython import embed
from numpy import *
import pandas as pd
import sys
import time


###
# plotting
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def adjust_spines(ax, spines = ['left','bottom'], shift_pos = False):
    for loc, spine in ax.spines.items():
        if loc in spines:
            if shift_pos:
                spine.set_position(('outward', 10))  # outward by 10 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def custom_fig(name, size=(9, 8)):
    return plt.figure(name, figsize=cm2inch(size))


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Missing arguments. Please specify a pkl file.')
        exit()
    pkl_file = sys.argv[1]

    # load data
    data = data_from_file(pkl_file)
    # average data for distance, condition, year and height
    avg_data = average_duplicates(data, ['envelopes', 'times'])


    #####
    # sort data
    sorted_data = dict()
    for rowidx, rowdata in avg_data.iterrows():
        catid = (rowdata.condition, 'height:' + str(rowdata.height))

        if catid not in sorted_data.keys():
            sorted_data[catid] = dict()
            sorted_data[catid]['distance'] = []
            sorted_data[catid]['envelopes'] = []
            sorted_data[catid]['times'] = []
            sorted_data[catid]['env_means'] = []
            sorted_data[catid]['env_stds'] = []

        sorted_data[catid]['distance'].append(rowdata.distance)
        sorted_data[catid]['envelopes'].append(rowdata.envelopes)
        sorted_data[catid]['times'].append(rowdata.times)

        env_parts = [env[(rowdata.times > 0.5) & (rowdata.times < 1.5)] for env in rowdata.envelopes]
        env_means = [mean(env) for env in env_parts]
        env_stds = [std(env) for env in env_parts]
        sorted_data[catid]['env_means'].append(env_means)
        sorted_data[catid]['env_stds'].append(env_stds)



    #####
    # plot data
    figs = dict()

    # raw ENVELOPES
    for catid in sorted_data.keys():
        fig = custom_fig('raw ' + str(catid), (13, 16))
        figs[catid] = []

        # load data
        figdata = sorted_data[catid]
        distance = figdata['distance']
        envelopes = figdata['envelopes']
        times = figdata['times']

        # filter distances
        dist = asarray(distance, dtype=int)
        dist_cond = (dist <= 1000) & (dist != 700) & (dist != 500) & (dist != 900)
        dist_range = arange(0, len(distance))
        dist_range = dist_range[dist_cond]


        plotnum = len(dist_range)
        ylims = [[-0.1, 1.25], [-0.05, 0.7], [-0.02, 0.5], [-0.01, 0.25], [-0.01, 0.25], [-0.01, 0.25], [-0.01, 0.25]]
        for pltidx, (idx, ylim) in enumerate(zip(dist_range, ylims)):
            if idx == 0:
                figs[catid].append(fig.add_subplot(plotnum, 1, pltidx + 1))
            else:
                figs[catid].append(fig.add_subplot(plotnum, 1, pltidx + 1))

            t = times[idx]
            env = envelopes[idx]
            dist = distance[idx]


            mEnv = mean(env, axis=0)
            mEnvStd = std(env, axis=0)
            # plot
            figs[catid][pltidx].fill_between(t, mEnv - mEnvStd, mEnv + mEnvStd, color='gray')
            figs[catid][pltidx].plot(t, mEnv)

            # format
            figs[catid][pltidx].set_xlim(0, 2.5)
            figs[catid][pltidx].set_ylim(*ylim)

            xtick = [0, 0.5, 1, 1.5, 2, 2.5]
            figs[catid][pltidx].set_xticks(xtick)
            figs[catid][pltidx].set_xticklabels([])

            figs[catid][pltidx].text(2.1, ylim[1] - 0.25 * (diff(ylim)), str(round(dist)) + 'cm')

            adjust_spines(figs[catid][-1])

        figs[catid][6].set_xticklabels(xtick)
        figs[catid][3].set_ylabel('Envelope magnitude')
        figs[catid][6].set_xlabel('Time [s]')


    #  mean(envelope)
    fig = custom_fig('call_mean', (13, 11))
    figs['mean'] = fig.add_subplot(1, 1, 1)
    for catid in sorted_data.keys():
        # data
        figdata = sorted_data[catid]
        distance = asarray(figdata['distance'])
        mean_env = figdata['env_means']
        mean_mean_env = [mean(env) for env in mean_env]

        # error bar
        std_mean_env = [std(env) for env in mean_env]

        # plot
        #figs['mean'].semilogx(distance, mean_mean_env, label=catid[0])
        figs['mean'].semilogx()
        figs['mean'].errorbar(distance, mean_mean_env, yerr=std_mean_env, label=catid[0])
        figs['mean'].legend()
        figs['mean'].set_xlabel('Distance [cm]')
        figs['mean'].set_xlim(100, 2000)
        figs['mean'].set_ylabel('mean(Env) [a.u.]')

        adjust_spines(figs['mean'])

    # amplitude transfer std(envelope) / mean(envelope)
    fig = custom_fig('call_std/mean', (13, 11))
    figs['stdmean'] = fig.add_subplot(1,1,1)
    for catid in sorted_data.keys():

        # data
        figdata = sorted_data[catid]
        mean_env = figdata['env_means']
        std_env = figdata['env_stds']
        mean_means_env = asarray([mean(env) for env in mean_env])
        mean_stds_env = asarray([mean(env) for env in std_env])

        # error bar
        std_mean_env = asarray([std(env) for env in mean_env])

        distance = asarray(figdata['distance'])

        # plot
        #figs['stdmean'].semilogx(distance, mean_stds_env / mean_means_env, label=catid[0])
        figs['stdmean'].semilogx()
        figs['stdmean'].errorbar(distance, mean_stds_env / mean_means_env, yerr=std_mean_env, label=catid[0])
        figs['stdmean'].legend()
        figs['stdmean'].set_xlabel('Distance [cm]')
        figs['stdmean'].set_xlim(100, 2000)
        figs['stdmean'].set_ylabel('Modulation Depth [a.u.]')

        adjust_spines(figs['stdmean'])






    plt.show()