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
            sorted_data[catid]['mean_env'] = []
            sorted_data[catid]['std_env'] = []

        sorted_data[catid]['distance'].append(rowdata.distance)
        sorted_data[catid]['envelopes'].append(rowdata.envelopes)
        sorted_data[catid]['times'].append(rowdata.times)
        envelopes = []
        envelopes.extend([env[(rowdata.times > 0.5) & (rowdata.times < 1.5)] for env in rowdata.envelopes])
        sorted_data[catid]['mean_env'].append(mean(envelopes))
        sorted_data[catid]['std_env'].append(std(envelopes))



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
    figs['gains'] = fig.add_subplot(1, 1, 1)
    for catid in sorted_data.keys():
        # data
        figdata = sorted_data[catid]
        distance = asarray(figdata['distance'])
        mean_env = asarray(figdata['mean_env'])

        # plot
        figs['gains'].semilogx(distance, mean_env, label=catid)
        figs['gains'].legend()
        figs['gains'].set_xlabel('Distance [cm]')
        figs['gains'].set_xlim(100, 3000)
        figs['gains'].set_ylabel('mean(Env) [a.u.]')


    # amplitude transfer std(envelope) / mean(envelope)
    fig = custom_fig('call_std/mean', (13, 11))
    figs['gains'] = fig.add_subplot(1,1,1)
    for catid in sorted_data.keys():

        # data
        figdata = sorted_data[catid]
        distance = asarray(figdata['distance'])
        mean_env = asarray(figdata['mean_env'])
        std_env = asarray(figdata['std_env'])

        # plot
        figs['gains'].semilogx(distance, std_env / mean_env, label=catid)
        figs['gains'].legend()
        figs['gains'].set_xlabel('Distance [cm]')
        figs['gains'].set_xlim(100, 3000)
        figs['gains'].set_ylabel('SD(Env) / mean(Env) [a.u.]')






    plt.show()