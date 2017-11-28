from distance_attenuation import *
from IPython import embed
from numpy import *
import pandas as pd
import sys


###
# plotting
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
        catid = (rowdata.condition, 'height:' + str(rowdata.height), 'year:' + str(rowdata.year))

        if catid not in sorted_data.keys():
            sorted_data[catid] = dict()
            sorted_data[catid]['distance'] = []
            sorted_data[catid]['envelopes'] = []
            sorted_data[catid]['times'] = []
            sorted_data[catid]['gains'] = []

        sorted_data[catid]['distance'].append(rowdata.distance)
        sorted_data[catid]['envelopes'].append(rowdata.envelopes)
        sorted_data[catid]['times'].append(rowdata.times)
        sorted_data[catid]['gains'].append(std(rowdata.envelopes))



    #####
    # plot data
    figs = dict()

    # ENVELOPES
    for catid in sorted_data.keys():
        fig = plt.figure('raw ' + str(catid))
        figs[catid] = []

        figdata = sorted_data[catid]
        distance = asarray(figdata['distance'])
        envelopes = figdata['envelopes']
        times = figdata['times']

        plotnum = distance.shape[0]
        for idx, (dist, env, t) in enumerate(zip(distance, envelopes, times)):
            if idx == 0:
                figs[catid].append(fig.add_subplot(plotnum, 1, idx + 1))
            else:
                figs[catid].append(fig.add_subplot(plotnum, 1, idx + 1, sharex=figs[catid][-1], sharey=figs[catid][-1]))

            mEnv = mean(env, axis=0)
            mEnvStd = std(env, axis=0)
            figs[catid][-1].fill_between(t, mEnv - mEnvStd, mEnv + mEnvStd, color='gray')
            figs[catid][-1].plot(t, mEnv, color='red')
            figs[catid][-1].set_title('distance: ' + str(dist))

        figs[catid][int(round(len(figs[catid])/2))].set_ylabel('Envelope magnitude')
        figs[catid][-1].set_xlabel('Time [s]')


    # GAIN std(envelope)
    fig = plt.figure('gains')
    figs['gains'] = fig.add_subplot(1,1,1)
    open_distances = asarray(sorted_data[('Open', 'height:80.0', 'year:2016.0')]['distance'])
    open_gains = asarray(sorted_data[('Open', 'height:80.0', 'year:2016.0')]['gains'])
    max_gain = open_gains[open_distances == 100]
    for catid in sorted_data.keys():

        # data
        figdata = sorted_data[catid]
        distance = asarray(figdata['distance'])
        gains = asarray(figdata['gains'])

        # plot
        figs['gains'].semilogx(distance, gains / max_gain, label=catid)
        figs['gains'].legend()
        figs['gains'].set_xlabel('Distance [cm]')
        figs['gains'].set_xlim(100, 2000)
        figs['gains'].set_ylabel('Gain')



    plt.show()