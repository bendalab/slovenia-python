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
    noise_data = data_from_file(pkl_file)
    # average data for distance, condition, year and height
    avg_data = average_duplicates(noise_data)

    # create dictionary for axes' handles
    figs = dict()

    #####
    # sort data
    sorted_data = dict()
    for rowidx, rowdata in avg_data.iterrows():
        catid = (rowdata.condition, 'height:' + str(rowdata.height), 'year:' + str(rowdata.year))

        if catid not in sorted_data.keys():


            sorted_data[catid] = dict()
            sorted_data[catid]['freqs'] = rowdata.freqs
            sorted_data[catid]['distance'] = []
            sorted_data[catid]['H_sr'] = []

        sorted_data[catid]['distance'].append(rowdata.distance)
        sorted_data[catid]['H_sr'].append(rowdata.H_sr)

    #####
    # calculate average transfer for frequency bins
    bwidth = 2500
    freq_bins = arange(5000, 25000, bwidth)
    mfreqs = freq_bins + bwidth / 2
    for catid in sorted_data.keys():
        figdata = sorted_data[catid]
        freqs = figdata['freqs']
        distance = asarray(figdata['distance'])
        H_sr = abs(asarray(figdata['H_sr']))

        # calculate average transfer for frequency range
        mH_sr = empty((distance.shape[0], mfreqs.shape[0]))
        for fidx, mf in enumerate(mfreqs):
            for didx, dist in enumerate(distance):
                mH_sr[didx, fidx] = mean(H_sr[didx, (freqs > (mf - bwidth / 2)) & (freqs < (mf + bwidth / 2))])

        # add to dictionary
        figdata['mfreqs'] = mfreqs
        figdata['mH_sr'] = mH_sr



    figs = dict()

    #####
    # 3d
    # plot surface plot
    if False:
        for catid in sorted_data.keys():
            if not catid[2] == 'year:2015.0':
                continue

            fig = plt.figure('3d ' + str(catid))
            figs[catid] = fig.gca(projection='3d')

            figdata = sorted_data[catid]
            distance = asarray(figdata['distance'])
            mfreqs = figdata['mfreqs']
            mH_sr = abs(asarray(figdata['mH_sr']))


            # plot
            X, Y = meshgrid(distance, mfreqs)
            Z = log10(mH_sr.transpose())
            surf = figs[catid].plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
            figs[catid].set_xlabel('Sender-receiver distance [cm]')
            figs[catid].set_ylabel('Frequency [Hz]')
            figs[catid].set_zlabel('log10(Gain)')

    # 2d plot
    if True:
        for catid in sorted_data.keys():
            if not catid[2] == 'year:2015.0':
                continue

            fig = plt.figure('2d ' + str(catid))
            figs[catid] = fig.gca()

            figdata = sorted_data[catid]
            distance = asarray(figdata['distance'])
            mfreqs = figdata['mfreqs']
            mH_sr = abs(asarray(figdata['mH_sr']))

            # plot
            for freq in mfreqs:
                figs[catid].loglog(distance, mH_sr[:, mfreqs == freq], label='f(' + str(freq - bwidth / 2) + ' - ' + str(freq + bwidth / 2) + ')')
            figs[catid].loglog(distance, min(distance)/distance, '--k', label='1/distance')
            figs[catid].set_xlabel('Distance [cm]')
            figs[catid].set_xlim(min(distance), max(distance))
            figs[catid].set_ylabel('Gain [V/V]')
            figs[catid].set_ylim(floor(min(mH_sr.flatten())), ceil(max(mH_sr.flatten())))

            figs[catid].legend()

    plt.show()
