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
    avg_data = average_duplicates(noise_data, avg_cols = ['freqs', 'H_sr', 'coherence', 'ml_coh'])

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
            sorted_data[catid]['coh'] = []

        sorted_data[catid]['distance'].append(rowdata.distance)
        sorted_data[catid]['H_sr'].append(abs(rowdata.H_sr))
        sorted_data[catid]['coh'].append(abs(rowdata.ml_coh))

    #####
    # calculate average transfer for frequency bins
    bwidth = 1000
    freq_bins = arange(5000, 25000, bwidth)
    mfreqs = freq_bins + bwidth / 2
    for catid in sorted_data.keys():
        figdata = sorted_data[catid]
        freqs = figdata['freqs']
        distance = asarray(figdata['distance'])
        H_sr = abs(asarray(figdata['H_sr']))
        coherence = abs(asarray(figdata['coh']))

        # calculate average transfer for frequency range
        mH_sr = empty((distance.shape[0], mfreqs.shape[0]))
        mCoh = empty((distance.shape[0], mfreqs.shape[0]))
        for fidx, mf in enumerate(mfreqs):
            for didx, dist in enumerate(distance):
                mH_sr[didx, fidx] = mean(H_sr[didx, (freqs > (mf - bwidth / 2)) & (freqs < (mf + bwidth / 2))])
                mCoh[didx, fidx] = mean(coherence[didx, (freqs > (mf - bwidth / 2)) & (freqs < (mf + bwidth / 2))])

        # add to dictionary
        figdata['mfreqs'] = mfreqs
        figdata['mH_sr'] = mH_sr
        figdata['mCoh'] = mCoh

    figs = dict()

    #####
    # 3d
    # plot surface plot
    if '3d' in sys.argv:
        for catid in sorted_data.keys():
            if not catid[2] == 'year:2015.0':
                continue

            fig = plt.figure('3d ' + str(catid))
            figs[catid] = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]

            figdata = sorted_data[catid]
            distance = asarray(figdata['distance'])
            mfreqs = figdata['mfreqs']
            mH_sr = abs(asarray(figdata['mH_sr']))
            mCoh = abs(asarray(figdata['mCoh']))

            dist_cond = distance > 0
            # plot
            X, Y = meshgrid(distance[dist_cond], mfreqs)

            # signal-response transfer
            Z_H = log10(mH_sr[dist_cond, :].transpose())
            surf = figs[catid][0].plot_surface(X, Y, Z_H, cmap='viridis', linewidth=0, antialiased=False)
            figs[catid][0].set_xlabel('Sender-receiver distance [cm]')
            figs[catid][0].set_ylabel('Frequency [Hz]')
            figs[catid][0].set_zlabel('log10(Gain)')
            figs[catid][0].set_zlim(-4, 1)

            # signal-response coherence
            Z_C = mCoh[dist_cond, :].transpose()
            surf = figs[catid][1].plot_surface(X, Y, Z_C, cmap='viridis', linewidth=0, antialiased=False)
            figs[catid][1].set_xlabel('Sender-receiver distance [cm]')
            figs[catid][1].set_ylabel('Frequency [Hz]')
            figs[catid][1].set_zlabel('Coherence')

    # 2d plot
    if '2d' in sys.argv:
        for catid in sorted_data.keys():
            if not catid[2] == 'year:2015.0':
                continue

            # create
            fig = plt.figure('2d ' + str(catid))
            figs[catid] = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

            # get data
            figdata = sorted_data[catid]
            distance = asarray(figdata['distance'])
            mfreqs = figdata['mfreqs']
            mH_sr = abs(asarray(figdata['mH_sr']))
            mCoh = abs(asarray(figdata['mCoh']))

            # plot
            cmap = plt.get_cmap('viridis', lut=mfreqs.shape[0])
            for freq, color in zip(mfreqs, cmap.colors):
                lbl = 'f(' + str(freq - bwidth / 2) + ' - ' + str(freq + bwidth / 2) + ')'
                figs[catid][0].loglog(distance, mH_sr[:, mfreqs == freq], color=color, label=lbl)
                figs[catid][1].loglog(distance, mCoh[:, mfreqs == freq], color=color, label=lbl)
            figs[catid][0].loglog(distance, min(distance)/distance, '--k', label='1/distance')

            # format
            figs[catid][0].set_xlabel('Distance [cm]')
            figs[catid][1].set_xlabel('Distance [cm]')
            figs[catid][0].set_xlim(min(distance), max(distance))
            figs[catid][1].set_xlim(min(distance), max(distance))
            figs[catid][0].set_ylabel('Gain [V/V]')
            figs[catid][1].set_ylabel('Coherence')
            figs[catid][0].set_ylim(0.001, 10)

            figs[catid][0].legend()
            figs[catid][1].legend()

    plt.show()
