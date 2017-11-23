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
    noise_data = data_from_file(glob_noise_file)

    avg_data = average_duplicates(noise_data)

    # create dictionary for axes' handles
    figs = dict()
    if '2d' in sys.argv:  # plot 2d
        for rowidx, rowdata in avg_data.iterrows():
            print('Plot row', rowidx, '-', rowdata.distance, 'cm')

            figid = ('2d', rowdata.condition, 'height:' + str(rowdata.height), 'year:' + str(rowdata.year))
            if figid not in figs.keys():
                plt.figure(str(figid))
                figs[figid] = plt.subplot()

            figs[figid].semilogy(rowdata.freqs, abs(rowdata.H_sr), label=rowdata.distance)
            figs[figid].set_xlim(5000, 30000)
            figs[figid].legend()

        plt.show()

    if '3d' in sys.argv:
        # sort data
        sorted_data = dict()
        for rowidx, rowdata in avg_data.iterrows():
            figid = ('3d', rowdata.condition, 'height:' + str(rowdata.height), 'year:' + str(rowdata.year))

            if figid not in sorted_data.keys():
                sorted_data[figid] = dict()
                sorted_data[figid]['freqs'] = rowdata.freqs
                sorted_data[figid]['distance'] = []
                sorted_data[figid]['H_sr'] = []

            sorted_data[figid]['distance'].append(rowdata.distance)
            sorted_data[figid]['H_sr'].append(rowdata.H_sr)

        # plot data in surface plot
        for figid in sorted_data.keys():
            figdata = sorted_data[figid]
            freqs = figdata['freqs']
            distance = asarray(figdata['distance'])
            H_sr = abs(asarray(figdata['H_sr']))

            # plot
            fig = plt.figure(str(figid))
            figs[figid] = fig.gca(projection='3d')

            freq_range = (freqs >= 5000) & (freqs <= 25000)
            X, Y = meshgrid(distance, freqs[freq_range])
            Z = log10(H_sr.transpose()[freq_range, :])
            surf = figs[figid].plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

            figs[figid].set_xlabel('Sender-receiver distance [cm]')
            figs[figid].set_ylabel('Frequency [Hz]')
            figs[figid].set_zlabel('log10(Gain)')

        plt.show()