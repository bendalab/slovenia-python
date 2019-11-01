# Author: Tim Hladnik

from glob import glob
from IPython import embed
import matplotlib.mlab as ml
import numpy as np
import os
import pandas as pd
import pickle
from pyrelacs.DataClasses import load as pyre_load
from scipy import signal
from scipy.io import wavfile
import sys
import time


###############
# plotting
###############
if __name__ == '__main__':
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt


###############
# globals
###############
glob_data_path = ['..', '..', 'data', 'distance_attenuation']
glob_pkl_path = ['..', '..', 'pkl']
glob_fig_path = ['..', '..', 'figures']


###############
# methods
###############



def add_data(data, rowdata, rowidx = None):
    # expects two arguments
    #   data: the DataFrame
    #   rowdata: dictionary containing keys pertaining to columns in data
    #           and values consisting of lists with one or multiple entries
    #   (optional) rowidx: int, list of integers or logical array to add content to existing rows
    #           if rowidx is provided, the number of rows addressed
    #           must match the number of list-elements in rowdata[key]

    if not isinstance(data, dict):
        print('ERROR: data in add_data() is not a dictionary.')
        exit()
    if not isinstance(rowdata, dict):
        print('ERROR: rowdata in add_data() is not a dictionary.')
        exit()

    for colkey in rowdata.keys():
        if rowidx is not None:
            if colkey not in data.keys():
                data[colkey] = [None] * data['_count']
            data[colkey][rowidx] = rowdata[colkey]
        else:
            if colkey not in data.keys():
                data[colkey] = []
            data[colkey].append(rowdata[colkey])


    return data

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

def data_to_file(pkl_file, pkl_data):
    savepath = os.path.join(*glob_pkl_path)

    if os.path.exists(savepath):
        print('Saving data to ' + os.path.join(savepath, pkl_file))
        with open(os.path.join(savepath, pkl_file), 'wb') as fobj:
            pickle.dump(pkl_data, fobj, protocol=pickle.HIGHEST_PROTOCOL)
        fobj.close()
    else:
        print('Creating directory ' + savepath)
        os.mkdir(savepath)
        data_to_file(pkl_file, pkl_data)

    return True

def data_from_file(pkl_file):
    savefile = os.path.join(*(glob_pkl_path + [pkl_file]))

    print('Loading data from ' + savefile)
    if os.path.exists(savefile):
        with open(savefile, 'rb') as fobj:
            data = pickle.load(fobj)
        fobj.close()

        return data
    else:
        print('ERROR: pkl file not found.')
        exit()

def load_traces_dat(folderpath, filename):
    traces_file = os.path.join(*(glob_data_path + folderpath + [filename]))
    if not os.path.exists(traces_file):
        print('ERROR: no *-traces.dat -', traces_file)
        exit()

    metadata = []
    traces = []
    for n, trace_content in enumerate(pyre_load(traces_file).data_blocks()):
        print('Loadings trial', n+1, '...')
        metadata.append(trace_content[:-1])
        traces.append(np.asarray([row.split() for row in trace_content[-1]], dtype=float).transpose())


    return metadata, traces

def sanitize_string(instr):
    outstr = instr.replace('"', '').replace(' ', '')
    return outstr


###############
# program
###############

if len(sys.argv) == 1:
    print('Missing argument.')
    exit()


# pkl file to save to
pkl_file = 'noise.pkl'


#################
# ILUUSTRATION of PSD, CSD and TRANSFER
#################
if 'illustrate' in sys.argv:
    folderpath = ['2015', '2015-08-01-aa-free']
    sr = 100000
    nfft = 2 ** 14
    params = {'NFFT': nfft, 'noverlap': nfft / 2}

    metadata, traces = load_traces_dat(folderpath, 'transferfunction-traces.dat')

    t = traces[0]
    x = t[1, :]
    y = t[2, :]

    # subtract mean
    x = x - np.mean(x)
    y = y - np.mean(y)

    Pxx, _ = ml.psd(x, Fs=sr, **params)
    Pyy, _ = ml.psd(y, Fs=sr, **params)
    Pxy, f = ml.csd(x, y, Fs=sr, **params)

    fig = custom_fig('CSD illustration', (13, 15))
    ax1 = plt.subplot(2, 1, 1)
    ax1.fill_between([0, 20000], -0.1, 1.1, color='lightgray')
    ax1.plot(f, Pxx / max(Pxx), label='PSD(x)')
    ax1.plot(f, Pyy / max(Pyy), label='PSD(y)')
    ax1.set_ylabel('Auto spectral density')
    ax1.set_yticks([])
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(-500, 30000)
    ax1.set_xticklabels([])
    ax1.legend()

    adjust_spines(ax1)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(f, np.absolute(Pxy), color='black', label='CSD(x, y)')
    ax2.set_ylabel('Cross spectral density')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_yticks([])
    ax2.set_xlim(-500, 30000)
    ax2.legend()

    adjust_spines(ax2)

    plt.show()


#################
# MANUAL
#################
if 'manual' in sys.argv:
    embed()


#################
# LOAD DATA
#################
if 'load' in sys.argv:

    # data folder to load
    years = ['2015']  #, '2016']

    folder_list = []
    for year in years:
        new_folders = glob(os.path.join(*(glob_data_path + [year] + [year + '*'])))
        folder_list.extend([folder.split(os.sep) for folder in new_folders])


    data = dict()

    entry_num = len(folder_list)
    for idx, folderpath in enumerate(folder_list):
        print('Entry', idx + 1, '/', entry_num, ' - Processing ', os.path.join(*folderpath), '...')

        info = pyre_load(os.path.join(*(glob_data_path + folderpath[-2:] + ['info.dat'])))

        metadata, traces = load_traces_dat(folderpath[-2:], 'transferfunction-traces.dat')

        Pxxs = []
        Pyys = []
        Pxys = []
        Pyxs = []
        Cxys = []
        print('Processing trials ...')
        for t in traces:
            # get recordings
            sr = round(1000. / np.mean(np.diff(t[0, :])))
            x = t[1, :]
            y = t[2, :]

            # subtract mean
            x = x - np.mean(x)
            y = y - np.mean(y)

            # set spectrogram parameters
            nfft = 2 ** 14
            params = {'NFFT': nfft, 'noverlap': nfft / 2}

            # calc spectra
            Pxx, _ = ml.psd(x, Fs=sr, **params)
            Pyy, _ = ml.psd(y, Fs=sr, **params)
            Pxy, _ = ml.csd(x, y, Fs=sr, **params)
            Pyx, _ = ml.csd(y, x, Fs=sr, **params)
            Cxy, f = ml.cohere(x, y, Fs=sr, **params)

            Pxxs.append(np.absolute(Pxx))
            Pyys.append(np.absolute(Pyy))
            Pxys.append(np.absolute(Pxy))
            Pyxs.append(np.absolute(Pyx))
            Cxys.append(np.absolute(Cxy))
        freqs = f

        # calculate mean
        Pxx = np.mean(Pxxs, axis=0)
        Pyy = np.mean(Pyys, axis=0)
        Pxy = np.mean(Pxys, axis=0)
        Pyx = np.mean(Pyxs, axis=0)
        Cxy_or = np.mean(Cxys, axis=0)

        # plot
        plotname = 'noise_' + folderpath[-1]
        fig = custom_fig(plotname, (17, 30))
        ax1 = fig.add_subplot(5, 1, 1)
        ax1.set_title('Pxx')
        ax1.plot(f, np.absolute(Pxx))

        ax2 = fig.add_subplot(5, 1, 2)
        ax2.set_title('Pyy')
        ax2.plot(f, np.absolute(Pyy))

        ax3 = fig.add_subplot(5, 1, 3)
        ax3.set_title('Pxy')
        ax3.plot(f, np.absolute(Pxy))

        ax4 = fig.add_subplot(5, 1, 4)
        ax4.set_title('Pyx')
        ax4.plot(f, np.absolute(Pyx))

        ax5 = fig.add_subplot(5, 1, 5)
        ax5.set_title('Coherence')
        ax5.plot(f, np.absolute(Cxy_or))
        ax5.set_xlabel('Frequency [Hz]')

        figpath = os.path.join(*(glob_fig_path + [plotname + '.pdf']))
        print(figpath)
        plt.savefig(figpath, format='pdf')
        plt.close()

        # add new row
        newrow = dict(
            # calculated from raw data
            Pxx = Pxx,
            Pyy = Pyy,
            Pxy = Pxy,
            Pyx = Pyx,
            Cxy_or = Cxy_or,
            f = f,

            # metadata and additional data
            info_relacs = [info],
            meta_relacs = [metadata],
            foldername = folderpath[-1],
            condition = sanitize_string(info[0]['Recording']['Condition']),
            distance = float(sanitize_string(info[0]['Recording']['Distance'])[:-2]),
            height = float(sanitize_string(info[0]['Recording']['Height'])[:-2]),
            temperature = sanitize_string(info[0]['Recording']['Temperature']),
            date = sanitize_string(info[0]['Recording']['Date']),
            year = float(sanitize_string(info[0]['Recording']['Date'])[:4])
        )
        data = add_data(data, newrow)
    data['_count'] = entry_num
    # save data
    data_to_file(pkl_file, data)


#################
# ANALYSE
#################
if 'analyse' in sys.argv:

    # load data
    data = data_from_file(pkl_file)

    print('Calculate output-response transfer functions...')
    for rowidx, (Pxx, Pyy, Pxy, Pyx) in enumerate(zip(data['Pxx'], data['Pyy'], data['Pxy'], data['Pyx'])):
        newdata = dict()

        # output-response transfer
        newdata['H_or'] = Pxy / Pxx

        # response-output transfer
        newdata['H_ro'] = Pyx / Pyy

        # add data
        data = add_data(data, newdata, rowidx)

    print('Calculate signal-response transfer functions and coherence...')
    # conditions for calibration-recordings (smallest distance in an open environment)
    indcs = np.arange(0, data['_count'])
    calib_cond = indcs[(np.asarray(data['distance']) == 50) & (np.asarray(data['condition']) == 'Cutmeadow') & (np.asarray(data['height']) == 150)]

    # calculate mean output-response transfer function for equipment
    # using the recordings made in the open with the smallest speaker-microphone distance
    H_or_calib = data['H_or'][calib_cond[0]]
    H_ro_calib = data['H_ro'][calib_cond[0]]

    for rowidx, (H_or, H_ro) in enumerate(zip(data['H_or'], data['H_ro'])):
        # calculate signal-response transfer functions (forward transfer)
        H_sr = H_or / H_or_calib

        # calculate response-signal transfer functions (backward transfer)
        H_rs = H_ro / H_ro_calib

        Cxy_sr = H_sr * H_rs

        # add data
        newdata = dict(
            H_sr = H_sr,
            H_rs = H_rs,
            Cxy_sr = Cxy_sr
        )
        data = add_data(data, newdata, rowidx)

    # save data
    data_to_file(pkl_file, data)


#################
# PLOT
#################
if 'plot' in sys.argv:

    # load data
    data = data_from_file(pkl_file)

    #####
    # sort data into categories
    sorted_data = dict()
    for rowidx, (f, distance, condition, height, H_sr, Cxy_sr, Pxy, Pxx, Pyy) in \
            enumerate(zip(data['f'],
                          data['distance'],
                          data['condition'],
                          data['height'],
                          data['H_sr'],
                          data['Cxy_or'],
                          data['Pxy'],
                          data['Pxx'],
                          data['Pyy'])):

        catid = (condition, 'height:' + str(height))

        if catid not in sorted_data.keys():
            sorted_data[catid] = dict()
            sorted_data[catid]['f'] = f
            sorted_data[catid]['distance'] = []
            sorted_data[catid]['H_sr'] = []
            sorted_data[catid]['Cxy_sr'] = []

        sorted_data[catid]['distance'].append(distance)
        sorted_data[catid]['H_sr'].append(H_sr)
        sorted_data[catid]['Cxy_sr'].append(np.absolute(Cxy_sr))
        # test
        #sorted_data[catid]['Cxy_sr'].append(Pxy ** 2 / (Pxx * Pyy)) # calc coherence manually, should be approx. equal to Cxy_or

    #####
    # calculate average transfer for frequency bins
    bwidth = 2500
    freq_bins = np.arange(5000, 30000, bwidth)
    mfreqs = freq_bins + bwidth / 2
    for catid in sorted_data.keys():
        figdata = sorted_data[catid]
        distance = np.asarray(figdata['distance'])
        freqs = figdata['f']
        H_sr = np.asarray(figdata['H_sr'])
        coherence = np.asarray(figdata['Cxy_sr'])

        # calculate average transfer for frequency range
        mH_sr = np.empty((distance.shape[0], mfreqs.shape[0]))
        mCxy_sr = np.empty((distance.shape[0], mfreqs.shape[0]))
        for fidx, mf in enumerate(mfreqs):
            for didx, dist in enumerate(distance):
                mH_sr[didx, fidx] = np.mean(H_sr[didx, (freqs > (mf - bwidth / 2)) & (freqs < (mf + bwidth / 2))])
                mCxy_sr[didx, fidx] = np.mean(coherence[didx, (freqs > (mf - bwidth / 2)) & (freqs < (mf + bwidth / 2))])

        # add to dictionary
        sorted_data[catid]['mfreqs'] = mfreqs
        sorted_data[catid]['mH_sr'] = mH_sr
        sorted_data[catid]['mCxy_sr'] = mCxy_sr

    #####
    # sort data according to distance
    for catid in sorted_data.keys():
        figdata = sorted_data[catid]
        dist = figdata['distance']
        mH_sr = figdata['mH_sr']
        mCxy_sr = figdata['mCxy_sr']

        dist, mH_sr, mCxy_sr = zip(*sorted(zip(dist, mH_sr, mCxy_sr)))

        figdata['distance'] = dist
        figdata['mH_sr'] = mH_sr
        figdata['mCxy_sr'] = mCxy_sr


    figs = dict()

    # 3d
    # plot surface plot
    if '3d' in sys.argv:

        psize = (13, 15)

        for catid in sorted_data.keys():

            fig = custom_fig('3d gain' + str(catid), psize)
            figs[catid] = [fig.add_subplot(1, 1, 1, projection='3d')]

            fig = custom_fig('3d coh' + str(catid), psize)
            figs[catid].append(fig.add_subplot(1, 1, 1, projection='3d'))

            figdata = sorted_data[catid]
            distance = np.asarray(figdata['distance'])
            mfreqs = figdata['mfreqs']
            mH_sr = np.asarray(figdata['mH_sr'])
            mCxy_sr = np.asarray(figdata['mCxy_sr'])

            dist_cond = distance > 0
            # plot
            X, Y = np.meshgrid(np.log10(distance[dist_cond]), mfreqs)

            # signal-response transfer
            Z_H = np.log10(mH_sr[dist_cond, :].transpose())
            surf = figs[catid][0].plot_surface(X, Y, Z_H, cmap='viridis', linewidth=0, antialiased=False)
            figs[catid][0].set_xlabel('log10(Distance [cm])')
            figs[catid][0].set_ylabel('Frequency [Hz]')
            figs[catid][0].set_zlabel('log10(Gain)')
            figs[catid][0].set_zlim(-4, 1)

            # signal-response coherence
            Z_C = mCxy_sr[dist_cond, :].transpose()
            surf = figs[catid][1].plot_surface(X, Y, Z_C, cmap='viridis', linewidth=0, antialiased=False)
            figs[catid][1].set_xlabel('log10(Distance [cm])')
            figs[catid][1].set_ylabel('Frequency [Hz]')
            figs[catid][1].set_zlabel('Coherence')

    # 2d plot
    if '2d' in sys.argv:

        psize = (13.5, 11)

        for catid in sorted_data.keys():

            # get data
            figdata = sorted_data[catid]
            distance = np.asarray(figdata['distance'])
            mfreqs = figdata['mfreqs']
            mH_sr = np.asarray(figdata['mH_sr'])
            mCxy_sr = np.asarray(figdata['mCxy_sr'])

            # plot as function of distance
            fig = custom_fig('2d distance' + str(catid), psize)
            figs[catid] = [fig.add_subplot(1, 1, 1)]
            cmap = plt.get_cmap('viridis', lut = mfreqs.shape[0]).colors
            for freq, color in zip(mfreqs, cmap):
                lbl =  str(int(freq - bwidth / 2)) + ' - ' + str(int(freq + bwidth / 2)) + ' Hz'
                figs[catid][0].semilogy(distance, mH_sr[:, mfreqs == freq], label=lbl, color=color)
            figs[catid][0].semilogy(distance, min(distance) / distance, '--k', label='1 / distance')

            # format
            figs[catid][0].set_xlabel('Distance [cm]')
            figs[catid][0].set_xlim(min(distance), max(distance))
            figs[catid][0].set_ylabel('Gain [a.u.]')
            figs[catid][0].set_ylim(0.0001, 100)

            figs[catid][0].legend(loc='upper right')


            # plot as function of frequency
            fig = custom_fig('2d frequency' + str(catid), psize)
            figs[catid].append(fig.add_subplot(1, 1, 1))
            # plot
            cmap = plt.get_cmap('viridis', lut=distance.shape[0]).colors
            for dist, color in zip(distance, cmap):
                lbl = str(int(dist)) + ' cm'
                figs[catid][1].semilogy(mfreqs, mH_sr[distance == dist, :].transpose(), label=lbl, color=color)
            figs[catid][1].set_xlabel('Frequency [Hz]')
            figs[catid][1].set_ylabel('Gain [a.u.]')
            figs[catid][1].legend(loc='upper left')

    plt.show()