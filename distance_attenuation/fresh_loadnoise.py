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


###############
# plotting
###############
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
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

    if not isinstance(data, pd.DataFrame):
        print('ERROR: data in add_data() is not a DataFrame.')
        exit()
    if not isinstance(rowdata, dict):
        print('ERROR: rowdata in add_data() is not a dictionary.')
        exit()

    if not rowidx is None:
        for colkey in rowdata.keys():
            if colkey not in data.columns:
                data[colkey] = None

            data.loc[rowidx, colkey] = rowdata[colkey]
    else:
        data = data.append(pd.DataFrame(rowdata), ignore_index=True)

    return data

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


###############
# get data folders

# pkl file to save to
pkl_file = 'noise.pkl'

# data folder to load
years = ['2015', '2016']

folder_list = []
for year in years:
    new_folders = glob(os.path.join(*(glob_data_path + [year] + [year + '*'])))
    folder_list.extend([folder.split(os.sep) for folder in new_folders])


data = pd.DataFrame()
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

        Pxxs.append(Pxx)
        Pyys.append(Pyy)
        Pxys.append(Pxy)
        Pyxs.append(Pyx)
        Cxys.append(Cxy)
    freqs = f

    newrow = dict(
        # calculated from raw data
        Pxx = [np.mean(Pxxs, axis=0)],
        Pyy = [np.mean(Pyys, axis=0)],
        Pxy = [np.mean(Pxys, axis=0)],
        Pyx = [np.mean(Pyxs, axis=0)],
        Cxy_or = [np.mean(Pyxs, axis=0)],
        f = [f],

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


data_to_file(pkl_file, data)
embed()