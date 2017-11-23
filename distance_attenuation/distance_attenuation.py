from glob import glob
from IPython import embed
import matplotlib.mlab as ml
from numpy import *
import os
import pandas as pd
from pyrelacs.DataClasses import load as pyre_load
import sys


###
# globals
glob_data_path = ['..', '..', 'data', 'distance_attenuation']


###
# methods

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


def calc_spectra(x, y, sr, params = None):
    # expects two 1d arrays as signals and a number for the sampling rate
    # (optional) parameters for FFT

    # returns PSD(x,x), PSD(y,y), PSD(x,y), PSD(y,x) and frequency

    if params is None:
        nfft = 2 ** 11
        params = {'NFFT': nfft, 'noverlap': nfft / 2}
    if not isinstance(params, dict):
        print('ERROR: params in calc_spectra() is not a dictionary.')

    Pxx, _ = ml.psd(x, Fs=sr, **params)
    Pyy, _ = ml.psd(y, Fs=sr, **params)
    Pxy, _ = ml.csd(x, y, Fs=sr, **params)
    Pyx, f = ml.csd(y, x, Fs=sr, **params)

    return Pxx, Pyy, Pxy, Pyx, f


def load_info_dat(folderpath):
    info_file = os.path.join(*folderpath, 'info.dat')
    if not os.path.exists(info_file):
        print('ERROR: no info.dat -', info_file)
        exit()

    return pyre_load(info_file)


def load_traces_dat(folderpath, filename):
    traces_file = os.path.join(*glob_data_path, *folderpath, filename)
    if not os.path.exists(traces_file):
        print('ERROR: no *-traces.dat -', traces_file)
        exit()

    metadata = []
    traces = []
    for n, trace_content in enumerate(pyre_load(traces_file).data_blocks()):
        print('Loadings trial', n+1, '...')
        metadata.append(trace_content[:-1])
        traces.append(asarray([row.split() for row in trace_content[-1]], dtype=float).transpose())


    return metadata, traces


def read_noise_traces(folderpath, nfft = None):
    metadata, traces = load_traces_dat(folderpath, 'transferfunction-traces.dat')

    Pxxs = []
    Pyys = []
    Pxys = []
    Pyxs = []
    print('Processing trials ...')
    for t in traces:

        # get recordings
        sr = round(1000. / mean(diff(t[0, :])))
        output = t[1, :]
        response = t[2, :]

        # get spectra
        Pxx, Pyy, Pxy, Pyx, f = calc_spectra(output, response, sr, nfft)

        Pxxs.append(Pxx)
        Pyys.append(Pyy)
        Pxys.append(Pxy)
        Pyxs.append(Pyx)
    freqs = f

    return asarray(Pxxs), asarray(Pyys), asarray(Pxys), asarray(Pyxs), freqs


def gather_folders(years):
        # function gathers all folders in a given subdirectory on the global data path

        if not os.path.exists(os.path.join(*glob_data_path)):
            print('Data directory does not exist.')
            exit()

        folder_list = []
        for year in years:
            new_folders = glob(os.path.join(*glob_data_path, year, year+'*'))
            folder_list.extend([folder.split(os.sep) for folder in new_folders])

        return folder_list


def sanitize_string(instr):
    outstr = instr.replace('"', '').replace(' ', '')
    return outstr

###
# run script

if __name__ == '__main__':

    if 'load_noise' in sys.argv:
        folder_list = gather_folders(['2015', '2016'])

    Pxxs, Pyys, Pxys, Pyxs, freqs = read_noise_traces(['2016', '2016-07-21-ae-meadow'])

    embed()