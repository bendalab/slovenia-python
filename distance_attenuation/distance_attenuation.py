from glob import glob
from IPython import embed
import matplotlib.mlab as ml
from numpy import *
import os
import pandas as pd
import pickle
from pyrelacs.DataClasses import load as pyre_load
from scipy.io import wavfile
import sys
import wave

###
# plotting
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt



###
# globals
glob_data_path = ['..', '..', 'data', 'distance_attenuation']
glob_pkl_path = ['..', '..', 'pkl', 'distance_attenuation']

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

    x = x - mean(x)
    y = y - mean(y)

    Pxx, _ = ml.psd(x, Fs=sr, **params)
    Pyy, _ = ml.psd(y, Fs=sr, **params)
    Pxy, _ = ml.csd(x, y, Fs=sr, **params)
    Pyx, f = ml.csd(y, x, Fs=sr, **params)

    return Pxx, Pyy, Pxy, Pyx, f


def calc_H_out_resp(data):

    print('Calculate output-response transfer functions...')
    for rowidx, rowdata in data.iterrows():

        newdata = dict()

        # mean PSD(output)
        newdata['Pxx'] = mean(rowdata['Pxxs'], axis=0)
        newdata['Pxx_sd'] = std(rowdata['Pxxs'], axis=0)

        # mean PSD(response)
        newdata['Pyy'] = mean(rowdata['Pyys'], axis=0)
        newdata['Pyy_sd'] = std(rowdata['Pyys'], axis=0)

        # mean CSD(output, response)
        newdata['Pxy'] = mean(rowdata['Pxys'], axis=0)
        newdata['Pxy_sd'] = std(rowdata['Pxys'], axis=0)

        # mean CSD(response, output)
        newdata['Pyx'] = mean(rowdata['Pyxs'], axis=0)
        newdata['Pyx_sd'] = std(rowdata['Pyxs'], axis=0)

        # output-response transfer
        newdata['H_or'] = newdata['Pxy'] / newdata['Pxx']
        newdata['H_or_sd'] = newdata['Pxy_sd'] / newdata['Pxx_sd']

        # response-output transfer
        newdata['H_ro'] = newdata['Pyx'] / newdata['Pyy']
        newdata['H_ro_sd'] = newdata['Pyx_sd'] / newdata['Pyy_sd']

        # add data
        data = add_data(data, newdata, rowidx)

    return data


def calc_H_sign_resp(data):

    print('Calculate signal-response transfer functions and coherence...')
    # conditions for calibration-recordings (smallest distance in an open environment)
    calib_cond = {
        2015: (data.distance == 50) & (data.condition == 'Cutmeadow'),
        2016: (data.distance == 100) & (data.condition == 'Open')
    }

    for year in calib_cond.keys():
        # basic condition for all datasets
        dataset_rows = (data.year == year)

        # calculate mean output-response transfer function for equipment during this year
        # using the recordings made in the open with the smallest speaker-microphone distance (50 and 100cm)
        H_or_calib = mean(data.H_or[calib_cond[year] & dataset_rows].values, axis=0)
        H_ro_calib = mean(data.H_ro[calib_cond[year] & dataset_rows].values, axis=0)

        # get output-response transfer function for this year
        Hs_or = asarray([row for row in data.H_or[dataset_rows].values])
        # get output-response transfer function for this year
        Hs_ro = asarray([row for row in data.H_ro[dataset_rows].values])

        # calculate signal-response transfer functions
        Hs_sr = Hs_or / H_or_calib

        # calculate response-signal transfer functions
        Hs_rs = Hs_ro / H_ro_calib

        coh = Hs_sr * Hs_rs

        newdata = dict(
            H_sr = [row for row in Hs_sr],
            H_rs = [row for row in Hs_rs],
            coherence = [row for row in coh]
        )

        # add data
        data = add_data(data, newdata, dataset_rows)

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
        print('WARN: No data file found. Returning empty DataFrame')
        return pd.DataFrame(dict(data_folder = []))


def load_info_dat(folderpath):
    info_file = os.path.join(*(folderpath + ['info.dat']))
    if not os.path.exists(info_file):
        print('ERROR: no info.dat -', info_file)
        exit()

    return pyre_load(info_file)


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
        traces.append(asarray([row.split() for row in trace_content[-1]], dtype=float).transpose())


    return metadata, traces


def read_call_traces(folderpath, nfft = None):

    # get recorded data
    metadata, traces = load_traces_dat(folderpath, 'stimulus-file-traces.dat')

    # get stimulus file contents (recording of a pholidoptera call)
    sr, data = wavfile.read(os.path.join(*(glob_data_path + ['Pholidoptera_littoralis-HP1kHz-T25C.wav'])))
    output = data[:, 0]  # use first channel

    Pxxs = []
    Pyys = []
    Pxys = []
    Pyxs = []
    print('Processing trials ...')
    for t in traces:

        # get recordings
        sr = round(1000. / mean(diff(t[0, :])))
        response = t[1, :]

        # get spectra
        Pxx, Pyy, Pxy, Pyx, f = calc_spectra(output, response, sr, nfft)

        Pxxs.append(Pxx)
        Pyys.append(Pyy)
        Pxys.append(Pxy)
        Pyxs.append(Pyx)
    freqs = f

    return asarray(Pxxs), asarray(Pyys), asarray(Pxys), asarray(Pyxs), freqs

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
            new_folders = glob(os.path.join(*(glob_data_path + [year] + [year+'*'])))
            folder_list.extend([folder.split(os.sep) for folder in new_folders])

        return folder_list


def sanitize_string(instr):
    outstr = instr.replace('"', '').replace(' ', '')
    return outstr


###
# run script

if __name__ == '__main__':

    # trial with band-limited white noise
    if 'load_noise' in sys.argv:

        # initiate DataFrame
        data = pd.DataFrame()

        pkl_file = 'noise_data.pkl'

        # get folders
        folder_list = gather_folders(['2015', '2016'])

        entry_num = len(folder_list)
        for idx, folder in enumerate(folder_list):
            print('Entry', idx+1, '/', entry_num, ' - Processing ', os.path.join(*folder), '...')

            Pxxs, Pyys, Pxys, Pyxs, freqs = read_noise_traces(folder[-2:])

            data = add_data(data, dict(Pxxs=[Pxxs], Pyys=[Pyys], Pxys=[Pxys], Pyxs=[Pyxs], freqs=[freqs]))

        data_to_file(pkl_file, data)

    ###
    # trials with recorded bushcricket calls
    elif 'load_call' in sys.argv:

        # initiate DataFrame
        data = pd.DataFrame()

        pkl_file = 'call_data.pkl'

        # get folders
        folder_list = gather_folders(['2016'])

        entry_num = len(folder_list)
        for idx, folder in enumerate(folder_list):
            print('Entry', idx + 1, '/', entry_num, ' - Processing ', os.path.join(*folder), '...')

            Pxxs, Pyys, Pxys, Pyxs, freqs = read_noise_traces(folder[-2:])

            data = add_data(data, dict(Pxxs=[Pxxs], Pyys=[Pyys], Pxys=[Pxys], Pyxs=[Pyxs], freqs=[freqs]))

        data_to_file(pkl_file, data)

    # for testing
    else:

        sr_out, data = wavfile.read(os.path.join(*(glob_data_path + ['Pholidoptera_littoralis-HP1kHz-T25C.wav'])))
        output = data[:, 0]  # use first channel
        t_out = arange(0, output.shape[0] / sr_out, 1. / sr_out)

        metadata, traces = load_traces_dat(['2016', '2016-07-22-aa-open'], 'stimulus-file-traces.dat')
        recordings = asarray([t[1, :] for t in traces])
        sr_rec = round(1000. / mean(diff(traces[0][0, :])))
        t_rec = arange(0, recordings.shape[1] / sr_rec, 1. / sr_rec)

        embed()

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t_out, output)

        plt.subplot(2,1,2)
        for rec in recordings:
            plt.plot(t_rec, rec)

        plt.show()


