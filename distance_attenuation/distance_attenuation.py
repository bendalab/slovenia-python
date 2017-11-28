# Author: Tim Hladnik

from glob import glob
from IPython import embed
import matplotlib.mlab as ml
#from numba import jit, int32, float64
from numpy import *
import os
import pandas as pd
import pickle
from pyrelacs.DataClasses import load as pyre_load
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
import sys
import wave


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


def add_metadata(data):
    # takes importan metadata from the dictionaries (as provided by pyrelacs)
    # and adds it to the DataFrame

    print('Add metadata information...')
    for rowidx, row in data.iterrows():
        metadata = row.metadata

        newdata = dict(condition=sanitize_string(metadata[0]['Recording']['Condition']),
                       distance=float(sanitize_string(metadata[0]['Recording']['Distance'])[:-2]),
                       height=float(sanitize_string(metadata[0]['Recording']['Height'])[:-2]),
                       temperature=sanitize_string(metadata[0]['Recording']['Temperature']),
                       date=sanitize_string(metadata[0]['Recording']['Date']),
                       year=float(sanitize_string(metadata[0]['Recording']['Date'])[:4]))

        # add to dataframe
        data = add_data(data, newdata, rowidx)

    return data


def average_duplicates(data, avg_cols = None):
    if avg_cols is None:
        avg_cols = ['freqs', 'H_sr']

    keycols = ['year', 'distance', 'condition', 'height']

    uniq_vals = []
    for keycol in keycols:
        uniq_vals.append(unique(data.loc[:, keycol].values))

    avg_df = pd.DataFrame()
    for year in uniq_vals[0]:
        for distance in uniq_vals[1]:
            for condition in uniq_vals[2]:
                for height in uniq_vals[3]:
                    filter_cond = (data.year == year) & \
                                  (data.condition == condition) & \
                                  (data.distance == distance) & \
                                  (data.height == height)

                    if not any(filter_cond):
                        continue

                    newrow = dict(
                        year = year,
                        condition = condition,
                        distance = distance,
                        height = height
                    )

                    for col in avg_cols:
                        newrow[col] = [mean(data[col][filter_cond].values, axis=0)]

                    avg_df = avg_df.append(pd.DataFrame(newrow), ignore_index=True)

    return avg_df


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    [b, a] = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    [b, a] = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def calc_spectra(x, y, sr, nfft = None):
    # expects two 1d arrays as signals and a number for the sampling rate
    # (optional) parameters for FFT

    # returns PSD(x,x), PSD(y,y), PSD(x,y), PSD(y,x) and frequency

    if nfft is None:
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
        newdata['Pxy'] = mean(abs(rowdata['Pxys']), axis=0)
        newdata['Pxy_sd'] = std(abs(rowdata['Pxys']), axis=0)

        # mean CSD(response, output)
        newdata['Pyx'] = mean(abs(rowdata['Pyxs']), axis=0)
        newdata['Pyx_sd'] = std(abs(rowdata['Pyxs']), axis=0)

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
        2015: (data.distance == 50) & (data.condition == 'Cutmeadow') & (data.height == 150),
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

        # calculate signal-response transfer functions (forward transfer)
        Hs_sr = Hs_or / H_or_calib

        # calculate response-signal transfer functions (backward transfer)
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
        print('ERROR: pkl file not found.')
        exit()


def extract_envelope(x, sr):
    # hpf
    b, a = butter_highpass(50, sr)
    y1 = signal.filtfilt(b, a, x)

    # lpf
    b, a = butter_lowpass(100, sr)
    y2 = signal.filtfilt(b, a, abs(y1))

    return y2


def load_info_dat(folderpath):
    info_file = os.path.join(*(glob_data_path + folderpath + ['info.dat']))
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


def read_call_traces(folderpath, plot = True):

    # get recorded data
    metadata, traces = load_traces_dat(folderpath, 'stimulus-file-traces.dat')

    if plot:
        plt.figure(folderpath[-1])
        plotnum = len(traces)

    print('Processing trials ...')
    envelopes = []
    sample_points = False
    for idx, trial in enumerate(traces):

        # get recordings
        sr = round(1000. / mean(diff(trial[0, :])))
        rec = trial[1, :]

        # discard if trials have unequal lengths (error while recording)
        if not sample_points:
            sample_points = rec.shape[0]
        if sample_points != rec.shape[0]:
            return False, False, False

        # get envelope
        env = extract_envelope(rec, sr)

        # downsampling
        reduce_order = 10
        reduce_num = 2
        for num in range(reduce_num):
            env = signal.decimate(env, reduce_order, zero_phase=True)

        envelopes.append(env)
        t = arange(0, float(len(rec)) / sr, 1. / (sr / (reduce_order ** reduce_num)))

        if plot:
            plt.subplot(plotnum, 1, idx + 1)
            plt.plot(trial[0, :]/1000, rec - mean(rec), label='Raw')
            plt.plot(t, env, label='Envelope')

    if plot:
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.savefig(os.path.join(*(glob_fig_path + ['calls_' + folderpath[-1] + '.pdf'])), format='pdf')
        #plt.show()
        plt.close()



    return metadata, t, asarray(envelopes)


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

    return metadata, asarray(Pxxs), asarray(Pyys), asarray(Pxys), asarray(Pyxs), freqs


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


###############
# run script
###############

if __name__ == '__main__':

    ###
    # for testing
    if 'test' == sys.argv[1]:



        # ignore wav-sampling rate because it is not used for stimulation
        _, data = wavfile.read(os.path.join(*(glob_data_path + ['Pholidoptera_littoralis-HP1kHz-T25C.wav'])))
        output = data[:, 0]  # use first channel

        folder = ['2016', '2016-07-22-aa-open']

        metadata, traces = load_traces_dat(folder, 'stimulus-file-traces.dat')
        recordings = asarray([t[1, :] for t in traces])
        sr = round(1000. / mean(diff(traces[0][0, :])))

        trialmeta, t, envelopes = read_call_traces(folder)
        if trialmeta is False:
            print('ERROR')
            exit()

        t_rec = arange(0, recordings.shape[1] / sr, 1. / sr)
        t_out = arange(0, output.shape[0] / sr, 1. / sr)


        plt.figure()
        pltnum = len(recordings) + 1
        plt.subplot(pltnum, 1, 1)
        plt.plot(t_out, output)
        plt.xlim(0, 3)


        for idx, (rec, env) in enumerate(zip(recordings, envelopes)):
            rec = rec - mean(rec)
            plt.subplot(pltnum, 1, idx + 2)
            plt.plot(t_rec, rec)

            plt.plot(t, env)
            plt.xlim(0, 3)

        plt.show()


    ###
    # trial with band-limited white noise
    if 'load_noise' == sys.argv[1]:

        if len(sys.argv) < 3:
            print('Too few arguments. Please add a pkl-filename.')
            exit()

        pkl_file = sys.argv[2]

        nfft = None
        if len(sys.argv) > 3:
            try:
                nfft = int(sys.argv[-1])
            except:
                print('WARNING: invalid NFFT parameter in script call. Needs to be integer.')

        # initiate DataFrame
        data = pd.DataFrame()

        # get folders
        folder_list = gather_folders(['2015', '2016'])

        entry_num = len(folder_list)
        for idx, folder in enumerate(folder_list):
            print('Entry', idx+1, '/', entry_num, ' - Processing ', os.path.join(*folder), '...')

            # recording metadata
            metadata = load_info_dat(folder[-2:])

            # get spectra for stimulus condition
            trialmeta, Pxxs, Pyys, Pxys, Pyxs, freqs = read_noise_traces(folder[-2:], nfft=nfft)

            # add row to DataFrame
            newdata = dict(Pxxs = [Pxxs],
                           Pyys = [Pyys],
                           Pxys = [Pxys],
                           Pyxs = [Pyxs],
                           freqs = [freqs],
                           trialmeta=[trialmeta],
                           metadata=[metadata])
            data = add_data(data, newdata)

        # save to file
        data_to_file(pkl_file, data)

        # extract metadata from RELACS output
        data = add_metadata(data)

        # calculate transfer functions
        data = calc_H_sign_resp(calc_H_out_resp(data))
        data_to_file(pkl_file, data)


    ###
    # trials with recorded bushcricket calls
    if 'load_call' == sys.argv[1]:

        if len(sys.argv) < 3:
            print('Too few arguments. Please add a pkl-filename.')
            exit()

        pkl_file = sys.argv[2]

        # initiate DataFrame
        data = pd.DataFrame()

        # get folders
        folder_list = gather_folders(['2016'])

        entry_num = len(folder_list)
        for idx, folder in enumerate(folder_list):
            print('Entry', idx + 1, '/', entry_num, ' - Processing ', os.path.join(*folder), '...')

            # recording metadata
            metadata = load_info_dat(folder[-2:])

            # get spectra for stimulus condition
            trialmeta, t, envelopes = read_call_traces(folder[-2:])
            if trialmeta is False:
                print('Skipping dataset because some trials may be corrupted.')
                continue


            # add row to DataFrame
            newdata = dict(times=[t],
                           envelopes=[envelopes],
                           trialmeta=[trialmeta],
                           metadata=[metadata])
            data = add_data(data, newdata)


        # save to file
        data_to_file(pkl_file, data)

        # extract metadata from RELACS output
        data = add_metadata(data)
        data_to_file(pkl_file, data)

        embed()


    if 'recalc' == sys.argv[1]:

        # load pkl
        data_file = sys.argv[-1]
        data = data_from_file(data_file)

        # update metadata
        data = add_metadata(data)
        # calculate transfer functions
        data = calc_H_sign_resp(calc_H_out_resp(data))
        data_to_file(data_file, data)

