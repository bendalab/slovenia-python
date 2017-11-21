from pyrelacs.DataClasses import load as pyre_load
from numpy import *
from path_settings import *
import pandas as pd
import pickle
import os
import gc
import sys
from glob import glob
from IPython import embed
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d import Axes3D


class data_cruncher:

    data_file = 'data.pkl'

    def __init__(self):
        self.data = self.data_from_file()


    def gather_folders(self, years):
        # function gathers all folderpaths on given subdirs

        if not os.path.exists(os.path.join(*data_dir)):
            print('Data directory does not exist.')
            exit()

        newcount = 0
        for year in years:
            new_folders = glob(os.path.join(*data_dir, year, year+'*'))
            folder_list = [folder.split(os.sep) for folder in new_folders]

            # add folders that are not already included in DataFrame
            for data_path in folder_list:
                if not data_path[-1] in self.data.data_folder.values:
                    newrow = dict(data_folder=data_path[-1], data_dir=[data_path[:-1]], digested=False)
                    self.data = self.data.append(pd.DataFrame(newrow), ignore_index=True)

                    newcount += 1

        print(newcount, 'new folders have been added to DataFrame', '...')


    def digest_rawdata(self, dat_files = dict(noise='transferfunction-traces')):

        # iterate through all folders that have not already been processed
        cond = self.data.loc[:, 'digested'] == False
        for rowidx, data_folder, data_dir in zip(self.data.index[cond], self.data.data_folder[cond], self.data.data_dir[cond]):
            print('Entry', rowidx, '-', data_folder, '...')

            data_path = data_dir + [data_folder]

            # get metadata
            info_path = os.path.join(*data_path, 'info.dat')
            if os.path.exists(info_path):
                rec_info = pyre_load(info_path)
            else:
                print('Info.dat missing')

            # load recordings
            for abbrv in dat_files.keys():
                dat_file = dat_files[abbrv] + '.dat'
                print(dat_file)

                transfer_file = os.path.join(*data_path, dat_file)
                if not os.path.exists(transfer_file):
                    print('File missing')
                    exit()


                Pxxs = []
                Pxys = []
                Pyys = []
                Pyxs = []
                # iterate through trials
                transfer_data = pyre_load(transfer_file)
                for trial_idx, trial_data in enumerate(transfer_data.data_blocks()):
                    print('Trial', trial_idx)
                    # load traces for this trial
                    traces = asarray([row.split() for row in trial_data[-1]], dtype=float).transpose()

                    # get data for spectral analysis
                    sr = 1000. / mean(diff(traces[0, :]))
                    output = traces[1, :]
                    output -= mean(output)
                    response = traces[2, :]
                    response -= mean(response)

                    nfft = 2 ** 11
                    sd_params = {'Fs': sr, 'NFFT': nfft, 'noverlap': nfft / 2}

                    Pxx, _ = ml.psd(output, **sd_params)
                    Pyy, _ = ml.psd(response, **sd_params)
                    Pxy, _ = ml.csd(output, response, **sd_params)
                    Pyx, f = ml.csd(response, output, **sd_params)

                    Pxxs.append(Pxx)
                    Pyys.append(Pyy)
                    Pxys.append(Pxy)
                    Pyxs.append(Pyx)

                    # free memory
                    del Pxx, Pyy, Pxy, Pyx, traces, output, response
                    gc.collect()

                # generate new dictionary containing spectra and metadata of all trials
                row_content = {
                    abbrv + '_Pxxs': asarray(Pxxs),
                    abbrv + '_Pyys': asarray(Pyys),
                    abbrv + '_Pxys': asarray(Pxys),
                    abbrv + '_Pyxs': asarray(Pyxs),
                    abbrv + '_freqs': f,
                    abbrv + '_trialmeta': [trial_data[0]],
                    'metadata': rec_info
                }

                # add to DataFrame
                self.add_data(rowidx, row_content)

                # free memory
                del Pxxs, Pyys, Pxys, Pyxs, f, transfer_data, rec_info, trial_data
                gc.collect()

            # mark dataset as processed
            self.add_data(rowidx, dict(digested=True))

            # save to file
            self.data_to_file()


    def extract_metadata(self, rawstr):
        return rawstr.replace('"', '').replace(' ', '')


    def add_relevant_metadata(self):
        # takes importan metadata from the dictionaries (as provided by pyrelacs)
        # and adds it to the DataFrame

        print('Add metadata information...')
        for rowidx, row in self.data.iterrows():
            metadata = row.metadata

            newdata = dict(condition = self.extract_metadata(metadata[0]['Recording']['Condition']),
                           distance = float(self.extract_metadata(metadata[0]['Recording']['Distance'])[:-2]),
                           height = float(self.extract_metadata(metadata[0]['Recording']['Height'])[:-2]),
                           temperature = self.extract_metadata(metadata[0]['Recording']['Temperature']),
                           date = self.extract_metadata(metadata[0]['Recording']['Date']),
                           year = float(self.extract_metadata(metadata[0]['Recording']['Date'])[:4]))

            # add to dataframe
            self.add_data(rowidx, newdata)
        # save file
        self.data_to_file()


    def calculate_stuff(self, abbrvs):

        print('Calculate output-response transfer functions...')
        for abbrv in abbrvs:
            for rowidx, rowdata in self.data.iterrows():

                newdata = dict()
                # mean PSD(output)
                newdata[abbrv + '_Pxx'] = mean(rowdata[abbrv + '_Pxxs'], axis=0)
                newdata[abbrv + '_Pxx_sd'] = std(rowdata[abbrv + '_Pxxs'], axis=0)

                # mean PSD(response)
                newdata[abbrv + '_Pyy'] = mean(rowdata[abbrv + '_Pyys'], axis=0)
                newdata[abbrv + '_Pyy_sd'] = std(rowdata[abbrv + '_Pyys'], axis=0)

                # mean CSD(output, response)
                newdata[abbrv + '_Pxy'] = mean(rowdata[abbrv + '_Pxys'], axis=0)
                newdata[abbrv + '_Pxy_sd'] = std(rowdata[abbrv + '_Pxys'], axis=0)

                # mean CSD(response, output)
                newdata[abbrv + '_Pyx'] = mean(rowdata[abbrv + '_Pyxs'], axis=0)
                newdata[abbrv + '_Pyx_sd'] = std(rowdata[abbrv + '_Pyxs'], axis=0)

                # output-response transfer
                newdata[abbrv + '_H_or'] = newdata[abbrv + '_Pxy'] / newdata[abbrv + '_Pxx']
                newdata[abbrv + '_H_or_sd'] = newdata[abbrv + '_Pxy_sd'] / newdata[abbrv + '_Pxx_sd']

                # response-output transfer
                newdata[abbrv + '_H_ro'] = newdata[abbrv + '_Pyx'] / newdata[abbrv + '_Pyy']
                newdata[abbrv + '_H_ro_sd'] = newdata[abbrv + '_Pyx_sd'] / newdata[abbrv + '_Pyy_sd']

                # add data
                self.add_data(rowidx, newdata)

            # save to file
            self.data_to_file()


        print('Calculate signal-response transfer functions and coherence...')
        # conditions for calibration-recordings (smallest distance in an open environment)
        calib_cond = {
            2015: (self.data.distance == 50) & (self.data.condition == 'Cutmeadow'),
            2016: (self.data.distance == 100) & (self.data.condition == 'Open')
        }

        for abbrv in abbrvs:
            for year in calib_cond.keys():
                # basic condition for all datasets
                dataset_cond = (self.data.year == year)

                # calculate mean output-response transfer function for equipment during this year
                # using the recordings made in the open with the smallest speaker-microphone distance (50 and 100cm)
                H_or_calib = mean(self.data[abbrv + '_H_or'][calib_cond[year] & dataset_cond].values, axis=0)
                H_ro_calib = mean(self.data[abbrv + '_H_ro'][calib_cond[year] & dataset_cond].values, axis=0)

                # get output-response transfer function for this year
                Hs_or = asarray([row for row in self.data[abbrv + '_H_or'][dataset_cond].values])
                # get output-response transfer function for this year
                Hs_ro = asarray([row for row in self.data[abbrv + '_H_ro'][dataset_cond].values])

                # calculate signal-response transfer functions
                Hs_sr = Hs_or / H_or_calib

                # calculate response-signal transfer functions
                Hs_rs = Hs_ro / H_ro_calib

                coh = Hs_sr * Hs_rs

                newdata = {
                    abbrv + '_H_sr': [row for row in Hs_sr],
                    abbrv + '_H_rs': [row for row in Hs_rs],
                    abbrv + '_coherence': [row for row in coh]
                }

                # add data to DataFrame
                self.add_data(dataset_cond, newdata)


        # save DataFrame to file
        self.data_to_file()


    def average_duplicates(self, abbrv):

        keycols = ['year', 'distance', 'condition', 'height']

        uniq_vals = []
        for keycol in keycols:
            uniq_vals.append(unique(self.data.loc[:, keycol].values))

        avg_df = pd.DataFrame()
        for year in uniq_vals[0]:
            for distance in uniq_vals[1]:
                for condition in uniq_vals[2]:
                    for height in uniq_vals[3]:
                        filter_cond = (self.data.year == year) & \
                                      (self.data.condition == condition) & \
                                      (self.data.distance == distance) & \
                                      (self.data.height == height)

                        if not any(filter_cond):
                            continue

                        newrow = {
                            'year': year,
                            'condition': condition,
                            'distance': distance,
                            'height': height,
                            'transfer_freqs': [self.data.transfer_freqs[filter_cond].values[0]],
                            abbrv + '_H_sr': [mean(self.data.transfer_H_sr[filter_cond].values, axis=0)]
                        }
                        avg_df = avg_df.append(pd.DataFrame(newrow), ignore_index=True)

        return avg_df


    def add_data(self, rowidx, rowdata):
        # expects newdata to be dictionary and adds dict-entries to specified row of DataFrame

        for colkey in rowdata.keys():
            if colkey not in self.data.columns:
                self.data[colkey] = None

            self.data.loc[rowidx, colkey] = rowdata[colkey]


    def data_to_file(self, filename = None, data = None):
        if filename is None:
            filename = self.data_file
            data = self.data

        savepath = os.path.join(*pkl_path)
        if os.path.exists(savepath):
            print('Saving data to ' + os.path.join(savepath, filename))
            with open(os.path.join(savepath, filename), 'wb') as fobj:
                pickle.dump(data, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            fobj.close()
        else:
            print('Creating directory ' + savepath)
            os.mkdir(savepath)
            self.data_to_file()


    def data_from_file(self, filename = None):


        if filename is None:
            filename = self.data_file

        savefile = os.path.join(*pkl_path, filename)

        print('Loading data from ' + savefile)
        if os.path.exists(savefile):
            with open(savefile, 'rb') as fobj:
                data = pickle.load(fobj)
            fobj.close()

            return data
        else:
            print('WARN: No data file found. Returning empty DataFrame')
            return pd.DataFrame(dict(data_folder = []))


if __name__ == '__main__':

    cruncher = data_cruncher()

    datasets = dict(
        #calls = 'stimulus-rectangular-traces'
        transfer = 'transferfunction-traces'
    )

    if 'digest' in sys.argv:
        # add folders to be processed
        cruncher.gather_folders(['2015', '2016'])

        # iterate through all new folders and calc spectra from raw traces
        cruncher.digest_rawdata(datasets)

    elif 'calculate' in sys.argv:
        cruncher.add_relevant_metadata()
        cruncher.calculate_stuff(datasets.keys())

    elif 'plot' in sys.argv:

        avg_data = cruncher.average_duplicates(abbrv = 'transfer')

        figs = dict()
        for rowidx, rowdata in avg_data.iterrows():
            print('Plot row', rowidx, '-', rowdata.distance, 'cm')

            figid = (rowdata.condition, 'height:'+str(rowdata.height), 'year:'+str(rowdata.year))
            if figid not in figs.keys():
                plt.figure(str(figid))
                figs[figid] = plt.subplot()

            figs[figid].plot(rowdata.transfer_freqs, abs(rowdata.transfer_H_sr), label=rowdata.distance)
            figs[figid].set_xlim(5000, 30000)
            plt.legend()

        plt.show()


    else:
        embed()