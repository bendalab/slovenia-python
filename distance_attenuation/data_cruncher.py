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


class data_cruncher:

    data_file = 'data.pkl'
    avg_data_file = 'avg_data.pkl'

    def __init__(self):
        self.folder_list = []
        self.data = pd.DataFrame()
        self.avg_data = pd.DataFrame()


    def gather_folders(self, year):
        # function gathers all folderpaths on given subdir

        if os.path.exists(os.path.join(*data_dir)):
            new_folders = glob(os.path.join(*data_dir, year, year+'*'))
            self.folder_list.extend([folder.split(os.sep) for folder in new_folders])
            print(str(len(new_folders)), 'new folders have been added to list for subdir', year, '...')
        else:
            print('Data directory does not exist.')
            exit()


    def digest_rawdata(self, dat_file, abbrv):

        # check if DataFrame contains any data was loaded and if not, warn user that any existing  data
        # on the data path will be overwritten in the process
        if self.data.empty:
            confirmation = input('WARNING: if you continue, the current '+self.data_file+' will be overwritten! Confirm? (Y/n): ')
            if confirmation.lower() != 'y':
                exit()

        self.data['data_dir'] = None
        self.data['data_folder'] = None
        self.data['digested'] = None

        # add folders that are not already included in DataFrame
        for data_path in self.folder_list:
                if data_path[-1] not in self.data.data_folder.values:
                    newrow = dict(data_folder=data_path[-1], data_dir=[data_path[:-1]], digested=False)
                    self.data = self.data.append(pd.DataFrame(newrow), ignore_index=True)


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

            # load recordings of white-noise stimuli
            transfer_file = os.path.join(*data_path, dat_file+'.dat')
            if not os.path.exists(transfer_file):
                print(dat_file+'.dat missing')
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
            row_content = dict(Pxxs = asarray(Pxxs),
                               Pyys = asarray(Pyys),
                               Pxys = asarray(Pxys),
                               Pyxs = asarray(Pyxs),
                               freqs = f,
                               digested = True,
                               metadata = rec_info,
                               trialmeta = [trial_data[0]])

            # add to DataFrame
            self.add_data(rowidx, row_content)

            # free memory
            del Pxxs, Pyys, Pxys, Pyxs, f, transfer_data, rec_info, trial_data
            gc.collect()

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


    def calculate_stuff(self):

        print('Calculate output-response transfer functions...')
        for rowidx, rowdata in self.data.iterrows():

            newdata = dict()
            # mean PSD(output)
            newdata['Pxx'] = mean(rowdata.Pxxs, axis=0)
            newdata['Pxx_sd'] = std(rowdata.Pxxs, axis=0)

            # mean PSD(response)
            newdata['Pyy'] = mean(rowdata.Pyys, axis=0)
            newdata['Pyy_sd'] = std(rowdata.Pyys, axis=0)

            # mean CSD(output, response)
            newdata['Pxy'] = mean(rowdata.Pxys, axis=0)
            newdata['Pxy_sd'] = std(rowdata.Pxys, axis=0)

            # mean CSD(response, output)
            newdata['Pyx'] = mean(rowdata.Pyxs, axis=0)
            newdata['Pyx_sd'] = std(rowdata.Pyxs, axis=0)

            # output-response transfer
            newdata['H_or'] = abs(newdata['Pxy']) / newdata['Pxx']
            newdata['H_or_sd'] = abs(newdata['Pxy_sd']) / newdata['Pxx_sd']

            # response-output transfer
            newdata['H_ro'] = abs(newdata['Pyx']) / newdata['Pyy']
            newdata['H_ro_sd'] = abs(newdata['Pyx_sd']) / newdata['Pyy_sd']

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

        for year in calib_cond.keys():
            # basic condition for all datasets
            dataset_cond = (self.data.year == year)

            # calculate mean output-response transfer function for equipment during this year
            # using the recordings made in the open with the smallest speaker-microphone distance (50 and 100cm)
            H_or_calib = mean(self.data.H_or[calib_cond[year] & dataset_cond].values, axis=0)
            H_ro_calib = mean(self.data.H_ro[calib_cond[year] & dataset_cond].values, axis=0)

            # get output-response transfer function for this year
            Hs_or = asarray([row for row in self.data.H_or[dataset_cond].values])
            # get output-response transfer function for this year
            Hs_ro = asarray([row for row in self.data.H_ro[dataset_cond].values])

            # calculate signal-response transfer functions
            Hs_sr = Hs_or / H_or_calib

            # calculate response-signal transfer functions
            Hs_rs = Hs_ro / H_ro_calib

            coh = Hs_sr * Hs_rs

            newdata = dict(H_sr = [row for row in Hs_sr],
                           H_rs = [row for row in Hs_rs],
                           coherence = [row for row in coh])

            # add data to DataFrame
            self.add_data(dataset_cond, newdata)


        # save DataFrame to file
        self.data_to_file()


    def average_duplicates(self):
        keycols = ['year', 'distance', 'condition', 'height']

        uniq_vals = []
        for keycol in keycols:
            uniq_vals.append(unique(self.data.loc[:, keycol].values))

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

                        newrow = dict(
                            year = year,
                            condition = condition,
                            distance = distance,
                            height = height,
                            freqs = [self.data.freqs[filter_cond].values[0]],
                            H_sr = [mean(self.data.H_sr[filter_cond].values, axis=0)]
                        )
                        self.avg_data = self.avg_data.append(pd.DataFrame(newrow), ignore_index=True)

        self.avg_data_to_file()


    def add_data(self, rowidx, rowdata):
        # expects newdata to be dictionary and adds dict-entries to specified row of DataFrame

        for colkey in rowdata.keys():
            if colkey not in self.data.columns:
                self.data[colkey] = None

            self.data.loc[rowidx, colkey] = rowdata[colkey]


    def avg_data_to_file(self):
        self.data_to_file(self.avg_data_file, self.avg_data)


    def avg_data_from_file(self):
        self.avg_data = self.data_from_file(self.avg_data_file)


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
        if os.path.exists(savefile):
            print('Loading data from ' + savefile)
            with open(savefile, 'rb') as fobj:
                data = pickle.load(fobj)
            fobj.close()

            if filename == self.data_file:
                self.data = data
            else:
                return data
        else:
            print('WARN: No master file found.')


if __name__ == '__main__':

    cruncher = data_cruncher()

    if 'gather' in sys.argv:
        # add folders to be processed
        cruncher.gather_folders('2016')
        cruncher.gather_folders('2015')


        # load data from .pkl file if it exists
        cruncher.data_from_file()

        # iterate through all new folders and calc spectra from raw traces
        cruncher.digest_rawdata('transferfunction-traces')

    elif 'calculate' in sys.argv:
        # load DataFrame from file
        cruncher.data_from_file()

        cruncher.add_relevant_metadata()
        cruncher.calculate_stuff()

    elif 'average' in sys.argv:
        cruncher.data_from_file()
        cruncher.average_duplicates()

    elif 'plot' in sys.argv:
        cruncher.avg_data_from_file()

        figs = dict()
        for rowidx, rowdata in cruncher.avg_data.iterrows():
            print('Plot row', rowidx, '-', rowdata.distance, 'cm')

            figid = (rowdata.condition, 'height:'+str(rowdata.height), 'year:'+str(rowdata.year))
            if figid not in figs.keys():
                plt.figure(str(figid))
                figs[figid] = plt.subplot()

            figs[figid].plot(rowdata.freqs, rowdata.H_sr, label=rowdata.distance)
            plt.legend()

        plt.show()


    else:
        cruncher.data_from_file()
        embed()