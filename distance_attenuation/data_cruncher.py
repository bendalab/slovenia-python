import gc
from glob import glob
from IPython import embed
import matplotlib
import matplotlib.mlab as ml
from numpy import *
import os
import pandas as pd
from path_settings import *
import pickle
from pyrelacs.DataClasses import load as pyre_load
import sys

# matplotlib plotting
matplotlib.use('Qt5Agg')
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


class data_cruncher:

    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.data_from_file()


    def add_data(self, rowidx, rowdata):
        # expects newdata to be dictionary and adds dict-entries to specified row of DataFrame

        for colkey in rowdata.keys():
            if colkey not in self.data.columns:
                self.data[colkey] = None

            self.data.loc[rowidx, colkey] = rowdata[colkey]


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


    def average_duplicates(self):

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

                        newrow = dict(
                            year = year,
                            condition = condition,
                            distance = distance,
                            height = height,
                            freqs = [self.data.freqs[filter_cond].values[0]],
                            H_sr =  [mean(self.data.H_sr[filter_cond].values, axis=0)]
                        )

                        avg_df = avg_df.append(pd.DataFrame(newrow), ignore_index=True)

        return avg_df


    def calculate_stuff(self):

        print('Calculate output-response transfer functions...')
        for rowidx, rowdata in self.data.iterrows():

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

            newdata = dict(
                H_sr = [row for row in Hs_sr],
                H_rs = [row for row in Hs_rs],
                coherence = [row for row in coh]
            )

            # add data to DataFrame
            self.add_data(dataset_cond, newdata)


        # save DataFrame to file
        self.data_to_file()


    def data_to_file(self):
        savepath = os.path.join(*pkl_path)
        if os.path.exists(savepath):
            print('Saving data to ' + os.path.join(savepath, self.data_file))
            with open(os.path.join(savepath, self.data_file), 'wb') as fobj:
                pickle.dump(self.data, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            fobj.close()
        else:
            print('Creating directory ' + savepath)
            os.mkdir(savepath)
            self.data_to_file()


    def data_from_file(self):
        savefile = os.path.join(*pkl_path, self.data_file)

        print('Loading data from ' + savefile)
        if os.path.exists(savefile):
            with open(savefile, 'rb') as fobj:
                data = pickle.load(fobj)
            fobj.close()

            return data
        else:
            print('WARN: No data file found. Returning empty DataFrame')
            return pd.DataFrame(dict(data_folder = []))


    def digest_rawdata(self, dat_file, sd_params = None):
        if sd_params is None:
            nfft = 2 ** 11
            sd_params = {'NFFT': nfft, 'noverlap': nfft / 2}

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
                sr = round(1000. / mean(diff(traces[0, :])))
                output = traces[1, :]
                output -= mean(output)
                response = traces[2, :]
                response -= mean(response)

                sd_params['Fs'] = sr

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
            row_content = dict(
                Pxxs = asarray(Pxxs),
                Pyys = asarray(Pyys),
                Pxys = asarray(Pxys),
                Pyxs = asarray(Pyxs),
                freqs = f,
                trialmeta = [trial_data[0]],
                metadata = rec_info
            )

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


if __name__ == '__main__':

    transfer_crunch = data_cruncher('data_transfer.pkl')

    if 'digest' in sys.argv:
        # add folders to be processed
        transfer_crunch.gather_folders(['2015', '2016'])

        # iterate through all new folders and calc spectra from raw traces
        transfer_crunch.digest_rawdata(dat_file = 'transferfunction-traces.dat')

    elif 'calculate' in sys.argv:
        transfer_crunch.add_relevant_metadata()
        transfer_crunch.calculate_stuff()

    elif 'plot' in sys.argv:

        # get data averaged over height, condition, distance and year of recording
        avg_data = transfer_crunch.average_duplicates()

        figs = dict()
        if False:  # plot 2d
            for rowidx, rowdata in avg_data.iterrows():
                print('Plot row', rowidx, '-', rowdata.distance, 'cm')

                figid = ('2d', rowdata.condition, 'height:' + str(rowdata.height), 'year:' + str(rowdata.year))
                if figid not in figs.keys():
                    plt.figure(str(figid))
                    figs[figid] = plt.subplot()

                figs[figid].semilogy(rowdata.freqs, abs(rowdata.H_sr), label=rowdata.distance)
                figs[figid].set_xlim(5000, 30000)
                figs[figid].legend()

        if True:  # plot 3d

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

                # white noise from 5 to 30 kHz, but speakers may have been too weak above 25kHz
                figs[figid].set_ylim(5000, 25000)





        plt.show()


    else:
        embed()
