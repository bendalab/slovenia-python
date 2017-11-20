from pyrelacs.DataClasses import load as pyre_load
from numpy import *
from path_settings import *
import pandas as pd
import pickle
import os
import sys
from glob import glob
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.mlab as ml


class read_data:

    master_file = 'master.pkl'

    def __init__(self):
        self.folder_list = []
        self.data = pd.DataFrame()

    def gather_folders(self, year):
        if os.path.exists(os.path.join(*data_dir)):
            new_folders = glob(os.path.join(*data_dir, year, year + '*'))
            self.folder_list.extend(new_folders)
            print(str(len(new_folders)) + ' folders have been added to list.')
        else:
            print('Data directorty does not exist.')
            exit()


    def read_rawdata(self):
        dataset_num = len(self.folder_list)
        for fentry, dataset_id in zip(self.folder_list, range(dataset_num)):
            # for testing
            #if dataset_id > 2:
                #break
            print('Entry ' + str(dataset_id) + ': ' + fentry + '...')

            info_path = os.path.join(fentry, 'info.dat')
            if os.path.exists(info_path):
                rec_info = pyre_load(info_path)

            # iterate through measurements
            transfer_file = os.path.join(fentry, 'transferfunction-traces.dat')
            if not os.path.exists(transfer_file):
                print('No transferfunction-traces.dat in ' + transfer_file)
                break

            transfer_data = pyre_load(transfer_file)

            outputs = []
            responses = []
            for trial_data in transfer_data.data_blocks():
                traces = asarray([row.split() for row in trial_data[-1]], dtype=float).transpose()
                outputs.append(traces[1, :])
                responses.append(traces[2, :])
            t = traces[0, :]

            # generate new DataFrame-row to be appended to self.data
            newrow = pd.DataFrame()

            # recording data
            # first column value needs to be provided as a list with one entry
            # or else pandas does not understand what to do with the information
            newrow['path'] = [fentry]  # do not remove brackets!
            newrow['foldername'] = rec_info[0]['Recording']['Name'].replace('"', '')
            newrow['distance'] = float(rec_info[0]['Recording']['Distance'][:-2])  # cm
            newrow['height'] = float(rec_info[0]['Recording']['Height'][:-2])  # cm
            newrow['condition'] = rec_info[0]['Recording']['Condition']
            newrow['temperature'] = float(rec_info[0]['Recording']['Temperature'][:-1])  # C

            # stimulus data
            stim_settings = trial_data[0]['Settings']['Stimulus']

            newrow['transfer_fmax'] = float(stim_settings['fmax'][:-2])  # Hz
            if 'fmin' in stim_settings.keys():
                f_min = float(stim_settings['fmin'][:-2])  # Hz
            else:
                f_min = None
            newrow['transfer_fmin'] = f_min

            try:
                duration = float(stim_settings['duration'][:-1])  # s
            except:
                duration = float(stim_settings['duration'][:-2]) / 1000  # ms
            newrow['transfer_duration'] = float(duration)  # s

            newrow['transfer_intensity'] = float(stim_settings['intensity'][:-2])  # dB
            newrow['transfer_amplitude'] = float(stim_settings['amplitude'][:-1])  # V


            # raw trial and metadata
            newrow['trial_data'] = [trial_data]
            newrow['info'] = [rec_info]

            # trace data
            newrow['transfer_time'] = [t]  # ms
            newrow['transfer_outputs'] = [asarray(outputs)]  # V
            newrow['transfer_responses'] = [asarray(responses)]  # V
            newrow['sr'] = 1000. / mean(diff(t))  # 1000 / dt => Hz

            # append new row
            self.data = self.data.append(newrow, ignore_index=True)


    def add_metadata(self):

        newdata = dict(rec_year=[])
        for idx in self.data.index:
            newdata['rec_year'].append(self.data.foldername[idx][:4])

        for newkey in newdata.keys():
            self.data[newkey] = newdata[newkey]


    def out_resp_transfer(self):
        # calculates the output-response transfer functions (program-microphone transfer)

        # add new columns
        newdata = dict(freqs=[], out_Pxxs=[], out_resp_Pxys=[], out_resp_transfer=[])

        print('Calculating output-response transfer...')
        for idx in self.data.index:
            print('Entry ' + str(idx) + ': distance ' + str(self.data.distance[idx]))
            outputs = self.data.transfer_outputs[idx]
            responses = self.data.transfer_responses[idx]
            sr = self.data.sr[idx]
            f, Pxys, Pxxs, H = self.transfer_fun(X = outputs, Y = responses, fs = sr)

            newdata['freqs'].append(f)
            newdata['out_Pxxs'].append(Pxxs)
            newdata['out_resp_Pxys'].append(Pxys)
            newdata['out_resp_transfer'].append(H)

            year = self.data.rec_year[idx]
            cond = self.data.condition[idx]
            dist = str(round(self.data.distance[idx]))
            plt.savefig(os.path.join(*fig_path, year + '-out-resp-transfer' + cond + dist + 'cm.pdf'), format='pdf')
            plt.close()

        self.add_data_to_frame(newdata)

    def sign_resp_transfer(self):
        # calculates to signal-response transfer function:
        # divides the out-resp transfer-fun by the out-resp for the minimum distance (100cm)
        # assumption: no change of spectral content for short distance

        newdata = dict(sign_resp_transfer=[])
        for idx in self.data.index:
            cond = (self.data.distance == 100) and (self.data.year == self.data.year[idx])
            sign_resp_transfer = self.data.out_resp_transfer[idx] / self.data.out_resp_transfer[cond]

            newdata['sign_resp_transfer'].append(sign_resp_transfer)

        self.add_data_to_frame(newdata)


    def add_data_to_frame(self, newdata):
        print('Adding new data to DataFrame')
        for newkey in newdata.keys():
            self.data[newkey] = newdata[newkey]


    def transfer_fun(self, X, Y, fs, nfft=2**11):

        xlim = [4000, 30000]
        set_num = X.shape[0]
        plt.figure()
        Pxxs = []
        Pxys = []
        for x, y, pltidx in zip(X, Y, range(set_num)):
            params = {'Fs':fs, 'NFFT':nfft, 'noverlap':nfft/2}
            x = x - mean(x)
            y = y - mean(y)

            # calc PSD and CSD
            Pxx, f = ml.psd(x, **params)
            Pxy, _ = ml.csd(x, y, **params)

            Pxys.append(Pxy)
            Pxxs.append(Pxx)

            # plot transfer function
            plt.subplot(set_num + 1, 1, pltidx + 1)
            plt.plot(f, abs(Pxy / Pxx), label='H = Pxy / Pxx')
            plt.xlim(xlim)
            plt.legend()

        # convert to arrays
        Pxxs = asarray(Pxxs)
        Pxys = asarray(Pxys)

        H = abs(mean(Pxys, axis=0)/ mean(Pxxs, axis=0))

        # plot transfer function
        plt.subplot(set_num + 1, 1, set_num + 1)
        plt.plot(f, H, label='H = abs(mean(Pxys) / mean(Pxxs))')
        plt.xlim(xlim)
        plt.legend()

        return f, Pxys, Pxxs, H


    def save_data(self):
        savepath = os.path.join(*pkl_path)
        if os.path.exists(savepath):
            print('Saving data to ' + os.path.join(savepath, self.master_file))
            with open(os.path.join(savepath, self.master_file), 'wb') as fobj:
                pickle.dump(self.data, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            fobj.close()
        else:
            print('Creating directory ' + savepath)
            os.mkdir(savepath)
            self.save_data()


    def load_data(self):
        savefile = os.path.join(*pkl_path, self.master_file)
        if os.path.exists(savefile):
            print('Loading data from ' + savefile)
            with open(savefile, 'rb') as fobj:
                self.data = pickle.load(fobj)
            fobj.close()
        else:
            print('No master file found.')
            exit()



if __name__ == '__main__':

    rd_handle = read_data()

    if 'gather' in sys.argv:
        rd_handle.gather_folders('2016')
        rd_handle.gather_folders('2015')

        rd_handle.read_rawdata()
        rd_handle.save_data()

        rd_handle.add_metadata()
        rd_handle.save_data()

    elif 'calculate' in sys.argv:
        rd_handle.load_data()

        rd_handle.out_resp_transfer()
        rd_handle.save_data()

        rd_handle.sign_resp_transfer()
        rd_handle.save_data()

    else:
        rd_handle.load_data()

        embed()


