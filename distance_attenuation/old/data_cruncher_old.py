from pyrelacs.DataClasses import load as pyre_load
from numpy import *
from path_settings import *
import pandas as pd
import pickle
import os
import json
import gc
import sys
from glob import glob
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.mlab as ml


class data_cruncher:

    index_file = 'index.json'

    def __init__(self):
        self.folder_list = []
        self.index = None


    def gather_folders(self, year):
        # function gathers all folderpaths on given subdir for which there exists no .json-file

        if os.path.exists(os.path.join(*data_dir)):
            new_folders = glob(os.path.join(*data_dir, year, year+'*'))
            counter = 0
            for folder in new_folders:
                folder = folder.split(os.sep)
                # if there exists no .pkl file for this dataset
                if not os.path.exists(os.path.join(*json_path, folder[-1]+'.json')):
                    self.folder_list.append(folder)
                    counter += 1
            print(str(counter), 'new folders have been added to list for subdir', year)
        else:
            print('Data directorty does not exist.')
            exit()


    def update_index(self):
        print('--------\nUpdating dataset index...\n--------')

        datasets = os.listdir(os.path.join(*json_path))

        self.index = dict(json_file=[], distance=[], height=[], condition=[], year=[], dataset_count=0)

        counter = 0
        for fname in datasets:
            if (fname[-4:] != 'json') or (fname == self.index_file):
                continue

            print('Entry:', counter)

            content = self.load_data(fname)

            self.index['json_file'].append(fname)

            self.index['distance'].append(content['distance'])
            self.index['height'].append(content['height'])
            self.index['condition'].append(content['condition'])
            self.index['year'].append(content['year'])

            counter += 1

        self.index['dataset_count'] = counter

        self.save_index()


    def digest_rawdata(self):
        dataset_num = len(self.folder_list)
        for fentry, dataset_id in zip(self.folder_list, range(dataset_num)):
            # for testing
            #if dataset_id > 4:
                #break
            print('Entry', str(dataset_id), '-', os.path.join(*fentry), '...')

            info_path = os.path.join(*fentry, 'info.dat')
            if os.path.exists(info_path):
                rec_info = pyre_load(info_path)
            else:
                print('Info.dat missing')

            # iterate through measurements
            transfer_file = os.path.join(*fentry, 'transferfunction-traces.dat')
            if not os.path.exists(transfer_file):
                print('transferfunction-traces.dat missing')
                exit()

            transfer_data = pyre_load(transfer_file)

            outputs = []
            responses = []

            for trial_data in transfer_data.data_blocks():
                traces = asarray([row.split() for row in trial_data[-1]], dtype=float).transpose()
                outputs.append(traces[1, :])
                responses.append(traces[2, :])
            t = traces[0, :]

            # generate new dictionary
            newrow = dict()

            # recording data
            newrow['path'] = fentry
            newrow['foldername'] = fentry[-1]
            newrow['year'] = newrow['foldername'][:4]
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
            newrow['trial_data'] = trial_data
            newrow['info'] = rec_info

            # trace data
            newrow['transfer_time'] = t  # ms
            newrow['transfer_outputs'] = asarray(outputs)  # V
            newrow['transfer_responses'] = asarray(responses)  # V
            newrow['sr'] = 1000. / mean(diff(t))  # 1000 / dt => Hz

            # save data to individual pickle0file
            self.save_data(fentry[-1], newrow)

            # clear all variables
            del newrow, rec_info, transfer_data, traces, outputs, responses, t, stim_settings
            gc.collect()

        # update file index
        self.update_index()


    def prepare_data(self):
        self.out_resp_transfer()
        self.sign_resp_transfer()


    def out_resp_transfer(self):
        # calculates the output-response transfer functions (program-microphone transfer)

        print('Calculating output-response transfer...')
        for idx, json_file in enumerate(self.index['json_file']):

            # add new columns
            newdata = dict(freqs=None, out_Pxxs=None, out_resp_Pxys=None, out_resp_transfer=None)

            # load dataset
            content = self.load_data(json_file)

            # get program outputs and recorded responses
            outputs = content['transfer_outputs']
            responses = content['transfer_responses']
            sr = content['sr']

            # calculate transfer function
            f, Pxys, Pxxs, H = self.transfer_fun(X = outputs, Y = responses, fs = sr)

            # append to newdata
            newdata['freqs'] = f
            newdata['out_Pxxs'] = abs(Pxxs)
            newdata['out_resp_Pxys'] = abs(Pxys)
            newdata['out_resp_transfer'] = H

            year = content['year']
            condition = content['condition']
            distance = str(round(content['distance']))
            plt.savefig(os.path.join(*fig_path, year+'-out-resp-transfer'+condition+distance+'cm.pdf'), format='pdf')
            plt.close()

            self.add_data(json_file, content, newdata)


    def sign_resp_transfer(self):
        # calculates to signal-response transfer function:
        # divides the out-resp transfer-fun by the out-resp for the minimum distance (100cm) in the 'free' condtition
        # assumption: no change of spectral content for short distance

        for year in unique(self.index['year']):
            embed()
            if year == '2015':
                cond = (self.index['distance'] == 50) & (self.index['condition'] == 'Cut meadow')
            elif year == '2016':
                cond = (self.index['distance'] == 100) & (self.index['condition'] == 'Open')

            baseline_transfer = self.load_data(self.index['json_file'][cond])['out_resp_transfer']

            for json_file in self.index['json_file'][self.index['year'] == year]:

                # add new columns
                newdata = dict(sign_resp_transfer=[])

                # load dataset
                content = self.load_data(json_file)

                # calculate signal-response transfer function
                sign_resp_transfer = content['out_resp_transfer'] / baseline_transfer

                newdata['sign_resp_transfer'].append(sign_resp_transfer)

                plt.figure()
                plt.plot(content['freqs'], sign_resp_transfer, label='Signal-response-transfer')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Gain')

                plt.legend()
                distance = content['distance']
                condition = content['condition']
                plt.savefig(os.path.join(*fig_path, year + '-sign-resp-transfer'+condition+distance+'cm.pdf'), format='pdf')
                plt.close()

                self.add_data(json_file, content, newdata)


    def add_data(self, json_file, filecontent, newdata):
        print('Adding new data to file', json_file)
        for newkey in newdata.keys():
            filecontent[newkey] = newdata[newkey]

        self.save_data(json_file, filecontent)


    def transfer_fun(self, X, Y, fs, nfft=2**11):

        xlim = [5000, 30000]
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


    def save_index(self):
        self.save_data(self.index_file, self.index)


    def save_data(self, filename, content):
        if filename[-4:] != 'json':
            filename += '.json'

        if os.path.exists(os.path.join(*json_path)):
            savepath = json_path + [filename]
            print('Saving data to', os.path.join(*savepath))
            with open(os.path.join(*savepath), 'w') as fobj:

                # convert arrays to lists, because json does not accept ndarray
                for keyname in content.keys():
                    value = content[keyname]
                    if isinstance(value, ndarray):
                        content[keyname] = value.tolist()

                # dump contents
                json.dump(content, fobj)
            fobj.close()

        else:
            print('Creating directory', os.path.join(*json_path))
            os.mkdir(os.path.join(*json_path))
            self.save_data(filename, content)


    def load_index(self):
        self.index = self.load_data(self.index_file)


    def load_data(self, filename):
        if filename[-4:] != 'json':
            filename += '.json'
        savefile = os.path.join(*json_path, filename)

        if os.path.exists(savefile):
            print('Loading data from', savefile)
            with open(savefile, 'r') as fobj:

                # load file contents
                rdata = json.load(fobj)

                # convert lists to arrays if possible
                for keyname in rdata.keys():
                    value = rdata[keyname]
                    if isinstance(value, list):
                        try:
                            rdata[keyname] = asarray(value)
                        except:
                            rdata[keyname] = value
            fobj.close()
        else:
            print('File', savefile, 'not found.')
            exit()

        return rdata


if __name__ == '__main__':

    cruncher = data_cruncher()

    if 'gather' in sys.argv:
        cruncher.gather_folders('2016')
        cruncher.gather_folders('2015')

        cruncher.digest_rawdata()


    elif 'prepare' in sys.argv:
        cruncher.load_index()
        cruncher.prepare_data()


    else:
        cruncher.load_index()

        embed()


