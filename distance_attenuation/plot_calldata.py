from distance_attenuation import *
from IPython import embed
from numpy import *
import pandas as pd
import sys


###
# plotting
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Missing arguments. Please specify a pkl file.')
        exit()
    pkl_file = sys.argv[1]

    # load data
    data = data_from_file(pkl_file)
    # average data for distance, condition, year and height
    avg_data = average_duplicates(data, ['envelopes'])


    #####
    # sort data
    sorted_data = dict()
    for rowidx, rowdata in avg_data.iterrows():
        catid = (rowdata.condition, 'height:' + str(rowdata.height), 'year:' + str(rowdata.year))

        if catid not in sorted_data.keys():
            sorted_data[catid] = dict()
            sorted_data[catid]['distance'] = []
            sorted_data[catid]['envelopes'] = []

        sorted_data[catid]['distance'].append(rowdata.distance)
        sorted_data[catid]['envelopes'].append(rowdata.envelopes)

    embed()