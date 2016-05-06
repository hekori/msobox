import os
import numpy
import vplan
import json

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rc('xtick', labelsize=6)
matplotlib.rc('ytick', labelsize=6)
matplotlib.rc('font', size=10)
matplotlib.rc('font', size=10)
matplotlib.rc('text', usetex=True)


class OfflinePlotData(object):
    """
    Store data for offline plotting 
    """


    def __init__(self, plot_data):
        self.plot_data = plot_data
        self.reset()

    def reset(self):
        self.t = []
        self.x = []
        self.u = []
        self.p = []

    def append_data(self):
        """

        takes as input an object of the class PlotData and stores
        the data in a list


        I.e., append t, x, u, p to a list where

        plot_data.t.shape = (M, NIS)
        plot_data.x.shape = (M, NIS, NX)
        plot_data.u.shape = (M, NIS, NU)
        plot_data.p.shape = (M, NP)

        """

        self.t.append(self.plot_data.t.tolist())
        self.x.append(self.plot_data.x.tolist())
        self.u.append(self.plot_data.u.tolist())
        self.p.append(self.plot_data.p.tolist())


    def save_data(self, f_path):
        """
        saves data to a txt file

        f_path:  e.g. './mydata.txt'

        """


        dat = {}
        dat['t'] = self.t
        dat['x'] = self.x
        dat['u'] = self.u
        dat['p'] = self.p

        # save data container content to file
        fname = os.path.basename(f_path)
        dname = os.path.dirname(f_path)

        if not '.' in fname:
            raise ValueError('expected *.txt  in file path')

        # create dirs if they don't exist
        if not os.path.isdir(dname):
            os.makedirs(dname)
        path = os.path.join(dname)
        path = os.path.join(path, fname)

        print('MHE writes data into file:\n')
        print('{}\n'.format(path))

        with open(path, 'w') as outf:
            json_str = json.dumps(dat, sort_keys=True, indent=4,
                                  separators=(',', ': '))
            outf.write(json_str)

