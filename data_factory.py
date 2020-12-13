import sys

import numpy as np
import tensorflow.math as tf_math
import math

# save to mongoDB
import time
import pickle
from pymongo import MongoClient

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from tensorflow.python.framework import ops
from tensorflow.python.data.ops import iterator_ops
import tensorflow as tf

class Timeseries_Factory():
    '''
    Class to create function-definedd data
    '''

    def __init__(self):
        self.signal = 'hi'

    def time_signal(self, n_points, x_1=0, x_n=1):

        # create time element, for sinusoids
        t = np.linspace(x_1,x_n,n_points)

        return t

    def sinusoids_numpy(self, t, c1):
        # create sinusoids
        # c1: scales angular speed
        # scaled for f=10
        sig = list(np.sin(10 * 2 * c1 * np.pi * t))

        return sig

    def sinusoids(self, t, c1, c2=1):
        sig =tf_math.scalar_mul( c2 , tf_math.sin(2 * math.pi * 10 * c1 * t) )

        return sig

    def plot(self, t, sig):
        pg.plot(t, sig, pen=None, symbol='o', title='first graph')
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def save_to_mongo(self, data_to_save):
        client = MongoClient(port=27017)
        db = client.timeseries
        db_collection = db['const_amp']

        Data_pkl = pickle.dumps(data_to_save)
        db_collection.insert({'model': Data_pkl, 'created': time.time()})



if __name__ == '__main__':
    '''
    ========================================================
    Timeseries Factory
    ========================================================
    '''
    n_points = 100

    Data = Timeseries_Factory()

    t = Data.time_signal(n_points)
    sig = Data.sinusoids(t, 0.5, 1)
    Data.save_to_mongo(sig)

    Data.plot(t,sig)

print('end')