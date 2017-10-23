#!/usr/bin/env python

""" Generate spectrograms from data recordings of raw data """

import constants as c
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
import scipy as sp
from scipy import signal
import os

# May not need this
def read_samples(file_path):
    """ Reads samples from file and stores them in a list """
    data = scipy.fromfile(file_path, dtype=sp.complex64)
    data = data.tolist()
    return data

def plot_spectrum(file_path, bin_size=2**10, sample_rate=10e6, plot_path=None, colormap='jet'):
    """ plot the spectrum as a waterfall (waterrise????) """
    data = sp.fromfile(file_path, count=count)
    # f, t, Sxx = signal.spectrogram(data, sample_rate, nfft=128, nperseg=128)
    f, t, Sxx = signal.spectrogram(data,
                                   fs=10e6,
                                   mode='psd',
                                   return_onesided=False,
                                   nperseg=NFFT,
                                   noverlap=0)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


abs_path = os.path.join(c.PRE_PATH, '..', 'final_pu', 'no_dc', 'scn_2_snr_15.dat')
plot_spectrum(abs_path)

from matplotlib.pyplot import specgram
data = sp.fromfile(abs_path, count=1000000)
specgram(data, NFFT=128, Fs=10e6, noverlap=10)
plt.show()
