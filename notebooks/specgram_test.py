#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Generates Spectrograms based on Raw data recorded with GNURadio """

import os

import numpy as np
import scipy as sp
from scipy import signal
from scipy.misc import imsave
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
# Set up constants and data
SCN = 8
SNR = 15


# While using sp.signal.specgram there are two fields that regard the FFT size:
# * nfft:    Length of the FFT used, if a zero padded FFT is desired. If None,
#            the  FFT length is nperseg. Defaults to None.
# * nperseg: Length of each segment. Defaults to None, but if window is str or
#            tuple, is set to 256, and if window is array_like, is set to the
#            length of the window.
# Length of window segments to later avg. avg over 120FFTs = 0.768ms
NFFT = 64
NUM_TRAIN_IMG = 2000
# TODO: change NFFT name?
# 5e5 samples for 50ms time window
COUNT = int(5e5) # TODO do I use this?
DIR_PATH = os.path.join('..', '..', 'data', 'final_pu', 'with_dc')
# plt.gray()
TRAIN_PATH = os.path.join(DIR_PATH, '..', '..', 'pic_set', 'SNR_{}'.format(SNR), 'train')
TEST_PATH = os.path.join(DIR_PATH, '..', '..', 'pic_set', 'SNR_{}'.format(SNR), 'test')
# Count = 1.25 e 9 samples for 2500 pictures back to back
# + 500 samples for overflow avoidance (not expected to use them)

# TODO: I need to check first if the file to be analyzed exists, otherwise
# this is pointless

# Check if the dirs for the images exists. If not, create
## Checking for train dir
print("Checking for directories...")
# if not os.path.exists(TRAIN_PATH):
    # print("Creating directory at ", os.path.realpath(TRAIN_PATH))
    # os.makedirs(TRAIN_PATH)
## Checking for test dir
# if not os.path.exists(TEST_PATH):
    # print("Creating directory at ", os.path.realpath(TEST_PATH))
    # os.makedirs(TEST_PATH)
## Checking for class scenario directory
### Train
TRAIN_SCN_PATH = os.path.join(TRAIN_PATH, 'scn_{}'.format(SCN))
# if not os.path.exists(TRAIN_SCN_PATH):
    # print("Creating directory at ", os.path.realpath(TRAIN_SCN_PATH))
    # os.makedirs(TRAIN_SCN_PATH)
### Test
TEST_SCN_PATH = os.path.join(TEST_PATH, 'scn_{}'.format(SCN))
# if not os.path.exists(TEST_SCN_PATH):
    # print("Creating directory at ", os.path.realpath(TEST_SCN_PATH))
    # os.makedirs(TEST_SCN_PATH)

AF = open(os.path.join(DIR_PATH, 'scn_{}_snr_{}.dat'.format(SCN, SNR)), 'rb')

for j in range(1):
    for i in range(64):
        # af.seek(7700*i) # every sample has 4 bytes THIS ONE IS TOTALLY WRONG!!!
        # af.seek(7700*8*i, 0) # every sample has 4 bytes I DUNNO WHY THIS IS NOT THE SAME AS BELOW
        # From https://stackoverflow.com/questions/39834345/scipy-signal-spectrogram-output-not-as-expected
        # I got that # of segments = 1 + floor( (datalen - NFFT) / (NFFT - overlap))
        # With:
        # * NFFT = 64
        # # segments = 120 (to record around 50ms of data)
        # the datalen required is ~7700 samples
        # seek uses the offset in bytes, so offset = #samples * bytes per sample
        # Remember: here we are using samples of type np.complex64
        AF.seek(7700*8, 1) # every sample has 4 bytes THIS ONE WORKS

        # in fromfile(...) the count includes the datatype, so no need of
        # multiplying samples times bytes per sample
        data = sp.fromfile(AF, dtype=sp.complex64, count=7700)

        # spectrogram(...) returns also the frequency bins and the times:
        # f, t, Sxx = signal.spectrogram(...)
        # but we won't use them
        # f, t, Sxx = signal.spectrogram(data,
                                       # fs=10e6,
                                       # mode='magnitude',
                                       # return_onesided=False,
                                       # nperseg=NFFT,
                                       # window=np.hamming(NFFT),
                                       # detrend=False,
                                       # scaling='density',
                                       # noverlap=0)
                                       # noverlap=NFFT/2)
        f, t, Sxx = signal.spectrogram(data,
                                       fs=10e6,
                                       mode='psd',
                                       return_onesided=False,
                                       nperseg=NFFT,
                                       detrend=False,
                                       noverlap=0)
        # The spectrum will be reversed, so we shift it
        f = sp.fftpack.fftshift(f)
        Sxx = sp.fftpack.fftshift(Sxx, axes=0)
        # Sxx = 20* np.log10(np.abs(Sxx)) # PSD with complex Sxx
        avgd = np.average(Sxx, axis=1)
        if i == 0:
            stacked = np.array(avgd)
        else:
            stacked = np.vstack([stacked, avgd])
    if i < NUM_TRAIN_IMG:
        dani = plt.figure(1)
        Pxx, freqs, bins, im = specgram(data, NFFT=NFFT, Fs=10e6,
                                window=np.hamming(NFFT), noverlap=NFFT-1,
                                Fc=3.195e9)
        sai = plt.figure(2)
        plt.pcolormesh(t, f, Sxx)
        # imsave(os.path.join("what.jpg"), stacked)
        plt.show()
    else:
        imsave(os.path.join(TEST_SCN_PATH, 'image_{}.jpg'.format(j)), stacked)


AF.close()
