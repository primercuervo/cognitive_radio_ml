#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.pyplot import specgram
import os
import sys

data = sp.fromfile('scn_6_PUgain_0_5.dat', dtype=sp.complex64)
NFFT =2**10
count = 4000
plt.gray()
for i in range(2500):
    fig = plt.figure()
    data_plot = [data[i] for i in range(count - 3500, count + 3500)]
    Pxx, freqs, bins, im = specgram(data_plot, NFFT=NFFT, Fs=10e6,
                                    window=np.blackman(NFFT), noverlap=NFFT-1,
                                    Fc=3.195e9)
    count += 4000
    if i < 2000:
        plt.savefig('pics/train/image_{}.jpg'.format(i))
    else:
        plt.savefig('pics/validation/image_{}.jpg'.format(i))
    plt.close()
