#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.pyplot import specgram
import os
import sys

# Set up constants and data
scn = 0
NFFT = 2**6
# 5e5 samples for 50ms time window
count = int(5e5)
# gain_path = 'gain_0_0'
snr = 5
dir_path = os.path.join('..', '..', 'data', 'final_pu', 'with_dc')
plt.gray()
train_path = os.path.join(dir_path, '..', '..', 'pic_set', 'SNR_{}'.format(snr), 'train')
test_path = os.path.join(dir_path, '..', '..', 'pic_set', 'SNR_{}'.format(snr), 'test')
print("Loading data...")
# Count = 1.25 e 9 samples for 2500 pictures back to back
# + 500 samples for overflow avoidance (not expected to use them)
data = sp.fromfile(os.path.join(dir_path, 'scn_{}_snr_{}'.format(scn, snr)), dtype=sp.complex64, count = (1250000000 + 500) )
print("Loading DONE")

# Check if the dirs for the images exists. If not, create
## Checking for train dir
print("Checking for directories...")
if not os.path.exists(train_path):
    print("Creating directory at ", os.path.realpath(train_path))
    os.makedirs(train_path)
## Checking for test dir
if not os.path.exists(test_path):
    print("Creating directory at ", os.path.realpath(test_path))
    os.makedirs(test_path)
## Checking for class scenario directory
### Train
train_scn_path = os.path.join(train_path, 'scn_{}'.format(scn))
if not os.path.exists(train_scn_path):
    print("Creating directory at ", os.path.realpath(train_scn_path))
    os.makedirs(train_scn_path)
### Test
test_scn_path = os.path.join(test_path, 'scn_{}'.format(scn))
if not os.path.exists(test_scn_path):
    print("Creating directory at ", os.path.realpath(test_scn_path))
    os.makedirs(test_scn_path)

for i in range(10000):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.,])
    ax.set_axis_off()
    fig.add_axes(ax)
    data_plot = [data[i] for i in range(count - 200000, count + 200000)]
    Pxx, freqs, bins, im = specgram(data_plot, NFFT=NFFT, Fs=10e6,
                                    window=np.blackman(NFFT), noverlap=NFFT-1,
                                    Fc=3.195e9)
    count += int(5e5)
    # ax.imshow(im, aspect='normal')
    if i < 8000:
        fig.savefig(os.path.join(train_scn_path, 'image_{}.jpg'.format(i)), dpi = 40)
    else:
        fig.savefig(os.path.join(test_scn_path, 'image_{}.jpg'.format(i)))
    plt.close()
