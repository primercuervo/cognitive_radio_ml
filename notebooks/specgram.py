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
scn = 3
NFFT = 2**6
count = 200000
gain_path = 'gain_0_0'
dir_path = os.path.join('..', '..', 'data', 'ota', gain_path)
plt.gray()
train_path = os.path.join(dir_path, 'train')
test_path = os.path.join(dir_path, 'test')
print("Loading data...")
data = sp.fromfile(os.path.join(dir_path, 'scn_{}_PU{}.dat'.format(scn, gain_path)), dtype=sp.complex64, count = 500000000)
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

for i in range(2500):
    fig = plt.figure()
    data_plot = [data[i] for i in range((2 * count) - 80000, (2 * count) + 80000)]
    Pxx, freqs, bins, im = specgram(data_plot, NFFT=NFFT, Fs=10e6,
                                    window=np.blackman(NFFT), noverlap=NFFT-1,
                                    Fc=3.195e9)
    count += 40000
    if i < 2000:
        plt.savefig(os.path.join(train_scn_path, 'image_{}.jpg'.format(i)))
    else:
        plt.savefig(os.path.join(test_scn_path, 'image_{}.jpg'.format(i)))
    plt.close()
