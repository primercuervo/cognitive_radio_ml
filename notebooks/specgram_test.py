#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
from scipy import signal
from scipy.misc import imsave

# Set up constants and data
scn = 0
NFFT = 64 # Length of window segments to later avg. avg over 120FFTs = 0.768ms
# TODO: change NFFT name?
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

af = open(os.path.join(dir_path, 'scn_{}_snr_{}'.format(scn, snr)), 'rb')

for i in range(64):
    af.seek(7700*i)
    data = sp.fromfile(af, dtype=sp.complex64, count = (7700))
    f, t, Sxx = signal.spectrogram(data, fs=10e6, mode='magnitude', return_onesided=False, nperseg=NFFT, noverlap=0)
    avgd = np.average(Sxx, axis=1)
    if i == 0:
        stacked = np.array(avgd)
    else:
        stacked = np.vstack([stacked, avgd])
af.close()
stacked_scaled = stacked * 255.0 / np.amax(stacked)
print("avgd_scaled shape: ", stacked_scaled.shape)
imsave('what.jpg', stacked_scaled)
