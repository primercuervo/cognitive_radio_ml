#!/usr/bin/env python
# Constant parameters given by the DySpan PU setup
DELAY_1 = 0.005 # tau1
DELAY_2 = 0.01  # tau2
TCONST = 0.002
MEAN1 = 0.02    # lambda1
MEAN2 = 0.01    # lambda2
MEAN3 = 0.005   # lambda3
N_CHAN = 4      # Number of channels
N_SCN = 10      # Number of scenarios
N_SAMPS = 4000  # Number of samples in the dataset per scenario
NUM_SLICES = 4  # Number of different dataset sizes
CLASS_NAMES = ["Scenario_{}".format(i) for i in range(10)]
FEATURE_NAMES = ["IF delay ch 1",
                 "IF delay ch 2",
                 "IF delay ch 3",
                 "IF delay ch 4",
                 "Packet Rate",
                 "IF variance"]

# List of Measured SNR
SNR_MEAS = ['-5', '-2_5', '0', '2_5', '5', '10', '15'] # dB
PRE_PATH = '../../data/feature_extraction/'
