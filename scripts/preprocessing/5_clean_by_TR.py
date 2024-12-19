# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 19, 2024
# Description: This script performs steps (3) and (4) described in Murphy et al., 2014 (Hum. Brain Mapp.)
#              Essentially, it removes noisy samples while performing downsampling to align pupil data to brain data

# From Murphy et al., (2014): "(3) Data were segmented into epochs from 0 to +2 s relative to the acquisition onset of each fMRI volume. 
#                              Within each epoch, amplitude (any sample < 1.5 mm) and variability (any sample ± 3 s.d. outside the epoch mean) thresholds 
#                              were applied to identify artefactual samples which survived Step 2. 
#                              An average pupil diameter measure was then calculated for the corresponding volume by taking the mean across 
#                              the remaining non-artifactual samples in that epoch. 
#                              This step is equivalent to time-locking the continuous pupil data to the onset of fMRI data acquisition
#                              and downsampling to the temporal resolution of the EPI sequence (0.5 Hz) using only clean data samples. 
#                              (4) Mean pupil diameter for any epoch characterized by >40% artifactual samples was replaced 
#                               via linear interpolation across adjacent clean epochs."

# Steps:
# 1. Load downsampled (50 Hz; i.e., sampled every 20 ms) pupil data
# 2. The TR for the brain data is 1 second. To align the pupil data to TRs, segment the data into 1-second epochs
#    (i.e., 1 epoch = 50 samples)
# 3. Identify the artifactual samples within each epoch by removing samples that are ± 3 s.d. outside the epoch mean
# 4. Calculate the mean pupil diameter for each epoch from the remaining non-artifactual samples
# 5. If an epoch is characterized by >40% artifactual samples, 
#    replace the mean pupil diameter for that epoch via linear interpolation across adjacent clean epochs

import numpy as np
import pandas as pd
import os
import math
import importlib
import scipy.stats as stats

# ------------------ Define functions ------------------ # 
def compute_epoch_noise(arr, mean, interval, prop):
    '''
    Determines if segment of data is considered noisy, which is when 40% or more of the samples
    are +/- 1 SD from the epoch mean.

    Inputs:
        - arr: (np.ndarray) containing samples from duration of 1 sec
        - mean: (float) mean of epoch set
        - interval: (float/int) how many SDs away from mean we want to measure
        - prop: (float) percent of data that is the noise limit 

    Outputs:
        - float/int/np.nan, the mean if not noisy, otherwise NaN
    '''

    sd = np.std(arr)

    upper_lim = mean + sd * interval
    lower_lim = mean - sd * interval

    count = 0

    for m in arr:
        if (m > upper_lim) or m < lower_lim:
            count +=1

    if count / len(arr) > prop:
        #print("this set is noisy")
        result = 0
    else:
        result = mean
    
    return result

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/4_downsampled'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_timelocked'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

f_sample = 50  # sampling rate (downsampled to)
module_name = '3_interpolate_blinks'
module = importlib.import_module(module_name)
zero_runs = getattr(module, 'zero_runs')
interpolate_blinks = getattr(module, 'interpolate_blinks')

SUBJ_IDS = range(1002, 1005)

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
    # Load clean pupil data
    file_path = os.path.join(DAT_PATH, str(sub) + "_downsampled_ET.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)
    
    

    n = len(pupilSize)
    epoch_set = np.array([])
    data_by_TR = np.array([])

    for idx, val in enumerate(pupilSize):
        epoch_set = np.append(val, epoch_set)

        if (idx + 1) % f_sample == 0 or idx == n-1:

            epoch_mean = np.average(epoch_set)
            output = compute_epoch_noise(epoch_set, epoch_mean, 1, 0.5)

            data_by_TR = np.append(output, data_by_TR)
            epoch_set = np.array([])
    

    # start interpolation
    # get array containing index values of ranges to interpolate over
    
    ranges = zero_runs(data_by_TR)

    for val in ranges:

        # get first consec. zeros
        start, end = (val[0], val[-1])
        i1, i2 = (start-1, end+1)
        data_TR_new = interpolate_blinks(i1, i2, data_by_TR)
        data_by_TR = data_TR_new


    filename = os.path.join(save_path, str(sub) + "_final_interp_ET.mat")
    sio.savemat(filename, {'pupilFinal': data_by_TR})
    


