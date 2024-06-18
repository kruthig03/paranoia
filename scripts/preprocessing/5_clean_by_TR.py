# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: June 12, 2024
# Description: This script segments data into 1 second epochs (1 TR) and averages and removes data +/- 1 SD from segment mean
# If 40% (changed to 50% for now) of the epoch data had to be removed, then interpolates across previous and following segments. (Murphy 2014)

import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math
import importlib

module_name = '3_interpolate_blinks'
module = importlib.import_module(module_name)
zero_runs = getattr(module, 'zero_runs')
interpolate_blinks = getattr(module, 'interpolate_blinks')

# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/4_downsampled')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/timestamps')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/5_last_interp')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1002, 1023)

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


f_sample = 50  # sampling rate (downsampled to)

for sub in subj_ids:
    
    # fetch data
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_downsampled_ET.mat"))
    pupilSize = mat['pupilDownsampled'].flatten()

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
    


