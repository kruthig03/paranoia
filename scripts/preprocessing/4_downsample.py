# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: May 28, 2024
# Description: The script downsamples the interpolated pupil data to 60 Hz using [](Murphy et al. 2014)


import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math


# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/3_interpolated')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/timestamps')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/4_downsampled')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1002, 1022)


# Define iterative downsampling method

def average_downsample(arr, downsample_factor):
    '''
    Perform downsampling of array by averaging across every n (downsampling_factor) elements

    Inputs:
        - arr: (np.ndarray) of samples to downsample
        - downsample_factor: (float) for every element to average across

    Outputs:
        - averaged_array: (np.ndarray) of downsampled samples
    '''
    downsample_factor = int(downsample_factor)

    # Calculate the number of elements to pad
    pad_size = int((downsample_factor - len(arr) % downsample_factor) % downsample_factor)

    # Pad the array with NaNs
    padded_array = np.pad(arr, (0, pad_size), mode='constant', constant_values=np.nan)

    # Reshape the array to group by every factor
    reshaped_array = padded_array.reshape(-1, downsample_factor)

    # Compute the mean, ignoring NaNs
    averaged_array = np.nanmean(reshaped_array, axis=1)

    return averaged_array


# define freq. parameters
f_sample = 500 
f_cutoff = 50

# iterate over subjects
for sub in subj_ids:

    # fetch data
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_interpolated_ET.mat"))
    pupilSize = mat['pupilInterpolated'].flatten()

    downsample_factor = f_sample / f_cutoff

    downsampled_array = average_downsample(pupilSize, downsample_factor)

    # save data
    filename = os.path.join(save_path, str(sub) + "_downsampled_ET.mat")
    sio.savemat(filename, {'pupilDownsampled': downsampled_array, 'stim_min': mat['stim_min']})



    

