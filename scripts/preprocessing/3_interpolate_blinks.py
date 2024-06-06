# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: May 28, 2024
# Description: The script interpolates over blinks using both Eyelink's blink detection algorithm and basic interpolation scheme for data loss < 1 sec (Murphy et al. 2014)


import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math

# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/1_aligned')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/timestamps')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/3_interpolated')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1003, 1022)


## ALTER FOLLOWING FUNCTION TO INCLUDE < 1 SEC INTERPOLATION

def interpolate_blinks(sblink_minus1, eblink_plus1, pupilSize):
    """
    This function performs linear interpolation to estimate pupil size during blinks
    
    Params:
        sblink_minus1: index of the sample right before blink
        eblink_plus1: index of the sample right after blink
        pupilSize: pupil size during the entire time course, where blinks are zeros
        
    Returns:
        np.ndarray: modified pupil size with interpolated values for blinks
    
    """
    
    # Two points must be present for interpolations; if the data begins or ends with a blink, you cannot interpolate
    if ((eblink_plus1 < len(pupilSize)) and (sblink_minus1 >= 0)):
        
        # Interpolate over these samples
        blink_data = pupilSize[sblink_minus1:eblink_plus1]

        # Pupil size right before and after blink
        toInterp = [blink_data[0], blink_data[-1]]

        # Timepoint to interpolate over
        toInterp_TP = [0, len(blink_data)-1] # x-coordinate of query points
        
        # Perform interpolation
        afterInterpolate = np.interp(range(len(blink_data)), toInterp_TP, toInterp)
        afterInterpolate = afterInterpolate[1:-1]
        
        # Put the interpolated data back in
        pupilSize[sblink_minus1 + 1:eblink_plus1-1] = afterInterpolate
        
    return pupilSize


def zero_runs(arr):
    """
    Takes in array and outputs new array where each row contains the first and 
    last index of the consecutive zeros present in the original array.
    
    Params:
        arr: (np.ndarray) containing values with consecutive zeros
        
    Returns:
        ranges: (np.ndarray) containing indices of consecutive zeros
    
    """

    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    result = np.where(absdiff == 1)[0].reshape(-1, 2)
    return result


WINSIZE = 1000 ## cap for ms needed for interpolation
f_sample = int(500) # Sampling frequency/rate(Hz)

for sub in subj_ids:

    # get data
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_aligned_ET.mat"))
    
    # Pupil size during the entire timecourse
    pupilSize = mat['pupilEncoding'].flatten()
    time = mat['time'].flatten()
    sample_num = mat['sample_num']
    stim_length = mat['stim_min']

    # get array containing index values of ranges to interpolate over
    ranges = zero_runs(pupilSize)

    for val in ranges:

        # get first consec. zeros
        start, end = (val[0], val[-1])

        if (end - start) <= f_sample:  # make sure that the range is less than or eq. to 1 sec

            i1, i2 = (start-1, end+1)
            pupilSize_new = interpolate_blinks(i1, i2, pupilSize)
            pupilSize = pupilSize_new

    filename = os.path.join(save_path, str(sub) + "_interpolated_ET.mat")
    sio.savemat(filename, {'pupilInterpolated':pupilSize, 'time': time, 'sample_num': sample_num, 'stim_min': stim_length})





    
    