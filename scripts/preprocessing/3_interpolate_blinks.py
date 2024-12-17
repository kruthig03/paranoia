# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 17, 2024
# Description: The script interpolates over blinks (detected by the Eyelink blink detection algorithm) and missing data points

# Steps:
# 1. Load aligned (and validated) pupil data
# 2. Identify blinks in the data using Eyelink's algorithm
# 3. Interpolate over blinks
# 4. Identify missing data points that are shorter than 1 second
# 5. Interpolate over missing data points

import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math
import mat73 # to load .mat files in MATLAB v7.3

# ------------------ Define functions ------------------ # 
def fetch_mat(mat_path, sub_id):
    """
    Grabs .mat file for a given subject and saves each struct as an array.
    
    Samples (1x1 Struct): contains time, posX, posY, pupilSize, etc.
    Events (1x1 Struct): contains Messages (another Struct), Sblink, Eblink, etc.
        Sblink: time of the start of the blink
        Eblink: time of the start and end of the blink, and blink duration
        Detailed description of the variables: http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    
    """
    mat = mat73.loadmat(os.path.join(mat_path, str(sub_id), str(sub_id) + "_ET.mat"))
    samples = mat['Samples']
    events = mat['Events']
        
    return samples, events


# ------------------ Define functions ------------------ #
def interpolate_blinks(sBlink_idx, eBlink_idx, pupilSize):
    """
    This function performs linear interpolation to estimate pupil size during blinks
    
    Params:
    - sblink (numpy array): index of the start of blink
    - eblink (numpy array): index of the end of blink
    - pupilSize (numpy array): pupil size
        
    Returns:
    - pupilSize (numpy array) : modified pupil size with interpolated values for blinks
    
    """
    
    # 1 point before the start of blink
    sBlink_minus1 = sBlink_idx - 1
    
    # 1 point after the end of blink
    eBlink_plus1 = eBlink_idx + 1
    
    # Two points must be present for interpolations 
    # If the data begins or ends with a blink, you cannot interpolate
    if ((eBlink_plus1 < len(pupilSize)) and (sBlink_minus1 >= 0)):
        
        # Interpolate over these samples
        blink_data = np.array(pupilSize[sBlink_minus1:eBlink_plus1])

        # Pupil size right before and after blink
        toInterp = [blink_data[0], blink_data[-1]]

        # Timepoint to interpolate over
        toInterp_TP = [0, len(blink_data)-1] # x-coordinate of query points
        
        # Perform interpolation
        afterInterpolate = np.interp(range(len(blink_data)), toInterp_TP, toInterp)
        afterInterpolate = afterInterpolate[1:-1] # Remove the point before and after blink
        
        # Put the interpolated data back in
        pupilSize[sBlink_idx:eBlink_idx] = afterInterpolate
        
    return pupilSize


def id_zeros(arr):
    """
    Identify segments in the data where there are <500 consequtive zeros
    
    Params:
    - arr (numpy array): blink-removed pupil size data
    
    Returns:
    - ranges (numpy array): indices of consecutive zeros
    
    """
    
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    
    
    
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

# ------------------ Hardcoded parameters ------------------ #
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts'))
MAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_mat'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/3_interpolated'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

SUBJ_IDS = range(1002, 1005)
#SUBJ_IDS = range(1002, 1029)

# Standard score cutoffs
SDSCORE = 2

WINSIZE = 1000 ## cap for ms needed for interpolation
SAMPLE_RATE = int(500) # Sampling frequency/rate(Hz)

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
    # Load aligned and validated data
    file_path = os.path.join(DAT_PATH, str(sub) + "_aligned_" + str(SDSCORE) + "SD_ET.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)
    
    pupilSize = dat['pupilSize']
    time_of_sample = np.array(dat['time_in_ms'])
    
    # Load .mat file to access Eyelink blink data
    samples, events = fetch_mat(MAT_PATH, sub)
    
    # ======================================================
    # Step 1. Interpolate over blinks, detected by Eyelink
    # ======================================================
    
    # Time of blink start
    sBlink_time = events['Eblink']['start']
    
    # Time of blink end
    eBlink_time = events['Eblink']['end']
    
    # Index of corresponding time in the pupil data
    sBlink_idx = [np.where(time_of_sample == blink_time)[0][0] for blink_time in sBlink_time 
                  if np.where(time_of_sample == blink_time)[0].size > 0]
    
    eBlink_idx = [np.where(time_of_sample == blink_time)[0][0] for blink_time in eBlink_time
                  if np.where(time_of_sample == blink_time)[0].size > 0]
    
    # How many blinks?
    nBlinks = len(sBlink_idx)
    
    # Interpolate over blinks
    for i in range(nBlinks):
        pupilSize_blinks_removed = interpolate_blinks(sBlink_idx[i], eBlink_idx[i], pupilSize)
        
    # Add interpolated pupil data to the dataframe
    dat['pupilSize_noBlinks'] = pupilSize_blinks_removed
    
    
    # ================================================================
    # Step 2. Interpolate over missing data points shorter than 1 sec
    # ================================================================
    
    arr = np.array(pupilSize_blinks_removed)
    
    
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    result = np.where(absdiff == 1)[0].reshape(-1, 2)
    
        
    # Save interpolated data
    # filename = os.path.join(SAVE_PATH, str(sub) + "_interpolated_ET.csv")
    # dat.to_csv(filename, index=False)
    
    
    
   

    
    
    
    """

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

    """




    
    