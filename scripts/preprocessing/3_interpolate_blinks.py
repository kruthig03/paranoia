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

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()

# Standard score cutoffs
SDSCORE = 2

DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts', str(SDSCORE) + "SD"))
MAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_mat'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/3_interpolated'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

SUBJ_IDS = range(1002, 1037)

WINSIZE = 1000 ## cap for ms needed for interpolation
SAMPLE_RATE = int(500) # Sampling frequency/rate(Hz)

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
    
    # 1 point after the end of blink (blink ends at eBlink_idx + 1)
    eBlink_plus1 = eBlink_idx + 2
    
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
        pupilSize[sBlink_idx:eBlink_idx+1] = afterInterpolate
        
    return pupilSize


def id_zeros(arr):
    """
    Identify segments in the data where there are zeros. The function takes in array and outputs new array
    where each row contains the first and last index of the consecutive zeros present in the original array.
    
    Params:
    - arr (numpy array): blink-removed pupil size data
    
    Returns:
    - ranges (numpy array): indices of consecutive zeros
    
    """
    
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    result = np.where(absdiff == 1)[0].reshape(-1, 2)
    
    return result

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
    # Load aligned and validated data
    file_path = os.path.join(DAT_PATH, str(sub) + "_aligned_" + str(SDSCORE) + "SD.csv")
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
                  # To prevent out of bounds error
                  if np.where(time_of_sample == blink_time)[0].size > 0 and 
                  np.where(time_of_sample == blink_time)[0].size < len(pupilSize)] 
    
    eBlink_idx = [np.where(time_of_sample == blink_time)[0][0] for blink_time in eBlink_time
                  # To prevent out of bounds error
                  if np.where(time_of_sample == blink_time)[0].size > 0 and
                  np.where(time_of_sample == blink_time)[0].size < len(pupilSize)]
    
    # How many blinks?
    nBlinks = len(sBlink_idx)
    
    # If the sample ends with a blink
    if len(sBlink_idx) > len(eBlink_idx):
        eBlink_idx.append(len(pupilSize))
    # If the sample starts with a blink
    elif len(sBlink_idx) < len(eBlink_idx):
        sBlink_idx.insert(0, 0)  
    # If the sample starts AND ends with a blink
    elif sBlink_idx[0] > eBlink_idx[0] and len(sBlink_idx) == len(eBlink_idx):
        sBlink_idx.insert(0, 0)
        eBlink_idx.append(len(pupilSize))
    
    # Interpolate over blinks
    pupilSize_blinks_removed = pupilSize.copy()
    
    for i in range(nBlinks):
        pupilSize_blinks_removed = interpolate_blinks(sBlink_idx[i], eBlink_idx[i], pupilSize_blinks_removed)
            
    # Add interpolated pupil data to the dataframe
    dat['pupilSize_noBlinks'] = pupilSize_blinks_removed
    
    
    # ================================================================
    # Step 2. Interpolate over missing data points shorter than 1 sec
    # ================================================================
    
    # Identify consecutive zeros (i.e., missing data) in blink-removed pupil data
    zeros_idx = id_zeros(pupilSize_blinks_removed)
    
    pupilSize_clean = pupilSize_blinks_removed.copy()
    for val in zeros_idx:
            
            # Idx of zero start and end
            start, end = (val[0], val[-1])
            
            # Are the consequtive zeros less than 1 sec?
            if (end - start) <= SAMPLE_RATE:
                
                pupilSize_clean = interpolate_blinks(start, end, pupilSize_clean)
    
    dat['pupilSize_clean'] = pupilSize_clean
    
    # Calculate the percentage of data with missing data longer than 1 seconds
    prop_missing_data = (np.sum(pupilSize_clean == 0)) / len(pupilSize_clean) * 100
    print("Subject", sub, "has", prop_missing_data, "% missing data points")
    
    
    # Save clean pupil data
    filename = os.path.join(SAVE_PATH, str(sub) + "_" + str(SDSCORE) + "SD_interpolated.csv")
    dat.to_csv(filename, index=False)