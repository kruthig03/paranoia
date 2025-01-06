# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 20, 2024
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
import matplotlib.pyplot as plt

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/4_downsampled'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_timelocked'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

TR = int(1000) # TR in ms 
CURRENT_SAMPLE_HZ = int(50) # Currently sampled at 50 Hz
CURRENT_SAMPLE_MS = 1/CURRENT_SAMPLE_HZ * 1000 # 50 Hz in ms (20 ms)

SAMPLES_PER_EPOCH = int(TR / CURRENT_SAMPLE_MS) # Number of samples per epoch (segment)

# Standard score for identifying cutoffs (SDSCORE = 1, 2, 3, ...)
# For example, if SDSCORE = 3, any sample ± 3 s.d. outside the epoch mean are considered artifacts
SDSCORE = 3

# Cutoff for identifying artifactual samples
# For example, if ARTIFACT_THRESHOLD = 0.4, an epoch with >40% artifactual samples is considered noisy
ARTIFACT_THRESHOLD = 0.4 # Range: 0-1

SUBJ_IDS = range(1002, 1037)

# ------------------ Plot settings ------------------ # 
plt.figure(figsize=(12, 3))
THIS_SUB = int(1010) # Manually define which subject you want to view

# ------------------ Define functions ------------------ # 
def calc_clean_mean(arr, z, artifact_threshold):
    """
    Calculate the mean of non-artifactual samples within each epoch.
    
    Inputs:
    - arr (np array): contains the samples for each epoch
    - z (float): specifies the number of standard deviations to consider
    
    Outputs:
    - mean (float): mean pupil size of the non-artifactual samples
        
    """
    
    # Calculate the mean and standard deviation of the epoch
    arr = pupilSize_epoch
    mean = np.mean(arr)
    sd = np.std(arr)
    
    # Identify the artifactual samples within each epoch
    upper_lim = mean + z * sd
    lower_lim = mean - z * sd
    
    # Remove artifactual samples
    arr_clean = arr[(arr > lower_lim) & (arr < upper_lim)]
    
    # Calculate the mean of the non-artifactual samples
    mean_clean = np.mean(arr_clean)
    
    # If the epoch is characterized by >40% artifactual samples, replace the mean pupil diameter with 0 for that epoch
    if len(arr_clean) / len(arr) < 1-artifact_threshold:
        mean_clean = 0
    
    return mean_clean

def tolerant_mean(arrs):
    """
    Calculate the mean of arrays with different lengths
    
    Input:
    - arrs (list of np arrays): contains the epoch samples from each subject
    
    Output:
    - y (np array): mean of the arrays
    - sem (np array): standard error of the mean of the arrays
    
    """
    
    # Get the length of each array (i.e., length of each subject's pupil size)
    lens = [len(i) for i in arrs]
    
    # Create a masked array (max_length, number of arrays)
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    
    # Fill the masked array with data
    # Shorter arrays are left empty
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    
    # Calculate standard error
    sem = arr.std(axis=-1) / np.sqrt(len(arrs))
    
    return arr.mean(axis=-1), sem

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

# ------------------- Main ------------------ #
# Create empty dictionary to store everyone's pupil data
pupil_allSub = {}

for sub in SUBJ_IDS:
    
    # Load clean pupil data
    file_path = os.path.join(DAT_PATH, str(sub) + "_2SD_downsampled.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)
    
    pupilSize = np.array(dat['pupilSize_clean'])
    
    # Create empty array to store time-locked pupil data
    pupilTimeLocked = np.array([])
    
    # Segment the data into 1-second epochs (segments)
    for i in range(0, len(pupilSize), SAMPLES_PER_EPOCH):
        pupilSize_epoch = pupilSize[i:i+SAMPLES_PER_EPOCH]
        
        # Identify the artifactual samples within each epoch
        # Calculate the mean pupil diameter for each epoch from the remaining non-artifactual samples
        epoch_clean = calc_clean_mean(pupilSize_epoch, SDSCORE, ARTIFACT_THRESHOLD)
        
        # Append the mean pupil diameter to the time-locked array
        pupilTimeLocked = np.append(pupilTimeLocked, epoch_clean)
            
        
    # Create a TR column
    TR = np.arange(1, len(pupilTimeLocked)+1)
    
    # If there are epochs with a zero (i.e., epochs with >40% artifactual samples), 
    # replace the mean pupil diameter for that epoch via linear interpolation across adjacent clean epochs  
    print("Subject", sub, ";", np.any(pupilTimeLocked == 0))
    skip_interpolation = False
    
    if np.any(pupilTimeLocked == 0) == True:
        
        # Get the index of the zero epochs
        zero_idx = np.where(pupilTimeLocked == 0)[0]
        
        # If the data begins or ends with a zero epoch, you cannot interpolate
        if zero_idx==0 or np.any(zero_idx==len(pupilTimeLocked)-1):
            skip_interpolation = True
        else:
            for idx in zero_idx:
                # Get the start and end of the zero epoch
                start = idx
                end = idx + 1
                
                # Interpolate over the zero epoch
                pupilTimeLocked = interpolate_blinks(start, end, pupilTimeLocked)
                
    # Save data for each subject
    df = pd.DataFrame({'TR': TR, 'pupilSize': pupilTimeLocked})
    df.to_csv(os.path.join(SAVE_PATH, str(sub) + "_timelocked.csv"), index=False)



    # ==========
    # Plotting
    # ==========
    # Standardize pupil data
    pupil_z = stats.zscore(pupilTimeLocked)
    
    # Save everyone's data in a dictionary
    pupil_allSub[sub] = {'TR': TR, 'pupilSize': pupil_z}
  
    # Plot the time-locked pupil data
    if sub == THIS_SUB:
         plt.plot(pupil_allSub[sub]['TR'], pupil_allSub[sub]['pupilSize'], color='red', linewidth=1)
    else: 
        plt.plot(pupil_allSub[sub]['TR'], pupil_allSub[sub]['pupilSize'], color='lightgray', linewidth=0.5)
    
# Calculate averate across all subjects
subs = list(pupil_allSub.keys())
allSub_data_list = [data['pupilSize'] for data in pupil_allSub.values()]
pupil_mean, sem = tolerant_mean(allSub_data_list)
    
# Plot average data
plt.plot(np.arange(len(pupil_mean)), pupil_mean, color='blue', linewidth=2)

# Add labels and title
plt.xlabel('Time (TR)')
plt.ylabel('Pupil Size')
plt.title('Pupil Size Time Course Across Subjects')

# Display the plot
plt.show()



