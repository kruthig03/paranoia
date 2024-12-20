# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 19, 2024
# Description: The script downsamples the clean (interpolated) pupil data to 50 Hz
# Downsampled via averaging (i.e., taking the mean of every N samples)

import numpy as np
import pandas as pd
import os
import math

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/3_interpolated'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/4_downsampled'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SAMPLE_RATE_HZ = int(500) # Sampling frequency in Hz
SAMPLE_RATE_MS = 1/SAMPLE_RATE_HZ * 1000 # 500 Hz in ms (2 ms)

DOWNSAMPLE_RATE_HZ = int(50) # Downsample to 50 Hz
DOWNSAMPLE_RATE_MS = 1/DOWNSAMPLE_RATE_HZ * 1000 # 50 Hz in ms (20 ms)
    
SUBJ_IDS = range(1002, 1037)

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
     # Load clean pupil data
    file_path = os.path.join(DAT_PATH, str(sub) + "_2SD_interpolated.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)
    
    # Convert time to datetime and set as index
    dat['time_in_ms_corrected'] = pd.to_datetime(dat['time_in_ms_corrected'], unit='ms')
    dat.set_index('time_in_ms_corrected', inplace=True)
    
    # Resample to 50 Hz (20 ms intervals) and aggregate (e.g., take mean of each chunk)
    pupilSize_downsampled = dat['pupilSize_clean'].resample(f"{DOWNSAMPLE_RATE_MS}ms").mean()
    pupilSize_downsampled.reset_index(drop=True, inplace=True)
    
    # Create a new time column
    time_in_ms_downsampled = np.arange(0, len(pupilSize_downsampled) * DOWNSAMPLE_RATE_MS, DOWNSAMPLE_RATE_MS)
    
    # Create new dataframe with downsampled data
    df_downsampled = pd.DataFrame({'time_in_ms': time_in_ms_downsampled, 'pupilSize_clean': pupilSize_downsampled})
    
    # Check if the length is more or less the same across subjects
    print("Subject", sub, "; num samples: ", len(time_in_ms_downsampled))
    
    # Save downsampled data
    filename = os.path.join(SAVE_PATH, str(sub) + "_2SD_downsampled.csv")
    df_downsampled.to_csv(filename, index=False)
