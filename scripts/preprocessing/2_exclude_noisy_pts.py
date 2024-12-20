# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 17, 2024
# Description: The script calculated group mean pupil dilation and excludes noisy participants
# Noise is calculated based on the "derivative" (i.e., change in pupil size relative to the preceding sample; 
#                                                aka,  sample N - sample N-1 pupil size)
# Steps:
# 1. Load aligned pupil data
# 2. Create a distribution of the "derivatives"
# 3. Calculate the cutoff from the distribution (e.g., 1 SD, 2 SD, 3 SD)
# 4. Exclude participants if 25% data points are above/below the cutoff
# i.e., if 25% of the data points have a derivative greater/less than the cutoff, exclude the participant

import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import math

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/1_aligned'))

# Standard score for identifying cutoffs (SDSCORE = 1, 2, 3, ...)
# The higher the SDSCORE, the more stringent the cutoff for noise and more participants will be included
SDSCORE = 3

SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts', str(SDSCORE) + "SD"))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1002, 1037)

# ------------------ Define functions ------------------ #
def calculate_derivative(arr):
    """
    Calculates the derivative of the pupil size data.

    Inputs:
    - arr (numpy array) containing the pupil size data

    Outputs:
    - diff (numpy array) containing the derivative of the pupil size data

    """

    diff = np.diff(arr)
    diff = np.insert(diff, 0, 0)
    
    return diff

def create_dist_find_cutoff(arr, z):
    """
    Creates a distribution of the data and finds the cutoff value based on the distribution's standard deviation.
    
    Inputs:
    - arr (numpy array) containing the pupil size data
    - z (float) specifying the number of standard deviations to consider
    
    Outputs:
    - cutoff (float) specifying the cutoff value for the data

    """
    
    # Calculates median absolute deviation (MAD), which is more robust to outliers
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    cutoff = median + z * (mad * 1.4826) # pseudo SD for a normal distribution
    
    return cutoff

def calc_prop_noisy(pupil_diff, cutoff):
    """
    Calculates what proportion of the data is considered noisy based on the cutoff value.
    
    Inputs:
    - pupil_diff (numpy array) containing the derivative of the pupil size data
    - cutoff (float) specifying the cutoff value for the data
    
    Outputs:
    - prop_noisy (float) specifying the proportion of noisy data points (in percentage)

    """
    
    noisy_pts = np.where(np.abs(pupil_diff > cutoff)) # Indices of the "noisy" data points
    prop_noisy = len(noisy_pts[0])/len(pupil_diff)*100
    
    return prop_noisy

# ------------------ Initialize arrays ------------------ #
pupil_diff_allsub = np.array([]) # To store everyone's derivative data

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
    # Load aligned data
    dat = pd.read_csv(os.path.join(DAT_PATH, str(sub) + "_aligned_ET.csv"))
    pupil_raw = dat['pupilSize']
    
    # Drop invalid samples
    pupil_raw = pupil_raw.dropna()  

    # Calculate the "derivative"
    pupil_diff = calculate_derivative(pupil_raw)
    
    # Concatenate all participants' data to create a distribution
    pupil_diff_allsub = np.append(pupil_diff_allsub, pupil_diff)
    
# Find the cutoff values from the distribution
cutoff = create_dist_find_cutoff(pupil_diff_allsub, SDSCORE)

for sub in SUBJ_IDS:
    
    # Load aligned data
    dat = pd.read_csv(os.path.join(DAT_PATH, str(sub) + "_aligned_ET.csv"))
    pupil_raw = dat['pupilSize'].dropna()
    
    # Calculate the "derivative"
    pupil_diff = calculate_derivative(pupil_raw)
    
    # Calculate the proportion of noisy data points
    prop_noisy = calc_prop_noisy(pupil_diff, cutoff)
   
    if prop_noisy >= 25:
        print(f"Participant {sub} excluded. {prop_noisy:.2f}% of the data are noisy :( ")
        
    # Save non-noisy participants' data
    else:
        dat.to_csv(os.path.join(SAVE_PATH, str(sub) + "_aligned_" + str(SDSCORE) + "SD.csv"), index=False)
        # 2 SD: Excluded pts 1003, 1005, 1016, 1017, 1023
        # 3 SD: Excluded pt 1023







