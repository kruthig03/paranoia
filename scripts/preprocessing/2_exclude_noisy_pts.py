# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 17, 2024
# Description: The script calculated group mean pupil dilation and excludes noisy participants
# e.g., a participant is excluded if 25% of the raw samples are more than 3 standard deviations from the mean

import numpy as np
import pandas as pd
import os
import scipy.io as sio
import scipy.stats as stats
import math

# ------------------ Define functions ------------------ #
def calculate_noise(arr1, arr2, p, diff, lower):
    """
    Determines if data in array is noisy based on percent of data allowed to be below lower limit and above upper limit.

    Inputs:
    - arr1 (numpy array) containing the pupil size data
    - arr2 (numpy array) containing the pupil size difference data
    - p (float) specifying threshold for determining if data is noisy
    - upper (float) limit for non-noisy data
    - lower (float) limit for non-noisy data

    Outputs:
    - noisy (bool), True if noisy and False if not

    """

    full_length = len(arr1)
    count = 0

    for idx, item in enumerate(arr1):
        if idx != 0:
            if (item < lower):
                count +=1
            elif arr2[idx] > diff:
                count +=1
        else:
            if (item < lower):
                count +=1

    # proportion of noise in subject
    prop_noise = count/full_length
    
    if prop_noise >= p:
        noisy = True
    else:
        noisy = False
    
    return noisy

# ------------------ Hardcoded parameters ------------------ #
# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/1_aligned')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/timestamps')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/2_excluded_participants')

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Set range of subjects
subj_ids = range(1002, 1030)

# Create list with all pupil data
all_pupil = np.array([])
#all_pupil_z = np.array([])
pupilDiff = np.array([])

# ------------------- Main ------------------ #
for sub in subj_ids:

    # get data files
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_aligned_ET.mat"))

    # Pupil size during the entire timecourse
    pupilSize = mat['pupilEncoding'].flatten()
    time = mat['time'].flatten()

    pupilSize_next = np.copy(pupilSize)
    pupilSize_next = np.delete(pupilSize_next, 0)
    pupilSize = np.delete(pupilSize, -1)
    differences = pupilSize_next - pupilSize
    #differences = np.insert(differences, 0, 0)
    pupilDiff = np.append(differences, pupilDiff)

    #for i, val in enumerate(pupilSize):
        #if i != 0:
            #diff = val - prev_val
            #pupilDiff = np.append(diff, pupilDiff)    
        #else:
            #pupilDiff = np.append(0, pupilDiff) 
        #prev_val = val
    

    # combine all data
    all_pupil = np.append(all_pupil, pupilSize)
    #all_pupil_z = np.append(all_pupil, pupil_z_scored)


print(len(pupilDiff), len(all_pupil))

# Get group statistics
all_avg = np.mean(all_pupil)
all_sd = np.std(all_pupil, ddof=0)

all_diff_mean = np.mean(pupilDiff)
all_diff_sd = np.std(pupilDiff, ddof=0)


# specify boundaries
z_all_data = 1
z_diff = 1

#upper_lim = all_avg + s*all_sd
lower_lim = all_avg - z_all_data*all_sd
diff_thresh = all_diff_mean + z_diff*all_diff_sd

print(lower_lim, diff_thresh)

exclusions = np.array([])
for sub in subj_ids:
    
    # get data files
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_aligned_ET.mat"))

    # Pupil size during the entire timecourse
    pupilSize = mat['pupilEncoding'].flatten()
    time = mat['time'].flatten()

    # calculate percentage of noise
    result = calculate_noise(pupilSize, pupilDiff, 0.25, diff_thresh, lower_lim)

    if result:
        print(str(sub), " data is too noisy, will be excluded")
        exclusions = np.append(str(sub), exclusions)

    #else:
        #pupil_z_scored = stats.zscore(pupilSize)
        # save text file with participants to exclude
        #filename = os.path.join(save_path, str(sub) + "_excluded_participants.mat")
        #sio.savemat(filename, {'pupilZScored':pupil_z_scored, 'time': time, 'sample_num': mat['sample_num'], 'stim_min': mat['stim_min']})

filename = os.path.join(save_path, str(z_all_data) + '_' + str(z_diff) + "_excluded_participants.mat")
sio.savemat(filename, {'excluded_participants': exclusions})







