# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: August 1, 2024
# Description: This script takes in event-by-event pupil data for all subjects and averages across all subs per event

import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math
import importlib

# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/7_avg_by_event')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/8_avg_across_subs')

if not os.path.exists(save_path):
    os.makedirs(save_path)

all_subs = {}

for filename in os.listdir(mat_path):
    
    pupil_data = os.path.join(mat_path, filename)
    mat = sio.loadmat(pupil_data)
    pupilSize = mat['pupilByEvent'].flatten()

    subid = "sub-" + filename[:4]
    all_subs[subid] = pupilSize

df_all_subs = pd.DataFrame(all_subs)
df_all_subs['avg'] = df_all_subs.mean(axis=1)

avg_across_subs = df_all_subs['avg'].values

filename_2 = os.path.join(save_path, "paranoia_across_subs_avg.mat")
sio.savemat(filename_2, {'pupilAcrossSubs': avg_across_subs})






