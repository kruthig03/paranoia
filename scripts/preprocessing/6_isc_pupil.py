# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 20, 2024
# Description: This script calculates one-to-average ISC


import os
import glob
import scipy.io as sio
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.utils import check_random_state
from numpy import interp

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_timelocked'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_isc'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1002, 1029)
STORY_LENGTH = 1302 # Length of the story in seconds

# ------------------ Define functions ------------------ # 
def isc_loo(df, thisSub_idx):
    """
    One-to-average ISC

    Parameters:
        df (pd.DataFrame): dataframe of pupilSize by subject
        thisSub_idx (int): index of this subject's data column
    
    Returns:
        corr (np.float): one-to-average ISC for a given subject

    """

    i = thisSub_idx
    thisSubj = df.iloc[:,i]
    everyoneElse = df.drop(df.columns[[i]], axis=1)
    
    # Average everyone else's data
    avg = everyoneElse.mean(axis=1)
    
    # Create a temporary df to store thisSubj and avg
    df_temp = pd.DataFrame({'thisSubj': thisSubj, 'avg': avg})
    
    # Correlate this Subject's data with the average of everyone else's
    corr = df_temp.corr(method='pearson').iloc[0,1]
    
    return corr

def phase_randomize(data, random_state=None):
    """Perform phase randomization on time-series signal (from nltools.stats)

    This procedure preserves the power spectrum/autocorrelation,
    but destroys any nonlinear behavior. Based on the algorithm
    described in:

    Theiler, J., Galdrikian, B., Longtin, A., Eubank, S., & Farmer, J. D. (1991).
    Testing for nonlinearity in time series: the method of surrogate data
    (No. LA-UR-91-3343; CONF-9108181-1). Los Alamos National Lab., NM (United States).

    Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
    Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

    1. Calculate the Fourier transform ftx of the original signal xn.
    2. Generate a vector of random phases in the range[0, 2π]) with
       length L/2,where L is the length of the time series.
    3. As the Fourier transform is symmetrical, to create the new phase
       randomized vector ftr , multiply the first half of ftx (i.e.the half
       corresponding to the positive frequencies) by exp(iφr) to create the
       first half of ftr.The remainder of ftr is then the horizontally flipped
       complex conjugate of the first half.
    4. Finally, the inverse Fourier transform of ftr gives the FT surrogate.

    Args:

        data: (np.array) data (can be 1d or 2d, time by features)
        random_state: (int, None, or np.random.RandomState) Initial random seed (default: None)

    Returns:

        shifted_data: (np.array) phase randomized data
    """
    random_state = check_random_state(random_state)

    data = np.array(data)
    fft_data = fft(data, axis=0)

    if data.shape[0] % 2 == 0:
        pos_freq = np.arange(1, data.shape[0] // 2)
        neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
    else:
        pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
        neg_freq = np.arange(data.shape[0] - 1, (data.shape[0] - 1) // 2, -1)

    if len(data.shape) == 1:
        phase_shifts = random_state.uniform(0, 2 * np.pi, size=(len(pos_freq)))
        fft_data[pos_freq] *= np.exp(1j * phase_shifts)
        fft_data[neg_freq] *= np.exp(-1j * phase_shifts)
    else:
        phase_shifts = random_state.uniform(
            0, 2 * np.pi, size=(len(pos_freq), data.shape[1])
        )
        fft_data[pos_freq, :] *= np.exp(1j * phase_shifts)
        fft_data[neg_freq, :] *= np.exp(-1j * phase_shifts)
        
    return np.real(ifft(fft_data, axis=0))


# ------------------ Main ------------------ #

# ========================================================================
# Step 1. Append everyone's standardized pupil data in a single dataframe
# ========================================================================
allSub = []
for sub in SUBJ_IDS:
    
    # Load time-locked pupil data
    file_path = os.path.join(DAT_PATH, str(sub) + "_timelocked.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)
    
    # Ensure that everyone's data is the same length
    if len(dat) > STORY_LENGTH:
        length_orig = len(dat)
        dat = dat[:STORY_LENGTH] # Drop the last few seconds
        length_new = len(dat)
        print(f"Subject {sub} has length of {length_orig}. Truncated to {length_new}")
    elif len(dat) < STORY_LENGTH:
        length_orig = len(dat)
        dat = dat.append([dat.iloc[-1]] * (STORY_LENGTH - len(dat)), ignore_index=True) # Pad with the last value
        length_new = len(dat)
        print(f"Subject {sub} has length of {length_orig}. Padded to {length_new}")

    # Store everyone's standardized pupil data in a df
    pupilSize = stats.zscore(dat['pupilSize'])
    allSub.append(pupilSize)

allSub_df = pd.DataFrame(allSub).T # Transpose so that each column is a subject, each row is a TR
allSub_df.columns = [str(sub) for sub in SUBJ_IDS if os.path.exists(os.path.join(DAT_PATH, str(sub) + "_timelocked.csv"))] # Rename columns


# ======================================
# Step 2. Calculate one-to-average ISC
# This is done at the event level
# ======================================
isc = []
subj_ids = allSub_df.columns

for i, subid in enumerate(subj_ids):
    
    df = allSub_df
    #i = thisSub_idx
    thisSubj = df.iloc[:,i]
    everyoneElse = df.drop(df.columns[[i]], axis=1)
    
    # Average everyone else's data
    avg = everyoneElse.mean(axis=1)
    
    # Create a temporary df to store thisSubj and avg
    df_temp = pd.DataFrame({'thisSubj': thisSubj, 'avg': avg})
    
    # Correlate this Subject's data with the average of everyone else's
    corr = df_temp.corr(method='pearson').iloc[0,1]
    
    # Save the one-to-average correlation for each subject
    # isc_loo_values[pupilSize_by_sub.columns[i]] = isc_loo(pupilSize_by_sub, i)
    isc = np.append(isc, corr)

# One-to-average ISC
# isc_df = pd.DataFrame([isc_loo_values], index=None)
isc_df = pd.DataFrame(isc, columns=['ISC'])

# Fisher-z transform, average, inverse fisher-z transform
# isc_loo_z = np.arctanh(list(isc_loo_values.values()))
isc_loo_z = np.arctanh(isc)
true_mean_z = np.nanmean(isc_loo_z)
true_mean_r = np.tanh(true_mean_z) # True one-to-average ISC

print('True mean r value: ', true_mean_r)


# Bootstrapping HERE


nIt = 5000
boot_ISC_mean = np.full([nIt,1], np.nan)

pupilSize_by_sub = allSub_df

for iteration in range(nIt):

    if iteration % 100 == 0:
        print('Iteration =', iteration)

    for i, subid in enumerate(subj_ids):

        # This subject's time series data
        thisSubj = pupilSize_by_sub.iloc[:, i]
        
        sub_idx = i
        
        

        # Interpolate all NaNs for phase randomization
        # This also pads edge cases (head/tail NaNs) with first/last occurring value
        nSample = len(thisSubj)
        x = np.arange(0, len(thisSubj), 1) # x-coordinate of query points
        nan_indices = np.isnan(thisSubj) 
        thisSubj_interp = np.interp(x, x[~nan_indices], thisSubj[~nan_indices])

        # Everyone else's time series data
        everyoneElse = pupilSize_by_sub.drop(pupilSize_by_sub.columns[[sub_idx]], axis=1)

        # Phase randomize this subject's (interpolated) data
        thisSubj_rand = phase_randomize(thisSubj_interp)

        # Average everyone else's data
        avg = everyoneElse.mean(axis=1, skipna=True)

        # Create a temporary df to store thisSubj_rand and avg
        df_temp = pd.DataFrame({'thisSubj_rand': thisSubj_rand, 'avg': avg})

        # Correlate this subject's phase randomized data with the average of everyone else's
        boot_ISC_loo = df_temp.corr(method='pearson').iloc[0,1]

        boot_ISC_mean[iteration] = np.tanh(np.nanmean(np.arctanh(boot_ISC_loo)))

# Difference between actual and bootstrapped means
boot_ISC_demean = boot_ISC_mean - true_mean_r

# p-value
p_value = np.mean(true_mean_r < boot_ISC_demean) + 1 / nIt
print('P-value: ', p_value)
print(f'ISC: {true_mean_r}, p-value: {p_value}')

isc_final_df = pd.DataFrame(
    {'P-value': p_value,
    'True-Mean-R': true_mean_r},
    index=[0]
)

filename = os.path.join(save_path, str(sub) + "_isc_values.csv")
isc_final_df.to_csv(filename)




    