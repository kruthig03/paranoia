# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: January 6, 2025
# Description: This script calculates one-to-average ISC at the event level

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
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_events'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/7_isc'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1002, 1037)

# Number of iterations for permutation test
ITERATIONS = 5000

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

# Create empty dataframe to store data
allSub_df = pd.DataFrame()

for sub in SUBJ_IDS:
    
    # Load time-locked pupil data
    file_path = os.path.join(DAT_PATH, str(sub) + "_events.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)

    pupilSize = np.array(dat['pupilSize'])
    
    # Add data to dataframe
    allSub_df.loc[:, sub] = pupilSize


# ======================================
# Step 2. Calculate one-to-average ISC
# This is done at the event level
# ======================================
isc = []
subj_ids = allSub_df.columns

for i, subid in enumerate(subj_ids):
    
    df = allSub_df
    
    # This subject's pupil size data
    thisSubj = df.iloc[:,i]
    
    # Everyone else's pupil size data
    everyoneElse = df.drop(df.columns[[i]], axis=1)
    
    # Average everyone else's data
    avg = everyoneElse.mean(axis=1)
    
    # Create a temporary df to store thisSubj and avg
    df_temp = pd.DataFrame({'thisSubj': thisSubj, 'avg': avg})
    
    # Correlate this Subject's data with the average of everyone else's
    corr = df_temp.corr(method='pearson').iloc[0,1]
    
    # Save the one-to-average correlation for each subject
    isc = np.append(isc, corr)

# One-to-average ISC values
# isc_df is a pandas dataframe where each row is a subject and the column is the correlation with the rest
isc_df = pd.DataFrame(isc, columns=['ISC'])

# Fisher-z (r-to-z) transform 
isc_loo_z = np.arctanh(isc)

# Find the mean 
true_mean_z = np.nanmean(isc_loo_z)

# Inverse Fisher-z transform (z-to-r); true one-to-average ISC
true_mean_r = np.tanh(true_mean_z) 

print('True mean r value: ', true_mean_r)


# ======================================================================
# Step 3. Find the statistical significance of the ISC using permutation
# ======================================================================
perm_ISC_mean = np.full([ITERATIONS,1], np.nan)

for iteration in range(ITERATIONS):

    if iteration % 100 == 0:
        print('Running iteration', iteration)
    
    # To save "null" ISC values
    isc_perm = []

    for i, subid in enumerate(subj_ids):
        
        df = allSub_df

        # This subject's pupil size
        thisSubj = df.iloc[:,i]
        
        # Everyone else's pupil size
        everyoneElse = df.drop(df.columns[[i]], axis=1)

        # Phase randomize this subject's data
        thisSubj_rand = phase_randomize(thisSubj)

        # Average everyone else's data
        avg = everyoneElse.mean(axis=1)

        # Create a temporary df to store thisSubj_rand and avg
        df_temp = pd.DataFrame({'thisSubj_rand': thisSubj_rand, 'avg': avg})

        # Correlate this subject's phase randomized data with the average of everyone else's
        perm_corr = df_temp.corr(method='pearson').iloc[0,1]
        
        # One-to-average ISC values for each subject with permuted data
        isc_perm = np.append(isc_perm, perm_corr)
        
        # Fisher z-transform
        perm_isc_z = np.arctanh(isc_perm)
        
        # Find the mean
        fake_mean_z = np.nanmean(perm_isc_z)
        
        # Inverse Fisher-z transform
        fake_mean_r = np.tanh(fake_mean_z)
        
        # Store for each iteration
        perm_ISC_mean[iteration] = fake_mean_r
        
# perm_ISC_mean is now the null distribution of r (ISC) values
# Calculate non-parameteric p-value (two tailed)
p_onetail = (1+sum(perm_ISC_mean > true_mean_r)) / (1+ITERATIONS)
p_opposite_tail = (1+sum(perm_ISC_mean < -true_mean_r)) / (1+ITERATIONS)
p_twotail = p_onetail + p_opposite_tail

print(f'ISC: {true_mean_r}, p-value: {p_twotail}')



    