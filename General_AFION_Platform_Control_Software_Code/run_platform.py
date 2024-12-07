#! /usr/bin/env python
# Import required libraries
import time
import numpy as np
import os
import subprocess
import pickle
from gryffin import Gryffin # Importing the Gryffin optimization algorithm
import copy

# Set the target value to be optimized for
TARGET = 880

# Define a function to check known constraints on parameter values
def known_constraints(param):
    ''' constraint that the sum of the parameter values must
    less than 0.8
    '''
    param_0_val = param['param_0']
    param_1_val = param['param_1']
    if param_0_val + param_1_val > 1.2:
        return False
    else:
        return True

# Initialize Gryffin optimizer with configuration from 'config.json'
gryffin = Gryffin('config.json')

# Define the maximum number of iterations for optimization
max_iter = 110

try:
    # Try to open the existing raw observations pickle file
    obs_file = open('current_exp/raw_observations.pkl','rb')
    # Load the raw observations from the pickle file
    raw_observations = pickle.load(obs_file)
     # Close the pickle file
    obs_file.close()
except:
    # If the file doesn't exist or there's an error, initialize raw_observations as an empty list
    raw_observations = []

# If there are observations available
observations = copy.deepcopy(raw_observations)
if (observations != []):
# Iterate through each observation
    for i in range(len(observations)):
        observations[i]['area_ratio'] = np.abs(observations[i]['area_ratio'])
        observations[i].pop('num_peak')
        observations[i].pop('I400')
        observations[i]['d_wl'] = np.abs(observations[i]['d_wl'] - TARGET) / 2
        observations[i]['FWHM'] = np.abs(observations[i]['FWHM'])
        observations[i]['inten_ratio'] = np.abs(observations[i]['inten_ratio'])

# print (observations)
# exit(0)

for num_iter in range(len(observations), max_iter):
    print(f'Iteration {num_iter + 1}')

    # query for new parameters
    samples = gryffin.recommend(observations=observations)
    # print (samples)
    # exit(0)

    # select one strategy (exploration vs exploitation)
    if len(samples) > 1:
        select = num_iter % len(samples)
        sample = samples[select]
    else:
        sample = samples[0]


    # save pickle
    with open('current_exp/next_sample.pkl', 'wb') as content:
        pickle.dump(sample, content)

    # run the experiment, which will update the pickle file
    bg_check = subprocess.run('/mnt/c/Users/Moien/AppData/Local/Programs/Python/Python37/python.exe run_experiment_isotropic.py', shell=True)

    if bg_check.returncode == 3:
        exit(3)

    # read pickle file
    with open('current_exp/next_sample.pkl', 'rb') as content:
        raw_sample = pickle.load(content)
        # print (sample)


    # add result to gryffin sample and observations
    if 'num_peak' in raw_sample:
        raw_observations.append(raw_sample)
        sample = copy.deepcopy(raw_sample)
        sample['area_ratio'] = np.abs(sample['area_ratio'])
        sample.pop('num_peak')
        sample.pop('I400')
        sample['d_wl'] = np.abs(sample['d_wl'] - TARGET) / 2
        sample['FWHM'] = np.abs(sample['FWHM'])
        sample['inten_ratio'] = np.abs(sample['inten_ratio'])
        observations.append(sample)

    # save observation log
    with open('current_exp/raw_observations.pkl', 'wb') as content:
        pickle.dump(raw_observations, content)
