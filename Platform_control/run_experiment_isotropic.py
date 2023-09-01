# Import required libraries
from PythonLab.microreactor_v12 import *
import pickle
import numpy as np
import os


try:
    # Try to open the existing param result file
    f = open('current_exp/param.txt')
    n = len(f.readlines()) + 1
    f.close()
except FileNotFoundError:
     # If the file doesn't exist, start from 1
    n = 1

# Load the selected sample from the pickle file
with open('current_exp/next_sample.pkl', 'rb') as content:
    sample = pickle.load(content)

# Round the parameter values to one decimal place
v1 = np.around(sample['gold'],decimals = 1)
v2 = np.around(sample['silver'],decimals = 1)
v3 = np.around(sample['CTAB'],decimals = 1)
v4 = np.around(sample['I-2959'],decimals = 1)
v5 = 0
height = np.around(sample['height'],decimals = 1)
rtime = np.around(sample['time'],decimals = 1)
cycle_time = np.around(sample['os'],decimals = 1)

# Print the parameter values in a formatted manner
print ('#\tHAuCl4\tAgNO3\tCTAB\tI-2959\tHCl\theight\ttime\tcycle_time')
print ('%i\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (n,v1,v2,v3,v4,v5,height,rtime,cycle_time))


# Call the 'running' function with the given parameters and store the returned values
area_ratio, num_peak, d_wl, FWHM, I400, inten_ratio, std = running(n,v1,v2,v3,v4,v5,height,rtime,cycle_time)

# Update the 'sample' dictionary with some of the calculated values
sample['num_peak'] = np.abs(num_peak)
sample['d_wl'] = d_wl
sample['FWHM'] = FWHM
sample['I400'] = I400
sample['inten_ratio'] = np.abs(inten_ratio)

# Save the updated 'sample' dictionary back to the pickle file
with open('current_exp/next_sample.pkl', 'wb') as content:
    pickle.dump(sample, content)
