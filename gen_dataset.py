import numpy as np
import matplotlib.pylab as plt
import os
import sys

# Self-written functions
from lkf_detection import *
from lkf_tracking import *
from interp_RGPS_drift import interp_RGPS_drift


# gen_dataset.py
# 
# Python script to generate the LKF data-set from RGPS data
#
# Requirements: - RGPS data: Lagrangian and Eulerian needs to
#                 be downloaded from Ron Kwok's homepage
#                 https://rkwok.jpl.nasa.gov/radarsat/index.html
#         
#               - RGPS data needs to be unzip. The data needs
#                 to be orgnaized in a seperate directory for
#                 each winter that are named w9798, w9899, ...
#
#               - RGPS_eul_path needs to be set to the path 
#                 eulerian RGPS data
#
#               - RGPS_lag_path needs to be set to the path 
#                 lagrangian RGPS data



# ------------- Helper functions ---------------------------

def lkf_detect_loop(path_RGPS,path_processed,max_kernel=5,min_kernel=1,dog_thres=0,dis_thres=4,ellp_fac=3,angle_thres=35,eps_thres=0.5,lmin=4,latlon=False,return_eps=False):
    
    files = [i for i in os.listdir(path_RGPS) if i.endswith('.DIV')]
    files.sort()
    
    for i in files:
        print i.split('.')[0]
        lkf = lkf_detect_rgps(path_RGPS + i.split('.')[0],
                              max_kernel=max_kernel, min_kernel=min_kernel,
                              dog_thres=dog_thres,dis_thres=dis_thres,ellp_fac=ellp_fac,
                              angle_thres=angle_thres,eps_thres=eps_thres,lmin=lmin,
                              latlon=latlon,return_eps=return_eps)
        lkf_T = [j.T for j in lkf]
        np.save(path_processed + 'lkf_' + i.split('.')[0] + '.npy', lkf_T)



# ------------- Run part -----------------------------------

# Input data
years = ['w0001',  'w0102',  'w0203',  'w0304',  
         'w0405',  'w0506',  'w0607',  'w0708',
         'w9697',  'w9798',  'w9899',  'w9900']


RGPS_eul_path = './RGPS_data/eulerian/'
RGPS_lag_path = './RGPS_data/lagrangian/'
drift_path = './RGPS_drift_interp/'

lkf_path = './lkf_output/'


# Iterate over years

for year_dic in years:
    # ---------- (1. Detection) --------------------------------

    print "Start detection with season: " + year_dic
    new_dir = lkf_path + year_dic + '/'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    
    lkf_detect_loop(RGPS_eul_path+year_dic+'/', new_dir,
                    max_kernel=5,min_kernel=1,
                    dog_thres=15,dis_thres=4,
                    ellp_fac=2,angle_thres=35,
                    eps_thres=1.25,lmin=3,latlon=True,return_eps=True)
    


    # ---------- (2. Interpolate drift) --------------------------------

    print "Start interpolating drift with season: " + year_dic
    interp_RGPS_drift(year_dic,RGPS_lag_path,RGPS_eul_path,drift_path)



    # ---------- (3. Tracking) --------------------------------

    print "Start tracking with season: " + year_dic
    new_dir = lkf_path + year_dic + '/tracked_pairs/'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    gen_tracking_dataset_rgps(lkf_path + year_dic + '/',
                              drift_path + year_dic + '/drift_int_',
                              new_dir)
    

