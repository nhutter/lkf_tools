""" Script interpolate ice drift velocities to the
regular RGPS grid
"""

import numpy as np
import read_RGPS_lagrangian as rrl
import calendar
from griddata_fast import griddata_fast
import matplotlib.pylab as plt
import os
import sys




def interp_RGPS_drift(RGPS_season,RGPS_lag_path,RGPS_eul_path,output_drift_path):
    # -------------------- Initialise integration -----------------------------

    # Set-up RGPS grid

    xg0 = -2300.
    xg1 = 1000.
    yg0 = -1000.
    yg1 = 2100.
    
    nx = 264 # number of cells in x direction
    ny = 248 # number of cells in x direction
    
    xg = np.linspace(xg0,xg1,nx+1)
    xg = 0.5*(xg[1:]+xg[:-1])
    yg = np.linspace(yg0,yg1,ny+1)
    yg = 0.5*(yg[1:]+yg[:-1])
    xg,yg = np.meshgrid(xg,yg)
    
    
    # Initialise start year, day and time intervals
    
    time_int = 3  # in days

    RGPS_dir_lag = RGPS_lag_path + RGPS_season
    RGPS_dir_eul = RGPS_eul_path + RGPS_season

    new_dir = output_drift_path + RGPS_season + '/'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    output_path = new_dir
        

    # --------------------- Load Lagrangian motion data -----------------------

    icemotion_org = rrl.get_icemotion_RGPS_season(RGPS_dir_lag)

    filelist = [i for i in os.listdir(RGPS_dir_eul) if i.endswith('.DIV')]
    
    filelist.sort()
    
    start_year_season = int(filelist[0][:4])
    
    for ifile in filelist:
        start_year = int(ifile[:4])
        start_day = int(ifile[4:7])
        end_year = int(ifile[8:12])
        end_day = int(ifile[12:15])
        
        print 'Interpolating drift for %i%03i to %i%03i' %(start_year,start_day,end_year,end_day)
    
        # Loop parameters
        iyear = start_year
        iday = start_day

        icemotion = icemotion_org.copy()

        # Convert year-day-format to continuing start-year-day format
        if calendar.isleap(start_year_season):
            icemotion[:,:,1] += (icemotion[:,:,0]-start_year_season)*366
            iday += (iyear-start_year_season)*366
        else:
            icemotion[:,:,1] += (icemotion[:,:,0]-start_year_season)*365
            iday += (iyear-start_year_season)*365

        # Find buoys with valid data before and after the time interval
        index = np.any((icemotion[:,:,1] < iday),axis=1) & np.any((icemotion[:,:,1] > iday+time_int),axis=1)
        num_val_buoy = np.sum(index)

        # Find position of buoys at start of interval
        ## Find last position of buoys before the start of interval
        icemotion_copy = icemotion[index,:,:].copy()
        icemotion_copy[icemotion_copy[:,:,1]>iday,1] = np.nan
        pos_before_start = icemotion_copy[[np.arange(num_val_buoy),np.nanargmax((icemotion_copy[:,:,1]-iday),axis=1)]][:,1:4]
    
        ## Find first position of buoys after the start of interval
        icemotion_copy = icemotion[index,:,:].copy()
        icemotion_copy[icemotion_copy[:,:,1]<=iday,1] = np.nan
        pos_after_start = icemotion_copy[[np.arange(num_val_buoy),np.nanargmin((icemotion_copy[:,:,1]-iday),axis=1)]][:,1:4]
        
        
        # Find position of buoys at end of interval
        ## Find last position of buoys before the end of interval
        icemotion_copy = icemotion[index,:,:].copy()
        icemotion_copy[icemotion_copy[:,:,1]>iday+time_int,1] = np.nan
        pos_before_end = icemotion_copy[[np.arange(num_val_buoy),np.nanargmax((icemotion_copy[:,:,1]-iday-time_int),axis=1)]][:,1:4]
        
        ## Find first position of buoys after the end of interval
        icemotion_copy = icemotion[index,:,:].copy()
        icemotion_copy[icemotion_copy[:,:,1]<=iday+time_int,1] = np.nan
        pos_after_end = icemotion_copy[[np.arange(num_val_buoy),np.nanargmin((icemotion_copy[:,:,1]-iday-time_int),axis=1)]][:,1:4]
        
        
        # Determine mean drift during for time interval and mean position
        pos_start = np.nansum([pos_before_start[:,1:].T,(pos_after_start[:,1:].T-pos_before_start[:,1:].T)/(pos_after_start[:,0]-pos_before_start[:,0])*(iday-pos_before_start[:,0])],axis=0).T
        
        pos_end = np.nansum([pos_before_end[:,1:].T,(pos_after_end[:,1:].T-pos_before_end[:,1:].T)/(pos_after_end[:,0]-pos_before_end[:,0])*(iday+time_int-pos_before_end[:,0])],axis=0).T
        
        drift = (pos_end-pos_start)*1e3/time_int/3600./24.
        
        pos_drift = 0.5 * (pos_start+pos_end)


        # Interpolation on the regular grid
        
        interp = griddata_fast(pos_drift[:,0],pos_drift[:,1],xg,yg)
        dis_thres = 4*12.5 # Filter threshold for filtering too large vertices
        interp.minimum_distance(dis_thres)
        
        drift_int = np.rollaxis(np.array([interp.interpolate(drift[:,0]),interp.interpolate(drift[:,1])]),0,3)
        
        
        # Save interpolated drift
        np.save(output_path + 'drift_int_%i%03i_%i%03i' %(start_year,start_day,end_year,end_day), drift_int)

    

