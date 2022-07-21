# -*- coding: utf-8 -*-

"""
Routines to process deformation and drift data and generate LKF data set
"""


# Package Metadata
__version__ = 0.1
__author__ = "Nils Hutter"
__author_email__ = "nhutter@uw.edu"


import numpy as np
import matplotlib.pylab as plt
import os
import sys
from multiprocessing import Pool
import warnings
from pathlib import Path

import xarray as xr

from .detection import *
from .tracking import *
from .rgps import *



class process_dataset(object):
    """
    Class to process deformation and drift dataset to LKF data set.
    """
    def __init__(self,netcdf_file,output_path='./',xarray=None,
                 max_kernel=5,min_kernel=1, dog_thres=0.01,skeleton_kernel=0,
                 dis_thres=4,ellp_fac=2,angle_thres=45,eps_thres=1.25,lmin=3,
                 latlon=True,return_eps=True,red_fac=1,t_red=3):
        """
        Processes deformation and drift dataset to LKF data set

        netcdf_file: expected variables U,V,A in shape (time,x,y)
        """
        # Set output path
        self.netcdf_file = str(netcdf_file)
        self.lkfpath = Path(output_path).joinpath(self.netcdf_file.split('/')[-1].split('.')[0])
        lkfpath = '/'
        for lkfpathseg in str(self.lkfpath.absolute()).split('/')[1:]:
            lkfpath += lkfpathseg + '/'
            if not os.path.exists(lkfpath):
                os.mkdir(lkfpath)
                
        # Store detection parameters
        self.max_kernel = max_kernel
        self.min_kernel = min_kernel
        self.dog_thres = dog_thres
        self.skeleton_kernel = skeleton_kernel
        self.dis_thres = dis_thres
        self.ellp_fac = ellp_fac
        self.angle_thres = angle_thres
        self.eps_thres = eps_thres
        self.lmin = lmin
        self.latlon = latlon
        self.return_eps = return_eps
        self.red_fac = red_fac
        self.t_red = t_red
        

        # Read netcdf file
        if xarray is None:
            self.data = xr.open_dataset(self.netcdf_file)
        else:
            self.data = xarray

        # Store variables
        self.time = self.data.time
        self.lon = self.data.ULON
        self.lat = self.data.ULAT

        self.lon = self.lon.where(self.lon<=1e30); self.lat = self.lat.where(self.lat<=1e30);
        self.lon = self.lon.where(self.lon<180,other=self.lon-360)

        if hasattr(self.data,'DXU') and hasattr(self.data,'DYV'):
            self.dxu  = self.data.DXU
            self.dyu  = self.data.DYV
        else:
            print("Warning: DXU and DYU are missing in netcdf file!")
            print("  -->  Compute dxu and dyu from lon,lat using SSMI projection")
            m = mSSMI()
            x,y = m(self.lon,self.lat)
            self.dxu = np.sqrt((x[:,1:]-x[:,:-1])**2 + (y[:,1:]-y[:,:-1])**2)
            self.dxu = np.concatenate([self.dxu,self.dxu[:,-1].reshape((self.dxu.shape[0],1))],axis=1)
            self.dyu = np.sqrt((x[1:,:]-x[:-1,:])**2 + (y[1:,:]-y[:-1,:])**2)
            self.dyu = np.concatenate([self.dyu,self.dyu[-1,:].reshape((1,self.dyu.shape[1]))],axis=0)
        

        # Generate Arctic Basin mask
        self.mask = ((((self.lon > -120) & (self.lon < 100)) & (self.lat >= 80)) |
                ((self.lon <= -120) & (self.lat >= 70)) |
                ((self.lon >= 100) & (self.lat >= 70)))
        self.index_x = np.where(np.sum(self.mask[1:-1,1:-1],axis=0)>0)
        self.index_y = np.where(np.sum(self.mask[1:-1,1:-1],axis=1)>0)


    def detect_lkfs(self,indexes=None,force_redetect=False):
        """
        Detects LKFs in data set given in netcdf file
        
        :param indexes: time indexes that should be detected. If None all time steps are detected
        """
        
        # Check for already dectected features
        if force_redetect:
            self.lkf_filelist = [i for i in os.listdir(self.lkfpath) if i.startswith('lkf') and i.endswith('.npy')]
            self.lkf_filelist.sort()
            self.ind_detect = [int(i.split('.')[0].split('_')[-1]) for i in self.lkf_filelist]
        else:
            self.ind_detect = []
            
        if indexes is None:
            self.indexes = np.arange(self.time.size/self.t_red)
        else:
            self.indexes = indexes
        
        for it in [int(j) for j in self.indexes if j+1 not in self.ind_detect]:
            
            print("Compute deformation rates and detect features for day %i" %(it+1))
        
            self.eps_tot_list = []

            for itr in range(self.t_red):
                # Read in velocities
                uice = self.data.U[it+itr,:,:]
                vice = self.data.V[it+itr,:,:]
                aice = self.data.A[it+itr,:,:]
        
                # Check if deformation rates are given
                if hasattr(self.data,'div') and hasattr(self.data,'shr') and hasattr(self.data,'vor'):
                    div = self.data.div[it+itr,:,:]
                    shr = self.data.shr[it+itr,:,:]
                    vor = self.data.vor[it+itr,:,:]
                else:
                    dudx = ((uice[2:,:]-uice[:-2,:])/(self.dxu[:-2,:]+self.dxu[1:-1,:]))[:,1:-1]
                    dvdx = ((vice[2:,:]-vice[:-2,:])/(self.dxu[:-2,:]+self.dxu[1:-1,:]))[:,1:-1]
                    dudy = ((uice[:,2:]-uice[:,:-2])/(self.dyu[:,:-2]+self.dyu[:,1:-1]))[1:-1,:]
                    dvdy = ((vice[:,2:]-vice[:,:-2])/(self.dyu[:,:-2]+self.dyu[:,1:-1]))[1:-1,:]

                    div = (dudx + dvdy) * 3600. *24. # in day^-1
                    shr = np.sqrt((dudx-dvdy)**2 + (dudy + dvdx)**2) * 3600. *24. # in day^-1
                    vor = 0.5*(dudy-dvdx) * 3600. *24. # in day^-1

                eps_tot = np.sqrt(div**2+shr**2)

                eps_tot = eps_tot.where((aice[1:-1,1:-1]>0) & (aice[1:-1,1:-1]<=1))

                # Mask Arctic basin and shrink array
                eps_tot = eps_tot.where(self.mask[1:-1,1:-1])
                eps_tot = eps_tot[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                  max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]
                eps_tot[0,:] = np.nan; eps_tot[-1,:] = np.nan
                eps_tot[:,0] = np.nan; eps_tot[:,-1] = np.nan
                eps_tot[1,:] = np.nan; eps_tot[-2,:] = np.nan
                eps_tot[:,1] = np.nan; eps_tot[:,-2] = np.nan
                
                self.eps_tot_list.append(np.array(eps_tot))
    

            # Apply detection algorithm
            # Correct detection parameters for different resolution
            self.corfac = 12.5e3/np.mean([np.nanmean(self.dxu),np.nanmean(self.dyu)])/float(self.red_fac)

            # Detect features
            print('Start detection routines')
            lkf = lkf_detect_eps_multday(self.eps_tot_list,max_kernel=self.max_kernel*(1+self.corfac)*0.5,
                                         min_kernel=self.min_kernel*(1+self.corfac)*0.5,
                                         dog_thres=self.dog_thres,dis_thres=self.dis_thres*self.corfac,
                                         ellp_fac=self.ellp_fac,angle_thres=self.angle_thres,
                                         eps_thres=self.eps_thres,lmin=self.lmin*self.corfac,
                                         max_ind=500*self.corfac,use_eps=True,skeleton_kernel=self.skeleton_kernel)

            # Save the detected features

            if self.latlon:
                lkf = segs2latlon_model(lkf,
                                        np.array(self.lon[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                            max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]),
                                        np.array(self.lat[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                            max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]))
            if self.return_eps:
                lkf =  segs2epsvor(lkf,
                                np.array(div[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                      max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]),
                                np.array(shr[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                       max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]),
                                np.array(vor[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                       max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]))

            lkf_T = [j.T for j in lkf]
            np.save(self.lkfpath.joinpath('lkf_%s_%03i.npy' %(self.netcdf_file.split('/')[-1].split('.')[0],(it+1))), lkf_T)
            
            
            
            
    def track_lkfs(self,indexes=None, force_recompute=False):
        """Function that generates tracking data set
        :param indexes: time indexes that should be tracked. If None all time steps are tracked.
        """

        # Set output path
        self.track_output_path = self.lkfpath.joinpath('tracked_pairs')
        if not os.path.exists(self.track_output_path):
            os.mkdir(self.track_output_path)

        self.nx,self.ny = self.mask[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                    max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac].shape

        self.lkf_filelist = [i for i in os.listdir(self.lkfpath) if i.startswith('lkf') and i.endswith('.npy')]
        self.lkf_filelist.sort()

        # Determine which files have been already tracked
        if force_recompute:
            self.tracked_lkfs = []
        else:
            self.tracked_lkfs = [int(i.split('.')[0].split('_')[-1])-1 for i in os.listdir(self.track_output_path) if i.startswith('lkf') and i.endswith('.npy')]
            self.tracked_lkfs.sort()

        if indexes is None:
            self.indexes = np.arange(self.time.size/self.t_red-1)
        else:
            self.indexes = indexes
        
        # Determine advection time step
        self.dt = float(self.time.diff(dim='time')[0]/1e9)
        self.adv_time = float(self.time.diff(dim='time')[0]/1e9)*self.t_red
        
        # Do the tracking
        for ilkf in [int(j) for j in self.indexes if j+1 not in self.tracked_lkfs]:
            print("Track features in %s to %s" %(self.lkf_filelist[ilkf],
                                                 self.lkf_filelist[ilkf+1]))
            
            # Open lkf0 and compute drift estimate
            lkf0_d = drift_estimate(self.lkfpath.joinpath(self.lkf_filelist[ilkf]),self.data,
                                    self.mask,self.index_x,self.index_y,self.red_fac,
                                    self.dxu,self.dyu,adv_time=self.adv_time,t=self.dt,dt=self.dt)

            # Filter zero length LKFs due to NaN drift
            ind_f   = np.where(np.array([iseg.size for iseg in lkf0_d])>0)[0]
            lkf0_df = [iseg for iseg in lkf0_d if iseg.size>0]

            # Read LKFs
            lkf1 = np.load(self.lkfpath.joinpath(self.lkf_filelist[ilkf+1]),allow_pickle=True)
            # lkf1_l = []
            # for ilkf,iseg in enumerate(lkf1):
            #     lkf1_l.append(iseg[:,:2])
            lkf1_l = lkf1
            if len(lkf1_l)==1:
                #lkf1_l = np.array([lkf1.squeeze()],dtype='object')
                lkf1_l = [lkf1.squeeze()]
            for ilkf1,iseg in enumerate(lkf1):
                lkf1_l[ilkf1] = iseg[:,:2]

            # Compute tracking
            tracked_pairs = track_lkf(lkf0_df, lkf1_l, self.nx, self.ny, 
                                      thres_frac=0.75, min_overlap=4,
                                      overlap_thres=1.5,angle_thres=25)

            if len(tracked_pairs)==0:
                tracked_pairs = np.array([[],[]])
            else:
                tracked_pairs = np.stack(tracked_pairs)
                tracked_pairs[:,0] = ind_f[np.stack(tracked_pairs)[:,0]]

            # Save tracked pairs
            np.save(self.track_output_path.joinpath('lkf_tracked_pairs_%s_to_%s' %(self.lkf_filelist[ilkf][4:-4],
                                                                 self.lkf_filelist[ilkf+1][4:-4])),
                    tracked_pairs)