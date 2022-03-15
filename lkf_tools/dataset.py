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
    def __init__(self,netcdf_file,output_path='./',max_kernel=5,min_kernel=1,
                 dog_thres=0.01,dis_thres=4,ellp_fac=2,angle_thres=45,
                 eps_thres=1.25,lmin=3,latlon=True,return_eps=True,red_fac=1,t_red=3):
        """
        Processes deformation and drift dataset to LKF data set

        netcdf_file: expected variables U,V,A in shape (time,x,y)
        """
        # Set output path
        self.lkfpath = Path(output_path).joinpath(netcdf_file.split('.')[0])
        for lkfpathseg in str(self.lkfpath.absolute()).split('/'):
            if not os.path.exists(self.lkfpath):
                os.mkdir(self.lkfpath)
                
        # Store detection parameters
        self.max_kernel = max_kernel
        self.min_kernel = min_kernel
        self.dog_thres = dog_thres
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
        self.netcdf_file = netcdf_file
        self.data = xr.open_dataset(self.netcdf_file)

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
            print("ERROR: DXU and DYU are missing in netcdf file!")
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
            self.indexes = np.arange(time.size/self.t_red)
        else:
            self.indexes = indexes
        
        for it in [j for j in self.indexes if j+1 not in self.ind_detect]:
            
            print("Compute deformation rates and detect features for day %i" %(it+1))
        
            self.eps_tot_list = []

            for itr in range(self.t_red):
                # Read in velocities
                uice = self.data.U[it+itr,:,:]
                vice = self.data.V[it+itr,:,:]
                aice = self.data.A[it+itr,:,:]
        
                # Check if deformation rates are given
                if hasattr(self.data,'div') and hasattr(self.data,'shr'):
                    div = self.data.div[it+itr,:,:]
                    shr = self.data.shr[it+itr,:,:]
                else:
                    dudx = ((uice[2:,:]-uice[:-2,:])/(self.dxu[:-2,:]+self.dxu[1:-1,:]))[:,1:-1]
                    dvdx = ((vice[2:,:]-vice[:-2,:])/(self.dxu[:-2,:]+self.dxu[1:-1,:]))[:,1:-1]
                    dudy = ((uice[:,2:]-uice[:,:-2])/(self.dyu[:,:-2]+self.dyu[:,1:-1]))[1:-1,:]
                    dvdy = ((vice[:,2:]-vice[:,:-2])/(self.dyu[:,:-2]+self.dyu[:,1:-1]))[1:-1,:]

                    div = (dudx + dvdy) * 3600. *24. # in day^-1
                    shr = np.sqrt((dudx-dvdy)**2 + (dudy + dvdx)**2) * 3600. *24. # in day^-1

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
                                         max_ind=500*self.corfac,use_eps=True)

            # Save the detected features

            if self.latlon:
                lkf = segs2latlon_model(lkf,
                                        self.lon[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                            max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac],
                                        self.lat[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                            max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac])
            if return_eps:
                lkf =  segs2eps(lkf,
                                div[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                      max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac],
                                shr[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                       max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac])

            lkf_T = [j.T for j in lkf]
            np.save(lkfpath + 'lkf_' + datafile.split('/')[-1][:-3] + '_%03i.npy' %(it+1), lkf_T)
            
            
            
            
            
            
            
            
            
            
def segs2latlon_model(segs,lon,lat):
    """ Function that converts index format of detected LKFs to
    lat,lon coordinates
    """
    segsf = []
    for iseg in segs:
        segsf.append(np.concatenate([iseg,
                                     np.stack([lon[iseg[0],iseg[1]],
                                               lat[iseg[0],iseg[1]]])],
                                     axis=0))
    return segsf

