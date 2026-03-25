# -*- coding: utf-8 -*-

"""
Routines to process deformation and drift data and generate LKF data set
"""


# Package Metadata
__version__ = 2.0
__author__ = "Nils Hutter"
__author_email__ = "nils.hutter@awi.de"


import numpy as np
import matplotlib.pylab as plt
import os
import sys
from multiprocessing import Pool
import warnings
from pathlib import Path
import pickle
from loguru import logger

import xarray as xr

from .detection import *
from .tracking import *
from .rgps import *

import pickle


class lkf_dataset(object):
    """
    Class to process deformation and drift dataset to LKF data set.
    """
    def __init__(self,netcdf_file,output_path='./',xarray=None,
                 max_kernel=5,min_kernel=1, dog_thres=0.01,skeleton_kernel=0,
                 dis_thres=4,ellp_fac=2,angle_thres=45,eps_thres=1.25,lmin=3,
                 latlon=True,return_eps=True,use_eps=True,red_fac=1,t_red=3):
        """
        Processes deformation and drift dataset to LKF data set

        netcdf_file: expected variables U,V,A in shape (time,x,y)
        """
        # Set output path
        self.netcdf_file = str(netcdf_file)
        self.output_path = output_path
        self.lkfpath = Path(output_path).joinpath(self.netcdf_file.split('/')[-1].split('.')[0])
        self.pickle_path = Path(output_path).joinpath(str(netcdf_file).split('/')[-1].split('.')[0]+'lkf_pickle.pkl')
        
        self._initialized = False
        if self.pickle_path.is_file():
            with open(self.pickle_path, 'rb') as f:
                logger.info('Opening existing lkf_dataset from pickle: %s' %self.pickle_path)
                inst = pickle.load(f)
            if isinstance(inst, lkf_dataset):
                for k in inst.__dict__.keys():
                    setattr(self, k, getattr(inst, k))
                self._initialized = True
            else:
                logger.warning('Existing pickle does not fit to lkf_dataset class! %s' %cls.pickle_path)
        
        if not self._initialized:
            logger.info('Creating new lkf_dataset from input data')
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
            self.use_eps = use_eps
            self.red_fac = red_fac
            self.t_red = t_red

            # Fixed column indices for segment arrays (N_cols, N_points)
            # Layout: [row, col, lon, lat, div, shr, vor, x, y]
            self.indd0 = 4   # divergence
            self.indd1 = 5   # shear
            self.indd2 = 6   # vorticity


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


            # Check for deformation rates in netcdf
            self._has_deformation_rates = (hasattr(self.data,'div') and
                                           hasattr(self.data,'shr') and
                                           hasattr(self.data,'vor'))
            if not self._has_deformation_rates:
                self.data['div'] = self.data['U']*np.nan
                self.data['shr'] = self.data['U']*np.nan
                self.data['vor'] = self.data['U']*np.nan

            # Initialize LKF and tracked LKF fields
            self.lkfs = np.empty((int(np.ceil(self.data.time.size/self.t_red))),dtype='object')
            self.timesteps_detected = np.zeros((int(np.ceil(self.data.time.size/self.t_red)))).astype('bool')
            self.tracked_lkfs = np.empty((int(np.ceil(self.data.time.size/self.t_red))),
                                         dtype='object')
            self.data['lkfs'] = self.data['U']*np.nan
            self.data['tracked_lkfs'] = self.data['U']*np.nan


    def save(self):
        with open(self.pickle_path, 'wb') as output_pkl:
            logger.info('Writing lkf_data to pickle file: %s' %self.pickle_path)
            pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
    
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
                if self._has_deformation_rates:
                    div = self.data.div[it+itr,:,:]
                    shr = self.data.shr[it+itr,:,:]
                    vor = self.data.vor[it+itr,:,:]
                else:
                    dudx = ((uice[2:,:]-uice[:-2,:])/(self.dxu[:-2,:]+self.dxu[1:-1,:]))[:,1:-1]
                    dvdx = ((vice[2:,:]-vice[:-2,:])/(self.dxu[:-2,:]+self.dxu[1:-1,:]))[:,1:-1]
                    dudy = ((uice[:,2:]-uice[:,:-2])/(self.dyu[:,:-2]+self.dyu[:,1:-1]))[1:-1,:]
                    dvdy = ((vice[:,2:]-vice[:,:-2])/(self.dyu[:,:-2]+self.dyu[:,1:-1]))[1:-1,:]

                    div = (dudx + dvdy) * 3600. *24. # in day^-1
                    self.data.div[it+itr,1:-1,1:-1] = div
                    shr = np.sqrt((dudx-dvdy)**2 + (dudy + dvdx)**2) * 3600. *24. # in day^-1
                    self.data.shr[it+itr,1:-1,1:-1] = shr
                    vor = 0.5*(dudy-dvdx) * 3600. *24. # in day^-1
                    self.data.vor[it+itr,1:-1,1:-1] = vor

                # Convert to numpy immediately to avoid xarray .where() overhead
                eps_tot = np.array(np.sqrt(div**2+shr**2))

                # Mask ice concentration and Arctic basin in one numpy pass
                aice_np = np.array(aice[1:-1,1:-1])
                valid = ((aice_np > 0) & (aice_np <= 1) &
                         np.array(self.mask[1:-1,1:-1]).astype(bool))
                eps_tot[~valid] = np.nan

                # Subsample and apply border NaNs
                eps_tot = eps_tot[max([0,self.index_y[0][0]-1]):self.index_y[0][-1]+2:self.red_fac,
                                  max([0,self.index_x[0][0]-1]):self.index_x[0][-1]+2:self.red_fac]
                eps_tot[0,:] = np.nan; eps_tot[-1,:] = np.nan
                eps_tot[:,0] = np.nan; eps_tot[:,-1] = np.nan
                eps_tot[1,:] = np.nan; eps_tot[-2,:] = np.nan
                eps_tot[:,1] = np.nan; eps_tot[:,-2] = np.nan

                self.eps_tot_list.append(eps_tot)
    

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
                                         max_ind=500*self.corfac,use_eps=self.use_eps,skeleton_kernel=self.skeleton_kernel)

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

            if self.latlon:
                if not hasattr(self, 'proj'):
                    from pyproj import Proj as _Proj
                    self.proj = _Proj(proj='stere', lat_0=90, lat_ts=75, lon_0=-45, ellps='WGS84')
                    self.indm0 = 7
                    self.indm1 = 8
                proj_segs = []
                for seg in lkf:
                    x, y = self.proj(seg[2, :], seg[3, :])
                    proj_segs.append(np.concatenate([seg, np.stack([x, y])], axis=0))
                lkf = proj_segs

            lkf_T = np.empty(len(lkf), dtype=object)
            lkf_T[:] = [j.T for j in lkf]
            np.save(self.lkfpath.joinpath('lkf_%s_%03i.npy' %(self.netcdf_file.split('/')[-1].split('.')[0],(it+1))), lkf_T)
            
            # Store lkf in object
            self.lkfs[it+itr] = lkf
            self.timesteps_detected[it] = True
            
            # Store lkf in xarray (bulk write via underlying numpy array)
            lkfs_np = self.data.lkfs.values
            for inum, ilkf in enumerate(lkf):
                rows = ilkf[0, :].astype('int') + self.index_y[0][0]
                cols = ilkf[1, :].astype('int') + self.index_x[0][0]
                lkfs_np[it+itr, rows, cols] = inum
            
            
            
            
    def compute_projected_coords(self, proj=None):
        """Append projected (x, y) coordinates to every LKF segment.

        Reads lon (row 2) and lat (row 3) from each segment, projects them to
        Cartesian metres using *proj*, and appends x and y as two new rows
        (indices 7 and 8).  The method is idempotent: calling it a second time
        with the same projection leaves the arrays unchanged.

        Parameters
        ----------
        proj : pyproj.Proj, optional
            Projection to use.  Defaults to a polar-stereographic projection
            centred on the North Pole (lat_ts=75, lon_0=-45, WGS84) — the same
            projection used by the legacy ``stats.py`` reader.

        Sets
        ----
        self.proj   : the Proj instance used
        self.indm0  : 7  (column index of x in the (N_cols, N_points) array)
        self.indm1  : 8  (column index of y)
        """
        from pyproj import Proj as _Proj
        if proj is None:
            proj = _Proj(proj='stere', lat_0=90, lat_ts=75, lon_0=-45, ellps='WGS84')
        self.proj  = proj
        self.indm0 = 7
        self.indm1 = 8

        for t in range(len(self.lkfs)):
            if self.lkfs[t] is None:
                continue
            new_segs = []
            for seg in self.lkfs[t]:
                # seg shape: (N_cols, N_points); row 2=lon, row 3=lat
                if seg.shape[0] == 9:
                    # Already projected — re-project with current proj
                    seg = seg[:7, :]
                x, y = proj(seg[2, :], seg[3, :])
                new_segs.append(np.concatenate([seg, np.stack([x, y])], axis=0))
            self.lkfs[t] = new_segs

    def compute_lengths(self):
        """Compute arc-length of every LKF segment and cache on ``self.lengths``.

        Calls :meth:`compute_projected_coords` automatically if projected
        coordinates have not yet been appended.

        Returns
        -------
        lkf_lengths
            The :class:`lkf_tools.stats.lkf_lengths` instance (also stored as
            ``self.lengths``).
        """
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_lengths
        self.lengths = lkf_lengths(self)
        return self.lengths

    def compute_density(self, grid_res=50e3):
        """Bin LKF points into a 2-D spatial density grid and cache on ``self.density``.

        Calls :meth:`compute_projected_coords` automatically if projected
        coordinates have not yet been appended.

        Parameters
        ----------
        grid_res : float, optional
            Grid cell size in metres.  Default is 50 km.

        Returns
        -------
        lkf_density
            The :class:`lkf_tools.stats.lkf_density` instance (also stored as
            ``self.density``).
        """
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_density
        self.density = lkf_density(self, res=grid_res)
        return self.density

    def compute_curvature(self):
        """Compute end-to-end distance of every LKF segment and cache on ``self.curvature``.

        Calls :meth:`compute_projected_coords` automatically if needed.

        Returns
        -------
        lkf_curvature
            The :class:`lkf_tools.stats.lkf_curvature` instance (also stored
            as ``self.curvature``).
        """
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_curvature
        self.curvature = lkf_curvature(self)
        return self.curvature

    def compute_deformation(self):
        """Compute mean divergence and shear per segment and cache on ``self.deformation``.

        Calls :meth:`compute_projected_coords` automatically if needed.

        Returns
        -------
        lkf_deformation
            The :class:`lkf_tools.stats.lkf_deformation` instance (also stored
            as ``self.deformation``).
        """
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_deformation
        self.deformation = lkf_deformation(self)
        return self.deformation

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
            
            
    def density(self):
        return (self.data.lkfs[self.timesteps_detected,:,:].notnull().sum(axis=0)/
                self.data.div[self.timesteps_detected,:,:].notnull().sum(axis=0))
            
            
            
class process_dataset(lkf_dataset):
    def __init__(self,*args,**kwargs):
        super(process_dataset,self).__init__(*args,**kwargs)
        
        self._warning_relict_super_class = True

