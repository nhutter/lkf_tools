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
import types
from multiprocessing import Pool
import warnings
from pathlib import Path
import pickle
from loguru import logger

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable

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
                 latlon=True,return_eps=True,use_eps=True,red_fac=1,t_red=3,
                 mask=None, label=None):
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
        
        self.label = label

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

            # Apply custom mask if provided (must match grid shape)
            if mask is not None:
                custom = np.asarray(mask)
                if custom.shape != np.asarray(self.mask).shape:
                    warnings.warn(
                        f"Custom mask shape {custom.shape} does not match grid shape "
                        f"{np.asarray(self.mask).shape}; using default Arctic basin mask.")
                else:
                    self.mask = custom

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
            self.timestep_dates = {}


    def save(self):
        with open(self.pickle_path, 'wb') as output_pkl:
            logger.info('Writing lkf_data to pickle file: %s' %self.pickle_path)
            pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)

    # ── precondition guards ────────────────────────────────────────────────────

    def _require_detection(self):
        """Raise RuntimeError if no LKFs have been detected yet."""
        if not self.timesteps_detected.any():
            raise RuntimeError(
                "No LKFs detected. Call detect_lkfs() before computing statistics.")

    def _require_tracking(self):
        """Raise RuntimeError if track_lkfs() has never been called."""
        self._require_detection()
        if not isinstance(self.tracked_lkfs, list):
            raise RuntimeError(
                "No tracking data found. Call track_lkfs() before "
                "computing lifetime or growth-rate statistics.")

    # ── stats cache helpers ───────────────────────────────────────────────────

    def _stats_cache_path(self, name):
        return self.lkfpath / f'stats_{name}.pkl'

    def _load_stats(self, name):
        p = self._stats_cache_path(name)
        if p.exists():
            with open(p, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_stats(self, name, obj):
        with open(self._stats_cache_path(name), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    def detect_lkfs(self,indexes=None,force_redetect=False):
        """
        Detects LKFs in data set given in netcdf file
        
        :param indexes: time indexes that should be detected. If None all time steps are detected
        """
        
        # Check for already detected features; skip them unless force_redetect=True
        if not force_redetect and os.path.exists(self.lkfpath):
            self.lkf_filelist = [i for i in os.listdir(self.lkfpath) if i.startswith('lkf') and i.endswith('.npy')]
            self.lkf_filelist.sort()
            self.ind_detect = [int(i.split('.')[0].split('_')[-1]) for i in self.lkf_filelist]
            # Load already-detected LKFs that are not yet in memory.
            # On disk segments are stored transposed (N_points, N_cols);
            # in memory we keep the non-transposed (N_cols, N_points) layout.
            for fname, file_idx in zip(self.lkf_filelist, self.ind_detect):
                it = file_idx - 1
                if it < len(self.lkfs) and self.lkfs[it] is None:
                    lkf_saved = np.load(self.lkfpath.joinpath(fname),
                                        allow_pickle=True)
                    self.lkfs[it] = [seg.T for seg in lkf_saved]
                    self.timesteps_detected[it] = True
        else:
            self.ind_detect = []

        if indexes is None:
            self.indexes = np.arange(self.time.size/self.t_red)
        else:
            self.indexes = indexes

        pending = [int(j) for j in self.indexes if j+1 not in self.ind_detect]
        for it in _tqdm(pending, desc='Detecting LKFs', unit='step', leave=True):

            self.eps_tot_list = []

            for itr in range(self.t_red):
                # Read in velocities
                uice = self.data.U[it+itr,:,:]
                vice = self.data.V[it+itr,:,:]
                aice = self.data.A[it+itr,:,:]
        
                # Check if deformation rates are given
                if self._has_deformation_rates:
                    div = self.data.div[it+itr,1:-1,1:-1]
                    shr = self.data.shr[it+itr,1:-1,1:-1]
                    vor = self.data.vor[it+itr,1:-1,1:-1]
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
            self.lkfs[it] = lkf
            self.timesteps_detected[it] = True
            if not hasattr(self, 'timestep_dates'):
                self.timestep_dates = {}
            self.timestep_dates[it] = self.time.values[it]

            # Store lkf in xarray (bulk write via underlying numpy array)
            lkfs_np = self.data.lkfs.values
            for inum, ilkf in enumerate(lkf):
                rows = ilkf[0, :].astype('int') + self.index_y[0][0]
                cols = ilkf[1, :].astype('int') + self.index_x[0][0]
                lkfs_np[it, rows, cols] = inum
            
            
            
            
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

    def compute_lengths(self, force_recompute=False):
        """Compute arc-length of every LKF segment and cache on ``self.lengths``.

        Calls :meth:`compute_projected_coords` automatically if projected
        coordinates have not yet been appended.

        Parameters
        ----------
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_lengths
            The :class:`lkf_tools.stats.lkf_lengths` instance (also stored as
            ``self.lengths``).
        """
        if not force_recompute and 'lengths' in self.__dict__ and self.__dict__['lengths'] is not None:
            return self.lengths
        if not force_recompute:
            cached = self._load_stats('lengths')
            if cached is not None:
                self.lengths = cached
                return self.lengths
        self._require_detection()
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_lengths
        self.lengths = lkf_lengths(self)
        self._save_stats('lengths', self.lengths)
        return self.lengths

    def compute_density(self, grid_res=50e3, force_recompute=False):
        """Bin LKF points into a 2-D spatial density grid and cache on ``self.density``.

        Calls :meth:`compute_projected_coords` automatically if projected
        coordinates have not yet been appended.

        Parameters
        ----------
        grid_res : float, optional
            Grid cell size in metres.  Default is 50 km.
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_density
            The :class:`lkf_tools.stats.lkf_density` instance (also stored as
            ``self.density``).
        """
        if not force_recompute and 'density' in self.__dict__ and self.__dict__['density'] is not None:
            return self.density
        if not force_recompute:
            cached = self._load_stats('density')
            if cached is not None:
                self.density = cached
                return self.density
        self._require_detection()
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_density
        self.density = lkf_density(self, res=grid_res)
        self._save_stats('density', self.density)
        return self.density

    def compute_curvature(self, force_recompute=False):
        """Compute end-to-end distance of every LKF segment and cache on ``self.curvature``.

        Calls :meth:`compute_projected_coords` automatically if needed.

        Parameters
        ----------
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_curvature
            The :class:`lkf_tools.stats.lkf_curvature` instance (also stored
            as ``self.curvature``).
        """
        if not force_recompute and 'curvature' in self.__dict__ and self.__dict__['curvature'] is not None:
            return self.curvature
        if not force_recompute:
            cached = self._load_stats('curvature')
            if cached is not None:
                self.curvature = cached
                return self.curvature
        self._require_detection()
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_curvature
        self.curvature = lkf_curvature(self)
        self._save_stats('curvature', self.curvature)
        return self.curvature

    def compute_deformation(self, force_recompute=False):
        """Compute mean divergence and shear per segment and cache on ``self.deformation``.

        Calls :meth:`compute_projected_coords` automatically if needed.

        Parameters
        ----------
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_deformation
            The :class:`lkf_tools.stats.lkf_deformation` instance (also stored
            as ``self.deformation``).
        """
        if not force_recompute and 'deformation' in self.__dict__ and self.__dict__['deformation'] is not None:
            return self.deformation
        if not force_recompute:
            cached = self._load_stats('deformation')
            if cached is not None:
                self.deformation = cached
                return self.deformation
        self._require_detection()
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_deformation
        self.deformation = lkf_deformation(self)
        self._save_stats('deformation', self.deformation)
        return self.deformation

    def compute_intersection(self, num_p=10, dis_par=1, use_vorticity=True,
                             force_recompute=False):
        """Find intersecting LKF pairs and compute crossing angles; cache on ``self.intersection``.

        Calls :meth:`compute_projected_coords` automatically if needed.

        Parameters
        ----------
        num_p : int, optional
            Points on each side of the contact used for the linear fit (default 10).
        dis_par : int, optional
            Search radius in grid cells for neighbour detection (default 1).
        use_vorticity : bool, optional
            If ``True`` (default), use per-segment vorticity sign to distinguish
            intersection types and produce angles in [0, 180] degrees.
            ``False`` folds angles into [0, 90].
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_intersection
            The :class:`lkf_tools.stats.lkf_intersection` instance.
        """
        if not force_recompute and 'intersection' in self.__dict__ and self.__dict__['intersection'] is not None:
            return self.intersection
        if not force_recompute:
            cached = self._load_stats('intersection')
            if cached is not None:
                self.intersection = cached
                return self.intersection
        self._require_detection()
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_intersection
        self.intersection = lkf_intersection(self, num_p=num_p, dis_par=dis_par,
                                             use_vorticity=use_vorticity)
        self._save_stats('intersection', self.intersection)
        return self.intersection

    def compute_orientation(self, force_recompute=False):
        """Compute per-segment orientation angle and cache on ``self.orientation``.

        Calls :meth:`compute_projected_coords` automatically if needed.

        Parameters
        ----------
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_orientation
            The :class:`lkf_tools.stats.lkf_orientation` instance.
        """
        if not force_recompute and 'orientation' in self.__dict__ and self.__dict__['orientation'] is not None:
            return self.orientation
        if not force_recompute:
            cached = self._load_stats('orientation')
            if cached is not None:
                self.orientation = cached
                return self.orientation
        self._require_detection()
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_orientation
        self.orientation = lkf_orientation(self)
        self._save_stats('orientation', self.orientation)
        return self.orientation

    def compute_lifetime(self, force_recompute=False):
        """Compute per-segment lifetime and cache on ``self.lifetime``.

        Requires that :meth:`track_lkfs` has been called beforehand so that
        tracked-pair files are available on disk.

        Parameters
        ----------
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_lifetime
            The :class:`lkf_tools.stats.lkf_lifetime` instance (also stored
            as ``self.lifetime``).
        """
        if not force_recompute and 'lifetime' in self.__dict__ and self.__dict__['lifetime'] is not None:
            return self.lifetime
        if not force_recompute:
            cached = self._load_stats('lifetime')
            if cached is not None:
                self.lifetime = cached
                return self.lifetime
        self._require_tracking()
        from lkf_tools.stats import lkf_lifetime
        self.lifetime = lkf_lifetime(self)
        self._save_stats('lifetime', self.lifetime)
        return self.lifetime

    def compute_growthrate(self, force_recompute=False):
        """Compute per-segment growth and shrink rates; cache on ``self.growthrate``.

        Calls :meth:`compute_lifetime` and :meth:`compute_lengths` automatically
        if not already cached.  Requires that :meth:`track_lkfs` has been called.

        Parameters
        ----------
        force_recompute : bool, optional
            Re-run the computation even if a cached result exists.  Default ``False``.

        Returns
        -------
        lkf_growthrate
            The :class:`lkf_tools.stats.lkf_growthrate` instance (also stored
            as ``self.growthrate``).
        """
        if not force_recompute and 'growthrate' in self.__dict__ and self.__dict__['growthrate'] is not None:
            return self.growthrate
        if not force_recompute:
            cached = self._load_stats('growthrate')
            if cached is not None:
                self.growthrate = cached
                return self.growthrate
        self._require_tracking()
        if not hasattr(self, 'lifetime') or self.lifetime is None:
            self.compute_lifetime(force_recompute=force_recompute)
        if not hasattr(self, 'lengths') or self.lengths is None:
            self.compute_lengths(force_recompute=force_recompute)
        if not hasattr(self, 'indm0'):
            self.compute_projected_coords()
        from lkf_tools.stats import lkf_growthrate
        self.growthrate = lkf_growthrate(self)
        self._save_stats('growthrate', self.growthrate)
        return self.growthrate

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
        pending_track = [int(j) for j in self.indexes if j+1 not in self.tracked_lkfs]
        for ilkf in _tqdm(pending_track, desc='Tracking LKFs', unit='pair', leave=True):
            
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
            if ilkf not in self.tracked_lkfs:
                self.tracked_lkfs.append(ilkf)


class process_dataset(lkf_dataset):
    def __init__(self,*args,**kwargs):
        super(process_dataset,self).__init__(*args,**kwargs)

        self._warning_relict_super_class = True


class lkf_collection:
    """Aggregate statistics across multiple :class:`lkf_dataset` instances.

    Wraps a list of ``lkf_dataset`` objects (typically one per year or per
    model run) and exposes the same ``compute_*()`` API as ``lkf_dataset``.
    Each call delegates to every dataset in the collection and caches the
    result on the individual dataset objects.  Flat properties provide
    concatenated per-segment arrays over all datasets and timesteps for
    direct use in statistics or plotting.

    Parameters
    ----------
    datasets : list of lkf_dataset
        Dataset instances to aggregate.

    Examples
    --------
    >>> col = lkf_collection([ds_2000, ds_2001, ds_2002])
    >>> col.compute_lengths()
    >>> all_lengths_km = col.lkf_length / 1e3
    """

    def __init__(self, datasets, label=None, by_dataset=False, color=None):
        self.datasets = list(datasets)
        self.label = label
        self.by_dataset = by_dataset
        self.color = color

    # ── internal helpers ──────────────────────────────────────────────────────

    def _detected(self, ds):
        """Return list of non-None timestep indices for *ds* (lkf_dataset only)."""
        return [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]

    def _flat_1d(self, ds_attr, obj_attr):
        """Concatenate per-segment 1-D arrays across all datasets and timesteps.

        Recurses into nested :class:`lkf_collection` instances.
        """
        parts = []
        for ds in self.datasets:
            if isinstance(ds, lkf_collection):
                arr = ds._flat_1d(ds_attr, obj_attr)
                if len(arr) > 0:
                    parts.append(arr)
            else:
                obj = getattr(ds, ds_attr, None)
                if obj is None:
                    continue
                arr = getattr(obj, obj_attr)
                for t in self._detected(ds):
                    a = arr[t]
                    if a is not None and len(a) > 0:
                        parts.append(np.asarray(a, dtype=float))
        return np.concatenate(parts) if parts else np.array([])

    # ── compute methods ───────────────────────────────────────────────────────

    def compute_projected_coords(self, proj=None):
        """Call :meth:`lkf_dataset.compute_projected_coords` on every dataset."""
        for ds in self.datasets:
            ds.compute_projected_coords(proj=proj)
        return self

    def compute_lengths(self, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_lengths` on every dataset."""
        for ds in self.datasets:
            ds.compute_lengths(force_recompute=force_recompute)
        return self

    def compute_density(self, grid_res=50e3, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_density` on every dataset."""
        for ds in self.datasets:
            ds.compute_density(grid_res=grid_res, force_recompute=force_recompute)
        return self

    def compute_curvature(self, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_curvature` on every dataset."""
        for ds in self.datasets:
            ds.compute_curvature(force_recompute=force_recompute)
        return self

    def compute_deformation(self, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_deformation` on every dataset."""
        for ds in self.datasets:
            ds.compute_deformation(force_recompute=force_recompute)
        return self

    def compute_intersection(self, num_p=10, dis_par=1, use_vorticity=True,
                             force_recompute=False):
        """Call :meth:`lkf_dataset.compute_intersection` on every dataset."""
        for ds in self.datasets:
            ds.compute_intersection(num_p=num_p, dis_par=dis_par,
                                    use_vorticity=use_vorticity,
                                    force_recompute=force_recompute)
        return self

    def compute_orientation(self, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_orientation` on every dataset."""
        for ds in self.datasets:
            ds.compute_orientation(force_recompute=force_recompute)
        return self

    def compute_lifetime(self, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_lifetime` on every dataset."""
        for ds in self.datasets:
            ds.compute_lifetime(force_recompute=force_recompute)
        return self

    def compute_growthrate(self, force_recompute=False):
        """Call :meth:`lkf_dataset.compute_growthrate` on every dataset."""
        for ds in self.datasets:
            ds.compute_growthrate(force_recompute=force_recompute)
        return self

    # ── flat aggregated arrays ────────────────────────────────────────────────

    @property
    def lkf_length(self):
        """Flat array of all segment arc-lengths (metres) across all datasets."""
        return self._flat_1d('lengths', 'lkf_length')

    @property
    def lkf_curvature(self):
        """Flat array of all segment end-to-end distances (metres)."""
        return self._flat_1d('curvature', 'lkf_curvature')

    @property
    def lkf_orientation(self):
        """Flat array of all segment orientation angles (degrees)."""
        return self._flat_1d('orientation', 'lkf_orientation')

    @property
    def lkf_lifetime(self):
        """Flat array of all segment lifetimes (timesteps)."""
        return self._flat_1d('lifetime', 'lkf_lifetime')

    @property
    def lkf_deformation(self):
        """(N, 2) array of per-segment ``[mean_div, mean_shr]`` across all datasets."""
        parts = []
        for ds in self.datasets:
            if isinstance(ds, lkf_collection):
                a = ds.lkf_deformation
                if len(a) > 0:
                    parts.append(a)
            else:
                if not hasattr(ds, 'deformation'):
                    continue
                for t in self._detected(ds):
                    a = ds.deformation.lkf_deformation[t]
                    if a is not None and len(a) > 0:
                        parts.append(np.asarray(a, dtype=float))
        return np.vstack(parts) if parts else np.empty((0, 2))

    @property
    def lkf_interc(self):
        """Flat array of intersection angles (degrees) across all datasets."""
        parts = []
        for ds in self.datasets:
            if isinstance(ds, lkf_collection):
                a = ds.lkf_interc
                if len(a) > 0:
                    parts.append(a)
            else:
                if not hasattr(ds, 'intersection'):
                    continue
                for t in self._detected(ds):
                    a = ds.intersection.lkf_interc[t]
                    if len(a) > 0:
                        parts.append(np.asarray(a, dtype=float))
        return np.concatenate(parts) if parts else np.array([])

    @property
    def lkf_interc_type(self):
        """Flat integer array of intersection types (0/1) across all datasets.

        Only populated when all datasets were computed with ``use_vorticity=True``.
        Returns an empty array if type information is unavailable.
        """
        parts = []
        for ds in self.datasets:
            if isinstance(ds, lkf_collection):
                a = ds.lkf_interc_type
                if len(a) > 0:
                    parts.append(a)
            else:
                if not hasattr(ds, 'intersection'):
                    continue
                itype = getattr(ds.intersection, 'lkf_interc_type', None)
                if itype is None:
                    continue
                for t in self._detected(ds):
                    a = itype[t]
                    if len(a) > 0:
                        parts.append(np.asarray(a, dtype=int))
        return np.concatenate(parts).astype(int) if parts else np.array([], dtype=int)

    @property
    def lkf_growth(self):
        """Flat array of segment growth rates (m per timestep) across all datasets."""
        return self._flat_1d('growthrate', 'lkf_growth')

    @property
    def lkf_shrink(self):
        """Flat array of segment shrink rates (m per timestep) across all datasets."""
        return self._flat_1d('growthrate', 'lkf_shrink')

    def _get_interc_arrays(self, lifetime_val=None):
        """Aggregate intersection angles and types across all flat datasets.

        Parameters
        ----------
        lifetime_val : int or None
            When not None, only include pairs where both segments have
            ``lifetime == lifetime_val`` according to each dataset's own
            ``lifetime`` object.

        Returns
        -------
        ang_flat : np.ndarray
            1-D array of intersection angles.
        type_flat : np.ndarray of int
            1-D integer array of types (0 = opp. vorticity, 1 = same vorticity).
            Empty if no vorticity data is available.
        """
        ang_parts = []
        type_parts = []
        for ds, _ in self._iter_flat_datasets():
            if not hasattr(ds, 'intersection') or ds.intersection is None:
                continue
            itype = getattr(ds.intersection, 'lkf_interc_type', None)
            ipar  = getattr(ds.intersection, 'lkf_interc_par', None)
            lt_obj = ds.__dict__.get('lifetime') if lifetime_val is not None else None
            for t in self._detected(ds):
                angs = ds.intersection.lkf_interc[t]
                if len(angs) == 0:
                    continue
                angs  = np.asarray(angs, dtype=float)
                types = (np.asarray(itype[t], dtype=int)
                         if (itype is not None and t < len(itype) and len(itype[t]))
                         else None)
                if (lifetime_val is not None and lt_obj is not None
                        and ipar is not None and t < len(ipar) and len(ipar[t])):
                    lts   = (lt_obj.lkf_lifetime[t]
                             if t < len(lt_obj.lkf_lifetime) else np.array([]))
                    pairs = ipar[t]
                    mask  = np.array([
                        (int(p[0]) < len(lts) and lts[int(p[0])] == lifetime_val and
                         int(p[1]) < len(lts) and lts[int(p[1])] == lifetime_val)
                        for p in pairs
                    ], dtype=bool)
                    angs  = angs[mask]
                    if types is not None:
                        types = types[mask]
                if len(angs) == 0:
                    continue
                ang_parts.append(angs)
                if types is not None and len(types) == len(angs):
                    type_parts.append(types)
        ang_flat  = np.concatenate(ang_parts)  if ang_parts  else np.array([])
        type_flat = np.concatenate(type_parts).astype(int) if type_parts else np.array([], dtype=int)
        return ang_flat, type_flat

    # ── plotting methods ──────────────────────────────────────────────────────

    @staticmethod
    def _wrap_flat(arr):
        """Wrap a flat 1-D array into a 1-element object array for _flatten_obj."""
        out = np.empty(1, dtype=object)
        out[0] = np.asarray(arr, dtype=float)
        return out

    @staticmethod
    def _elem_label(elem, i):
        """Return the label for dataset *elem* (index *i*), or a fallback."""
        return getattr(elem, 'label', None) or f'Dataset {i + 1}'

    @staticmethod
    def _elem_color(elem, i, colors):
        """Return the effective colour for *elem*: ``elem.color`` if set, else ``colors[i]``."""
        c = getattr(elem, 'color', None)
        return c if c is not None else colors[i]

    def _iter_flat_datasets(self):
        """Yield (lkf_dataset, label) pairs, recursing into nested collections."""
        for i, elem in enumerate(self.datasets):
            if isinstance(elem, lkf_collection):
                yield from elem._iter_flat_datasets()
            else:
                yield elem, self._elem_label(elem, i)

    def plot_lengths(self, by_dataset=None, ax=None, bins=50, fit=True,
                     colors=None, color=None):
        """Plot log-log PDF of LKF arc-lengths.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``True`` overlay one line per element with a power-law fit each.
            When ``False`` all data are pooled.  ``None`` (default) inherits
            ``self.by_dataset``.
        colors : list, optional
            Per-element colours (``tab10`` by default).
        color : str, optional
            Override colour for the pooled line (used by parent collections in
            nested mode).  Falls back to ``self.color`` when *None*.

        Returns
        -------
        fig, ax
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_lengths
        if not by_dataset:
            proxy = types.SimpleNamespace(
                lkf_length=self._wrap_flat(self.lkf_length))
            eff_color = color if color is not None else self.color
            kw = {'bins': bins, 'fit': fit}
            if eff_color is not None:
                kw['color'] = eff_color
            return lkf_lengths.plot(proxy, ax=ax, **kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(self.datasets), 1)))
        colors = colors if colors is not None else cmap
        for i, elem in enumerate(self.datasets):
            elem_label = self._elem_label(elem, i)
            elem_color = self._elem_color(elem, i, colors)
            if isinstance(elem, lkf_collection):
                elem.plot_lengths(ax=ax, by_dataset=elem.by_dataset,
                                  bins=bins, fit=fit, color=elem_color)
            else:
                if not hasattr(elem, 'lengths') or elem.lengths is None:
                    continue
                lkf_lengths.plot(elem.lengths, ax=ax, bins=bins,
                                 color=elem_color, fit=fit)
            ax.plot([], [], 'o-', color=elem_color, ms=4, lw=1.5,
                    label=elem_label)
        ax.legend(fontsize=8)
        ax.set_title('LKF length distribution')
        return fig, ax

    def plot_density(self, by_dataset=None, ax=None, **kw):
        """Plot spatial LKF density.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``False`` average all datasets onto a common grid and show one
            map.  When ``True`` produce one subplot row per top-level element.
            ``None`` (default) inherits ``self.by_dataset``.

        Returns
        -------
        by_dataset=False : fig, ax, pcm
        by_dataset=True  : fig, axes (list), pcms (list)
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_density as lkf_density_cls

        if not by_dataset:
            # ── merged mode ──────────────────────────────────────────────────
            # Flatten through nested collections using _iter_flat_datasets so
            # that nested lkf_collection elements are handled transparently.
            cubes, ref, ice_mask = [], None, None
            for ds, _ in self._iter_flat_datasets():
                d = ds.__dict__.get('density')
                if d is None:
                    continue
                if ref is None:
                    ref = d
                if d.lkf_density.shape[1:] == ref.lkf_density.shape[1:]:
                    cubes.append(d.lkf_density)
                    m = d.ice_mask
                    if m is not None:
                        ice_mask = m if ice_mask is None else (ice_mask | m)
            if ref is None:
                raise RuntimeError(
                    "No density data found. Call compute_density() on each "
                    "dataset before plotting.")
            merged = np.concatenate(cubes, axis=0)
            proxy = types.SimpleNamespace(
                lkf_density=merged, res=ref.res,
                xedg=ref.xedg, yedg=ref.yedg, ice_mask=ice_mask)
            return lkf_density_cls.plot(proxy, ax=ax, **kw)

        # ── per-dataset mode ──────────────────────────────────────────────────
        # Build a list of (proxy, label) pairs — one entry per top-level element.
        # lkf_dataset elements use their own density object; lkf_collection
        # elements are reduced to a merged density proxy so each inner
        # collection occupies exactly one subplot.
        def _merged_proxy(col):
            """Return a merged lkf_density-like namespace for an lkf_collection."""
            cubes, ref, ice_mask = [], None, None
            for ds, _ in col._iter_flat_datasets():
                d = ds.__dict__.get('density')
                if d is None:
                    continue
                if ref is None:
                    ref = d
                if d.lkf_density.shape[1:] == ref.lkf_density.shape[1:]:
                    cubes.append(d.lkf_density)
                    m = d.ice_mask
                    if m is not None:
                        ice_mask = m if ice_mask is None else (ice_mask | m)
            if ref is None:
                return None
            return types.SimpleNamespace(
                lkf_density=np.concatenate(cubes, axis=0),
                res=ref.res, xedg=ref.xedg, yedg=ref.yedg, ice_mask=ice_mask)

        panels = []
        for i, elem in enumerate(self.datasets):
            lbl = self._elem_label(elem, i)
            if isinstance(elem, lkf_collection):
                proxy = _merged_proxy(elem)
            else:
                proxy = elem.__dict__.get('density')
            if proxy is not None:
                panels.append((proxy, lbl))

        n = len(panels)
        if n == 0:
            raise RuntimeError(
                "No density data found for any element. Call compute_density() "
                "on the collection before plotting.")
        # Build GeoAxes when coastlines are requested (matches lkf_density.plot behaviour)
        subplot_kw = {}
        if kw.get('coastlines', True):
            try:
                import cartopy.crs as ccrs
                proj_crs = kw.get('proj_crs') or ccrs.NorthPolarStereo(
                    central_longitude=-45, true_scale_latitude=75)
                subplot_kw = {'projection': proj_crs}
            except ImportError:
                pass
        fig, axes = plt.subplots(n, 1, figsize=(8, 6 * n),
                                 squeeze=False, subplot_kw=subplot_kw)
        axes = axes[:, 0]
        pcms = []
        for idx, (proxy, lbl) in enumerate(panels):
            _, _, pcm = lkf_density_cls.plot(proxy, ax=axes[idx], **kw)
            axes[idx].set_title(lbl)
            pcms.append(pcm)
        fig.tight_layout()
        return fig, list(axes), pcms

    def plot_curvature(self, by_dataset=None, ax=None, colors=None, **kw):
        """Scatter plot of arc-length vs chord-length.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``True`` use one colour per element (with label).
            ``None`` (default) inherits ``self.by_dataset``.

        Returns
        -------
        fig, ax
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_curvature, lkf_lengths
        kw.pop('arc_lengths', None)
        if not by_dataset:
            curv_proxy = types.SimpleNamespace(
                lkf_curvature=self._wrap_flat(self.lkf_curvature))
            len_proxy = types.SimpleNamespace(
                lkf_length=self._wrap_flat(self.lkf_length))
            return lkf_curvature.plot(curv_proxy, arc_lengths=len_proxy,
                                      ax=ax, **kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(self.datasets), 1)))
        colors = colors if colors is not None else cmap
        for i, elem in enumerate(self.datasets):
            elem_label = self._elem_label(elem, i)
            elem_color = self._elem_color(elem, i, colors)
            if isinstance(elem, lkf_collection):
                elem.plot_curvature(ax=ax, by_dataset=elem.by_dataset,
                                    colors=[elem_color])
            else:
                if not hasattr(elem, 'curvature') or elem.curvature is None:
                    continue
                lkf_curvature.plot(elem.curvature, arc_lengths=elem.lengths,
                                   ax=ax, color=elem_color,
                                   alpha=kw.get('alpha', 0.2),
                                   s=kw.get('s', 3))
            ax.plot([], [], 'o', color=elem_color, ms=4, label=elem_label)
        ax.legend(fontsize=8)
        ax.set_title('LKF arc-length vs chord-length')
        return fig, ax

    def plot_deformation(self, ax=None, **kw):
        """2-D log-scale histogram of divergence vs shear rates merged over all datasets.

        Returns
        -------
        fig, ax
        """
        from lkf_tools.stats import lkf_deformation
        def_arr = self.lkf_deformation
        wrapped = np.empty(1, dtype=object)
        wrapped[0] = def_arr
        proxy = types.SimpleNamespace(lkf_deformation=wrapped)
        return lkf_deformation.plot(proxy, ax=ax, **kw)

    def plot_intersection(self, by_dataset=None, ax=None, bins=18,
                          colors=None, use_vorticity=None,
                          intersection_type='both', lifetime=None, **kw):
        """Plot histogram of LKF intersection angles.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``True`` overlay one faint histogram per element (with label)
            and a bold merged histogram.  ``None`` (default) inherits
            ``self.by_dataset``.
        use_vorticity : bool or None, optional
            Override whether to use the vorticity-based [0, 180°] mode.
            ``None`` (default) reads the flag stored on the first found
            ``intersection`` object.
        intersection_type : {'both', 'opp', 'same'}, optional
            Which intersection types to include (only effective when
            ``use_vorticity=True``).  ``'opp'`` → Type 0 (opposite vorticity);
            ``'same'`` → Type 1 (same vorticity); ``'both'`` (default) → both.
        lifetime : int or None, optional
            When set (e.g. ``lifetime=1``), restrict to pairs where **both**
            segments have that lifetime value, using each dataset's own
            ``lifetime`` object.

        Returns
        -------
        fig, ax
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_intersection
        # Determine use_vorticity from stored data unless caller overrides
        uv_stored = True
        for ds in self.datasets:
            if isinstance(ds, lkf_collection):
                for sub, _ in ds._iter_flat_datasets():
                    if hasattr(sub, 'intersection') and sub.intersection is not None:
                        uv_stored = getattr(sub.intersection, 'use_vorticity', True)
                        break
            elif hasattr(ds, 'intersection') and ds.intersection is not None:
                uv_stored = getattr(ds.intersection, 'use_vorticity', True)
                break
        uv = uv_stored if use_vorticity is None else use_vorticity

        def _make_proxy_from_arrays(ang_flat, type_flat):
            """Build a SimpleNamespace suitable for lkf_intersection.plot()."""
            ns = types.SimpleNamespace(
                lkf_interc=self._wrap_flat(ang_flat), use_vorticity=uv)
            if uv and len(type_flat):
                # _wrap_flat casts to float; types must stay int
                wrapped_type = np.empty(1, dtype=object)
                wrapped_type[0] = type_flat.astype(int)
                ns.lkf_interc_type = wrapped_type
            return ns

        if not by_dataset:
            ang_flat, type_flat = self._get_interc_arrays(lifetime_val=lifetime)
            proxy = _make_proxy_from_arrays(ang_flat, type_flat)
            return lkf_intersection.plot(proxy, ax=ax, bins=bins,
                                         intersection_type=intersection_type, **kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(self.datasets), 1)))
        colors = colors if colors is not None else cmap
        for i, elem in enumerate(self.datasets):
            elem_label = self._elem_label(elem, i)
            # Aggregate flat arrays for this element (dataset or nested collection)
            if isinstance(elem, lkf_collection):
                ang_e, type_e = elem._get_interc_arrays(lifetime_val=lifetime)
            else:
                if not hasattr(elem, 'intersection') or elem.intersection is None:
                    continue
                tmp = lkf_collection([elem])
                ang_e, type_e = tmp._get_interc_arrays(lifetime_val=lifetime)
            if len(ang_e) == 0:
                continue
            sub = _make_proxy_from_arrays(ang_e, type_e)
            lkf_intersection.plot(sub, ax=ax, bins=bins,
                                  color0=colors[i], color1=colors[i],
                                  intersection_type=intersection_type,
                                  label=elem_label)
        ax.legend(fontsize=8)
        ax.set_title('LKF intersection angles')
        return fig, ax

    def plot_orientation(self, by_dataset=None, ax=None, bins=36,
                         colors=None, color=None, **kw):
        """Polar rose diagram of LKF orientations.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``True`` overlay one rose per element.
            ``None`` (default) inherits ``self.by_dataset``.
        color : str, optional
            Override colour for the pooled rose.  Falls back to ``self.color``.

        Returns
        -------
        fig, ax
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_orientation
        if not by_dataset:
            proxy = types.SimpleNamespace(
                lkf_orientation=self._wrap_flat(self.lkf_orientation))
            eff_color = color if color is not None else self.color
            plot_kw = dict(kw)
            if eff_color is not None:
                plot_kw['color'] = eff_color
            return lkf_orientation.plot(proxy, ax=ax, bins=bins, **plot_kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5),
                                   subplot_kw={'projection': 'polar'})
        else:
            fig = ax.figure
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(self.datasets), 1)))
        colors = colors if colors is not None else cmap
        for i, elem in enumerate(self.datasets):
            elem_label = self._elem_label(elem, i)
            elem_color = self._elem_color(elem, i, colors)
            if isinstance(elem, lkf_collection):
                elem.plot_orientation(ax=ax, by_dataset=elem.by_dataset,
                                      bins=bins, color=elem_color)
            else:
                if not hasattr(elem, 'orientation') or elem.orientation is None:
                    continue
                lkf_orientation.plot(elem.orientation, ax=ax, bins=bins,
                                     color=elem_color,
                                     density=kw.get('density', True))
            ax.plot([], [], '-', color=elem_color, lw=2, label=elem_label)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title('LKF orientation')
        return fig, ax

    def plot_lifetime(self, by_dataset=None, ax=None, colors=None, color=None,
                      **kw):
        """Log-histogram of LKF lifetimes with exponential fit.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``True`` overlay one line per element with an exponential fit
            each.  ``None`` (default) inherits ``self.by_dataset``.
        color : str, optional
            Override colour for the pooled line.  Falls back to ``self.color``.

        Returns
        -------
        fig, ax
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_lifetime
        if not by_dataset:
            proxy = types.SimpleNamespace(
                lkf_lifetime=self._wrap_flat(self.lkf_lifetime))
            eff_color = color if color is not None else self.color
            plot_kw = dict(kw)
            if eff_color is not None:
                plot_kw['color'] = eff_color
            return lkf_lifetime.plot(proxy, ax=ax, **plot_kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(self.datasets), 1)))
        colors = colors if colors is not None else cmap
        for i, elem in enumerate(self.datasets):
            elem_label = self._elem_label(elem, i)
            elem_color = self._elem_color(elem, i, colors)
            if isinstance(elem, lkf_collection):
                elem.plot_lifetime(ax=ax, by_dataset=elem.by_dataset,
                                   color=elem_color, **kw)
            else:
                if not hasattr(elem, 'lifetime') or elem.lifetime is None:
                    continue
                lkf_lifetime.plot(elem.lifetime, ax=ax, color=elem_color,
                                  fit=kw.get('fit', True))
            ax.plot([], [], '-', color=elem_color, lw=2, label=elem_label)
        ax.legend(fontsize=8)
        ax.set_title('LKF lifetime distribution')
        return fig, ax

    def plot_growthrate(self, by_dataset=None, ax=None, colors=None, color=None,
                        **kw):
        """Log-log PDFs of LKF growth and shrink rates.

        Parameters
        ----------
        by_dataset : bool or None, optional
            When ``True`` overlay one line per element with a power-law fit each.
            ``None`` (default) inherits ``self.by_dataset``.
        color : str, optional
            Override colour for the pooled lines.  Falls back to ``self.color``.

        Returns
        -------
        fig, axes : Figure and array of two Axes.
        """
        if by_dataset is None:
            by_dataset = self.by_dataset
        from lkf_tools.stats import lkf_growthrate
        if not by_dataset:
            wrapped_g = np.empty(1, dtype=object)
            wrapped_g[0] = np.asarray(self.lkf_growth, dtype=float)
            wrapped_s = np.empty(1, dtype=object)
            wrapped_s[0] = np.asarray(self.lkf_shrink, dtype=float)
            proxy = types.SimpleNamespace(lkf_growth=wrapped_g,
                                          lkf_shrink=wrapped_s)
            eff_color = color if color is not None else self.color
            plot_kw = dict(kw)
            if eff_color is not None:
                plot_kw.setdefault('color_growth', eff_color)
                plot_kw.setdefault('color_shrink', eff_color)
            return lkf_growthrate.plot(proxy, ax=ax, **plot_kw)

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        else:
            fig = ax[0].figure
            axes = ax
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(self.datasets), 1)))
        colors = colors if colors is not None else cmap
        for i, elem in enumerate(self.datasets):
            elem_label = self._elem_label(elem, i)
            elem_color = self._elem_color(elem, i, colors)
            if isinstance(elem, lkf_collection):
                elem.plot_growthrate(ax=axes, by_dataset=elem.by_dataset,
                                     color=elem_color, **kw)
            else:
                if not hasattr(elem, 'growthrate') or elem.growthrate is None:
                    continue
                lkf_growthrate.plot(elem.growthrate, ax=axes,
                                    color_growth=elem_color,
                                    color_shrink=elem_color,
                                    fit=kw.get('fit', True))
            axes[0].plot([], [], '-', color=elem_color, lw=2, label=elem_label)
        axes[0].legend(fontsize=8)
        axes[0].set_title('Growth rate')
        axes[1].set_title('Shrink rate')
        return fig, axes

