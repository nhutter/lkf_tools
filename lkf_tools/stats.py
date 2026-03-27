# -*- coding: utf-8 -*-

"""
Statistic module of lkf_tools to perform spatial and temporal LKF statistics
"""


# Package Metadata
__version__ = 2.0
__author__ = "Nils Hutter"
__author_email__ = "nils.hutter@awi.de"


import numpy as np
import matplotlib.pylab as plt
import os
import sys
import datetime as dt
import scipy
import warnings
from collections import defaultdict
from pathlib import Path
from scipy.spatial import cKDTree
import pickle

from pyproj import Proj
import xarray as xr


# Define function for fit to polynom

def lkf_poly_fit(x,y,deg,return_p=False):
    if x.size-1<deg:
        deg=x.size-1
    t = np.arange(x.size)
    p_x = np.polyfit(t,x,deg)
    x_fit = np.polyval(p_x,t)
    p_y = np.polyfit(t,y,deg)
    y_fit = np.polyval(p_y,t)
    if return_p:
        return x_fit,y_fit,p_x,p_y
    else:
        return x_fit,y_fit

def lkf_poly_fit_p(x,y,deg):
    if x.size-1<deg:
        deg=x.size-1
    t = np.arange(x.size)
    p_x = np.polyfit(t,x,deg)
    p_y = np.polyfit(t,y,deg)
    return p_x,p_y


def power_law_fit(x, y):
    """Fit a power law y = A · x^α in log-log space.

    Parameters
    ----------
    x, y : array-like
        Positive values to fit (zeros/negatives are silently skipped).

    Returns
    -------
    coeff : ndarray, shape (2,)
        ``[α, log10(A)]`` from ``np.polyfit`` in log-log space.
    fit : ndarray
        Power-law values evaluated at *x*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    coeff = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
    fit = np.power(10, np.polyval(coeff, np.log10(x)))
    return coeff, fit


def runmean(data, win):
    """Running mean with window *win*."""
    datam = np.zeros(data.size - win + 1)
    for i in range(win):
        datam += data[i:(data.size + 1 - win + i)]
    return datam / float(win)


def _flatten_obj(arr):
    """Return flat 1-D array from a 1-D or 2-D numpy object array of arrays.

    Handles both the new flat API (``arr[t]`` = 1-D array) and the legacy
    nested API (``arr[year]`` = list of day-arrays).
    """
    if arr.ndim == 1:
        parts = [np.asarray(a) for a in arr
                 if a is not None and np.asarray(a).size > 0]
        return np.concatenate(parts) if parts else np.array([])
    # Legacy: 2-D nesting
    parts = []
    for year in arr:
        for day in year:
            a = np.asarray(day)
            if a.size > 0:
                parts.append(a)
    return np.concatenate(parts) if parts else np.array([])


# Define reading function
def load_lkf_dataset(lkf_path,output_path=None,datatype=None,subdirs=None,name=None,
                     read_tracking=True,m=None,polyfit=True,poly_deg=6,force_reread=False,
                     mask_rgps=False):
    """Load a legacy multi-year LKF dataset from a directory tree.

    .. note::
        This function loads the *legacy* ``stats.lkf_dataset`` format
        (nested year/day structure, RGPS or MITgcm binary files).
        For netCDF-based data processed with the new pipeline, use
        :class:`lkf_tools.dataset.lkf_dataset` (or the
        :func:`lkf_tools.dataset.process_dataset` convenience wrapper) instead.
        Multiple datasets can be aggregated with
        :class:`lkf_tools.dataset.lkf_collection`.
    """
    open_pkl = False; write_pkl = False;
    if output_path is None: output_path=lkf_path
    if name is None: 
        pkl_list = [ifile for ifile in os.listdir(output_path) if ifile.endswith('.pkl')]
        if len(pkl_list) != 1:
            print('Error: Cannot choose which pickle to load in ' + str(pkl_list))
        else:
            pickle_name = pkl_list[0]
            open_pkl = True
    else:
        pickle_name = 'lkf_data_' + name + '.pkl'
        if os.path.exists(output_path + pickle_name):
            open_pkl = True
        else:
            write_pkl = True

    if force_reread:
        if open_pkl:
            print('Old pickle exists, but is not read due to force_reread option')
            open_pkl = False
        write_pkl = True

    if open_pkl:
        print('Open old pickle: %s' %(output_path + pickle_name))
        with open(output_path + pickle_name, 'rb') as input_pkl:
            print(input_pkl)
            lkf_data = pickle.load(input_pkl)
        return lkf_data
    
    if write_pkl:
        # Check whether all necessary parameters are given
        if np.any([ipar is None for ipar in [datatype,subdirs,name]]):
            print('Error: For reading the following parameters are needed: ' +
                  str(np.array(['datatype','subdir','name'])[[ipar is None for ipar in [datatype,subdirs,name]]]))
        else:
            # Read in data
            lkf_data = lkf_dataset(output_path + pickle_name,
                                   lkf_path,output_path,datatype,subdirs,name,
                                   read_tracking=read_tracking,m=m,
                                   polyfit=polyfit,poly_deg=poly_deg,
                                   mask_rgps=mask_rgps)

            # Write object pickle
            with open(output_path + pickle_name, 'wb') as output_pkl:
                pickle.dump(lkf_data, output_pkl, pickle.HIGHEST_PROTOCOL)

            return lkf_data

        
class lkf_dataset(object):
    def __init__(self,pickle_in,lkf_path,output_path,datatype,subdirs,name,read_tracking=True,m=None,polyfit=True,poly_deg=6,mask_rgps=False,track_dir_name='tracked_pairs'):
        
        # Store information
        self.pickle      = pickle_in
        self.lkf_path    = lkf_path
        self.output_path = output_path
        self.datatype    = datatype
        self.subdirs     = subdirs
        self.years       = subdirs
        self.name        = name
        self.mask_rgps   = mask_rgps

        self.polyfit     = polyfit 
        self.poly_deg    = poly_deg 

        # Define projection
        if m is None:
            #m = Basemap(projection='stere',lat_ts=70,lat_0=90,lon_0=-45,resolution='l',
            #            llcrnrlon=-115,llcrnrlat=64,urcrnrlon=105,urcrnrlat=65,ellps='WGS84')
            m = Proj(proj='stere',lat_0=90, lat_ts=75, lon_0=-45, ellps='WGS84')
        self.m = m

        
        # -------------- Start reading data --------------------------------------

        lkf_dataset = []
        lkf_meta = []
        lkf_track_data = []
        
        years = self.subdirs
        
        # Test for deformation rate output
        lkffile_list = [ifile for ifile in os.listdir(lkf_path + years[0]) if ifile.startswith('lkf')]
        print(lkf_path + years[0] + '/' + lkffile_list[0])
        if lkffile_list[0].endswith('.npy'):
            test_lkf = np.load(lkf_path + years[0] + '/' + lkffile_list[0], allow_pickle=True,encoding='bytes')
        elif lkffile_list[0].endswith('.npz'):
            test_lkf = np.load(lkf_path + years[0] + '/' + lkffile_list[0], allow_pickle=True,encoding='bytes')['lkf']
        if test_lkf[0].shape[1] == 6:
            self.indm0 = 6; self.indm1 = 7;
            self.indp0 = 8; self.indp1 = 9;
            self.indd0 = 4; self.indd1 = 5;
        elif test_lkf[0].shape[1] == 7: # Vorticity added
            self.indm0 = 7; self.indm1 = 8;
            self.indp0 = 9; self.indp1 = 10;
            self.indd0 = 4; self.indd1 = 5; self.indd2 = 6
        else: # No deformation rates
            self.indm0 = 4; self.indm1 = 5;
            self.indp0 = 6; self.indp1 = 7;

        if self.mask_rgps:
            if self.datatype == 'mitgcm_2km':
                # MITgcm grid points
                grid_path = '/work/ollie/nhutter/arctic_2km/run_cor_cs/'
                lon_cov, lat_cov = read_latlon(grid_path)
                mask    = mask_arcticbasin(grid_path,read_latlon)
                index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
                index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
                red_fac = 3 # Take only every red_fac point to reduce array size
                lon_cov = lon_cov[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                  index_x[0][0]-1:index_x[0][-1]+2:red_fac]
                lat_cov = lat_cov[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                  index_x[0][0]-1:index_x[0][-1]+2:red_fac]
                x_cov,y_cov  = m(lon_cov,lat_cov)
            
            if self.datatype == 'sirex':
                # if self.lkf_path.split('/')[-2].split('_')[-1] == 'means':
                #     ind_yp = -2
                # elif self.lkf_path.split('/')[-2].split('_')[-1] == 'inst':
                #     ind_yp = -1
                # ncfile = ('/work/ollie/nhutter/sirex/data/' +
                #           '/'.join(lkf_path[47:-1].split('/')[:-1])+'/'+
                #           '_'.join(lkf_path[47:-1].split('/')[-1].split('_')[:ind_yp]) + '_' +
                #           years[-1] + '_' +
                #           '_'.join(lkf_path[47:-1].split('/')[-1].split('_')[ind_yp:]) + '.nc')
                lkf_ps=lkf_path.split('/') 
                sp = '/'.join(lkf_ps[:np.where(np.array(lkf_ps)=='analysis')[0][0]]) 
                ind_m = np.where(np.array(lkf_ps)=='lead_detect')[0][0]+1 
                ind_f = np.where(['means' in iseg or 'inst' in iseg for iseg in lkf_ps])[0][0] 
                ind_yp = np.array([-2,-1])[[np.any(['means' in iseg for iseg in lkf_ps]),np.any(['inst' in iseg for iseg in lkf_ps])]][0] 
                flnml = lkf_ps[ind_f].split('_'); flnml.insert(ind_yp,years[-1]) 
                ncfile = os.path.join(sp,'data','/'.join(lkf_ps[ind_m:ind_f]),'_'.join(flnml)+'.nc')  


                ncdata = Dataset(ncfile)
                lon  = ncdata.variables['ULON'][:,:]
                lat  = ncdata.variables['ULAT'][:,:]

                mask = ((((lon > -120) & (lon < 100)) & (lat >= 80)) |
                        ((lon <= -120) & (lat >= 70)) |
                        ((lon >= 100) & (lat >= 70)))
                index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
                index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
                red_fac = 1 # Take only every red_fac point to reduce array size
                x_cov,y_cov  = m(lon[max([index_y[0][0]-1,0]):index_y[0][-1]+2:red_fac,
                                     max([index_x[0][0]-1,0]):index_x[0][-1]+2:red_fac],
                                 lat[max([index_y[0][0]-1,0]):index_y[0][-1]+2:red_fac,
                                     max([index_x[0][0]-1,0]):index_x[0][-1]+2:red_fac])
                
            if self.datatype == 'rgps':
                cov = np.load('/work/ollie/nhutter/sirex/analysis/lead_detect/RGPS/coverage_rgps_1997_mod.npz')
                x_cov, y_cov = m(cov['lon'],cov['lat'])
                
    
            # RGPS coverage
            if self.datatype=='mitgcm_2km':
                cov = np.load('/work/ollie/nhutter/lkf_data/rgps_opt/coverage_rgps.npz')
            else:
                cov = np.load('/work/ollie/nhutter/sirex/analysis/lead_detect/RGPS/coverage_rgps_1997_mod.npz')
                print('Loading this coverage file for masking: /work/ollie/nhutter/sirex/analysis/lead_detect/RGPS/coverage_rgps_1997_mod.npz')
                
            x_rgps,y_rgps = m(cov['lon'],cov['lat'])
            
            # Generate KDTree for interpolation
            tree = cKDTree(np.rollaxis(np.stack([x_rgps.flatten(),y_rgps.flatten()]),0,2))
            distances, inds = tree.query(np.rollaxis(np.stack([x_cov.flatten(),y_cov.flatten()]),0,2), k = 1)
            mask_cov = (distances>=12.5e3).reshape(x_cov.shape)


        for year_dic in years:
            print("Start reading with year: " + year_dic)
            new_dir = lkf_path + year_dic + '/'
            # Generate list of files
            lkffile_list = [ifile for ifile in os.listdir(new_dir) if ifile.startswith('lkf')]
            lkffile_list.sort()
    
            # Initialize list for year
            lkf_data_year = []
            lkf_meta_year = []

            # Read yearly RGPS coverage
            if self.mask_rgps:
                if self.datatype=='mitgcm_2km':
                    cov_year = np.load('/work/ollie/nhutter/lkf_data/rgps_opt/coverage_rgps_%s.npz' %year_dic)['coverage']
                else:
                    cov_year = np.load('/work/ollie/nhutter/sirex/analysis/lead_detect/RGPS/coverage_rgps_%s_mod.npz' %year_dic)['coverage']
                    

                if len(lkffile_list)>cov_year.shape[0]:
                    lkffile_list = lkffile_list[:cov_year.shape[0]]
    
            # Loop over files to read an process
            for it,lkffile in enumerate(lkffile_list):
        
                # Save meta information: start/end time, number features
                if datatype == 'rgps':
                    startdate = (dt.date(int(lkffile[4:-15]),1,1)+
                                 dt.timedelta(int(lkffile[-15:-12])))
                    enddate   = (dt.date(int(lkffile[-11:-7]),1,1)+
                                 dt.timedelta(int(lkffile[-7:-4])))
                elif datatype == 'mitgcm_2km':
                    startdate = (dt.datetime(1992,1,1,0,0,0) + 
                                 dt.timedelta(0,int(lkffile[-14:-4])*120.))
                    enddate   = (dt.datetime(1992,1,1,0,0,0) + 
                                 dt.timedelta(0,int(lkffile[-14:-4])*120. +
                                              24*3600.))
                elif datatype == 'sirex':
                    if lkffile.split('_')[-2] == 'means':
                        ind_year = -4
                    elif lkffile.split('_')[-2] == 'inst':
                        ind_year = -3
                    startdate = (dt.date(int(lkffile.split('_')[ind_year]),1,1)+
                                         dt.timedelta(int(lkffile.split('_')[-1][:-4])))
                    enddate   = (dt.date(int(lkffile.split('_')[ind_year]),1,1)+
                                         dt.timedelta(int(lkffile.split('_')[-1][:-4])+3))
                
                    
                if lkffile.endswith('.npy'):
                    lkfi = np.load(new_dir+lkffile, allow_pickle=True,encoding='bytes')
                    lkf_meta_year.append(np.array([startdate,
                                                   enddate,
                                                   lkfi.size]))
                elif lkffile.endswith('.npz'):
                    lkfiz = np.load(new_dir+lkffile, allow_pickle=True,encoding='bytes')
                    lkfi = lkfiz['lkf']

                    if datatype == 'mosaic':
                        print(str(lkfiz['fname']).split('/')[-1].split('_'))
                        startdate = dt.datetime.strptime(str(lkfiz['fname']).split('/')[-1].split('_')[1][:-2],'%Y%m%dT%H%M%S')
                        enddate   = dt.datetime.strptime(str(lkfiz['fname']).split('/')[-1].split('_')[2][:-2],'%Y%m%dT%H%M%S')
   
                    lkf_meta_year.append(np.array([startdate,
                                                   enddate,
                                                   lkfi.size,lkfiz['fname'],lkfiz['shape']]))
        
                # Add projected coordinates
                lkfim = []
        
                if self.mask_rgps:
                    if self.datatype=='rgps':
                        cov_int = cov_year[it,100:-100,100:-100]
                        #if it==0:
                        #    fig,ax = plt.subplots(1,1)
                        #    ax.pcolormesh(cov_int)
                        #    for iseg in lkfi:
                        #        ax.plot(iseg[:,1].astype('int'),iseg[:,0].astype('int'))
                    else:
                        # Coverage mask of all LKFs in one day
                        cov_int = cov_year[it,:,:][np.unravel_index(inds,x_rgps.shape)].reshape(x_cov.shape)
                        cov_int[mask_cov] = np.nan

                for iseg in lkfi:
                    if self.mask_rgps:
                        mask_seg = cov_int[iseg[:,0].astype('int'),
                                           iseg[:,1].astype('int')]
                        ind_mask = np.where(mask_seg)[0]
                        if np.any(np.diff(ind_mask)!=1):
                            ind_c = np.concatenate([np.array([-1]),
                                                    np.where(np.diff(ind_mask)!=1)[0],
                                                    np.array([ind_mask.size-1])])
                            for ic in range(ind_c.size-1):
                                if ind_c[ic]+1!=ind_c[ic+1]:
                                    iseg_c = iseg[ind_mask[ind_c[ic]+1]:ind_mask[ind_c[ic+1]]+1,:]
                                    isegm = np.rollaxis(np.stack(m(iseg_c[:,2],
                                                                   iseg_c[:,3])),1,0)
                                    lkfim.append(np.concatenate([iseg_c, isegm], axis=1))
                                    if polyfit:
                                        isegf = np.rollaxis(np.stack(lkf_poly_fit(lkfim[-1][:,self.indm0],
                                                                                  lkfim[-1][:,self.indm1],
                                                                                  poly_deg)),1,0)
                                        lkfim[-1] = np.concatenate([lkfim[-1], isegf], axis=1)

                        else:
                            iseg = iseg[mask_seg,:]
                            if iseg.shape[0]>1:
                                isegm = np.rollaxis(np.stack(m(iseg[:,2],
                                                               iseg[:,3])),1,0)
                                #print iseg, isegm
                                lkfim.append(np.concatenate([iseg, isegm], axis=1))
                                if polyfit:
                                    isegf = np.rollaxis(np.stack(lkf_poly_fit(lkfim[-1][:,self.indm0],
                                                                              lkfim[-1][:,self.indm1],
                                                                              poly_deg)),1,0)
                                    lkfim[-1] = np.concatenate([lkfim[-1], isegf], axis=1)
                    else:
                        isegm = np.rollaxis(np.stack(m(iseg[:,2],
                                                       iseg[:,3])),1,0)
                        lkfim.append(np.concatenate([iseg, isegm], axis=1))

                        if polyfit:
                            isegf = np.rollaxis(np.stack(lkf_poly_fit(lkfim[-1][:,self.indm0],
                                                                      lkfim[-1][:,self.indm1],
                                                                      poly_deg)),1,0)
                            lkfim[-1] = np.concatenate([lkfim[-1], isegf], axis=1)

                lkf_data_year.append(lkfim)

            if read_tracking:
                print( "Start reading tracking data of year: " + year_dic)
                track_dir = os.path.join(lkf_path,year_dic,track_dir_name)
                # Generate list of files
                trackfile_list = os.listdir(track_dir)
                trackfile_list.sort()

                track_year = []

                for itrack, trackfile_i in enumerate(trackfile_list[:len(lkf_data_year)-1]):
                    tracked_pairs = np.load(os.path.join(track_dir, trackfile_i))
                
                    #track_day = np.empty((lkf_meta_year[-1][itrack][2],2),dtype=object)
                    track_year.append(tracked_pairs)
                

            # Append to global data set
            lkf_dataset.append(lkf_data_year)
            lkf_meta.append(np.stack(lkf_meta_year))
            if read_tracking: lkf_track_data.append(track_year)

        # Store read and processed data
        self.lkf_dataset    = lkf_dataset
        self.lkf_meta       = lkf_meta
        self.lkf_track_data = lkf_track_data

        
        # Set all statistical fields to None
        self.length       = None
        self.density      = None
        self.curvature    = None
        self.deformation  = None
        self.intersection = None
        self.orientation  = None
        self.lifetime     = None
        self.growthrate   = None


# ---------------------------------------------------------------------------
# Stat helper classes (new flat-dataset API; also support legacy nested API)
# ---------------------------------------------------------------------------

class lkf_lengths:
    """Compute the arc-length of each detected LKF segment.

    Supports both the legacy nested ``stats.lkf_dataset`` (detected via
    ``hasattr(lkf, 'lkf_dataset')``) and the new flat ``dataset.lkf_dataset``
    whose ``lkfs[t]`` is a list of ``(N_cols, N_points)`` arrays.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with projected coordinates.  ``lkf.indm0`` and
        ``lkf.indm1`` must point to the x and y coordinate rows/columns.

    Attributes
    ----------
    lkf_length : np.ndarray of object
        1-D object array, one element per timestep slot.  Each element is a
        1-D float array of per-segment lengths in metres.  Slots without
        detected LKFs contain an empty array.
    """

    def __init__(self, lkf):
        if hasattr(lkf, 'lkf_dataset'):
            # Legacy nested year/day structure: segments are (N_points, N_cols)
            lkf_length = []
            for lkf_year in lkf.lkf_dataset:
                len_year = []
                for lkf_day in lkf_year:
                    len_day = []
                    for ilkf in lkf_day:
                        len_day.append(np.sum(np.sqrt(np.diff(ilkf[:, lkf.indm0])**2 +
                                                      np.diff(ilkf[:, lkf.indm1])**2)))
                    len_year.append(np.array(len_day)[np.isfinite(len_day)])
                lkf_length.append(len_year)
            self.lkf_length = np.array(lkf_length)
        else:
            # New flat structure: lkf.lkfs[t] is a list of (N_cols, N_points) arrays
            lkf_length = []
            for t in range(len(lkf.lkfs)):
                if lkf.lkfs[t] is None:
                    lkf_length.append(np.array([]))
                    continue
                len_t = [np.sum(np.sqrt(np.diff(seg[lkf.indm0, :])**2 +
                                        np.diff(seg[lkf.indm1, :])**2))
                         for seg in lkf.lkfs[t]]
                lkf_length.append(np.array(len_t))
            self.lkf_length = np.array(lkf_length, dtype=object)


    def plot(self, ax=None, bins=50, color='C0', fit=True):
        """Plot log-log PDF of LKF arc-lengths with optional power-law fit.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created if *None*.
        bins : int, optional
            Number of logarithmically spaced histogram bins.  Default 50.
        color : str, optional
            Marker and line colour.  Default ``'C0'``.
        fit : bool, optional
            Overlay a power-law fit line.  Default ``True``.

        Returns
        -------
        fig, ax : Figure and Axes objects.
        """
        all_len = _flatten_obj(self.lkf_length)
        all_len = all_len[all_len > 0] / 1e3  # convert to km
        log_edges = np.logspace(np.log10(all_len.min()),
                                np.log10(all_len.max()), bins + 1)
        counts, edges = np.histogram(all_len, bins=log_edges, density=True)
        centres = np.sqrt(edges[:-1] * edges[1:])
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure
        mask = counts > 0
        ax.plot(centres[mask], counts[mask], 'o', color=color, ms=4)
        if fit and mask.sum() > 2:
            coeff, fit_vals = power_law_fit(centres[mask], counts[mask])
            ax.plot(centres[mask], fit_vals, '-', color=color, lw=1.5,
                    label=r'$\alpha={:.2f}$'.format(coeff[0]))
            ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('LKF length (km)')
        ax.set_ylabel('PDF (km$^{-1}$)')
        ax.set_title('LKF length distribution')
        fig.tight_layout()
        return fig, ax


class lkf_density:
    """Compute the spatial density of LKF points on a regular projected grid.

    Bins all projected (x, y) points from each timestep into a 2-D histogram.
    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with projected coordinates.  ``lkf.indm0`` and
        ``lkf.indm1`` must point to the x and y coordinate rows/columns.
    res : float, optional
        Grid cell size in metres.  Default is 50 km.

    Attributes
    ----------
    lkf_density : np.ndarray, shape ``(n_timesteps, nx, ny)``
        Raw point counts per grid cell per timestep.
    x, y : np.ndarray
        2-D arrays of grid cell-centre coordinates (metres).
    xedg, yedg : np.ndarray
        1-D grid edge arrays (metres).
    res : float
        Grid resolution in metres.
    """

    def __init__(self, lkf, res=50e3):
        self.res = res

        if hasattr(lkf, 'lkf_dataset'):
            # Legacy: projection object has xmin/xmax/ymin/ymax
            xedg = np.arange(lkf.m.xmin, lkf.m.xmax, res)
            yedg = np.arange(lkf.m.ymin, lkf.m.ymax, res)
            y, x = np.meshgrid(yedg[1:], xedg[1:])
            self.x, self.y, self.xedg, self.yedg = x, y, xedg, yedg

            density = np.zeros((len(lkf.lkf_dataset), xedg.size - 1, yedg.size - 1))
            for iyear in range(len(lkf.lkf_dataset)):
                pts = np.concatenate([np.concatenate(lkfday)
                                      for lkfday in lkf.lkf_dataset[iyear]
                                      if len(lkfday) > 0])
                H, _, _ = np.histogram2d(pts[:, lkf.indm0], pts[:, lkf.indm1],
                                         bins=(xedg, yedg))
                density[iyear] = H
            self.lkf_density = density
        else:
            # New flat structure: determine grid extent from data
            xs, ys = [], []
            for t in range(len(lkf.lkfs)):
                if lkf.lkfs[t] is None:
                    continue
                for seg in lkf.lkfs[t]:
                    xs.append(seg[lkf.indm0, :])
                    ys.append(seg[lkf.indm1, :])
            all_x = np.concatenate(xs)
            all_y = np.concatenate(ys)

            xedg = np.arange(np.floor(all_x.min() / res) * res,
                             np.ceil(all_x.max() / res) * res + res, res)
            yedg = np.arange(np.floor(all_y.min() / res) * res,
                             np.ceil(all_y.max() / res) * res + res, res)
            y, x = np.meshgrid(yedg[1:], xedg[1:])
            self.x, self.y, self.xedg, self.yedg = x, y, xedg, yedg

            n_t = len(lkf.lkfs)
            density = np.zeros((n_t, xedg.size - 1, yedg.size - 1))
            for t in range(n_t):
                if lkf.lkfs[t] is None:
                    continue
                pts = np.concatenate([seg[[lkf.indm0, lkf.indm1], :]
                                      for seg in lkf.lkfs[t]], axis=1)
                H, _, _ = np.histogram2d(pts[0], pts[1], bins=(xedg, yedg))
                density[t] = H
            self.lkf_density = density

            # --- Ice-presence mask on the density grid ---
            # Project every model grid point (once) and bin the mean ice
            # concentration A.  Cells with mean A == 0 are open water or land
            # and should be masked in the plot.
            self.ice_mask = None
            try:
                lon_np = np.array(lkf.lon).flatten()
                lat_np = np.array(lkf.lat).flatten()
                # Project using the dataset's pyproj instance
                x_m, y_m = lkf.proj(lon_np, lat_np)
                # Mean A over detected timesteps
                t_det = [t for t in range(n_t) if lkf.lkfs[t] is not None]
                if t_det:
                    A_stack = np.stack([np.array(lkf.data.A[t]).flatten()
                                        for t in t_det], axis=0)
                    A_mean = np.nanmean(A_stack, axis=0)
                    valid = (np.isfinite(x_m) & np.isfinite(y_m) &
                             np.isfinite(A_mean))
                    H_A, _, _ = np.histogram2d(x_m[valid], y_m[valid],
                                               bins=(xedg, yedg),
                                               weights=A_mean[valid])
                    H_n, _, _ = np.histogram2d(x_m[valid], y_m[valid],
                                               bins=(xedg, yedg))
                    with np.errstate(invalid='ignore', divide='ignore'):
                        A_grid = np.where(H_n > 0, H_A / H_n, 0.0)
                    self.ice_mask = A_grid > 0   # True where sea ice present
            except Exception:
                pass  # no proj available — skip masking


    def plot(self, ax=None, cmap='YlOrRd', vmax=None, coastlines=True,
             proj_crs=None):
        """Plot the mean spatial density as a pcolormesh with optional coastlines.

        Averages the density over all timesteps and normalises to a rate
        (feature points per 100 km² per timestep).  If *cartopy* is
        available, coastlines and land fill are overlaid on the projected map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes, optional
            Axes to draw on.  When *coastlines* is ``True`` and *ax* is
            ``None``, a new cartopy GeoAxes is created automatically.
        cmap : str, optional
            Colormap name.  Default ``'YlOrRd'``.
        vmax : float, optional
            Upper colour-scale limit.  Derived from the 95th percentile if
            *None*.
        coastlines : bool, optional
            Overlay cartopy coastlines and land fill.  Default ``True``.
            Silently ignored if *cartopy* is not installed.
        proj_crs : cartopy.crs.CRS, optional
            Map projection for the cartopy axes.  Defaults to a North-Polar
            Stereographic projection (lat_0=90, lon_0=-45) matching the
            default ``pyproj`` projection used by ``compute_projected_coords``.

        Returns
        -------
        fig : Figure
        ax : Axes (GeoAxes when *coastlines* is ``True``).
        pcm : QuadMesh from pcolormesh.
        """
        # Mean over timesteps, normalise to density per 100 km²
        cell_area_km2 = (self.res / 1e3) ** 2
        dens = np.sum(self.lkf_density, axis=0) / self.lkf_density.shape[0]
        dens_norm = dens / cell_area_km2 * 100.0

        # Apply sea-ice mask: NaN out cells with no sea ice (A == 0)
        if self.ice_mask is not None:
            dens_norm = np.where(self.ice_mask, dens_norm, np.nan)

        xedg = self.xedg  # metres
        yedg = self.yedg

        if vmax is None:
            valid_dn = dens_norm[np.isfinite(dens_norm) & (dens_norm > 0)]
            vmax = np.nanpercentile(valid_dn, 95) if valid_dn.size > 0 else 1.0

        # Try to set up cartopy axes
        _cartopy_ok = False
        if coastlines:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                if proj_crs is None:
                    proj_crs = ccrs.NorthPolarStereo(central_longitude=-45,
                                                     true_scale_latitude=75)
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 6),
                                           subplot_kw={'projection': proj_crs})
                else:
                    fig = ax.figure
                _cartopy_ok = True
            except ImportError:
                pass

        if not _cartopy_ok:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 6))
            else:
                fig = ax.figure

        if _cartopy_ok:
            pcm = ax.pcolormesh(xedg, yedg, dens_norm.T,
                                cmap=cmap, vmin=0, vmax=vmax,
                                transform=proj_crs)
            ax.add_feature(cfeature.LAND, zorder=1, facecolor='#d0cfc7',
                           edgecolor='none')
            ax.add_feature(cfeature.COASTLINE, zorder=2, linewidth=0.5,
                           edgecolor='0.3')
            ax.set_title('Mean LKF density')
        else:
            pcm = ax.pcolormesh(xedg / 1e3, yedg / 1e3, dens_norm.T,
                                cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.set_aspect('equal')
            ax.set_title('Mean LKF density')

        fig.tight_layout()
        return fig, ax, pcm


class lkf_curvature:
    """Compute the end-to-end straight-line distance of each LKF segment.

    This is sometimes called the *chord length* or *end-to-end distance*.
    Comparing it to the arc-length (from :class:`lkf_lengths`) gives a
    straightness (sinuosity) measure: a ratio close to 1 means a nearly
    straight feature; a ratio close to 0 means a highly curved one.

    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with projected coordinates (``indm0``, ``indm1``).

    Attributes
    ----------
    lkf_curvature : np.ndarray of object
        1-D object array, one element per timestep slot.  Each element is a
        list of per-segment end-to-end distances in metres.
    """

    def __init__(self, lkf):
        if hasattr(lkf, 'lkf_dataset'):
            # Legacy: segments are (N_points, N_cols), first/last row = first/last point
            lkf_curvature = []
            for lkf_year in lkf.lkf_dataset:
                curv_year = []
                for lkf_day in lkf_year:
                    curv_day = []
                    for ilkf in lkf_day:
                        curv_day.append(np.sqrt(
                            (ilkf[0, lkf.indm0] - ilkf[-1, lkf.indm0])**2 +
                            (ilkf[0, lkf.indm1] - ilkf[-1, lkf.indm1])**2))
                    curv_year.append(curv_day)
                lkf_curvature.append(curv_year)
            self.lkf_curvature = lkf_curvature
        else:
            # New flat structure: segments are (N_cols, N_points)
            lkf_curvature = []
            for t in range(len(lkf.lkfs)):
                if lkf.lkfs[t] is None:
                    lkf_curvature.append([])
                    continue
                curv_t = [np.sqrt((seg[lkf.indm0, 0] - seg[lkf.indm0, -1])**2 +
                                  (seg[lkf.indm1, 0] - seg[lkf.indm1, -1])**2)
                          for seg in lkf.lkfs[t]]
                lkf_curvature.append(curv_t)
            self.lkf_curvature = np.array(lkf_curvature, dtype=object)


    def plot(self, arc_lengths=None, ax=None, color='C0', alpha=0.3, s=4):
        """Scatter plot of arc-length vs chord-length (end-to-end distance).

        If *arc_lengths* are provided (a :class:`lkf_lengths` instance or
        compatible object array), a scatter plot of chord vs arc-length is
        drawn on log-log axes with a 1:1 reference line.  Otherwise a
        histogram of raw chord lengths is shown.

        Parameters
        ----------
        arc_lengths : lkf_lengths or array-like, optional
            Arc-length data.  Pass ``dataset.lengths`` (a
            :class:`lkf_lengths` instance) or a numpy object array.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        color : str, optional
            Marker colour.  Default ``'C0'``.
        alpha : float, optional
            Marker transparency.  Default 0.3.
        s : float, optional
            Marker size.  Default 4.

        Returns
        -------
        fig, ax : Figure and Axes objects.
        """
        chord = _flatten_obj(self.lkf_curvature)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure

        if arc_lengths is not None:
            arr = getattr(arc_lengths, 'lkf_length', arc_lengths)
            arc = _flatten_obj(np.asarray(arr, dtype=object))
            n = min(len(chord), len(arc))
            chord, arc = chord[:n] / 1e3, arc[:n] / 1e3  # → km
            valid = (arc > 0) & (chord > 0)
            ax.scatter(arc[valid], chord[valid], s=s, color=color,
                       alpha=alpha, lw=0)
            lim = max(arc[valid].max(), chord[valid].max()) * 1.05
            ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='1:1')
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel('Arc-length (km)')
            ax.set_ylabel('Chord length (km)')
            ax.set_title('LKF arc-length vs chord-length')
            ax.legend(fontsize=9)
        else:
            chord_km = chord[chord > 0] / 1e3
            ax.hist(chord_km, bins=50, density=True, color=color,
                    edgecolor='white', lw=0.4)
            ax.set_xlabel('Chord length (km)')
            ax.set_ylabel('PDF (km$^{-1}$)')
            ax.set_title('LKF chord-length distribution')

        fig.tight_layout()
        return fig, ax


class lkf_deformation:
    """Compute mean divergence and shear rates along each LKF segment.

    For each segment, the mean of the per-point divergence and shear values
    is returned as a ``[mean_div, mean_shr]`` pair.

    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance.  ``lkf.indd0`` must point to the divergence column
        and ``lkf.indd1`` to the shear column.

    Attributes
    ----------
    lkf_deformation : np.ndarray of object
        1-D object array, one element per timestep slot.  Each element is a
        list of ``[mean_div, mean_shr]`` pairs (one per segment) in day⁻¹.
    """

    def __init__(self, lkf):
        if hasattr(lkf, 'lkf_dataset'):
            # Legacy: segments are (N_points, N_cols)
            lkf_deformation = []
            for lkf_year in lkf.lkf_dataset:
                defo_year = []
                for lkf_day in lkf_year:
                    defo_day = []
                    for ilkf in lkf_day:
                        defo_day.append([np.mean(ilkf[:, lkf.indd0]),
                                         np.mean(ilkf[:, lkf.indd1])])
                    defo_year.append(defo_day)
                lkf_deformation.append(defo_year)
            self.lkf_deformation = np.array(lkf_deformation)
        else:
            # New flat structure: segments are (N_cols, N_points)
            lkf_deformation = []
            for t in range(len(lkf.lkfs)):
                if lkf.lkfs[t] is None:
                    lkf_deformation.append([])
                    continue
                defo_t = [[np.mean(seg[lkf.indd0, :]), np.mean(seg[lkf.indd1, :])]
                          for seg in lkf.lkfs[t]]
                lkf_deformation.append(defo_t)
            self.lkf_deformation = np.array(lkf_deformation, dtype=object)


    def plot(self, ax=None, bins=80, cmap='Blues', vmin=None, vmax=None,
             clim_pct=99):
        """Plot a 2-D histogram of mean divergence vs mean shear rate.

        The colour scale is logarithmic so that both the dense core and the
        sparse tails are visible simultaneously.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        bins : int, optional
            Number of bins per axis.  Default 80.
        cmap : str, optional
            Colormap name.  Default ``'Blues'``.
        vmin, vmax : float, optional
            Colour-scale limits (counts).  If *None*, derived automatically.
        clim_pct : float, optional
            Upper percentile for *vmax* when *vmax* is *None*.

        Returns
        -------
        fig, ax : Figure and Axes objects.
        """
        from matplotlib.colors import LogNorm

        defo = _flatten_obj(self.lkf_deformation)
        if defo.ndim == 1:
            parts = []
            for a in self.lkf_deformation:
                if a is not None and len(a) > 0:
                    parts.append(np.array(a))
            defo = np.concatenate(parts) if parts else np.empty((0, 2))
        if defo.ndim == 1:
            defo = defo.reshape(-1, 2)

        div = defo[:, 0] * 1e3  # → 10⁻³ day⁻¹
        shr = defo[:, 1] * 1e3

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure

        lim = np.nanpercentile(np.abs(np.concatenate([div, shr])), 99)
        h, xe, ye = np.histogram2d(div, shr,
                                   bins=bins,
                                   range=[[-lim, lim], [0, lim]])
        if vmax is None:
            vmax = np.nanpercentile(h[h > 0], clim_pct) if np.any(h > 0) else 1
        if vmin is None:
            vmin = max(1, h[h > 0].min()) if np.any(h > 0) else 1

        norm = LogNorm(vmin=vmin, vmax=vmax)
        h_plot = np.where(h > 0, h, np.nan)
        pcm = ax.pcolormesh(xe, ye, h_plot.T, cmap=cmap, norm=norm)
        fig.colorbar(pcm, ax=ax, label='count')
        ax.axvline(0, color='k', lw=0.7, ls='--')
        ax.set_xlabel(r'Mean divergence ($10^{-3}$ day$^{-1}$)')
        ax.set_ylabel(r'Mean shear ($10^{-3}$ day$^{-1}$)')
        ax.set_title('LKF deformation rates')
        fig.tight_layout()
        return fig, ax


class lkf_intersection:
    """Find pairs of intersecting LKF segments and compute their crossing angle.

    Two segments are considered intersecting when at least one point of one
    segment falls within *dis_par* grid cells of a point of the other segment.
    The crossing angle is computed from linear fits to the *num_p* points on
    each segment that are closest to the contact point.

    Without vorticity (``use_vorticity=False``, default) the angle is folded
    into [0, 90] degrees (parallel=0, perpendicular=90).

    With ``use_vorticity=True`` the vorticity sign of each segment near the
    contact point is used to distinguish intersection types, producing angles
    in [0, 180] degrees:

    * **Type 0** (opposite-sign vorticity): a directed angle is computed so
      that the rotation from segment *i* to segment *j* is always positive,
      yielding one angle per pair in [0, 180].  Each pair is appended twice
      (both "sides" of the crossing).
    * **Type 1** (same-sign vorticity): both supplementary angles
      ``|θ_i − θ_j|`` and ``180 − |θ_i − θ_j|`` are appended, representing
      the two distinct wedges of the intersection.

    The intersection type for each entry is stored in ``lkf_interc_type``
    (0 = opposite vorticity, 1 = same vorticity) when ``use_vorticity=True``.

    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.  Vorticity-based mode is only supported for the
    new flat API where vorticity is stored as column ``indd2`` of each segment.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with projected coordinates (``indm0``, ``indm1``).
    num_p : int, optional
        Number of points on each side of the contact used for the linear fit.
        Default is 10.
    dis_par : int, optional
        Search radius in grid cells for neighbour detection.  Default is 1.
    use_vorticity : bool, optional
        If ``True``, use vorticity sign to distinguish intersection types and
        produce angles in [0, 180].  Default is ``False``.

    Attributes
    ----------
    lkf_interc : np.ndarray of object
        Crossing angles in degrees, one list per timestep slot.
    lkf_interc_par : np.ndarray of object
        Corresponding segment index pairs ``[i, j]`` with ``i < j``.
    lkf_interc_type : np.ndarray of object
        Only present when ``use_vorticity=True``.  Integer type per angle
        entry: 0 = opposite-vorticity pair, 1 = same-vorticity pair.
    """

    @staticmethod
    def _angle_near_point(seg_rows, seg_cols, seg_xy, indm0, indm1, intcp, num_p,
                          is_legacy, indd2=None):
        """Return (angle, vor_mean) of *seg* at the point nearest to *intcp*.

        *vor_mean* is ``None`` when *indd2* is ``None``.

        The local orientation is estimated by a degree-1 linear fit of the
        projected (x, y) coordinates against the point index t.  Instead of
        calling ``np.polyfit`` (which builds a Vandermonde matrix and runs
        lstsq/QR), the degree-1 slope is computed directly as the ratio of
        dot products:  slope = dot(t−t̄, z) / dot(t−t̄, t−t̄).  This is
        numerically equivalent to polyfit but avoids the O(n²) matrix
        allocation and SVD for small windows.
        """
        idx = int(np.argmin(np.sqrt((seg_rows - intcp[0])**2 + (seg_cols - intcp[1])**2)))
        if is_legacy:
            n = seg_xy.shape[0]
            sl = slice(max(0, idx - num_p), min(n, idx + num_p + 1))
            x_sl = seg_xy[sl, indm0].astype(float)
            y_sl = seg_xy[sl, indm1].astype(float)
            vor_mean = float(np.mean(seg_xy[sl, indd2])) if indd2 is not None else None
        else:
            n = seg_xy.shape[1]
            sl = slice(max(0, idx - num_p), min(n, idx + num_p + 1))
            x_sl = seg_xy[indm0, sl].astype(float)
            y_sl = seg_xy[indm1, sl].astype(float)
            vor_mean = float(np.mean(seg_xy[indd2, sl])) if indd2 is not None else None
        t = np.arange(x_sl.size, dtype=float)
        t -= t.mean()
        tt = np.dot(t, t)
        if tt == 0:
            p = np.nan
        else:
            slope_x = np.dot(t, x_sl) / tt
            slope_y = np.dot(t, y_sl) / tt
            p = slope_y / slope_x if slope_x != 0 else np.nan
        angle = 90. if np.isnan(p) else np.arctan(p) / np.pi * 180.
        return angle, vor_mean

    def _process_timestep(self, segs, indm0, indm1, num_p, dis_par, is_legacy,
                          use_vorticity=False, indd2=None):
        """Core intersection logic shared by both API variants.

        Uses a sparse dict-based neighbourhood search instead of dense boolean
        grids.  This avoids the O(grid_size) scan per segment that previously
        dominated runtime, and also fixes a latent correctness bug where the
        accumulative ``lkf_map += iseg`` encoding produced wrong partner IDs
        when two or more segments share a grid cell.
        """
        if is_legacy:
            get_rows = lambda s: s[:, 0].astype(int)
            get_cols = lambda s: s[:, 1].astype(int)
        else:
            get_rows = lambda s: s[0, :].astype(int)
            get_cols = lambda s: s[1, :].astype(int)

        all_r = np.concatenate([get_rows(s) for s in segs])
        all_c = np.concatenate([get_cols(s) for s in segs])
        grid_shape = (int(all_r.max()) + dis_par + 2,
                      int(all_c.max()) + dis_par + 2)
        gs1 = grid_shape[1]

        # Precompute ravel indices per segment + reverse map pixel → seg list.
        # Replaces the dense lkf_map grid; exact (no accumulation aliasing).
        seg_ravel = []
        pixel_to_segs = defaultdict(list)
        for iseg, seg in enumerate(segs):
            rv = get_rows(seg) * gs1 + get_cols(seg)
            seg_ravel.append(rv)
            for px in rv.tolist():
                pixel_to_segs[px].append(iseg)

        # Vectorised offset grid for neighbourhood dilation: (2·dp+1)² offsets.
        dr = np.arange(-dis_par, dis_par + 1)
        drow = np.repeat(dr, len(dr))   # shape (k²,): row offsets
        dcol = np.tile(dr, len(dr))     # shape (k²,): col offsets

        intc_ang = []
        intc_par = []
        intc_type = []
        _indd2 = indd2 if use_vorticity else None

        for iseg, seg_i in enumerate(segs):
            rows_i = get_rows(seg_i)
            cols_i = get_cols(seg_i)
            rv_i   = seg_ravel[iseg]

            # Expand neighbourhood: all in-bounds pixels within dis_par of seg_i.
            # Broadcasting (N_pts, k²) avoids the Python for-ix/iy loop.
            r_exp = (rows_i[:, None] + drow[None, :]).ravel()
            c_exp = (cols_i[:, None] + dcol[None, :]).ravel()
            valid = ((r_exp >= 0) & (r_exp < grid_shape[0]) &
                     (c_exp >= 0) & (c_exp < grid_shape[1]))
            expanded = np.unique(r_exp[valid] * gs1 + c_exp[valid])
            # Remove own pixels
            expanded = expanded[~np.isin(expanded, rv_i)]

            # Collect partner segments and their contact pixel coords.
            # Replaces np.where(search_ind & (lkf_map != 0)) scan.
            partners_contact = defaultdict(list)   # jseg → [(r, c), …]
            for px in expanded.tolist():
                for jseg in pixel_to_segs.get(px, []):
                    if jseg > iseg:
                        partners_contact[jseg].append(
                            (int(px // gs1), int(px % gs1)))

            for pari in sorted(partners_contact):
                cpts = partners_contact[pari]
                cpts_r = np.array([p[0] for p in cpts])
                cpts_c = np.array([p[1] for p in cpts])
                # Best contact point: nearest to any pixel of seg_i (vectorised)
                min_dists = np.sqrt(
                    (cpts_r[:, None] - rows_i[None, :])**2 +
                    (cpts_c[:, None] - cols_i[None, :])**2
                ).min(axis=1)
                intcp = cpts[int(np.argmin(min_dists))]

                ang_i, vor0 = self._angle_near_point(rows_i, cols_i, seg_i,
                                                     indm0, indm1, intcp, num_p,
                                                     is_legacy, _indd2)
                seg_p = segs[pari]
                ang_ii, vor1 = self._angle_near_point(get_rows(seg_p), get_cols(seg_p), seg_p,
                                                      indm0, indm1, intcp, num_p,
                                                      is_legacy, _indd2)
                if use_vorticity and vor0 is not None and vor1 is not None:
                    pair_arr = np.array([iseg, pari])
                    if vor0 > 0 and vor1 < 0:
                        if ang_ii < ang_i:
                            ang_ii += 180
                        intc_ang.extend([ang_ii - ang_i, ang_ii - ang_i])
                        intc_type.extend([0, 0])
                    elif vor1 > 0 and vor0 < 0:
                        if ang_i < ang_ii:
                            ang_i += 180
                        intc_ang.extend([ang_i - ang_ii, ang_i - ang_ii])
                        intc_type.extend([0, 0])
                    else:
                        intc_ang.extend([abs(ang_ii - ang_i),
                                         180 - abs(ang_ii - ang_i)])
                        intc_type.extend([1, 1])
                    intc_par.extend([pair_arr, pair_arr])
                else:
                    angdiff = abs(ang_ii - ang_i)
                    if angdiff > 90:
                        angdiff = 180 - angdiff
                    intc_ang.append(angdiff)
                    intc_par.append(np.array([iseg, pari]))
        return intc_ang, intc_par, intc_type

    def __init__(self, lkf, num_p=10, dis_par=1, use_vorticity=True):
        self.use_vorticity = use_vorticity
        self._lkf = lkf  # back-reference used by plot(lifetime=<int>)
        indd2 = getattr(lkf, 'indd2', None)
        if hasattr(lkf, 'lkf_dataset'):
            lkf_interc = []
            lkf_interc_par = []
            for lkf_year in lkf.lkf_dataset:
                ang_year = []
                par_year = []
                for lkf_day in lkf_year:
                    if len(lkf_day) == 0:
                        ang_year.append([]); par_year.append([]); continue
                    a, p, _ = self._process_timestep(lkf_day, lkf.indm0, lkf.indm1,
                                                     num_p, dis_par, is_legacy=True,
                                                     use_vorticity=False)
                    ang_year.append(a); par_year.append(p)
                lkf_interc.append(ang_year)
                lkf_interc_par.append(par_year)
            self.lkf_interc = np.array(lkf_interc)
            self.lkf_interc_par = np.array(lkf_interc_par)
        else:
            lkf_interc = []
            lkf_interc_par = []
            lkf_interc_type = []
            for t in range(len(lkf.lkfs)):
                if lkf.lkfs[t] is None or len(lkf.lkfs[t]) == 0:
                    lkf_interc.append([]); lkf_interc_par.append([])
                    if use_vorticity:
                        lkf_interc_type.append([])
                    continue
                a, p, ty = self._process_timestep(lkf.lkfs[t], lkf.indm0, lkf.indm1,
                                                  num_p, dis_par, is_legacy=False,
                                                  use_vorticity=use_vorticity,
                                                  indd2=indd2)
                lkf_interc.append(a)
                lkf_interc_par.append(p)
                if use_vorticity:
                    lkf_interc_type.append(ty)
            self.lkf_interc = np.array(lkf_interc, dtype=object)
            self.lkf_interc_par = np.array(lkf_interc_par, dtype=object)
            if use_vorticity:
                self.lkf_interc_type = np.array(lkf_interc_type, dtype=object)

    def plot(self, ax=None, bins=18, color0='C0', color1='C1', lifetime=None,
             intersection_type='both', label=None):
        """Plot a histogram of LKF intersection angles.

        Without vorticity (``use_vorticity=False``) angles are folded into
        [0, 90] and a single histogram is shown.

        With ``use_vorticity=True`` angles span [0, 180] and two overlapping
        bars distinguish:

        * **Type 0** (opposite vorticity — diverging wedge), colour *color0*.
        * **Type 1** (same vorticity — shear wedge), colour *color1*.

        A vertical dashed line marks the median angle.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        bins : int, optional
            Number of histogram bins.  Default 18 (5° bins for 90°; 10° for 180°).
        color0, color1 : str, optional
            Bar colours for type 0 and type 1 respectively.
        lifetime : int or None, optional
            When provided, restrict the histogram to intersection pairs where
            **both** segments have the given lifetime value (e.g. ``lifetime=1``
            for newly formed LKFs only).  Requires that ``compute_lifetime()``
            has been called on the parent dataset.
        intersection_type : {'both', 'opp', 'same'}, optional
            Which intersection types to include (only effective when
            ``use_vorticity=True``).  ``'opp'`` shows only Type 0 (opposite
            vorticity); ``'same'`` shows only Type 1 (same vorticity);
            ``'both'`` (default) shows both.
        label : str or None, optional
            Label prefix attached to histogram bars.  When ``None`` the
            default type descriptions are used.  When set, type-0 bars are
            labelled ``'{label} (opp.)'``, type-1 bars ``'{label} (same)'``
            (with vorticity), or just ``label`` for single-bar mode.

        Returns
        -------
        fig, ax : Figure and Axes objects.
        """
        use_vorticity = getattr(self, 'use_vorticity', False)

        all_ang = _flatten_obj(self.lkf_interc)
        # Populate type array early so filters can use it
        all_type = None
        if use_vorticity and hasattr(self, 'lkf_interc_type'):
            all_type = _flatten_obj(self.lkf_interc_type)

        # Optionally filter to pairs where both segments have a given lifetime.
        # ``lifetime`` may be:
        #   - an int/float : filter value; lifetime arrays come from self._lkf
        #   - a lkf_lifetime instance (legacy): use it directly, filter value == 1
        if lifetime is not None and hasattr(self, 'lkf_interc_par'):
            if isinstance(lifetime, (int, float, np.integer, np.floating)):
                lt_val = int(lifetime)
                lt_obj = getattr(getattr(self, '_lkf', None), 'lifetime', None)
                if lt_obj is None:
                    raise RuntimeError(
                        "lifetime filter requires compute_lifetime() to have been called first.")
            else:
                # Legacy: caller passes a lkf_lifetime object directly
                lt_obj = lifetime
                lt_val = 1
            filtered_ang = []
            filtered_type = []
            for t in range(len(self.lkf_interc)):
                if t >= len(self.lkf_interc_par):
                    continue
                angles = self.lkf_interc[t]
                pairs  = self.lkf_interc_par[t]
                lts    = lt_obj.lkf_lifetime[t] if t < len(lt_obj.lkf_lifetime) else np.array([])
                ty_t   = (self.lkf_interc_type[t]
                          if (use_vorticity and hasattr(self, 'lkf_interc_type')
                              and t < len(self.lkf_interc_type))
                          else [])
                for k, (ang, pair) in enumerate(zip(angles, pairs)):
                    i_lt = lts[int(pair[0])] if int(pair[0]) < len(lts) else np.nan
                    j_lt = lts[int(pair[1])] if int(pair[1]) < len(lts) else np.nan
                    if i_lt == lt_val and j_lt == lt_val:
                        filtered_ang.append(ang)
                        if use_vorticity and k < len(ty_t):
                            filtered_type.append(ty_t[k])
            all_ang = np.array(filtered_ang)
            all_type = np.array(filtered_type) if (use_vorticity and filtered_type) else None

        # Filter by intersection type (only meaningful with vorticity)
        if intersection_type != 'both' and use_vorticity and all_type is not None and len(all_type):
            type_val = 0 if intersection_type == 'opp' else 1
            mask = all_type == type_val
            all_ang = all_ang[mask]
            all_type = all_type[mask]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure

        if use_vorticity and all_type is not None and len(all_type):
            bin_edges = np.linspace(0, 180, bins + 1)
            ang0 = all_ang[all_type == 0]
            ang1 = all_ang[all_type == 1]
            if label is not None:
                lbl0 = f'{label} (opp.)' if intersection_type == 'both' else label
                lbl1 = f'{label} (same)' if intersection_type == 'both' else label
            else:
                lbl0 = 'Type 0 (opp. vorticity)'
                lbl1 = 'Type 1 (same vorticity)'
            if len(ang0):
                ax.hist(ang0, bins=bin_edges, density=True, alpha=0.7,
                        color=color0, edgecolor='white', lw=0.4, label=lbl0)
            if len(ang1):
                ax.hist(ang1, bins=bin_edges, density=True, alpha=0.7,
                        color=color1, edgecolor='white', lw=0.4, label=lbl1)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 180)
            ax.set_xlabel('Intersection angle (°)')
        else:
            bin_edges = np.linspace(0, 90, bins + 1)
            ax.hist(all_ang[all_ang >= 0], bins=bin_edges, density=True,
                    color=color0, edgecolor='white', lw=0.4, label=label)
            ax.set_xlim(0, 90)
            ax.set_xlabel('Intersection angle (°)')

        if len(all_ang):
            med = np.median(all_ang[all_ang >= 0])
            ax.axvline(med, color=color0, ls='--', lw=1,
                       label=f'median = {med:.1f}°')
            ax.legend(fontsize=8)

        ax.set_ylabel('PDF')
        ax.set_title('LKF intersection angles')
        fig.tight_layout()
        return fig, ax


class lkf_orientation:
    """Compute the orientation angle of each LKF segment.

    For each segment a linear polynomial is fitted to the projected (x, y)
    coordinates and the resulting slope is converted to an angle in degrees
    in the range (-90, 90].  An angle near 0 means East-West orientation;
    near +/-90 means North-South.

    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with projected coordinates (``indm0``, ``indm1``).

    Attributes
    ----------
    lkf_orientation : np.ndarray of object
        1-D object array, one element per timestep slot.  Each element is a
        1-D float array of per-segment orientation angles in degrees.
    """

    def __init__(self, lkf):
        if hasattr(lkf, 'lkf_dataset'):
            # Legacy: segments are (N_points, N_cols)
            lkf_orientation = []
            for lkf_year in lkf.lkf_dataset:
                ori_year = []
                for lkf_day in lkf_year:
                    ori_day = []
                    for ilkf in lkf_day:
                        px, py = lkf_poly_fit_p(ilkf[:, lkf.indm0], ilkf[:, lkf.indm1], 1)
                        p = py[0] / px[0]
                        ori_day.append(90. if np.isnan(p) else np.arctan(p) / np.pi * 180.)
                    ori_year.append(np.array(ori_day))
                lkf_orientation.append(ori_year)
            self.lkf_orientation = np.array(lkf_orientation)
        else:
            # New flat structure: segments are (N_cols, N_points)
            lkf_orientation = []
            for t in range(len(lkf.lkfs)):
                if lkf.lkfs[t] is None:
                    lkf_orientation.append(np.array([]))
                    continue
                angles = []
                for seg in lkf.lkfs[t]:
                    px, py = lkf_poly_fit_p(seg[lkf.indm0, :], seg[lkf.indm1, :], 1)
                    p = py[0] / px[0]
                    angles.append(90. if np.isnan(p) else np.arctan(p) / np.pi * 180.)
                lkf_orientation.append(np.array(angles))
            self.lkf_orientation = np.array(lkf_orientation, dtype=object)

    def plot(self, ax=None, bins=36, color='C0', density=True):
        """Plot a polar rose diagram of LKF orientations.

        Orientation angles are in (−90, 90] degrees; the rose is mirrored to
        the full circle so that opposite directions (e.g. +45° and −135°) are
        combined.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            A *polar* axes to draw on.  A new polar figure is created if
            *None*.
        bins : int, optional
            Number of directional bins over the half-circle [0°, 180°).
            Default 36 (5° bins).
        color : str, optional
            Bar colour.  Default ``'C0'``.
        density : bool, optional
            Normalise bars to probability density.  Default ``True``.

        Returns
        -------
        fig, ax : Figure and Axes objects.
        """
        all_ori = _flatten_obj(self.lkf_orientation)
        all_ori = all_ori[np.isfinite(all_ori)]

        # Map from (-90, 90] to [0, 180)
        ori_pos = all_ori % 180

        bin_edges = np.linspace(0, 180, bins + 1)
        counts, edges = np.histogram(ori_pos, bins=bin_edges,
                                     density=density)
        theta = np.deg2rad((edges[:-1] + edges[1:]) / 2)
        width = np.deg2rad(np.diff(edges)[0])

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5),
                                   subplot_kw={'projection': 'polar'})
        else:
            fig = ax.figure

        # Mirror to full circle
        theta_full = np.concatenate([theta, theta + np.pi])
        counts_full = np.concatenate([counts, counts])
        ax.bar(theta_full, counts_full, width=width, color=color,
               alpha=0.8, edgecolor='white', lw=0.4)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.set_title('LKF orientation', pad=15)
        fig.tight_layout()
        return fig, ax


class lkf_lifetime:
    """Compute the lifetime of each LKF segment in timesteps.

    Lifetime is initialised to 1 for every detected segment.  For each
    consecutive tracked pair (t → t+1), the lifetime of the matched segment
    at t+1 is incremented by the lifetime of its predecessor at t.  As a
    result, a feature continuously tracked for *N* timesteps reaches
    lifetime *N*.

    Requires tracking data (``track_lkfs()`` must have been called on the
    dataset before instantiating this class).

    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with tracking output available.

    Attributes
    ----------
    lkf_lifetime : np.ndarray of object
        Object array, one element per timestep slot.  Each element is a
        1-D float array of per-segment lifetimes (≥ 1).
    """

    def __init__(self, lkf):
        if hasattr(lkf, 'lkf_dataset'):
            # Legacy nested API
            lkf_lifetime_out = []
            for iyear, lkf_year in enumerate(lkf.lkf_dataset):
                life_year = [np.ones(len(segs)) for segs in lkf_year]
                for it, itrack in enumerate(lkf.lkf_track_data[iyear]):
                    if itrack.size > 0:
                        life_year[it + 1][itrack[:, 1].astype(int)] += \
                            life_year[it][itrack[:, 0].astype(int)]
                lkf_lifetime_out.append(life_year)
            self.lkf_lifetime = np.array(lkf_lifetime_out)
        else:
            # New flat API
            if not hasattr(lkf, 'track_output_path'):
                raise RuntimeError(
                    "No tracking data found. Call track_lkfs() before compute_lifetime().")

            life = []
            for t in range(len(lkf.lkfs)):
                life.append(np.ones(len(lkf.lkfs[t])) if lkf.lkfs[t] is not None else None)

            # File suffix NNN (1-based) maps to lkfs[NNN-1] (0-based outer index).
            track_path = lkf.track_output_path
            track_files = sorted([f for f in os.listdir(track_path)
                                   if f.startswith('lkf') and f.endswith('.npy')])
            for tf in track_files:
                stem = tf[:-4]  # strip .npy
                parts = stem.split('_to_')
                t0 = int(parts[0].split('_')[-1]) - 1
                t1 = int(parts[1].split('_')[-1]) - 1
                pairs = np.load(os.path.join(track_path, tf))
                # Non-empty case: shape (N_pairs, 2)
                if pairs.size > 0 and pairs.ndim == 2 and pairs.shape[1] == 2:
                    if life[t0] is not None and life[t1] is not None:
                        life[t1][pairs[:, 1].astype(int)] += \
                            life[t0][pairs[:, 0].astype(int)]

            self.lkf_lifetime = np.array(life, dtype=object)

    def plot(self, ax=None, color='C0', fit=True):
        """Plot a histogram of LKF lifetimes with an optional exponential fit.

        The y-axis uses a logarithmic scale so that an exponential decay
        (geometric distribution) appears as a straight line.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        color : str, optional
            Bar colour.  Default ``'C0'``.
        fit : bool, optional
            Overlay an exponential fit.  Default ``True``.

        Returns
        -------
        fig, ax : Figure and Axes objects.
        """
        all_life = _flatten_obj(self.lkf_lifetime)
        all_life = all_life[np.isfinite(all_life) & (all_life >= 1)]
        max_life = int(all_life.max()) if len(all_life) else 1
        bins = np.arange(0.5, max_life + 1.5, 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure

        counts, edges = np.histogram(all_life, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, counts, width=0.8, color=color,
               edgecolor='white', lw=0.4)

        if fit and len(counts) > 1:
            mask = counts > 0
            xf = centres[mask]
            yf = counts[mask].astype(float)
            coeff = np.polyfit(xf, np.log(yf), 1)
            x_fit = np.linspace(centres[0], centres[-1], 200)
            ax.plot(x_fit, np.exp(np.polyval(coeff, x_fit)), '-',
                    color=color, lw=1.5,
                    label=r'$\lambda={:.2f}$'.format(-coeff[0]))
            ax.legend(fontsize=9)

        ax.set_yscale('log')
        ax.set_xlabel('LKF lifetime (timesteps)')
        ax.set_ylabel('Count')
        ax.set_title('LKF lifetime distribution')
        fig.tight_layout()
        return fig, ax


class lkf_growthrate:
    """Compute per-segment growth and shrink rates from consecutive tracked pairs.

    For each segment at timestep t+1 that is tracked from a predecessor at t:

    * **growth[t][j]**: arc-length of the non-overlapping part of the
      predecessor (the portion of the older feature *not* carried over to the
      new position — i.e., how much the feature "lost" at its old location).
    * **shrink[t][j]**: arc-length of the non-overlapping part of the
      successor (the new extent not present in the predecessor — i.e., how
      much the feature "grew" at its new location).

    For newly born segments (lifetime = 1) both values are set to the full
    arc-length of the segment.

    Requires prior calls to ``compute_lifetime()`` and ``compute_lengths()``
    on the dataset.

    Supports both the legacy nested ``stats.lkf_dataset`` and the new flat
    ``dataset.lkf_dataset``.

    Parameters
    ----------
    lkf : lkf_dataset
        Dataset instance with ``lifetime``, ``lengths``, and tracking output.

    Attributes
    ----------
    lkf_growth : np.ndarray of object
        Object array (one entry per timestep from t=1 onward).  Each entry
        is a 1-D float array of per-segment growth values in metres
        (NaN for segments that have no tracked predecessor and lifetime > 1).
    lkf_shrink : np.ndarray of object
        Same shape as ``lkf_growth`` but for the shrink rate.
    """

    @staticmethod
    def _arc_length(xy_2xN):
        """Arc-length of a (2, N) array with NaN masking."""
        diffs = np.diff(xy_2xN, axis=1)           # (2, N-1)
        return float(np.nansum(np.sqrt(np.sum(diffs ** 2, axis=0))))

    def __init__(self, lkf):
        from .tracking import compute_MHD_segment

        if hasattr(lkf, 'lkf_dataset'):
            # Legacy nested API (kept for compatibility)
            lkf_growth_out = []
            lkf_shrink_out = []
            for iyear, lkf_year in enumerate(lkf.lkf_dataset):
                growth_year = [np.full(len(segs), np.nan) for segs in lkf_year[1:]]
                shrink_year = [np.full(len(segs), np.nan) for segs in lkf_year[1:]]
                for iday, day_track in enumerate(lkf.lkf_track_data[iyear]):
                    for itrack in day_track:
                        if len(itrack) == 0:
                            continue
                        a_idx = int(itrack[0])
                        b_idx = int(itrack[1])
                        segA = lkf_year[iday][a_idx]
                        segB = lkf_year[iday + 1][b_idx]
                        _, ov, [A_o, B_o] = compute_MHD_segment(
                            segA[:, :2].T, segB[:, :2].T,
                            overlap_thres=1.5, angle_thres=25,
                            return_overlap=True, return_overlaping_area=True,
                            mask_instead=True)
                        A_xy = segA[:, lkf.indm0:lkf.indm1 + 1].copy().T  # (2, N)
                        B_xy = segB[:, lkf.indm0:lkf.indm1 + 1].copy().T
                        A_xy[:, A_o] = np.nan
                        B_xy[:, B_o] = np.nan
                        growth_year[iday][b_idx] = self._arc_length(A_xy)
                        shrink_year[iday][b_idx] = self._arc_length(B_xy)
                    ind_new = lkf.lifetime.lkf_lifetime[iyear][iday + 1] == 1
                    growth_year[iday][ind_new] = lkf.lengths.lkf_length[iyear][iday + 1][ind_new]
                    shrink_year[iday][ind_new] = lkf.lengths.lkf_length[iyear][iday + 1][ind_new]
                lkf_growth_out.append(growth_year)
                lkf_shrink_out.append(shrink_year)
            self.lkf_growth = np.array(lkf_growth_out)
            self.lkf_shrink = np.array(lkf_shrink_out)

        else:
            # New flat API
            if not hasattr(lkf, 'track_output_path'):
                raise RuntimeError(
                    "No tracking data found. Call track_lkfs() before compute_growthrate().")

            n_t = len(lkf.lkfs)
            growth = [np.full(len(lkf.lkfs[t]), np.nan)
                      if lkf.lkfs[t] is not None else None
                      for t in range(n_t)]
            shrink = [np.full(len(lkf.lkfs[t]), np.nan)
                      if lkf.lkfs[t] is not None else None
                      for t in range(n_t)]

            track_path = lkf.track_output_path
            track_files = sorted([f for f in os.listdir(track_path)
                                   if f.startswith('lkf') and f.endswith('.npy')])
            for tf in track_files:
                stem = tf[:-4]
                parts = stem.split('_to_')
                t0 = int(parts[0].split('_')[-1]) - 1
                t1 = int(parts[1].split('_')[-1]) - 1
                pairs = np.load(os.path.join(track_path, tf))
                if pairs.size == 0 or pairs.ndim != 2 or pairs.shape[1] != 2:
                    # Empty tracking step — all segments at t1 are new
                    if lkf.lkfs[t1] is not None:
                        ind_new = lkf.lifetime.lkf_lifetime[t1] == 1
                        growth[t1][ind_new] = lkf.lengths.lkf_length[t1][ind_new]
                        shrink[t1][ind_new] = lkf.lengths.lkf_length[t1][ind_new]
                    continue

                segs_t0 = lkf.lkfs[t0]
                segs_t1 = lkf.lkfs[t1]
                for a_idx, b_idx in pairs.astype(int):
                    segA = segs_t0[a_idx]   # (N_cols, N_pts_A)
                    segB = segs_t1[b_idx]   # (N_cols, N_pts_B)
                    _, ov, [mask_A, mask_B] = compute_MHD_segment(
                        segA[:2, :], segB[:2, :],
                        overlap_thres=1.5, angle_thres=25,
                        return_overlap=True, return_overlaping_area=True,
                        mask_instead=True)
                    A_xy = segA[lkf.indm0:lkf.indm1 + 1, :].copy()  # (2, N_pts_A)
                    B_xy = segB[lkf.indm0:lkf.indm1 + 1, :].copy()  # (2, N_pts_B)
                    A_xy[:, mask_A] = np.nan
                    B_xy[:, mask_B] = np.nan
                    growth[t1][b_idx] = self._arc_length(A_xy)
                    shrink[t1][b_idx] = self._arc_length(B_xy)

                # Newly born segments at t1 (lifetime=1)
                if lkf.lkfs[t1] is not None:
                    ind_new = lkf.lifetime.lkf_lifetime[t1] == 1
                    growth[t1][ind_new] = lkf.lengths.lkf_length[t1][ind_new]
                    shrink[t1][ind_new] = lkf.lengths.lkf_length[t1][ind_new]

            self.lkf_growth = np.array(growth, dtype=object)
            self.lkf_shrink = np.array(shrink, dtype=object)

    def plot(self, ax=None, bins=50, color_growth='C0', color_shrink='C3',
             fit=True):
        """Plot PDFs of the growth and shrink rates on a two-panel figure.

        Both rates are expressed in km per timestep.  Rates are log-binned
        and displayed on a log-log scale; a power-law fit is overlaid on
        each panel.

        Parameters
        ----------
        ax : array of two matplotlib.axes.Axes, optional
            A pair of axes (growth, shrink).  A new 1×2 figure is created
            if *None*.
        bins : int, optional
            Number of logarithmically spaced histogram bins.  Default 50.
        color_growth, color_shrink : str, optional
            Colours for the growth and shrink panels.
        fit : bool, optional
            Overlay power-law fits.  Default ``True``.

        Returns
        -------
        fig : Figure
        axes : ndarray of two Axes objects.
        """
        def _panel(data, ax, color, label):
            data = data[np.isfinite(data) & (data > 0)] / 1e3  # → km
            if len(data) < 2:
                ax.set_title(label + ' (no data)')
                return
            log_edges = np.logspace(np.log10(data.min()),
                                    np.log10(data.max()), bins + 1)
            counts, edges = np.histogram(data, bins=log_edges, density=True)
            centres = np.sqrt(edges[:-1] * edges[1:])
            mask = counts > 0
            ax.plot(centres[mask], counts[mask], 'o', color=color, ms=4)
            if fit and mask.sum() > 2:
                coeff, fit_vals = power_law_fit(centres[mask], counts[mask])
                ax.plot(centres[mask], fit_vals, '-', color=color, lw=1.5,
                        label=r'$\alpha={:.2f}$'.format(coeff[0]))
                ax.legend(fontsize=9)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f'{label} (km)')
            ax.set_ylabel('PDF (km$^{-1}$)')
            ax.set_title(f'LKF {label.lower()} distribution')

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        else:
            fig = ax[0].figure
            axes = ax

        growth_flat = _flatten_obj(self.lkf_growth)
        shrink_flat = _flatten_obj(self.lkf_shrink)
        _panel(growth_flat, axes[0], color_growth, 'Growth rate')
        _panel(shrink_flat, axes[1], color_shrink, 'Shrink rate')
        fig.tight_layout()
        return fig, axes
