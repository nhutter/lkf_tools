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


# Define reading function
def load_lkf_dataset(lkf_path,output_path=None,datatype=None,subdirs=None,name=None,
                     read_tracking=True,m=None,polyfit=True,poly_deg=6,force_reread=False,
                     mask_rgps=False):
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
