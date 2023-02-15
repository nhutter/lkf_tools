import numpy as np
import matplotlib.pylab as plt
import os
import sys
import datetime as dt
from mpl_toolkits.basemap import Basemap
import scipy
import matplotlib as mpl
from netCDF4 import Dataset, MFDataset
import pickle
from scipy.spatial import cKDTree

# Self-written functions
from read_RGPS import *
from model_utils import *
from lkf_utils import *
from griddata_fast import griddata_fast
from local_paths import *

# Suppress rank warnings for fit to polynom
import warnings
warnings.simplefilter('ignore', np.RankWarning)


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



#Projection used for analysielif datatype == 'mitgcm_2km_test_xy':
#m = mSSMI()
m = Basemap(projection='stere',lat_ts=70,lat_0=90,lon_0=-45,resolution='l',llcrnrlon=-115,llcrnrlat=64,urcrnrlon=105,urcrnrlat=65,ellps='WGS84')


# # Read in LKF data set
# force_reread = False
# read_tracking = True


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


# Define LKF dataset class as container for LKF data and processed information

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
            m = Basemap(projection='stere',lat_ts=70,lat_0=90,lon_0=-45,resolution='l',
                        llcrnrlon=-115,llcrnrlat=64,urcrnrlon=105,urcrnrlat=65,ellps='WGS84')
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
        else:
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



    def gen_length(self,overwrite=False,write_pickle=True):
        if self.length is None or overwrite:
            self.length = lkf_lengths(self)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: length object exists already, to overwrite active overwrite=True option')

    def gen_density(self,overwrite=False,write_pickle=True,**kwargs):
        if self.density is None or overwrite:
            self.density = lkf_density(self,lkf_path=self.lkf_path,years=self.years,**kwargs)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: density object exists already, to overwrite active overwrite=True option')

    def gen_density_len_class(self,len_class=[0e3,100e3,np.inf],write_pickle=True,**kwargs):
        if self.density is None:
            self.density = lkf_density(self,lkf_path=self.lkf_path,years=self.years,**kwargs)
        if self.length is None:
            self.length = lkf_lengths(self)
        self.density.density_len_class(self,len_class=len_class)
        if write_pickle:
            with open(self.pickle, 'wb') as output_pkl:
                pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        

    def gen_curvature(self,overwrite=False,write_pickle=True):
        if self.curvature is None or overwrite:
            self.gen_length(write_pickle=False)
            self.curvature = lkf_curvature(self)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: curvature object exists already, to overwrite active overwrite=True option')

    def gen_deformation(self,overwrite=False,write_pickle=True):
        if self.deformation is None or overwrite:
            self.deformation = lkf_deformation(self)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: deformation object exists already, to overwrite active overwrite=True option')

    def gen_intersection(self,overwrite=False,write_pickle=True,
                         link_def_life_len=True,link_def_len=False,**kwargs):
        if self.intersection is None or overwrite:
            if link_def_life_len:
                self.gen_deformation(write_pickle=False)
                self.gen_length(write_pickle=False)
                self.gen_lifetime(write_pickle=False)
            if link_def_len:
                self.gen_deformation(write_pickle=False)
                self.gen_length(write_pickle=False)
            self.intersection = lkf_intersection(self,link_def_life_len=link_def_life_len,
                                                 link_def_len=link_def_len,
                                                 lkf_path=self.lkf_path,years=self.years,**kwargs)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: intersection object exists already, to overwrite active overwrite=True option')

    def gen_orientation(self,overwrite=False,write_pickle=True,**kwargs):
        if self.orientation is None or overwrite:
            self.orientation = lkf_orientation(self,**kwargs)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: orientation object exists already, to overwrite active overwrite=True option')

    def gen_lifetime(self,overwrite=False,write_pickle=True):
        if self.lifetime is None or overwrite:
            self.lifetime = lkf_lifetime(self)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: lifetime object exists already, to overwrite active overwrite=True option')

    def gen_growthrate(self,overwrite=False,write_pickle=True):
        if self.growthrate is None or overwrite:
            self.gen_length(write_pickle=False)
            self.gen_lifetime(write_pickle=False)
            self.growthrate = lkf_growthrate(self)
            if write_pickle:
                with open(self.pickle, 'wb') as output_pkl:
                    pickle.dump(self, output_pkl, pickle.HIGHEST_PROTOCOL)
        else:
            print('Warning: growth rate object exists already, to overwrite active overwrite=True option')



    def write2tex(self,output_path,output_name):
        for iyear in range(len(self.years)):
            print('Write output file: %s%s_%s.txt' %(output_path,output_name,self.years[iyear]))
            output_file = open('%s%s_%s.txt' %(output_path,output_name,self.years[iyear]),'w')

            # Write header
            output_file.write('Start_Year\tStart_Month\tStart_Day\tEnd_Year\tEnd_Month\tEnd_Day\tDate(RGPS_format)\tLKF_No.\tParent_LKF_No.\tind_x\tind_y\tlon\tlat\tdivergence_rate\tshear_rate\n')
    
            # Loop over days
            id_year = []
            id_c    = 1
            for iday in range(len(self.lkf_dataset[iyear])):
                id_day = []
                # Loop over LKFs
                for ilkf in range(len(self.lkf_dataset[iyear][iday])):
                    # Determine LKF ID
                    id_lkf = int(np.copy(id_c)); 
                    id_c+=1
                    if iday!=0:
                        if self.lkf_track_data[iyear][iday-1].size>0:
                            if np.any(self.lkf_track_data[iyear][iday-1][:,1]==ilkf):
                                id_parent = ','.join([str(id_year[-1][int(it)]) for it in self.lkf_track_data[iyear][iday-1][:,0][self.lkf_track_data[iyear][iday-1][:,1]==ilkf]])
                            else:
                                id_parent = '0'
                        else:
                            id_parent = '0'
                    else:
                        id_parent = '0'

                    # Loop over all points of LKF and write data to file
                    for ip in range(self.lkf_dataset[iyear][iday][ilkf].shape[0]):
                        output_file.write('\t'.join([self.lkf_meta[iyear][iday][0].strftime('%Y'),
                                                     self.lkf_meta[iyear][iday][0].strftime('%m'),
                                                     self.lkf_meta[iyear][iday][0].strftime('%d'),
                                                     self.lkf_meta[iyear][iday][1].strftime('%Y'),
                                                     self.lkf_meta[iyear][iday][1].strftime('%m'),
                                                     self.lkf_meta[iyear][iday][1].strftime('%d'),
                                                     '_'.join([self.lkf_meta[iyear][iday][idate].strftime('%Y%j') for idate in [0,1]]),
                                                     '%i' %id_lkf,
                                                     id_parent,
                                                     '%i' %self.lkf_dataset[iyear][iday][ilkf][ip,0],
                                                     '%i' %self.lkf_dataset[iyear][iday][ilkf][ip,1],
                                                     '%.020e' %self.lkf_dataset[iyear][iday][ilkf][ip,2],
                                                     '%.020e' %self.lkf_dataset[iyear][iday][ilkf][ip,3],
                                                     '%.020e' %self.lkf_dataset[iyear][iday][ilkf][ip,4],
                                                     '%.020e\n' %self.lkf_dataset[iyear][iday][ilkf][ip,5]]))
                        
                    id_day.append(id_c)
            
                id_year.append(id_day)
                
            output_file.close()




# ------------ Statistic functions ----------------------------

# 1. Length

def ks(cdf_sample,cdf_model):
    """Computes Komologorov-Smirnov (KS) statistic:
    D = max( abs( S(x) - N(x) ) / sqrt( N(x) - (1 - N(x)) ) )
    S(x): CDF of sample, N(x): CDF of model"""
    return np.max((np.abs(cdf_sample-cdf_model)/np.sqrt(cdf_model*(1-cdf_model)))[1:])

class lkf_lengths:    
    #def compute_lengths(self):
    def __init__(self,lkf):
        self.output_path = lkf.output_path
    
        print("Compute length of segments")
        lkf_length = []

        for lkf_year in lkf.lkf_dataset:
            len_year = []
            for lkf_day in lkf_year:
                len_day = []
                for ilkf in lkf_day:
                    len_day.append(np.sum(np.sqrt(np.diff(ilkf[:,lkf.indm0])**2 +
                                                  np.diff(ilkf[:,lkf.indm1])**2)))

                len_year.append(np.array(len_day)[np.isfinite(len_day)])

            lkf_length.append(len_year)
        
        self.lkf_length = np.array(lkf_length)


    def plot_length_hist(self, years=None, bins=np.linspace(50,1000,80),pow_law_lim=[50,600],
                         output_plot_data=False,gen_fig=True,save_fig=False,
                         fig_name=None):
        #if self.lkf_length is None:
        #    self.compute_lengths()
        if years is None:
            years=range(len(self.lkf_length))

        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(6,5))
            ax.set_xlabel('LKF length in km')
            ax.set_ylabel('PDF')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim([bins.min(),bins.max()])
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        for iyear in years:
            pdf_length, bins_length = np.histogram(np.concatenate(self.lkf_length[iyear])/1e3,
                                                   bins=bins, density=True)
            bins_mean = 0.5*(bins_length[1:]+bins_length[:-1])
            if gen_fig:
                if iyear==0:
                    ax.plot(bins_mean, pdf_length,color='0.5',alpha=0.5,label="single years")
                else:
                    ax.plot(bins_mean, pdf_length,color='0.5',alpha=0.5)

        pdf_length, bins_length = np.histogram(np.concatenate([np.concatenate(self.lkf_length[iyear]) for iyear in years])/1e3,
                                               bins=bins, density=True)
        bins_mean = 0.5*(bins_length[1:]+bins_length[:-1])
        if gen_fig:
            ax.plot(bins_mean, pdf_length,color=colors[1],alpha=1.0,label="all years")
        
        coeff,pl_fit = power_law_fit(bins_mean[(bins_mean>=pow_law_lim[0]) & (bins_mean<=pow_law_lim[1])], 
                                     pdf_length[(bins_mean>=pow_law_lim[0]) & (bins_mean<=pow_law_lim[1])])

        if gen_fig:
            ax.plot(bins_mean[bins_mean<=600], pl_fit,color=colors[1],alpha=1.0,linestyle='--',label="power-law fit\nexponent %.2f" %(-coeff[0]))
            ax.plot(bins_mean[bins_mean>600], 
                    np.power(10,np.polyval(coeff,np.log10(bins_mean[bins_mean>600]))),
                    color=colors[1],alpha=1.0,linestyle=':')
        
            ax.legend()
            
            if save_fig:
                if fig_name is None:
                    fig.savefig(self.output_path + 'length_pdf.pdf')
                else:
                    fig.savefig(self.output_path + fig_name)

        if output_plot_data:
            return pdf_length, bins_mean, coeff, pl_fit

    def fit_stretched_exponential(self,bins=np.linspace(50,1000,80),xmin=100,xmax=1000,mctest=True,mctest_plots=False,mciter=1000):
        import powerlaw

        # Compute PDF and power-law fit
        (pdf_length, bins_mean, 
         coeff, pl_fit)          = self.plot_length_hist(bins=bins,
                                                         output_plot_data=True,
                                                         gen_fig=False)

        # Fit to stretched exponential
        len_all = np.concatenate([np.concatenate(self.lkf_length[iyear]) for iyear in range(len(self.lkf_length))])/1e3

        fit = powerlaw.Fit(len_all,xmin=xmin,xmax=len_all.max())

        lamd = fit.stretched_exponential.parameter1
        beta = fit.stretched_exponential.parameter2

        print('Fit to stretched exponential:\n   Parameter of fit: lambda = %.010e, beta = %.010e' %(lamd,beta))
        
        # Perform Monte-Carlo Simulation for KS test
        if mctest:
            bins_cdf   = np.logspace(np.log10(xmin),np.log10(xmax),40)
            bins_cdf[0] = xmin; bins_cdf[-1] = xmax
            bins_cdf_m = 10**(np.log10(bins_cdf[:-1])+0.5*np.diff(np.log10(bins_cdf)))
            
            hist,bins  = np.histogram(len_all,bins=bins_cdf,density=True)
            cdf_org    = (hist*np.diff(bins_cdf))[::-1].cumsum()[::-1]
            
            cdf_model  = fit.stretched_exponential.cdf(bins_cdf[:-1],survival=True)
            #cdf_model  = cdf_stretched_exponential(fit.stretched_exponential,bins_cdf[:-1])
            ks_org     = ks(cdf_org,cdf_model)
        
            n_sample   = len_all.size
        
            ks_list = []
            
            discrete_values = np.unique(np.hstack([i*12.5 + np.array([j*np.sqrt(2)*12.5 for j in range(140)]) for i in range(140)]))
            bins_discrete   = discrete_values[:-1] - 0.5*np.diff(discrete_values)
        
            if mctest_plots:
                fig3,ax3 = plt.subplots(1,1)
                fig2,ax2 = plt.subplots(1,1)
            
            for itest in range(mciter):
                #print(itest)
                sample     = fit.stretched_exponential.generate_random(n_sample)
                # Bin to discrete values
                hist_s,bins_s = np.histogram(sample,bins=bins_discrete)
                sample_dis = np.hstack([np.ones(hist_s[i])*discrete_values[i] for i in range(hist_s.size)])
            
                # Compute cdf
                hist,bins  = np.histogram(sample_dis,bins=bins_cdf,density=True)
                cdf_sample  = (hist*np.diff(bins_cdf))[::-1].cumsum()[::-1]
                
                if mctest_plots:
                    ax3.plot(bins_cdf_m,cdf_sample)
                    ax2.plot(bins_cdf_m,np.abs(cdf_sample-cdf_model)/np.sqrt(cdf_model*(1-cdf_model)))
            
                ks_list.append(ks(cdf_sample,cdf_model))
 
            if mctest_plots:
                ax3.plot(bins_cdf_m,cdf_model,'k',linewidth=2.)
                ax3.plot(bins_cdf_m,cdf_org,'r',linewidth=2.)
                ax2.plot(bins_cdf_m,np.abs(cdf_org-cdf_model)/np.sqrt(cdf_model*(1-cdf_model)),'r',linewidth=2.)

            \\

        cf = (pdf_length[bins_mean>=xmin]*np.diff(bins)[bins_mean>=xmin]).sum()
        bins_fit = np.logspace(np.log10(xmin),np.log10(xmax),100)
        pdf_fit = fit.stretched_exponential.pdf(bins_fit)*cf

        return ks_org<np.percentile(ks_list,95), bins_fit, pdf_fit, lamd, beta





# 2. Density

class lkf_density:    
    #def compute_density(self,res=50e3,norm_coverage=True):
    def __init__(self,lkf,res=50e3,norm_coverage=True,lkf_path=None,years=None):
        print("Compute LKF density")

        self.output_path = lkf.output_path
        self.m           = lkf.m
        self.years       = lkf.subdirs

        # Mapping grid
        xedg = np.arange(lkf.m.xmin,lkf.m.xmax,res)
        yedg = np.arange(lkf.m.ymin,lkf.m.ymax,res)
        y,x  = np.meshgrid(yedg[1:],xedg[1:])

        self.res  = res
        self.x    = x
        self.y    = y
        self.xedg = xedg
        self.yedg = yedg

        lkf_density = np.zeros((len(lkf.lkf_dataset),xedg.size-1,yedg.size-1))
        
        for iyear in range(len(lkf.lkf_dataset)):
            #lkf_year = np.concatenate(np.concatenate(lkf.lkf_dataset[iyear]))
            lkf_year = np.concatenate([np.concatenate(lkfday) for lkfday in lkf.lkf_dataset[iyear] if len(lkfday)>0])
            H, xedges, yedges = np.histogram2d(lkf_year[:,lkf.indm0], lkf_year[:,lkf.indm1], 
                                               bins=(xedg, yedg))
            lkf_density[iyear,:,:] = H
        
        #Save output
        self.lkf_density = lkf_density

        if norm_coverage:
            if lkf.datatype=='rgps':
                cov_dict = np.load(lkf.lkf_path + 'coverage_%s.npz' %lkf.datatype)
                coverage = cov_dict['coverage']
                lon_cov  = cov_dict['lon']; lat_cov = cov_dict['lat']
                x_cov,y_cov = m(lon_cov,lat_cov)
                coverage_map = np.zeros((len(lkf.lkf_dataset),xedg.size-1,yedg.size-1))
                for iyear in range(len(lkf.lkf_dataset)):
                    coverage_map[iyear,:,:], xedges, yedges = np.histogram2d(x_cov.flatten(),
                                                                             y_cov.flatten(),
                                                                             bins=(self.xedg, self.yedg),
                                                                             weights=coverage[iyear,:,:].flatten())
                self.coverage_map = coverage_map
                    
            elif lkf.datatype == 'mitgcm_2km':
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
                mask    =    mask[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                  index_x[0][0]-1:index_x[0][-1]+2:red_fac]
                x_cov,y_cov  = m(lon_cov[mask],lat_cov[mask])
                coverage_map = np.zeros((len(years),xedg.size-1,yedg.size-1))
                for iyear in range(len(lkf.lkf_dataset)):
                    coverage_map[iyear,:,:], xedges, yedges = np.histogram2d(x_cov.flatten(),
                                                                             y_cov.flatten(),
                                                                             bins=(self.xedg, self.yedg))
                    coverage_map[iyear,:,:] *= len(lkf.lkf_dataset[iyear])

                self.coverage_map = coverage_map

            elif lkf.datatype =='sirex':
                # if lkf_path.split('/')[-2].split('_')[-1] == 'means':
                #     ind_yp = -2
                # elif lkf_path.split('/')[-2].split('_')[-1] == 'inst':
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
                x_cov,y_cov  = m(lon[max([0,index_y[0][0]-1]):index_y[0][-1]+2:red_fac,
                                     max([0,index_x[0][0]-1]):index_x[0][-1]+2:red_fac],
                                 lat[max([0,index_y[0][0]-1]):index_y[0][-1]+2:red_fac,
                                     max([0,index_x[0][0]-1]):index_x[0][-1]+2:red_fac])
                
                coverage_map = np.zeros((len(years),xedg.size-1,yedg.size-1))
                for iyear in range(len(lkf.lkf_dataset)):
                    coverage_map[iyear,:,:], xedges, yedges = np.histogram2d(x_cov.flatten(),
                                                                             y_cov.flatten(),
                                                                             bins=(self.xedg, self.yedg))
                    coverage_map[iyear,:,:] *= len(lkf.lkf_dataset[iyear])

                self.coverage_map = coverage_map


    def density_len_class(self,lkf,len_class=[0e3,100e3,np.inf],filt_rgps_temp=False,dt=3.):
        dt_rgps = 3.
        
        if not lkf.length is None:
            self.len_class = len_class
            
            density_len_class = np.zeros((len(len_class)-1,len(lkf.lkf_dataset),
                                          self.xedg.size-1,self.yedg.size-1))

            for iclass in range(len(len_class)-1):
                for iyear in range(len(lkf.lkf_dataset)):
                    if ~filt_rgps_temp:
                        lkf_year = np.concatenate(np.concatenate([np.array(lkf.lkf_dataset[iyear][iday])[(lkf.length.lkf_length[iyear][iday]>=len_class[iclass]) & (lkf.length.lkf_length[iyear][iday]<len_class[iclass+1])] for iday in range(len(lkf.lkf_dataset[iyear]))]))
                    else:
                        lkf_year = np.concatenate([np.concatenate(np.concatenate([np.array(lkf.lkf_dataset[iyear][iday])[(lkf.length.lkf_length[iyear][iday]>=len_class[iclass]) & (lkf.length.lkf_length[iyear][iday]<len_class[iclass+1])] for iday in range(0,len(lkf.lkf_dataset[iyear]),int(dt_rgps/dt))])),
                                                   np.concatenate(np.concatenate([np.array(lkf.lkf_dataset[iyear][iday])[((lkf.length.lkf_length[iyear][iday]>=len_class[iclass]) & (lkf.length.lkf_length[iyear][iday]<len_class[iclass+1])) & (lkf.lifetime.lkf_lifetime[iyear][iday]==1)] for iday in [i for i in range(len(lkf.lkf_dataset[iyear])) if i%int(dt_rgps/dt)!=0]]))])

                    H, xedges, yedges = np.histogram2d(lkf_year[:,lkf.indm0], lkf_year[:,lkf.indm1], 
                                                       bins=(self.xedg, self.yedg))
                    density_len_class[iclass,iyear,:,:] = H
                    
            self.lkf_density_len_class = density_len_class
        
        else:
            print('ERROR: Need to compute LKF length first!')



    
    def plot_density(self, norm_coverage=True, plot_single_years=False, min_obs=500,
                     gen_fig=True,save_fig=False,
                     fig_name=None):

        if gen_fig:
            if plot_single_years:
                for iyear in range(len(self.lkf_density)):
                    H = self.lkf_density[iyear,:,:].copy()
        
                    # Plot density for year
                    fig,ax = plt.subplots(nrows=1,ncols=1)
                    if norm_coverage:
                        H /= self.coverage_map[iyear,:,:]
                        pcm = m.pcolormesh(self.x,self.y,
                                           np.ma.masked_where(np.isnan(H) | (H==0),H),
                                           vmin=0,vmax=0.2)
                    self.m.drawcoastlines()
                    cb = plt.colorbar(pcm)
                    cb.set_label('Relative LKF frequency')
                    ax.set_title('Year: %s' %years[iyear])

            fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8.4,6),gridspec_kw={'width_ratios':[12,1]})
            H = np.sum(self.lkf_density,axis=0)
            if norm_coverage:
                H /= np.sum(self.coverage_map,axis=0)
            H = np.ma.masked_where(np.sum(self.coverage_map,axis=0)<min_obs,H)
            pcm = ax[0].pcolormesh(self.x,self.y,
                                   np.ma.masked_where(np.isnan(H) | (H==0),H),
                                   vmin=0,vmax=0.2)
            self.m.drawcoastlines(ax=ax[0])
            self.m.fillcontinents(ax=ax[0],color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0),lake_color='w')
            cb = plt.colorbar(pcm,cax=ax[1])
            cb.set_label('Relative LKF frequency')
            cb.outline.set_visible(False)
            #ax.set_title('Average over entire data set')
            ax[0].axis('off')

            if fig_name is None:
                fig.savefig(self.output_path + 'Density_all_years.pdf')
            else:
                fig.savefig(self.output_path + fig_name)
            
            


# 3. Curvature
    
class lkf_curvature:
    def __init__(self,lkf):
        print("Compute LKF curvature")

        self.years = lkf.subdirs
        self.output_path = lkf.output_path

        lkf_curvature = []

        for lkf_year in lkf.lkf_dataset:
            curv_year = []
            for lkf_day in lkf_year:
                curv_day = []
                for ilkf in lkf_day:
                    curv_day.append(np.sum(np.sqrt((ilkf[0,lkf.indm0]-ilkf[-1,lkf.indm0])**2 +
                                                   (ilkf[0,lkf.indm1]-ilkf[-1,lkf.indm1])**2)))

                curv_year.append(curv_day)

            lkf_curvature.append(curv_year)

        self.lkf_curvature = lkf_curvature
        
    
    def plot(self,lkf_length,save_fig=False,fig_name=None):
        # Plot curvature
        style_label = 'seaborn-darkgrid'
        with plt.style.context(style_label):
            fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.plot(np.concatenate([np.concatenate(il) for il in lkf_length.lkf_length])/1e3,
                np.concatenate([np.concatenate(ic) for ic in self.lkf_curvature])/1e3,'.')
        ax.plot([0,1700],[0,1700],'k--')
        ax.set_xlabel('LKF Length')
        ax.set_ylabel('Distance between LKF endpoints')

        if save_fig:
            if fig_name is None:
                fig.savefig(self.output_path + 'Curvature_all_years.pdf')
            else:
                fig.savefig(self.output_path + fig_name)
            



# 4. Deformation rates

class lkf_deformation:
    def __init__(self,lkf):
        print("Compute deformation of LKFs")

        self.output_path = lkf.output_path

        lkf_deformation = []

        for lkf_year in lkf.lkf_dataset:
            defo_year = []
            for lkf_day in lkf_year:
                defo_day = []
                for ilkf in lkf_day:
                    defo_day.append([np.mean(ilkf[:,lkf.indd0]),np.mean(ilkf[:,lkf.indd1])])

                defo_year.append(defo_day)

            lkf_deformation.append(defo_year)
        
        #Save output
        self.lkf_deformation = np.array(lkf_deformation)


    def plot_deform_hist2d(self,
                           bins_shear = np.linspace(0,0.2,500),
                           bins_div   = np.linspace(-0.1,0.1,500),
                           hist_lims  = [2,300/4],
                           output_plot_data=False,gen_fig=True,save_fig=False,
                           fig_name=None):

        deform_all = np.vstack([np.vstack([np.stack([np.array(iseg) for iseg in self.lkf_deformation[i][j]]) 
                                           for j in range(len(self.lkf_deformation[i]))]) 
                                for i in range(len(self.lkf_deformation))])
    
        hist2d,div_edg,shr_edg = np.histogram2d(deform_all[:,0], deform_all[:,1],
                                                [bins_div,bins_shear])
        hist2d = np.ma.masked_where(hist2d==0,hist2d)
        self.hist2d = hist2d

        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=1)
            pcm = ax.pcolormesh(shr_edg,div_edg,hist2d,
                                norm=mpl.colors.LogNorm(vmin=hist_lims[0], vmax=hist_lims[1]))
            ax.plot([shr_edg[0],shr_edg[-1]],[0,0],'k--')
            ax.set_ylabel('Divergence rate [1/day]')
            ax.set_xlabel('Shear rate [1/day]')
            ax.set_aspect('equal')
            cb = fig.colorbar(pcm, ax=ax, extend='both')
            cb.set_label('Number of LKFs')
            
            if save_fig:
                if fig_name is None:
                    fig.savefig(self.output_path + 'Deformation_hist2d_all_years.pdf')
                else:
                    fig.savefig(self.output_path + fig_name)

        if output_plot_data:
            return hist2d



# 5. Intersections angles

class lkf_intersection:
    def __init__(self,lkf,pos_type='poly',num_p = 10,link_def_life_len=True,link_def_len=False,use_vorticity=False,lkf_path=None,years=None,dis_par=1):
        """num_p : Number of points on each side of the intersection
                   contribute to the orientation computation"""

        print("Compute intersection angle of LKFs")

        if pos_type=='ind':
            indc0 = 0; indc1 = 1;
        if pos_type=='m':
            indc0 = lkf.indm0; indc1 = lkf.indm1;
        if pos_type=='poly':
            indc0 = lkf.indp0; indc1 = lkf.indp1;

        print("Compute interc of segments")

        self.use_vorticity = use_vorticity

        self.years = lkf.years
    
        if lkf.datatype == 'mitgcm_2km':
            grid_path = '/work/ollie/nhutter/arctic_2km/run_cor_cs/'
            mask = mask_arcticbasin(grid_path,read_latlon)
            index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
            index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
            red_fac = 3. # Take only every red_fac point to reduce array size

        if lkf.datatype == 'sirex':
            # if lkf_path is None:
            #     print('ERROR: No lkf_path to netcdf_file is given!')
            # if lkf_path.split('/')[-2].split('_')[-1] == 'means':
            #     ind_yp = -2
            # elif lkf_path.split('/')[-2].split('_')[-1] == 'inst':
            #     ind_yp = -1
            # else:
            #     print(lkf_path.split('/')[-1].split('_')[-1])
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
            red_fac = 1. # Take only every red_fac point to reduce array size
  
        lkf_interc     = []
        lkf_interc_par = []
        if self.use_vorticity:
            lkf_interc_type = []
            
        for iyear in range(len(lkf.lkf_dataset)):
            intc_ang_year = []
            intc_par_year = []
            if self.use_vorticity:
                intc_type_year = []
            
            for iday in range(len(lkf.lkf_dataset[iyear])):
                if lkf.datatype == 'rgps':
                    lkf_map = np.zeros((248,264))
                    if self.use_vorticity:
                        vor_file = os.listdir(os.path.join(lkf_path,str(lkf.years[iyear])));
                        vor_file.sort(); vor_file = vor_file[iday][4:-4]
                        rgps_path = '/work/ollie/nhutter/RGPS/eulerian/'
                        vor_path = os.path.join(rgps_path,'w%02i%s' %(int(str(lkf.years[iyear])[-2:])-1,str(lkf.years[iyear])[-2:]))
                        (vor,xg0,xg1,yg0,yg1,nxcell,nycell) = read_RGPS(os.path.join(vor_path,vor_file + ".VRT"), land_fill=np.NaN, nodata_fill=np.NaN)
                elif lkf.datatype == 'mitgcm_2km' or lkf.datatype == 'sirex':
                    lkf_map = np.zeros((int(np.ceil((index_y[0][-1]+1-index_y[0][0]+1)/red_fac)),
                                        int(np.ceil((index_x[0][-1]+1-index_x[0][0]+1)/red_fac))))
                elif lkf.datatype == 'mosaic':
                    lkf_map = np.zeros(lkf.lkf_meta[iyear][iday][-1])
                    if self.use_vorticity:
                        print('vorticity used in mosaic data')
                        vor = Dataset(lkf.lkf_meta[iyear][iday][3],'r')['vorticity'][:]
                # if lkf.datatype == 'sirex':
                #     if use_vorticity:
                #         flnml = lkf_ps[ind_f].split('_'); flnml.insert(ind_yp,lkf.years[iyear]) 
                #         ncfile = os.path.join(sp,'data','/'.join(lkf_ps[ind_m:ind_f]),'_'.join(flnml)+'.nc')
                #         data = Dataset(ncfile)
                #         time = data.variables['time'][:]
                #         lon  = data.variables['ULON'][:,:]
                #         lat  = data.variables['ULAT'][:,:]
                #         lon[lon==1e30] = np.nan; lat[lat==1e30] = np.nan;
                #         if np.any(np.array(data.variables.keys())=='DXU') and np.any(np.array(data.variables.keys())=='DYU'):
                #             dxu  = data.variables['DXU'][:,:]
                #             dyu  = data.variables['DYU'][:,:]
                #         else:
                #             print("ERROR: DXU and DYU are missing in netcdf file!")
                #             print("  -->  Compute dxu and dyu from lon,lat using SSMI projection")
                #             m = mSSMI()
                #             x,y = m(lon,lat)
                #             dxu = np.sqrt((x[:,1:]-x[:,:-1])**2 + (y[:,1:]-y[:,:-1])**2)
                #             dxu = np.concatenate([dxu,dxu[:,-1].reshape((dxu.shape[0],1))],axis=1)
                #             dyu = np.sqrt((x[1:,:]-x[:-1,:])**2 + (y[1:,:]-y[:-1,:])**2)
                #             dyu = np.concatenate([dyu,dyu[-1,:].reshape((1,dyu.shape[1]))],axis=0)





                for iseg, seg_i in enumerate(lkf.lkf_dataset[iyear][iday]):
                    lkf_map[seg_i[:,0].astype('int'),seg_i[:,1].astype('int')] += iseg

                intc_ang_day = []
                intc_par_day = []
                if self.use_vorticity:
                    intc_type_day = []

                # Check for possible intersection partners
                for iseg, seg_i in enumerate(lkf.lkf_dataset[iyear][iday]):
                    search_ind = np.zeros(lkf_map.shape).astype('bool')
                    
                    # search_ind[seg_i[:,0].astype('int')+1,seg_i[:,1].astype('int')  ] = True
                    # search_ind[seg_i[:,0].astype('int')-1,seg_i[:,1].astype('int')  ] = True
                    # search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')+1] = True
                    # search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')-1] = True
                    # search_ind[seg_i[:,0].astype('int')+1,seg_i[:,1].astype('int')+1] = True
                    # search_ind[seg_i[:,0].astype('int')-1,seg_i[:,1].astype('int')-1] = True
                    # search_ind[seg_i[:,0].astype('int')-1,seg_i[:,1].astype('int')+1] = True
                    # search_ind[seg_i[:,0].astype('int')+1,seg_i[:,1].astype('int')-1] = True
                    # search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')  ] = False

                    for ix in range(-dis_par,dis_par+1):
                        for iy in range(-dis_par,dis_par+1):
                            if np.all([seg_i[:,0].astype('int')+ix >= 0, seg_i[:,0].astype('int')+ix < search_ind.shape[0]]):
                                if np.all([seg_i[:,1].astype('int')+iy >= 0, seg_i[:,1].astype('int')+iy < search_ind.shape[1]]):
                                    search_ind[seg_i[:,0].astype('int')+ix,seg_i[:,1].astype('int')+iy] = True
                    search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')  ] = False
                    

                    intercep_points = np.where(search_ind & (lkf_map!=0))

                    intercep_partners, intercep_counts = np.unique(lkf_map[intercep_points],
                                                                   return_counts=True)

                    for ipar,pari in enumerate(intercep_partners):
                        if pari > iseg and pari < len(lkf.lkf_dataset[iyear][iday]):
                            # Determine one intercetion point for pair
                            dis_intercep = np.zeros(intercep_counts[ipar])
                            for iintc in range(intercep_counts[ipar]):
                                dis_intercep[iintc] = np.min(np.sqrt((seg_i[:,0] - 
                                                                      intercep_points[0][lkf_map[intercep_points]==pari][iintc])**2 + 
                                                                     (seg_i[:,1] - 
                                                                      intercep_points[1][lkf_map[intercep_points]==pari][iintc])**2))
                            intcp = (intercep_points[0][lkf_map[intercep_points]==pari][np.argmin(dis_intercep)],
                                     intercep_points[1][lkf_map[intercep_points]==pari][np.argmin(dis_intercep)])
            
                            # Determine angle between both pairs
                            # # Determine orientation of seg_i
                            ind = np.argmin(np.sqrt((seg_i[:,0] - intcp[0])**2 + 
                                                    (seg_i[:,1] - intcp[1])**2))
                            ind = np.array([np.max([0,ind-num_p]),
                                            np.min([seg_i.shape[0],ind+num_p+1])])
                            p_x,p_y = lkf_poly_fit_p(seg_i[ind[0]:ind[1],indc0],
                                                     seg_i[ind[0]:ind[1],indc1],1) # Linear fit
                            p = p_y[0]/p_x[0]
                            # # Determin angle from linear fit
                            if np.isnan(p):
                                ang_i = 90.
                            else:
                                ang_i = np.arctan(p)/np.pi*180.

                            if self.use_vorticity:
                                vor0 = np.mean(vor[seg_i[ind[0]:ind[1],0].astype('int'),
                                                   seg_i[ind[0]:ind[1],1].astype('int')])

                            # # Determine orientation of pari
                            lkf_par = lkf.lkf_dataset[iyear][iday][int(pari)]
                            ind = np.argmin(np.sqrt((lkf_par[:,0] - intcp[0])**2 + 
                                                    (lkf_par[:,1] - intcp[1])**2))
                            ind = np.array([np.max([0,ind-num_p]),
                                            np.min([lkf_par.shape[0],ind+num_p+1])])
                            p_x,p_y = lkf_poly_fit_p(lkf_par[ind[0]:ind[1],indc0],
                                                     lkf_par[ind[0]:ind[1],indc1],1) # Linear fit
                            p = p_y[0]/p_x[0]
                            # # Determin angle from linear fit
                            if np.isnan(p):
                                ang_ii = 90.
                            else:
                                ang_ii = np.arctan(p)/np.pi*180.
                            if self.use_vorticity:
                                vor1 = np.mean(vor[lkf_par[ind[0]:ind[1],0].astype('int'),
                                                   lkf_par[ind[0]:ind[1],1].astype('int')])
                                # angdiff = np.abs(ang_ii-ang_i)
                                # if vor1*vor0>0:
                                #     # vorticity same sign
                                #     if angdiff > 90: angdiff=180-angdiff
                                #     intc_ang_day.append(180-angdiff)
                                # else:
                                #     if vor1<0:
                                #         angdiff = 180-angdiff
                                #     intc_ang_day.append(angdiff)
                                # intc_ang_day.append(angdiff)
                                if vor0>0 and vor1<0:
                                    if ang_ii<ang_i: ang_ii+=180
                                    intc_ang_day.append(ang_ii-ang_i)
                                    intc_ang_day.append(ang_ii-ang_i)
                                    intc_type_day.append(0); intc_type_day.append(0)
                                elif vor1>0 and vor0<0:
                                    if ang_i<ang_ii: ang_i+=180
                                    intc_ang_day.append(ang_i-ang_ii)
                                    intc_ang_day.append(ang_i-ang_ii)
                                    intc_type_day.append(0); intc_type_day.append(0)
                                else:
                                    intc_type_day.append(1); intc_type_day.append(1)
                                    intc_ang_day.append(np.abs(ang_ii-ang_i))
                                    intc_ang_day.append(180-np.abs(ang_ii-ang_i))
                                intc_par_day.append(np.array([iseg,pari]))
                            else:
                                angdiff = np.abs(ang_ii-ang_i)
                                if angdiff > 90: angdiff=180-angdiff
                                intc_ang_day.append(angdiff)
                            intc_par_day.append(np.array([iseg,pari]))

                intc_ang_year.append(intc_ang_day)
                intc_par_year.append(intc_par_day)
                if self.use_vorticity:
                    intc_type_year.append(intc_type_day)
            lkf_interc.append(intc_ang_year)
            lkf_interc_par.append(intc_par_year)
            if self.use_vorticity:
                lkf_interc_type.append(intc_type_year)

        self.lkf_interc     = np.array(lkf_interc)
        self.lkf_interc_par = np.array(lkf_interc_par)
        if self.use_vorticity:
            self.lkf_interc_type = np.array(lkf_interc_type)


        if link_def_life_len:
            # Compute mean deformation of intersecting partners
            self.def_par = []; self.diff_def_par = [];
            self.life_par = []; self.len_par = []
            for iyear in range(len(lkf.lkf_dataset)):
                def_par_year = []; diff_def_par_year = [];
                life_par_year = []; len_par_year = []
                for iday in range(len(lkf.lkf_dataset[iyear])):
                    if len(self.lkf_interc_par[iyear][iday]) > 0:
                        def_par_day = np.array([np.sqrt(np.sum(np.array(lkf.deformation.lkf_deformation[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,0].astype('int')]**2,axis=1)),
                                                np.sqrt(np.sum(np.array(lkf.deformation.lkf_deformation[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,1].astype('int')]**2,axis=1))])
                        diff_def_par_day = np.abs(np.diff(def_par_day,axis=0))
                        life_par_day = np.array([lkf.lifetime.lkf_lifetime[iyear][iday][np.stack(self.lkf_interc_par[iyear][iday])[:,0].astype('int')],
                                                 lkf.lifetime.lkf_lifetime[iyear][iday][np.stack(self.lkf_interc_par[iyear][iday])[:,1].astype('int')]])
                        len_par_day = np.array([np.array(lkf.length.lkf_length[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,0].astype('int')],
                                                np.array(lkf.length.lkf_length[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,1].astype('int')]])
                    else:
                        life_par_day = np.array([])
                        len_par_day  = np.array([])
                        def_par_day  = np.array([])
                        diff_def_par_day  = np.array([])

                    def_par_year.append(def_par_day)
                    diff_def_par_year.append(diff_def_par_day)
                    life_par_year.append(life_par_day)
                    len_par_year.append(len_par_day)
                self.def_par.append(def_par_year)
                self.diff_def_par.append(diff_def_par_year)
                self.life_par.append(life_par_year)
                self.len_par.append(len_par_year)

        if link_def_len:
            # Compute mean deformation of intersecting partners
            self.def_par = []; self.diff_def_par = [];
            #self.life_par = [];
            self.len_par = []
            for iyear in range(len(lkf.lkf_dataset)):
                def_par_year = []; diff_def_par_year = [];
                #life_par_year = [];
                len_par_year = []
                for iday in range(len(lkf.lkf_dataset[iyear])):
                    if len(self.lkf_interc_par[iyear][iday]) > 0:
                        def_par_day = np.array([np.sqrt(np.sum(np.array(lkf.deformation.lkf_deformation[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,0].astype('int')]**2,axis=1)),
                                                np.sqrt(np.sum(np.array(lkf.deformation.lkf_deformation[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,1].astype('int')]**2,axis=1))])
                        diff_def_par_day = np.abs(np.diff(def_par_day,axis=0))
                        #life_par_day = np.array([lkf.lifetime.lkf_lifetime[iyear][iday][np.stack(self.lkf_interc_par[iyear][iday])[:,0].astype('int')],
                        #                         lkf.lifetime.lkf_lifetime[iyear][iday][np.stack(self.lkf_interc_par[iyear][iday])[:,1].astype('int')]])
                        len_par_day = np.array([np.array(lkf.length.lkf_length[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,0].astype('int')],
                                                np.array(lkf.length.lkf_length[iyear][iday])[np.stack(self.lkf_interc_par[iyear][iday])[:,1].astype('int')]])
                    else:
                        #life_par_day = np.array([])
                        len_par_day  = np.array([])
                        def_par_day  = np.array([])
                        diff_def_par_day  = np.array([])

                    def_par_year.append(def_par_day)
                    diff_def_par_year.append(diff_def_par_day)
                    #life_par_year.append(life_par_day)
                    len_par_year.append(len_par_day)
                self.def_par.append(def_par_year)
                self.diff_def_par.append(diff_def_par_year)
                #self.life_par.append(life_par_year)
                self.len_par.append(len_par_year)



    def plot_hist(self,bins=np.linspace(0,90,45),
                  output_plot_data=False,gen_fig=True,save_fig=False,
                  fig_name=None):

        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=1)

            ax.set_xlabel('Intersection angle')
            ax.set_ylabel('PDF')
            #ax.set_xlim([0,90])
            for iyear in range(len(self.lkf_interc)):
                pdf_interc, bins_interc = np.histogram(np.concatenate(self.lkf_interc[iyear]),
                                                       bins=bins, density=True)
                bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
                ax.plot(bins_mean, pdf_interc,label=self.years[iyear],color='0.5',alpha=0.5)
            
        pdf_interc, bins_interc = np.histogram(np.concatenate([np.concatenate(self.lkf_interc[iyear]) for iyear in range(len(self.lkf_interc))]),
                                               bins=bins, density=True)
        bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])

        if gen_fig:
            ax.plot(bins_mean, pdf_interc,label=self.years[iyear])

            if save_fig:
                if fig_name is None:
                    fig.savefig(self.output_path + 'Interc_pdf_all_years.pdf')
                else:
                    fig.savefig(self.output_path + fig_name)
 
        if output_plot_data:
            return pdf_interc, bins_mean
        

    def plot_hist_def_life_len(self,def_class=None,bins=np.linspace(0,90,23),
                               len_thres = 10*12.5e3,
                               output_plot_data=False,gen_fig=True,
                               save_fig=False, fig_name=None,
                               return_num_meas=False):
        if def_class is None:
            if datatype=='rgps':
                def_class = [0,0.03,0.1,2]
            elif datatype == 'mitgcm_2km':
                def_class = [0,0.05,0.2,10]
        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=len(def_class)-1, figsize=(6*(len(def_class)-1),5))
                colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        def_masked_class = [[] for i in range(len(def_class))]
        
        pdf_all_class   = []
        pdf_years_class = []

        for iax in range(len(def_class)-1):
            if gen_fig:
                if (len(def_class)-1)==1:
                    axi=ax
                else:
                    axi = ax[iax]
            
                axi.set_title('Def. class: %.2f to %.2f [1/day]' %(def_class[iax],def_class[iax+1]))
                axi.set_xlabel('Intersection angle')
                axi.set_ylabel('PDF')
                axi.set_xlim([0,90])

            pdf_year_save = []

            for iyear in range(len(self.lkf_interc)):
                mask_def = np.all([np.all(np.hstack(self.def_par[iyear])>=def_class[iax],axis=0),
                                   np.all(np.hstack(self.def_par[iyear])<def_class[iax+1],axis=0)],
                                  axis=0)
                mask_life = np.all(np.hstack(self.life_par[iyear])==1,axis=0)
                mask_len = np.all(np.hstack(self.len_par[iyear])>=len_thres,axis=0)
                mask_life = mask_len & mask_life
                if self.use_vorticity:
                    mask_def  = np.stack([mask_def,mask_def]).T.flatten()
                    mask_life = np.stack([mask_life,mask_life]).T.flatten()
                pdf_interc, bins_interc = np.histogram(np.concatenate(self.lkf_interc[iyear])[mask_def & mask_life],
                                                       bins=bins, density=True)
                pdf_year_save.append(pdf_interc)
                bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
                if gen_fig:
                    if iyear==0:
                        axi.plot(bins_mean, pdf_interc,'.',label='single years',color='0.5',alpha=0.5)
                    else:
                        axi.plot(bins_mean, pdf_interc,'.',color='0.5',alpha=0.5)

                def_masked_class[iax].append(np.concatenate(self.lkf_interc[iyear])[mask_def & mask_life])
            pdf_interc, bins_interc = np.histogram(np.concatenate(def_masked_class[iax]),
                                                   bins=bins, density=True)
            bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])

            pdf_all_class.append(pdf_interc)
            pdf_years_class.append(pdf_year_save)

            if gen_fig:
                axi.plot(bins_mean, pdf_interc,label='all years',color=colors[0],alpha=1.0)

                axi.plot([],[],' ',label='Tot. num: %i' %np.concatenate(def_masked_class[iax]).size)
                axi.legend()

        self.def_masked_class = def_masked_class

        if save_fig:
            if fig_name is None:
                fig.savefig(self.output_path + 'Interc_pdf_def_len_life_all_years.pdf')
            else:
                fig.savefig(self.output_path + fig_name)
 
        if output_plot_data:
            if return_num_meas:
                return pdf_all_class, pdf_years_class, bins_mean, np.concatenate(def_masked_class[iax]).size
            else:
                return pdf_all_class, pdf_years_class, bins_mean


    def plot_hist_def_len(self,def_class=None,bins=np.linspace(0,90,23),
                               len_thres = 10*12.5e3,
                               output_plot_data=False,gen_fig=True,
                               save_fig=False, fig_name=None,
                               return_num_meas=False):
        if def_class is None:
            if datatype=='rgps':
                def_class = [0,0.03,0.1,2]
            elif datatype == 'mitgcm_2km':
                def_class = [0,0.05,0.2,10]
        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=len(def_class)-1, figsize=(6*(len(def_class)-1),5))
                colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        def_masked_class = [[] for i in range(len(def_class))]
        
        pdf_all_class   = []
        pdf_years_class = []

        for iax in range(len(def_class)-1):
            if gen_fig:
                if (len(def_class)-1)==1:
                    axi=ax
                else:
                    axi = ax[iax]
            
                axi.set_title('Def. class: %.2f to %.2f [1/day]' %(def_class[iax],def_class[iax+1]))
                axi.set_xlabel('Intersection angle')
                axi.set_ylabel('PDF')
                axi.set_xlim([0,90])

            pdf_year_save = []

            for iyear in range(len(self.lkf_interc)):
                mask_def = np.all([np.all(np.hstack(self.def_par[iyear])>=def_class[iax],axis=0),
                                   np.all(np.hstack(self.def_par[iyear])<def_class[iax+1],axis=0)],
                                  axis=0)
                #mask_life = np.all(np.hstack(self.life_par[iyear])==1,axis=0)
                mask_len = np.all(np.hstack(self.len_par[iyear])>=len_thres,axis=0)
                mask_life = mask_len #& mask_life
                if self.use_vorticity:
                    mask_def  = np.stack([mask_def,mask_def]).T.flatten()
                    mask_life = np.stack([mask_life,mask_life]).T.flatten()
                pdf_interc, bins_interc = np.histogram(np.concatenate(self.lkf_interc[iyear])[mask_def & mask_life],
                                                       bins=bins, density=True)
                pdf_year_save.append(pdf_interc)
                bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
                if gen_fig:
                    if iyear==0:
                        axi.plot(bins_mean, pdf_interc,'.',label='single years',color='0.5',alpha=0.5)
                    else:
                        axi.plot(bins_mean, pdf_interc,'.',color='0.5',alpha=0.5)

                def_masked_class[iax].append(np.concatenate(self.lkf_interc[iyear])[mask_def & mask_life])
            pdf_interc, bins_interc = np.histogram(np.concatenate(def_masked_class[iax]),
                                                   bins=bins, density=True)
            bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])

            pdf_all_class.append(pdf_interc)
            pdf_years_class.append(pdf_year_save)

            if gen_fig:
                axi.plot(bins_mean, pdf_interc,label='all years',color=colors[0],alpha=1.0)

                axi.plot([],[],' ',label='Tot. num: %i' %np.concatenate(def_masked_class[iax]).size)
                axi.legend()

        self.def_masked_class = def_masked_class

        if save_fig:
            if fig_name is None:
                fig.savefig(self.output_path + 'Interc_pdf_def_len_life_all_years.pdf')
            else:
                fig.savefig(self.output_path + fig_name)
 
        if output_plot_data:
            if return_num_meas:
                return pdf_all_class, pdf_years_class, bins_mean, np.concatenate(def_masked_class[iax]).size
            else:
                return pdf_all_class, pdf_years_class, bins_mean



            





# 6. Compute orientation

class lkf_orientation:
    def __init__(self,lkf,res=200e3,use_poly_ori=True):
        print("Compute orientation of LKFs")

        self.output_path = lkf.output_path
        self.m = lkf.m

        # Compute cell that an LKF contributes to
        # Mapping grid
        self.xedg     = np.arange(m.xmin,m.xmax,res)
        self.yedg     = np.arange(m.ymin,m.ymax,res)
        self.y,self.x = np.meshgrid(0.5*(self.yedg[1:]+self.yedg[:-1]),
                                    0.5*(self.xedg[1:]+self.xedg[:-1]))

        cell_contrib = []

        for lkf_year in lkf.lkf_dataset:
            cell_contrib_year = []
            for lkf_day in lkf_year:
                cell_contrib_day = []
                for ilkf in lkf_day:
                    if use_poly_ori:
                        H, xedges, yedges = np.histogram2d(ilkf[:,lkf.indp0], ilkf[:,lkf.indp1], 
                                                           bins=(self.xedg, self.yedg))
                    else:
                        H, xedges, yedges = np.histogram2d(ilkf[:,lkf.indm0], ilkf[:,lkf.indm1], 
                                                           bins=(self.xedg, self.yedg))
                    cell_contrib_day.append(np.where(H.flatten()>0)[0])

                cell_contrib_year.append(cell_contrib_day)

            cell_contrib.append(cell_contrib_year)
        
        self.cell_contrib = np.array(cell_contrib)



        # Compute orientation
        print("Compute orientation of segments")
        lkf_orientation = []
        lkf_ori_len_wght = []

        lkf_angle = []
        lkf_angle_len_wght = []

        ori_day_org = np.empty((self.x.shape),dtype=object)
        for ix in range(self.xedg.size-1):
            for iy in range(self.yedg.size-1):
                ori_day_org[ix,iy] = np.array([])

        for iyear,lkf_year in enumerate(lkf.lkf_dataset):
            
            ori_year = []
            ori_len_year = []
            ang_year = []
            ang_len_year = []
            for iday,lkf_day in enumerate(lkf_year):
                ori_day = ori_day_org.copy()
                ori_len_day = ori_day_org.copy()
                ang_day = []
                ang_len_day = []
                for ilkf,lkf_i in enumerate(lkf_day):
                    ang_lkf = []
                    ang_len_lkf = []
                    for i_cell in self.cell_contrib[iyear][iday][ilkf]:
                        # Find part of lkf inside box
                        ix,iy = np.unravel_index(i_cell,self.x.shape)
                        if use_poly_ori:
                            lkf_i_c = lkf_i[:,lkf.indp0:lkf.indp0+2][np.all([lkf_i[:,lkf.indp0]>=self.xedg[ix],
                                                                             lkf_i[:,lkf.indp0]<=self.xedg[ix+1],
                                                                             lkf_i[:,lkf.indp1]>=self.yedg[iy],
                                                                             lkf_i[:,lkf.indp1]<=self.yedg[iy+1]],
                                                                            axis=0),:]
                        else:
                            lkf_i_c = lkf_i[:,lkf.indm0:lkf.indm0+2][np.all([lkf_i[:,lkf.indm0]>=self.xedg[ix],
                                                                             lkf_i[:,lkf.indm0]<=self.xedg[ix+1],
                                                                             lkf_i[:,lkf.indm1]>=self.yedg[iy],
                                                                             lkf_i[:,lkf.indm1]<=self.yedg[iy+1]],
                                                                            axis=0),:]

                        # Linear fit & determine angle from linear fit
                        if lkf_i_c.size > 2:
                            # All cases that are not a line in y-direction
                            p_x,p_y = lkf_poly_fit_p(lkf_i_c[:,0],lkf_i_c[:,1],
                                                     1) # Linear fit
                            p = p_y[0]/p_x[0]
                            
                            # Determin angle from linear fit
                            if np.isnan(p):
                                ang = 90.
                            else:
                                ang = np.arctan(p)/np.pi*180.

                            ang_lkf.append(ang)
                            ang_len_lkf.append(lkf_i_c.shape[0])
                            
                            ori_day[ix,iy] = np.concatenate([ori_day[ix,iy],
                                                             np.array([ang])])
                            ori_len_day[ix,iy] = np.concatenate([ori_len_day[ix,iy],
                                                                 np.array([lkf_i_c.shape[0]])])
                        else:
                            ang_lkf.append(np.nan)
                            ang_len_lkf.append(np.nan)

                    ang_day.append(ang_lkf)
                    ang_len_day.append(ang_len_lkf)

                ang_year.append(ang_day)
                ang_len_year.append(ang_len_day)

                ori_year.append(ori_day)
                ori_len_year.append(ori_len_day)
    
            lkf_angle.append(ang_year)
            lkf_angle_len_wght.append(ang_len_year)

            lkf_orientation.append(ori_year)
            lkf_ori_len_wght.append(ori_len_year)

            
        
        #Save output
        self.lkf_angle = np.array(lkf_angle)
        self.lkf_angle_len_wght = np.array(lkf_angle_len_wght)
        self.lkf_orientation = np.array(lkf_orientation)
        self.lkf_ori_len_wght = np.array(lkf_ori_len_wght)




# 7. Lifetime of LKFs

class lkf_lifetime:
    def __init__(self,lkf):
        print("Compute lifetime of LKFs")

        self.output_path = lkf.output_path

        lkf_lifetime = []

        for iyear,lkf_year in enumerate(lkf.lkf_dataset):
            life_year = [np.ones((len(i_num_lkf),)) for i_num_lkf in lkf_year]
            #print(len(lkf_year),len(lkf.lkf_track_data[iyear]))
            #print(len(life_year))
            for it,itrack in enumerate(lkf.lkf_track_data[iyear]):
                #print(it,itrack)
                if itrack.size>0:
                    #print(life_year[it+1].shape)
                    life_year[it+1][itrack[:,1].astype('int')] += life_year[it][itrack[:,0].astype('int')]

            lkf_lifetime.append(life_year)

        #Save output
        self.lkf_lifetime = np.array(lkf_lifetime)


    def plot_pdf(self,xlim=[0,31],dt=3.,
                 output_plot_data=False,gen_fig=True,save_fig=False,
                 fig_name=None):
        # Compute histograms
        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=1)
            ax.set_xlabel('LKF lifetime')
            ax.set_ylabel('Relative frequency')
            ax.set_yscale('log')
            ax.set_xlim(xlim)
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
            for iyear in range(len(self.lkf_lifetime)):
                pdf_life = np.bincount(np.concatenate(self.lkf_lifetime[iyear]).astype('int')-1)
                bins_mean = np.arange(pdf_life.size)*dt+dt/2.
                if iyear==0:
                    ax.plot(bins_mean, pdf_life/float(np.sum(pdf_life)),'.',color='0.5',alpha=0.5,label="single years")
                else:
                    ax.plot(bins_mean, pdf_life/float(np.sum(pdf_life)),'.',color='0.5',alpha=0.5)
                    
        pdf_life = np.bincount(np.concatenate([np.concatenate(life_year) for life_year in self.lkf_lifetime]).astype('int')-1)
        bins_mean = np.arange(pdf_life.size)*dt+dt/2.

        if gen_fig:
            ax.plot(bins_mean, pdf_life/float(np.sum(pdf_life)),'.',color=colors[1],alpha=1.0,label="all years")
        
        coeff = np.polyfit(bins_mean, np.log(pdf_life/float(np.sum(pdf_life))),1)
        pdf_life_fit = np.exp(np.polyval(coeff,bins_mean))
        if gen_fig:
            ax.plot(bins_mean,pdf_life_fit,
                    color=colors[1],alpha=1.0,linestyle='--',
                    label="exponential fit\nexponent %.2f" %(-coeff[0]))
            ax.legend()

            if save_fig:
                if fig_name is None:
                    fig.savefig(self.output_path + 'Lifetime_pdf_exp_fit_all_years.pdf')
                else:
                    fig.savefig(self.output_path + fig_name)

        if output_plot_data:
            return pdf_life, bins_mean, coeff, pdf_life_fit





# 8. Growth rates

class lkf_growthrate:
    def __init__(self,lkf):
        print("Compute growth rates of LKFs")

        self.output_path  = lkf.output_path
        self.lkf_lifetime = lkf.lifetime.lkf_lifetime

        lkf_growth = []
        lkf_shrink = []

        for iyear,lkf_year in enumerate(lkf.lkf_dataset):
            growth_year = [np.ones((len(i_num_lkf),))*np.nan for i_num_lkf in lkf_year[1:]]
            shrink_year = [np.ones((len(i_num_lkf),))*np.nan for i_num_lkf in lkf_year[1:]]
            for iday,day_track in enumerate(lkf.lkf_track_data[iyear]):
                # Compute growth rate of all tracked features
                for it,itrack in enumerate(day_track):
                    if len(itrack)>0:
                        # Compute overlapping area for both features
                        mhd,overlap,[A_o,B_o] = compute_MHD_segment(lkf.lkf_dataset[iyear][iday][itrack[0].astype('int')][:,:2].T,
                                                                    lkf.lkf_dataset[iyear][iday+1][itrack[1].astype('int')][:,:2].T,
                                                                    overlap_thres=1.5,angle_thres=25,
                                                                    return_overlap=True,
                                                                    return_overlaping_area=True,
                                                                    mask_instead=True)
                        A = lkf.lkf_dataset[iyear][iday][itrack[0].astype('int')][:,lkf.indm0:lkf.indm1+1].copy()
                        B = lkf.lkf_dataset[iyear][iday+1][itrack[1].astype('int')][:,lkf.indm0:lkf.indm1+1].copy()

                        A[A_o,:] = np.nan; B[B_o,:] = np.nan;
                        
                        # Determine growth
                        growth_year[iday][itrack[1].astype('int')] = np.nansum(np.sqrt(np.sum(np.diff(A,axis=0)**2,axis=1)))
                        if np.isnan(growth_year[iday][itrack[1].astype('int')]):
                            growth_year[iday][itrack[1].astype('int')] = 0
                            
                        # Determine shrink
                        shrink_year[iday][itrack[1].astype('int')] = np.nansum(np.sqrt(np.sum(np.diff(B,axis=0)**2,axis=1)))
                        if np.isnan(shrink_year[iday][itrack[1].astype('int')]):
                            shrink_year[iday][itrack[1].astype('int')] = 0

                # Add growth rates of all not tracked features
                ind_life1 = (lkf.lifetime.lkf_lifetime[iyear][iday+1]==1)
                growth_year[iday][ind_life1] = lkf.length.lkf_length[iyear][iday+1][ind_life1]
                shrink_year[iday][ind_life1] = lkf.length.lkf_length[iyear][iday+1][ind_life1]

            lkf_growth.append(growth_year)
            lkf_shrink.append(shrink_year)


        #Save output
        self.lkf_growth = np.array(lkf_growth)
        self.lkf_shrink = np.array(lkf_shrink)
        

    def plot_pdf(self, bins=np.linspace(0,500,50),
                 output_plot_data=False,gen_fig=True,save_fig=False,
                 fig_name=None):
        #if self.lkf_length is None:
        #    self.compute_lengths()

        if gen_fig:
            style_label = 'seaborn-darkgrid'
            with plt.style.context(style_label):
                fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(6,5))
            ax.set_xlabel('LKF growth rates [km/day]')
            ax.set_ylabel('PDF')
            ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.set_xlim([bins.min(),bins.max()])
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        plot_list  = [self.lkf_growth,self.lkf_shrink]
        plot_label = ['positive (grow)', 'negative (shrink)']
        lifetime = self.lkf_lifetime
        coeffs   = []
        pdfs     = []
        binss    = []
        pdf_fits = []
        yerrs    = []

        for i,growthi in enumerate(plot_list):
            pdf_years_lifee1 = []
            pdf_years_lifel1 = []
            for iyear in range(len(growthi)):
                growth_year = np.concatenate(growthi[iyear])/3e3
                lifetime_year = np.concatenate(lifetime[iyear][1:])
                # All LKFs of year
                pdf_growth, bins_growth = np.histogram(growth_year[lifetime_year==1],
                                                       bins=bins, density=True)
                bins_mean = 0.5*(bins_growth[1:]+bins_growth[:-1])
                pdf_years_lifee1.append(pdf_growth)
                
                pdf_growth, bins_growth = np.histogram(growth_year[lifetime_year!=1],
                                                       bins=bins, density=True)
                bins_mean = 0.5*(bins_growth[1:]+bins_growth[:-1])
                pdf_years_lifel1.append(pdf_growth)

            pdf_years_lifee1 = np.stack(pdf_years_lifee1)
            pdf_years_lifel1 = np.stack(pdf_years_lifel1)

            growth_all = np.concatenate([np.concatenate(growthi[iyear]) for iyear in range(len(growthi))])/3e3
            lifetime_all = np.concatenate([np.concatenate(lifetime[iyear][1:]) for iyear in range(len(lifetime))])
            pdf_growth, bins_growth = np.histogram(growth_all[lifetime_all==1],
                                                   bins=bins, density=True)
            bins_mean = 0.5*(bins_growth[1:]+bins_growth[:-1])
            if i==0:
                ista = np.where(pdf_growth!=0)[0][0]
                if np.any(pdf_growth==0):
                    if np.where(np.where(pdf_growth==0)[0]>ista)[0].size>0:
                        iend = np.where(pdf_growth==0)[0][np.where(np.where(pdf_growth==0)[0]>ista)[0][0]]
                    else:
                        iend = -1
                else:
                    iend = -1
                coeff_l1 = np.polyfit(bins_mean[ista:iend], 
                                          np.log(pdf_growth[ista:iend]),1)
                pdf_growth_fit_l1 = np.exp(np.polyval(coeff_l1,bins_mean))
                yerr = [pdf_growth-pdf_years_lifee1.min(axis=0),
                        pdf_years_lifee1.max(axis=0)-pdf_growth]
                
                coeffs.append(coeff_l1)
                pdfs.append(pdf_growth)
                binss.append(bins_mean)
                pdf_fits.append(pdf_growth_fit_l1)
                yerrs.append(yerr)
            
                if gen_fig:
                    ax.errorbar(bins_mean, pdf_growth,yerr=yerr,
                                color=colors[2],alpha=0.5,
                                label='Newly formed (%.03f)' %coeff_l1[0],
                                fmt='.')
                    ax.plot(bins_mean,pdf_growth_fit_l1,'--',color=colors[2])


            pdf_growth, bins_growth = np.histogram(growth_all[lifetime_all!=1],
                                                   bins=bins, density=True)
            bins_mean = 0.5*(bins_growth[1:]+bins_growth[:-1])
            ista = np.where(pdf_growth!=0)[0][0]
            if np.any(pdf_growth==0):
                if np.where(np.where(pdf_growth==0)[0]>ista)[0].size>0:
                    iend = np.where(pdf_growth==0)[0][np.where(np.where(pdf_growth==0)[0]>ista)[0][0]]
                else:
                    iend = -1
            else:
                iend = -1
            coeff_e1 = np.polyfit(bins_mean[ista:iend], 
                                  np.log(pdf_growth[ista:iend]),1)
            pdf_growth_fit_e1 = np.exp(np.polyval(coeff_e1,bins_mean))
            yerr = [pdf_growth-pdf_years_lifel1.min(axis=0),
                    pdf_years_lifel1.max(axis=0)-pdf_growth]

            coeffs.append(coeff_e1)
            pdfs.append(pdf_growth)
            binss.append(bins_mean)
            pdf_fits.append(pdf_growth_fit_e1)
            yerrs.append(yerr)
            if gen_fig:
                ax.errorbar(bins_mean, pdf_growth,yerr=yerr,
                            color=colors[i],alpha=0.5,
                            label=plot_label[i]+' (%.03f)' %coeff_e1[0],fmt='.')
                ax.plot(bins_mean,pdf_growth_fit_e1,'--',color=colors[i])
        
                ax.legend()
                ax.set_ylim([10**np.floor(np.nanmin(np.log10(pdf_growth)[np.isfinite(np.log10(pdf_growth))])),
                             10**np.ceil(np.nanmax(np.log10(pdf_growth)))])
            
            if save_fig:
                if fig_name is None:
                    fig.savefig(self.output_path + 'growth_rate_pdf.pdf')
                else:
                    fig.savefig(self.output_path + fig_name)

        if output_plot_data:
            return pdfs, binss, yerrs ,coeffs, pdf_fits


















































def runmean(data,win):
    datam = np.zeros(data.size-win+1)
    for i in range(win):
        datam += data[i:(data.size+1-win+i)]
    return datam/float(win)


def power_law_fit(x,y):
    coeff = np.polyfit(np.log10(x),np.log10(y),1)
    fit = np.power(10,np.polyval(coeff,np.log10(x)))
    return coeff,fit
















# # --------------------- Statistics --------------------------------

# # if datatype == 'rgps':
# #     plot_output_path = '/work/ollie/nhutter/lkf_data/rgps_eps/stats/'
# # elif datatype == 'mitgcm_2km':
# #     plot_output_path = '/work/ollie/nhutter/lkf_data/mitgcm_2km/stats/'
# # elif datatype == 'mitgcm_2km_cor_cs':
# #     plot_output_path = '/work/ollie/nhutter/lkf_data/mitgcm_2km_cor_cs/stats/'

# num_time = False

# length = False

# density = False

# curvature = False

# comp_cell_contrib = False

# orientation = False
# use_poly_ori = True
# plot_ori_mean = True
# plot_ori_years = False
# plot_ori_months = True
# plot_rad_hist = False
# plot_broehan = True

# deformation = False

# intersection = False
# link_interc_def = False
# link_interc_lifetime = False
# link_interc_len = False

# lifetime = False

# if curvature: length = True
# if orientation: comp_cell_contrib = True
# if intersection: 
#     if link_interc_def: 
#         deformation = True
# if link_interc_len:
#     link_interc_lifetime = True
# if link_interc_lifetime:
#     intersection = True
#     lifetime = True
# if lifetime:
#     if not read_tracking:
# 	lifetime=False
# 	print "Please activate reading of tracking data first"


# force_recompute = False

# # Meta data statistics

# def runmean(data,win):
#     datam = np.zeros(data.size-win+1)
#     for i in range(win):
# 	datam += data[i:(data.size+1-win+i)]
#     return datam/float(win)


# if num_time:
#     fig,ax = plt.subplots(nrows=1,ncols=1)
#     for lkfyear in lkf_meta:
#         ax.plot(lkfyear[:,0],lkfyear[:,2],color='0.5',
# 	        linestyle='',marker='.')
# 	ax.plot(lkfyear[2:-2,0],runmean(lkfyear[:,2].astype('float'),5),'k')
#         ax.set_ylabel('Number of detected features')
#     fig.savefig(plot_output_path + 'Num_lkfs.pdf')


# # Data statistics

# # 1. Length of LKFs

# def power_law_fit(x,y):
#     coeff = np.polyfit(np.log10(x),np.log10(y),1)
#     fit = np.power(10,np.polyval(coeff,np.log10(x)))
#     return coeff,fit

# if length:
#     length_file = int_mem_path + 'length_%s_dataset.npy' %datatype
#     if os.path.exists(length_file) and not force_recompute:
#         print "Open already computed file: %s" %length_file
#         lkf_length = np.load(length_file)

#     else:
#         print "Compute length of segments"
#         lkf_length = []

#         for lkf_year in lkf_dataset:
#             len_year = []
#             for lkf_day in lkf_year:
#                 len_day = []
#                 for ilkf in lkf_day:
#                     len_day.append(np.sum(np.sqrt(np.diff(ilkf[:,indm0])**2 +
#                                                   np.diff(ilkf[:,indm1])**2)))

#                 len_year.append(len_day)

#             lkf_length.append(len_year)
        
#         #Save output
#         print "Saving computed file: %s" %length_file
#         np.save(length_file,lkf_length)
#         lkf_length = np.array(lkf_length)

#     # Compute histograms
#     # - one plot with lines for each year
#     nbins = 80
#     bins = np.logspace(1.68,3,nbins)
#     bins = np.linspace(50,1000,nbins)
#     style_label = 'seaborn-darkgrid'
#     with plt.style.context(style_label):
#         fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(6,5))
#     ax.set_xlabel('LKF length in km')
#     ax.set_ylabel('PDF')
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_xlim([50,1000])
#     colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
#     for iyear in range(len(lkf_length)):
#         pdf_length, bins_length = np.histogram(np.concatenate(lkf_length[iyear])/1e3,
#                                                bins=bins, density=True)
#         bins_mean = 0.5*(bins_length[1:]+bins_length[:-1])
# 	if iyear==0:
#             ax.plot(bins_mean, pdf_length,color='0.5',alpha=0.5,label="single years")
# 	else:
#             ax.plot(bins_mean, pdf_length,color='0.5',alpha=0.5)

#     pdf_length, bins_length = np.histogram(np.concatenate([np.concatenate(lkf_length[iyear]) for iyear in range(len(lkf_length))])/1e3,
#                                            bins=bins, density=True)
#     bins_mean = 0.5*(bins_length[1:]+bins_length[:-1])
#     ax.plot(bins_mean, pdf_length,color=colors[1],alpha=1.0,label="all years")

#     coeff,pl_fit = power_law_fit(bins_mean[bins_mean<=600], pdf_length[bins_mean<=600])

#     ax.plot(bins_mean[bins_mean<=600], pl_fit,color=colors[1],alpha=1.0,linestyle='--',label="power-law fit\nexponent %.2f" %(-coeff[0]))
#     ax.plot(bins_mean[bins_mean>600], 
#             np.power(10,np.polyval(coeff,np.log10(bins_mean[bins_mean>600]))),
#             color=colors[1],alpha=1.0,linestyle=':')

#     ax.legend()

#     fig.savefig(plot_output_path + 'length_pdf.pdf')


        
# # 2. Density

# if density:
#     density_file = int_mem_path + 'density_%s_dataset.npy' %datatype
#     # Mapping grid
#     res = 50e3
#     xedg = np.arange(m.xmin,m.xmax,res)
#     yedg = np.arange(m.ymin,m.ymax,res)
#     y,x = np.meshgrid(yedg[1:],xedg[1:])

#     if os.path.exists(density_file) and not force_recompute:
#         print "Open already computed file: %s" %density_file
#         lkf_density = np.load(density_file)

#     else:
#         print "Compute density of segments"
#         res = 50e3
#         lkf_density = np.zeros((len(lkf_dataset),xedg.size-1,yedg.size-1))
        
#         for iyear in range(len(lkf_dataset)):
#             lkf_year = np.concatenate(np.concatenate(lkf_dataset[iyear]))
#             H, xedges, yedges = np.histogram2d(lkf_year[:,indm0], lkf_year[:,indm1], 
#                                                bins=(xedg, yedg))
#             lkf_density[iyear,:,:] = H
        
#         #Save output
#         print "Saving computed file: %s" %density_file
#         np.save(density_file,lkf_density)

#     norm_coverage = True
#     if norm_coverage:
#         if datatype=='rgps':
#             cov_dict = np.load(lkf_path + 'coverage_%s.npz' %datatype)
#             coverage = cov_dict['coverage']
#             lon_cov = cov_dict['lon']; lat_cov = cov_dict['lat']
#             x_cov,y_cov = m(lon_cov,lat_cov)
#             coverage_map = np.zeros((coverage.shape[0],xedg.size-1,yedg.size-1))
#             for iyear in range(coverage.shape[0]):
#                 coverage_map[iyear,:,:], xedges, yedges = np.histogram2d(x_cov.flatten(),
#                                                                          y_cov.flatten(),
#                                                                          bins=(xedg, yedg),
#                                                                          weights=coverage[iyear,:,:].flatten())

#         elif datatype == 'mitgcm_2km':
#             lon_cov, lat_cov = read_latlon(grid_path)
#             mask = mask_arcticbasin(grid_path,read_latlon)
#             index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
#             index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
#             red_fac = 3 # Take only every red_fac point to reduce array size
#             lon_cov = lon_cov[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
#                               index_x[0][0]-1:index_x[0][-1]+2:red_fac]
#             lat_cov = lat_cov[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
#                               index_x[0][0]-1:index_x[0][-1]+2:red_fac]
#             mask    =    mask[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
#                               index_x[0][0]-1:index_x[0][-1]+2:red_fac]
#             x_cov,y_cov = m(lon_cov[mask],lat_cov[mask])
#             coverage_map = np.zeros((len(years),xedg.size-1,yedg.size-1))
#             for iyear in range(coverage_map.shape[0]):
#                 coverage_map[iyear,:,:], xedges, yedges = np.histogram2d(x_cov.flatten(),
#                                                                          y_cov.flatten(),
#                                                                          bins=(xedg, yedg))
#                 coverage_map[iyear,:,:] *= len(lkf_dataset[iyear])



#     for iyear in range(len(lkf_dataset)):
#         H = lkf_density[iyear,:,:].copy()
        
#         # Plot density for year
#         fig,ax = plt.subplots(nrows=1,ncols=1)
#         if norm_coverage:
#             H /= coverage_map[iyear,:,:]
#         pcm = m.pcolormesh(x,y,np.ma.masked_where(np.isnan(H) | (H==0),H),
#                            vmin=0,vmax=0.2)
#         m.drawcoastlines()
#         cb = plt.colorbar(pcm)
#         cb.set_label('Relative LKF frequency')
#         ax.set_title('Year: %s' %years[iyear])

#     # Plot Cummulated density
#     #style_label = 'seaborn-darkgrid'
#     #with plt.style.context(style_label):
#     fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8.4,6),gridspec_kw={'width_ratios':[12,1]})
#     H = np.sum(lkf_density,axis=0)
#     if norm_coverage:
#         H /= np.sum(coverage_map,axis=0)
#     H = np.ma.masked_where(np.sum(coverage_map,axis=0)<500,H)
#     pcm = ax[0].pcolormesh(x,y,np.ma.masked_where(np.isnan(H) | (H==0),H),
#                        vmin=0,vmax=0.2)
#     m.drawcoastlines(ax=ax[0])
#     m.fillcontinents(ax=ax[0],color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0),lake_color='w')
#     cb = plt.colorbar(pcm,cax=ax[1])
#     cb.set_label('Relative LKF frequency')
#     cb.outline.set_visible(False)
#     #ax.set_title('Average over entire data set')
#     ax[0].axis('off')

#     fig.savefig(plot_output_path + 'Density_all_paper.pdf')	




# # 3. Curvature
    
# if curvature:
#     curvature_file = int_mem_path + 'curvature_%s_dataset.npy' %datatype
#     if os.path.exists(curvature_file) and not force_recompute:
#         print "Open already computed file: %s" %curvature_file
#         lkf_curvature = np.load(curvature_file)

#     else:
#         print "Compute curvature of segments"
#         lkf_curvature = []

#         for lkf_year in lkf_dataset:
#             curv_year = []
#             for lkf_day in lkf_year:
#                 curv_day = []
#                 for ilkf in lkf_day:
#                     curv_day.append(np.sum(np.sqrt((ilkf[0,indm0]-ilkf[-1,indm0])**2 +
#                                                    (ilkf[0,indm1]-ilkf[-1,indm1])**2)))

#                 curv_year.append(curv_day)

#             lkf_curvature.append(curv_year)
        
#         #Save output
#         print "Saving computed file: %s" %curvature_file
#         np.save(curvature_file,lkf_curvature)
#         lkf_curvature = np.array(lkf_curvature)

#     # Plot curvature
#     for iyear in range(len(lkf_dataset)):
#         fig,ax = plt.subplots(nrows=1,ncols=1)
#         ax.plot(np.concatenate(lkf_length[iyear])/1e3,
#                 np.concatenate(lkf_curvature[iyear])/1e3,'.')
#         ax.plot([0,1700],[0,1700],'k--')
#         ax.set_xlabel('LKF Length')
#         ax.set_ylabel('Distance between LKF endpoints')
#         ax.set_title('Winter %s/%s' %(years[iyear][1:3],years[iyear][3:]))

# 	fig.savefig(plot_output_path + 'Curvature_year_' + years[iyear]+'.pdf')



# # 4. Compute orientation


# # Compute cell that an LKF contributes to

# # Mapping grid
# res = 200e3
# xedg = np.arange(m.xmin,m.xmax,res)
# yedg = np.arange(m.ymin,m.ymax,res)
# y,x = np.meshgrid(0.5*(yedg[1:]+yedg[:-1]),
#                   0.5*(xedg[1:]+xedg[:-1]))


# if comp_cell_contrib:
#     cell_contrib_file = int_mem_path + 'cell_contrib_lkf_%s_dataset_%i_poly_%i.npy' %(datatype,res,use_poly_ori)
#     if os.path.exists(cell_contrib_file) and not force_recompute:
#         print "Open already computed file: %s" %cell_contrib_file
#         cell_contrib = np.load(cell_contrib_file)

#     else:
#         print "Compute cell contributions of lkfs"
#         cell_contrib = []

#         for lkf_year in lkf_dataset:
#             cell_contrib_year = []
#             for lkf_day in lkf_year:
#                 cell_contrib_day = []
#                 for ilkf in lkf_day:
#                     if use_poly_ori:
#                         H, xedges, yedges = np.histogram2d(ilkf[:,indp0], ilkf[:,indp1], 
#                                                            bins=(xedg, yedg))
#                     else:
#                         H, xedges, yedges = np.histogram2d(ilkf[:,indm0], ilkf[:,indm1], 
#                                                            bins=(xedg, yedg))
#                     cell_contrib_day.append(np.where(H.flatten()>0)[0])

#                 cell_contrib_year.append(cell_contrib_day)

#             cell_contrib.append(cell_contrib_year)
        
#         #Save output
#         print "Saving computed file: %s" %cell_contrib_file
#         np.save(cell_contrib_file,cell_contrib)
#         cell_contrib = np.array(cell_contrib)



# # Compute orientation

# if orientation:
#     orientation_file = int_mem_path + 'orientation_%s_dataset_%i_poly_%i.npy' %(datatype,res,use_poly_ori)
#     ori_len_wght_file = int_mem_path + 'ori_len_wght_%s_dataset_%i_poly_%i.npy' %(datatype,res,use_poly_ori)
#     if os.path.exists(orientation_file) and os.path.exists(ori_len_wght_file) and not force_recompute:
#         print "Open already computed file: %s" %orientation_file
#         lkf_orientation = np.load(orientation_file)
#         lkf_ori_len_wght = np.load(ori_len_wght_file)

#     else:
#         print "Compute orientation of segments"
#         lkf_orientation = []
#         lkf_ori_len_wght = []

#         ori_day_org = np.empty((x.shape),dtype=object)
#         for ix in range(xedg.size-1):
#             for iy in range(yedg.size-1):
#                 ori_day_org[ix,iy] = np.array([])

#         for iyear,lkf_year in enumerate(lkf_dataset):
#             ori_year = []
#             ori_len_year = []
#             for iday,lkf_day in enumerate(lkf_year):
#                 ori_day = ori_day_org.copy()
#                 ori_len_day = ori_day_org.copy()
#                 for ilkf,lkf_i in enumerate(lkf_day):
#                     for i_cell in cell_contrib[iyear][iday][ilkf]:
#                         # Find part of lkf inside box
#                         ix,iy = np.unravel_index(i_cell,x.shape)
#                         if use_poly_ori:
#                             lkf_i_c = lkf_i[:,indp0:indp0+2][np.all([lkf_i[:,indp0]>=xedg[ix],
#                                                                      lkf_i[:,indp0]<=xedg[ix+1],
#                                                                      lkf_i[:,indp1]>=yedg[iy],
#                                                                      lkf_i[:,indp1]<=yedg[iy+1]],
#                                                                     axis=0),:]
#                         else:
#                             lkf_i_c = lkf_i[:,indm0:indm0+2][np.all([lkf_i[:,indm0]>=xedg[ix],
#                                                                      lkf_i[:,indm0]<=xedg[ix+1],
#                                                                      lkf_i[:,indm1]>=yedg[iy],
#                                                                      lkf_i[:,indm1]<=yedg[iy+1]],
#                                                                     axis=0),:]

#                         # Linear fit & determine angle from linear fit
#                         if lkf_i_c.size > 2:
#                             # All cases that are not a line in y-direction
#                             p_x,p_y = lkf_poly_fit_p(lkf_i_c[:,0],lkf_i_c[:,1],
#                                                      1) # Linear fit
#                             p = p_y[0]/p_x[0]
                            
#                             # Determin angle from linear fit
#                             if np.isnan(p):
#                                 ang = 90.
#                             else:
#                                 ang = np.arctan(p)/np.pi*180.
                            
#                             ori_day[ix,iy] = np.concatenate([ori_day[ix,iy],
#                                                              np.array([ang])])
#                             ori_len_day[ix,iy] = np.concatenate([ori_len_day[ix,iy],
#                                                                  np.array([lkf_i_c.shape[0]])])

                
#                 ori_year.append(ori_day)
#                 ori_len_year.append(ori_len_day)

#             lkf_orientation.append(ori_year)
#             lkf_ori_len_wght.append(ori_len_year)
        
#         #Save output
#         print "Saving computed file: %s" %orientation_file
#         np.save(orientation_file,lkf_orientation)
#         np.save(ori_len_wght_file,lkf_ori_len_wght)
#         lkf_orientation = np.array(lkf_orientation)
#         lkf_ori_len_wght = np.array(lkf_ori_len_wght)




# # Define function to plot radial histogram
# def plot_radial_hist(ang,wght,x0,y0,max_rad,nbins=10,ax=plt.gca,color='b'):
#     if nbins%2==1: nbins += 1.
#     binint=180/float(nbins)
#     bins = np.arange(-90+binint/2.,90+binint,binint)
#     ang[ang<bins[0]] += 180.
#     hist,bins_ang = np.histogram(ang,bins,weights=wght)

#     # Plot radial histogram
#     for ihist,hist_i in enumerate(hist):
#         ang_hist = 0.5*(bins[ihist]+bins[ihist+1])
#         r = hist_i/float(wght.sum())*max_rad
#         ax.plot([x0-r*np.cos(ang_hist/180.*np.pi),
#                  x0+r*np.cos(ang_hist/180.*np.pi)],
#                 [y0-r*np.sin(ang_hist/180.*np.pi),
#                  y0+r*np.sin(ang_hist/180.*np.pi)],
#                 color=color)
                

# # Define functions that compute mean angle and std, chi square test

# def average_angle(ang,wght):
#     x = np.sum(wght*np.cos(2*ang/180.*np.pi))
#     y = np.sum(wght*np.sin(2*ang/180.*np.pi))
#     return np.arctan2(y,x)/np.pi*180./2.
    
# def std_angle(ang,wght,ang_m):
#     ang_diff = np.abs(ang-ang_m)
#     ang_diff[ang_diff>90] = np.abs(ang[ang_diff>90]-180.-ang_m)
#     return np.sqrt(np.sum(ang_diff**2*wght)/np.sum(wght))

# def chisquare_sig(ang,wght,nchi=int(1e4),nbins=10,pmax=0.01):
#     p = np.zeros((nchi,))

#     # Relative frequency of observed orientations
#     binint=180/float(nbins); bins = np.arange(-90,90+binint/2.,binint)
#     hist_obs,bins_ang = np.histogram(ang,bins,weights=wght)

#     for i in range(nchi):
#         # Create random distribution
#         ang_rand = 180.*np.random.random((int(wght.sum()),)) - 90.
#         hist_rand,bins_ang = np.histogram(ang_rand,bins)
#         # Chi squared test
#         chisq, p[i] = scipy.stats.chisquare(hist_rand/float(ang.size),
#                                             hist_obs/float(wght.sum()))

#     p_mean = p.mean()
#     return p_mean<=pmax, p_mean


# def map_rad_hist(ori,ori_wght,x,y,res,nbins=8,color='b'):
#     fig,ax = plt.subplots(nrows=1,ncols=1)
#     m.drawcoastlines(ax=ax)
#     for ix in range(ori.shape[0]):
#         for iy in range(ori.shape[1]):
#             plot_radial_hist(ori[ix,iy],ori_wght[ix,iy],x[ix,iy],y[ix,iy],
#                              res/2,nbins=nbins,ax=ax,color=color)

# def map_mean_std_ori(ori,ori_wght,x,y,res,color='b',
#                      do_chi=False,nchi=int(1e4),nbins=10,pmax=0.01,
#                      color_dens=True):
#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
#     m.drawcoastlines(ax=ax)
#     std_bounds = [0,15,30,45,180]
#     std_linew = [0.5,1,1.5,2.]
#     if color_dens:
#         ori_dens = np.zeros(ori.shape)
#         for ix in range(ori.shape[0]):
#             for iy in range(ori.shape[1]):
#                 ori_dens[ix,iy] = ori[ix,iy].size
#         dens_max = np.ceil(ori_dens.max()/100.)*100.
#         dens_min = np.floor(ori_dens.min()/100.)*100.
#         cm_dens = plt.get_cmap('plasma',100)
#         # plt.figure();m.drawcoastlines(); m.pcolormesh(x,y,ori_dens);plt.colorbar()
#         # print dens_max,dens_min
        
#     for ix in range(ori.shape[0]):
#         for iy in range(ori.shape[1]):
#             # Compute mean, std and chi squared test
#             ang_mean = average_angle(ori[ix,iy],ori_wght[ix,iy])
#             ang_std = std_angle(ori[ix,iy],ori_wght[ix,iy],ang_mean)
#             std_class = np.floor(ang_std/15.);
#             if std_class>len(std_linew)-1: std_class=len(std_linew)-1
            

#             # Plot direction
#             if ~np.isnan(ang_std):
#                 if color_dens:
#                     color = cm_dens((ori_dens[ix,iy]-dens_min)/float(dens_max-dens_min))
#                 ax.plot([x[ix,iy]-res/2.*np.cos(ang_mean/180.*np.pi),
#                          x[ix,iy]+res/2.*np.cos(ang_mean/180.*np.pi)],
#                         [y[ix,iy]-res/2.*np.sin(ang_mean/180.*np.pi),
#                          y[ix,iy]+res/2.*np.sin(ang_mean/180.*np.pi)],
#                         color=color,linewidth=std_linew[int(std_class)])
#                 if do_chi:
#                     chi,p = chisquare_sig(ori[ix,iy],ori_wght[ix,iy],
#                                           nchi=nchi,nbins=nbins,pmax=pmax)
#                     if chi:
#                         ax.plot(x[ix,iy],y[ix,iy],'k.')

#     if color_dens:
#         ax_pos = ax.get_position().get_points()
#         mar = (ax_pos[0,0] + 1-(ax_pos[1,0]))/2.
#         mar_cbar = 0.005
#         ax.set_position([mar/2.,ax_pos[0,1],ax_pos[1,0]-ax_pos[0,0],
#                          ax_pos[1,1]-ax_pos[0,1]])
#         cax = fig.add_axes([mar/2.+ax_pos[1,0]-ax_pos[0,0]+mar_cbar, ax_pos[0,1], 
#                             0.3*mar-2*mar_cbar, ax_pos[1,1]-ax_pos[0,1]])
#         print ax.get_position(), cax.get_position()
#         norm = mpl.colors.Normalize(vmin=dens_min, vmax=dens_max)
#         cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm_dens,norm=norm)
#         cb1.set_label('Number of LKFs')

#     for i in range(len(std_linew)):
#         ax.plot([m.xmin,m.xmin],[m.ymin,m.ymin],linewidth=std_linew[i],color='k',
#                 label = "%i - %i deg std" %(std_bounds[i],std_bounds[i+1]))

#     ax.legend(loc="upper left",title='STD')

#     return fig

    
# # Plot orientation in radial histograms

# if orientation:
#     ori_org = np.empty((x.shape),dtype=object)
#     for ix in range(xedg.size-1):
#         for iy in range(yedg.size-1):
#             ori_org[ix,iy] = np.array([])

#     ori = []
#     ori_wght = []

#     for iyear,lkf_year in enumerate(lkf_dataset):
#         ori_month = np.stack([ori_org.copy() for i in range(12)])
#         ori_wght_month = np.stack([ori_org.copy() for i in range(12)])
#         for iday,lkf_day in enumerate(lkf_year):
#             imonth = lkf_meta[iyear][iday][0].month - 1
#             for ix in range(ori_month.shape[1]):
#                 for iy in range(ori_month.shape[2]):
#                     ori_month[imonth,ix,iy] = np.concatenate([ori_month[imonth,ix,iy],
#                                                               lkf_orientation[iyear][iday][ix,iy]])
#                     ori_wght_month[imonth,ix,iy] = np.concatenate([ori_wght_month[imonth,ix,iy],
#                                                             lkf_ori_len_wght[iyear][iday][ix,iy]])

#         ori.append(ori_month)
#         ori_wght.append(ori_wght_month)

    
#     ori_mean = ori_org.copy()
#     ori_wght_mean = ori_org.copy()
#     ori_year_mean = np.stack([ori_org.copy() for i in range(len(years))])
#     ori_wght_year_mean = np.stack([ori_org.copy() for i in range(len(years))])
#     ori_month_mean = np.stack([ori_org.copy() for i in range(12)])
#     ori_wght_month_mean = np.stack([ori_org.copy() for i in range(12)])

#     for iyear,lkf_year in enumerate(lkf_dataset):
#         for imonth in range(12):
#             for ix in range(ori_month_mean.shape[1]):
#                 for iy in range(ori_month_mean.shape[2]):
#                     ori_month_mean[imonth,ix,iy] = np.concatenate([ori_month_mean[imonth,ix,iy],
#                                                                    ori[iyear][imonth,ix,iy]])
#                     ori_wght_month_mean[imonth,ix,iy] = np.concatenate([ori_wght_month_mean[imonth,ix,iy],
#                                                                         ori_wght[iyear][imonth,ix,iy]])
#                     ori_year_mean[iyear,ix,iy] = np.concatenate([ori_year_mean[iyear,ix,iy],
#                                                                  ori[iyear][imonth,ix,iy]])
#                     ori_wght_year_mean[iyear,ix,iy] = np.concatenate([ori_wght_year_mean[iyear,ix,iy],
#                                                                       ori_wght[iyear][imonth,ix,iy]])
              
#                     ori_mean[ix,iy] = np.concatenate([ori_mean[ix,iy],ori[iyear][imonth,ix,iy]])
#                     ori_wght_mean[ix,iy] = np.concatenate([ori_wght_mean[ix,iy],
#                                                            ori_wght[iyear][imonth,ix,iy]])
        
        
                   
    

#     if plot_ori_mean:
#         if plot_rad_hist:
#             # Plot radial histogram for mean orientation of data set
#             map_rad_hist(ori_mean,ori_wght_mean,x,y,res,nbins=8,color='b')
        
#         if plot_broehan:
#             # Plot mean orientation of data set with std as linewidth
#             fig = map_mean_std_ori(ori_mean,ori_wght_mean,x,y,res,color='b',
#                   	           do_chi=True,nchi=int(1e4),nbins=20,pmax=0.01,
#                         	   color_dens=True)
#             fig.savefig(plot_output_path + 'Mean_ori_all_200.pdf')
        



# # 5. Deformation rate diagram

# if deformation:
#     deformation_file = int_mem_path + 'deformation_%s_dataset.npy' %datatype
#     if os.path.exists(deformation_file) and not force_recompute:
#         print "Open already computed file: %s" %deformation_file
#         lkf_deformation = np.load(deformation_file)

#     else:
#         print "Compute deformation of segments"
#         lkf_deformation = []

#         for lkf_year in lkf_dataset:
#             defo_year = []
#             for lkf_day in lkf_year:
#                 defo_day = []
#                 for ilkf in lkf_day:
#                     defo_day.append([np.mean(ilkf[:,indd0]),np.mean(ilkf[:,indd1])])

#                 defo_year.append(defo_day)

#             lkf_deformation.append(defo_year)
        
#         #Save output
#         print "Saving computed file: %s" %deformation_file
#         np.save(deformation_file,lkf_deformation)
#         lkf_deformation = np.array(lkf_deformation)

#     deform_all = np.vstack([np.vstack([np.stack([np.array(iseg) for iseg in lkf_deformation[i][j]]) 
#                                        for j in range(len(lkf_deformation[i]))]) 
#                             for i in range(len(lkf_dataset))])

#     shr_lim = [0,0.3]
#     div_lim = [-0.15,0.15]
#     nbins_shr = 500
#     nbins_div = 500
    
#     hist2d,div_edg,shr_edg = np.histogram2d(deform_all[:,0], deform_all[:,1],
#                                             [np.linspace(div_lim[0],div_lim[1],nbins_div),
#                                              np.linspace(shr_lim[0],shr_lim[1],nbins_shr)],
#                                             )
#     hist2d = np.ma.masked_where(hist2d==0,hist2d)

#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
#     hist_lims = [2,300/4]
#     pcm = ax.pcolormesh(shr_edg,div_edg,np.ma.masked_where(hist2d==0,hist2d),
#                         norm=mpl.colors.LogNorm(vmin=hist_lims[0], vmax=hist_lims[1]))
#     ax.plot([shr_edg[0],shr_edg[-1]],[0,0],'k--')
#     ax.set_ylabel('Divergence rate [1/day]')
#     ax.set_xlabel('Shear rate [1/day]')
#     ax.set_aspect('equal')
#     cb = fig.colorbar(pcm, ax=ax, extend='both')
#     cb.set_label('Number of LKFs')

#     fig.savefig(plot_output_path + 'deformation_shr_div_hist.pdf')


# # 6. Intersections angles

# if intersection:
#     pos_type = 'poly'
#     if pos_type=='ind':
#         indc0 = 0; indc1 = 1;
#     if pos_type=='m':
#         indc0 = indm0; indc1 = indm1;
#     if pos_type=='poly':
#         indc0 = indp0; indc1 = indp1;
#     num_p = 10 # Number of points on each side of the intersection
#     #            contribute to the orientation computation

#     interc_file     = int_mem_path + 'interc_%s_dataset_num%i_%s.npy' %(datatype, num_p, pos_type)
#     interc_par_file = int_mem_path + 'interc_par_%s_dataset_num%i_%s.npy' %(datatype, num_p, pos_type)

#     if os.path.exists(interc_file) and os.path.exists(interc_par_file) and not force_recompute:
#         print "Open already computed file: %s" %interc_file
#         lkf_interc     = np.load(interc_file)
#         lkf_interc_par = np.load(interc_par_file)

#     else:
#         print "Compute interc of segments"
 
#         if datatype == 'mitgcm_2km':
#             mask = mask_arcticbasin(grid_path,read_latlon)
#             index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
#             index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
#             red_fac = 3 # Take only every red_fac point to reduce array size


#         lkf_interc     = []
#         lkf_interc_par = []
#         for iyear in range(len(lkf_dataset)):
#             intc_ang_year = []
#             intc_par_year = []
#             for iday in range(len(lkf_dataset[iyear])):
#                 if datatype == 'rgps':
#                     lkf_map = np.zeros((248,264))
#                 elif datatype == 'mitgcm_2km':
#                     lkf_map = np.zeros((int(np.ceil((index_y[0][-1]+1-index_y[0][0]+1)/3.)),
#                                         int(np.ceil((index_x[0][-1]+1-index_x[0][0]+1)/3.))))


#                 for iseg, seg_i in enumerate(lkf_dataset[iyear][iday]):
#                     lkf_map[seg_i[:,0].astype('int'),seg_i[:,1].astype('int')] += iseg

#                 intc_ang_day = []
#                 intc_par_day = []

#                 # Check for possible intersection partners
#                 for iseg, seg_i in enumerate(lkf_dataset[iyear][iday]):
#                     search_ind = np.zeros(lkf_map.shape).astype('bool')
#                     search_ind[seg_i[:,0].astype('int')+1,seg_i[:,1].astype('int')  ] = True
#                     search_ind[seg_i[:,0].astype('int')-1,seg_i[:,1].astype('int')  ] = True
#                     search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')+1] = True
#                     search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')-1] = True
#                     search_ind[seg_i[:,0].astype('int')+1,seg_i[:,1].astype('int')+1] = True
#                     search_ind[seg_i[:,0].astype('int')-1,seg_i[:,1].astype('int')-1] = True
#                     search_ind[seg_i[:,0].astype('int')-1,seg_i[:,1].astype('int')+1] = True
#                     search_ind[seg_i[:,0].astype('int')+1,seg_i[:,1].astype('int')-1] = True
#                     search_ind[seg_i[:,0].astype('int')  ,seg_i[:,1].astype('int')  ] = False

#                     intercep_points = np.where(search_ind & (lkf_map!=0))

#                     intercep_partners, intercep_counts = np.unique(lkf_map[intercep_points],
#                                                                    return_counts=True)

#                     for ipar,pari in enumerate(intercep_partners):
#                         if pari > iseg and pari < len(lkf_dataset[iyear][iday]):
#                             # Determine one intercetion point for pair
#                             dis_intercep = np.zeros(intercep_counts[ipar])
#                             for iintc in range(intercep_counts[ipar]):
#                                 dis_intercep[iintc] = np.min(np.sqrt((seg_i[:,0] - 
#                                                                       intercep_points[0][lkf_map[intercep_points]==pari][iintc])**2 + 
#                                                                      (seg_i[:,1] - 
#                                                                       intercep_points[1][lkf_map[intercep_points]==pari][iintc])**2))
#                             intcp = (intercep_points[0][lkf_map[intercep_points]==pari][np.argmin(dis_intercep)],
#                                      intercep_points[1][lkf_map[intercep_points]==pari][np.argmin(dis_intercep)])
            
#                             # Determine angle between both pairs
#                             # # Determine orientation of seg_i
#                             ind = np.argmin(np.sqrt((seg_i[:,0] - intcp[0])**2 + 
#                                                     (seg_i[:,1] - intcp[1])**2))
#                             ind = np.array([np.max([0,ind-num_p]),
#                                             np.min([seg_i.shape[0],ind+num_p+1])])
#                             p_x,p_y = lkf_poly_fit_p(seg_i[ind[0]:ind[1],indc0],
#                                                      seg_i[ind[0]:ind[1],indc1],1) # Linear fit
#                             p = p_y[0]/p_x[0]
#                             # # Determin angle from linear fit
#                             if np.isnan(p):
#                                 ang_i = 90.
#                             else:
#                                 ang_i = np.arctan(p)/np.pi*180.

#                             # # Determine orientation of pari
#                             lkf_par = lkf_dataset[iyear][iday][int(pari)]
#                             ind = np.argmin(np.sqrt((lkf_par[:,0] - intcp[0])**2 + 
#                                                     (lkf_par[:,1] - intcp[1])**2))
#                             ind = np.array([np.max([0,ind-num_p]),
#                                             np.min([lkf_par.shape[0],ind+num_p+1])])
#                             p_x,p_y = lkf_poly_fit_p(lkf_par[ind[0]:ind[1],indc0],
#                                                      lkf_par[ind[0]:ind[1],indc1],1) # Linear fit
#                             p = p_y[0]/p_x[0]
#                             # # Determin angle from linear fit
#                             if np.isnan(p):
#                                 ang_ii = 90.
#                             else:
#                                 ang_ii = np.arctan(p)/np.pi*180.
                                
#                             angdiff = np.abs(ang_ii-ang_i)
#                             if angdiff > 90: angdiff=180-angdiff
#                             intc_ang_day.append(angdiff)
#                             intc_par_day.append(np.array([iseg,pari]))

#                 intc_ang_year.append(intc_ang_day)
#                 intc_par_year.append(intc_par_day)
#             lkf_interc.append(intc_ang_year)
#             lkf_interc_par.append(intc_par_year)
            
#         #Save output
#         print "Saving computed file: %s" %interc_file
#         np.save(interc_file,lkf_interc)
#         np.save(interc_par_file,lkf_interc_par)

#         lkf_interc     = np.array(lkf_interc)
#         lkf_interc_par = np.array(lkf_interc_par)


#     # Compute histograms
#     # - one plot with lines for each year
#     nbins = 45
#     bins = np.linspace(0,90,nbins)
#     fig,ax = plt.subplots(nrows=1,ncols=1)
#     ax.set_xlabel('Intersection angle')
#     ax.set_ylabel('PDF')
#     ax.set_xlim([0,90])
#     for iyear in range(len(lkf_interc)):
#         pdf_interc, bins_interc = np.histogram(np.concatenate(lkf_interc[iyear]),
#                                                bins=bins, density=True)
#         bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#         ax.plot(bins_mean, pdf_interc,label=years[iyear],color='0.5',alpha=0.5)

#     pdf_interc, bins_interc = np.histogram(np.concatenate([np.concatenate(lkf_interc[iyear]) for iyear in range(len(lkf_interc))]),
#                                            bins=bins, density=True)
#     bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#     ax.plot(bins_mean, pdf_interc,label=years[iyear],color='k',alpha=1.0)

#     #ax.legend()
#     fig.savefig(plot_output_path + 'interc_pdf.pdf')



#     # Plot intersection angle statistics depending on deformation rate 
#     if link_interc_def:
#         # Compute mean deformation of intersecting partners
#         def_par = []; diff_def_par = [];
#         for iyear in range(len(lkf_dataset)):
#             def_par_year = []; diff_def_par_year = [];
#             for iday in range(len(lkf_dataset[iyear])):
#                 def_par_day = np.array([np.sqrt(np.sum(np.array(lkf_deformation[iyear][iday])[np.stack(lkf_interc_par[iyear][iday])[:,0].astype('int')]**2,axis=1)),
#                                         np.sqrt(np.sum(np.array(lkf_deformation[iyear][iday])[np.stack(lkf_interc_par[iyear][iday])[:,1].astype('int')]**2,axis=1))])
#                 diff_def_par_day = np.abs(np.diff(def_par_day,axis=0))
                
#                 def_par_year.append(def_par_day)
#                 diff_def_par_year.append(diff_def_par_day)
#             def_par.append(def_par_year)
#             diff_def_par.append(diff_def_par_year)

            
#         # Plot histograms in different deformation rate classes
#         fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4.8))
#         if datatype=='rgps':
#             def_class = [0,0.03,0.05,2]
#         elif datatype == 'mitgcm_2km':
#             def_class = [0,0.05,0.2,10]
#         nbins = 45
#         bins = np.linspace(0,90,nbins)
#         def_masked_class = [[],[],[]]
#         for iax,axi in enumerate(ax):
#             axi.set_title('Def. class: %.2f to %.2f [1/day]' %(def_class[iax],def_class[iax+1]))
#             axi.set_xlabel('Intersection angle')
#             axi.set_ylabel('PDF')
#             axi.set_xlim([0,90])
            
#             for iyear in range(len(lkf_interc)):
#                 mask_def = np.all([np.all(np.hstack(def_par[iyear])>=def_class[iax],axis=0),
#                                    np.all(np.hstack(def_par[iyear])<def_class[iax+1],axis=0)],
#                                   axis=0)
#                 pdf_interc, bins_interc = np.histogram(np.concatenate(lkf_interc[iyear])[mask_def],
#                                                bins=bins, density=True)
#                 bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#                 axi.plot(bins_mean, pdf_interc,label=years[iyear],color='0.5',alpha=0.5)
#                 def_masked_class[iax].append(np.concatenate(lkf_interc[iyear])[mask_def])
#             pdf_interc, bins_interc = np.histogram(np.concatenate(def_masked_class[iax]),
#                                                    bins=bins, density=True)
#             bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#             axi.plot(bins_mean, pdf_interc,label=years[iyear],color='k',alpha=1.0)
#             axi.text(axi.get_xlim()[0]+0.1*axi.get_xlim()[1],
#                      axi.get_ylim()[0]+0.9*axi.get_ylim()[1],
#                      'Tot. num: %i' %np.concatenate(def_masked_class[iax]).size)

#             fig.savefig(plot_output_path + 'interc_pdf_def_class.pdf')





# # Statistics on tracking data

# # 1. Lifetime of LKFs

# if lifetime:
#     lifetime_file = int_mem_path + 'lifetime_%s_dataset.npy' %datatype
#     if os.path.exists(lifetime_file) and not force_recompute:
#         print "Open already computed file: %s" %lifetime_file
#         lkf_lifetime = np.load(lifetime_file)

#     else:
#         print "Compute lifetime of segments"
#         lkf_lifetime = []

#         for iyear,lkf_year in enumerate(lkf_dataset):
#             life_year = [np.ones((len(i_num_lkf),)) for i_num_lkf in lkf_year]
#             for it,itrack in enumerate(lkf_track_data[iyear]):
# 		if itrack.size>0:
# 		    life_year[it+1][itrack[:,1].astype('int')] += life_year[it][itrack[:,0].astype('int')]

#             lkf_lifetime.append(life_year)

#         #Save output
#         print "Saving computed file: %s" %lifetime_file
#         np.save(lifetime_file,lkf_lifetime)
#         lkf_lifetime = np.array(lkf_lifetime)


#     # Generate Plots
#     # One plot for each year
#     for iyear in range(len(lkf_lifetime)):
# 	#fig,ax = plt.subplots(nrows=1,ncols=2)
# 	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4),
#                                gridspec_kw={'width_ratios':[4,4,1]})

# 	# Determine all existing lifetimes
#         lt_max = np.max([np.max(ilife) for ilife in lkf_lifetime[iyear]])
#         lt_class = np.arange(lt_max)+1


# 	# Compute the percentage of each lifetime for each timeslice
#         lt_perc = np.zeros((len(lkf_lifetime[iyear]),lt_class.size))
#         lt_abs  = np.zeros((len(lkf_lifetime[iyear]),lt_class.size))
#         for it,ilife in enumerate(lkf_lifetime[iyear]):
#             for ilt in lt_class:
#                 lt_perc[it,int(ilt-1)] = np.sum(ilife==ilt)/float(ilife.size)
#                 lt_abs[it,int(ilt-1)]  = np.sum(ilife==ilt)

# 	# Make plot
# 	cmap = plt.get_cmap('viridis')
#         col = cmap(np.linspace(0,1,lt_class.size))

# 	for ic in range(lt_class.size):
#             ax[0].plot(3*np.arange(len(lkf_lifetime[iyear])), np.array(lt_perc[:,ic])*100,
#                          color = col[ic])

#         # Make line plot absolute lifetime numbers
#         for ic in range(lt_class.size):
#             ax[1].plot(3*np.arange(len(lkf_lifetime[iyear])), np.array(lt_abs[:,ic]),
#                          color = col[ic])

# 	# Labeling x axis
#         xax = [ax[0],ax[1]]
#         for iax in xax: iax.set_xlabel('Time [days]')

# 	# Plot colorbar
#     	norm = mpl.colors.Normalize(vmin=1, vmax=lt_max)
# 	cbar = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap,norm=norm)
# 	cbar.set_label('Lifetime [days]')

# 	# Labeling yaxis
# 	ax[0].set_ylabel('Fraction [%]')
# 	ax[1].set_ylabel('Absolute numbers')

#     # Compute histograms
#     # - one plot with lines for each year
#     style_label = 'seaborn-darkgrid'
#     with plt.style.context(style_label):
#         fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(6,5))
#     ax.set_xlabel('LKF lifetime')
#     ax.set_ylabel('Relative frequency')
#     ax.set_yscale('log')
#     ax.set_xlim([0,31])
#     colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
#     for iyear in range(len(lkf_length)):
#         pdf_life = np.bincount(np.concatenate(lkf_lifetime[iyear]).astype('int')-1)
#         bins_mean = np.arange(pdf_life.size)*3+1.5
#         if iyear==0:
#             ax.plot(bins_mean, pdf_life/float(np.sum(pdf_life)),'.',color='0.5',alpha=0.5,label="single years")
#         else:
#             ax.plot(bins_mean, pdf_life/float(np.sum(pdf_life)),'.',color='0.5',alpha=0.5)

#     pdf_life = np.bincount(np.concatenate([np.concatenate(lkf_lifetime[iyear]) for iyear in range(len(lkf_lifetime))]).astype('int')-1)
#     bins_mean = np.arange(pdf_life.size)*3+1.5
#     ax.plot(bins_mean, pdf_life/float(np.sum(pdf_life)),'.',color=colors[1],alpha=1.0,label="all years")

#     coeff = np.polyfit(bins_mean, np.log(pdf_life/float(np.sum(pdf_life))),1)
#     pdf_life_fit = np.exp(np.polyval(coeff,bins_mean))
#     ax.plot(bins_mean,pdf_life_fit,color=colors[1],alpha=1.0,linestyle='--',label="exponential fit\nexponent %.2f" %(-coeff[0]))

#     #ax.plot(bins_mean[bins_mean<=600], pl_fit,color=colors[1],alpha=1.0,linestyle='--',label="power-law fit\nexponent %.2f" %(-coeff[0]))
#     #ax.plot(bins_mean[bins_mean>600], 
#     #        np.power(10,np.polyval(coeff,np.log10(bins_mean[bins_mean>600]))),
#     #        color=colors[1],alpha=1.0,linestyle=':')

#     ax.legend()

#     fig.savefig(plot_output_path + "Lifetime_distribution_exp_fit.pdf")


# #     # Lifetime bar plot with atmospheric link
# # 
# #     # Load atmospheric grid
# #     coord = np.load(lkf_path + 'coord_jra55.npz')
# #     lat_cut = 52 + 1
# #     lon_atm = coord['lon']; lat_atm = coord['lat'][:lat_cut]
# #     lon_atm,lat_atm = np.meshgrid(lon_atm,lat_atm)
# #     x_atm,y_atm = m(lon_atm,lat_atm)
# # 
# #     # Load RGPS grid
# #     lon_rgps = np.load(lkf_path+'coverage_rgps.npz')['lon'][:]
# #     lat_rgps = np.load(lkf_path+'coverage_rgps.npz')['lat'][:]
# #     x_rgps,y_rgps = m(lon_rgps,lat_rgps)
# # 
# #     # Initialize interpolation routine
# #     interp_jra = griddata_fast(x_atm,y_atm,x_rgps,y_rgps)
# # 
# #     for iyear in range(len(lkf_lifetime)):
# # 	# Read Coverage of RGPS data
# # 	coverage = np.load(lkf_path+'coverage_rgps_%s.npz' %years[iyear])['coverage'][:]
# # 	
# # 	# Read surface pressure fields
# # 	year_cov = np.unique([lkf_meta[iyear][0][0].year,lkf_meta[iyear][-1][0].year])
# # 	atm_file = '/work/ollie/projects/clidyn/forcing/JRA55_3h/fcst_surf.001_pres.reg_tl319.'
# # 	year_lkf = [imeta[0].year for imeta in lkf_meta[iyear]]
# # 	ncfile = Dataset('/work/ollie/projects/clidyn/forcing/JRA55_3h/fcst_surf.001_pres.reg_tl319.2000.nc','r')	
# # 
# # 	
# # 	# Optinal filtering of lifetimes
# # 	lifetime_filt = np.copy(lkf_lifetime[iyear])
# # 
# # 	# Determine all existing lifetimes
# #         lt_max = np.max([np.max(ilife) for ilife in lifetime_filt])
# #         lt_class = np.arange(lt_max)+1
# # 
# #         # Compute the percentage of each lifetime for each timeslice
# #         lt_abs = np.zeros((len(lifetime_filt),lt_class.size))
# #       	for it,ilife in enumerate(lifetime_filt):
# # 	    for ilt in lt_class:
# # 	    	lt_abs[it,int(ilt-1)]  = np.sum(ilife==ilt)/float(np.sum(coverage[it,:,:]))
# # 
# # 	# Generate plot
# # 	fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(10,4),
# #         #                       gridspec_kw={'width_ratios':[4,4,1]})
# #         
# # 	# Make line plot absolute lifetime numbers
# #         for ic in range(lt_class.size):
# #             ax.plot(3*np.arange(len(lkf_lifetime[iyear])), np.array(lt_abs[:,ic]),
# #               	    color = col[ic])


#     # Lifetime bar plot linked to storms

#     # Load ncep strom data set
#     storms_path = '/work/ollie/nhutter/ncep_storms/'
#     sys.path.append(storms_path)
#     from read_ncepstorms import storms

#     storm = storms(storms_path)
#     storm.filt_storms()
#     storms_all = []

#     for iyear in range(len(lkf_lifetime)):
# 	if datatype=='rgps':
#             # Read Coverage of RGPS data
#             coverage = np.load(lkf_path+'coverage_rgps_%s.npz' %years[iyear])['coverage'][:]

#         # Read surface pressure fields
#         storms_year = []
# 	for it in range(len(lkf_lifetime[iyear])):
# 	    start_it  = lkf_meta[iyear][it][0]
# 	    end_it    = lkf_meta[iyear][it][1]
# 	    storms_it = storm.get_storms(start_it,end_it)
# 	    sto_it = []
# 	    for ist in np.unique(storms_it[:,-1]):
# 		sto_it.append([ist, storms_it[storms_it[:,-1]==ist,-2].max()])
# 	    storms_year.append(sto_it)
#         storms_all.append(storms_year)        

#         # Optinal filtering of lifetimes
#         lifetime_filt = np.copy(lkf_lifetime[iyear])

#         # Determine all existing lifetimes
#         lt_max = np.max([np.max(ilife) for ilife in lifetime_filt])
#         lt_class = np.arange(lt_max)+1

#         # Compute the percentage of each lifetime for each timeslice
#         lt_abs = np.zeros((len(lifetime_filt),lt_class.size))
#         for it,ilife in enumerate(lifetime_filt):
#             for ilt in lt_class:
#                 if datatype=='rgps':
#                     lt_abs[it,int(ilt-1)]  = np.sum(ilife==ilt)/float(np.sum(coverage[it,:,:]))
#                 else:
#                     lt_abs[it,int(ilt-1)]  = np.sum(ilife==ilt)

#         # Generate plot
#         fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(10,4),
#         #                       gridspec_kw={'width_ratios':[4,4,1]})

#         axt = ax.twinx()

#         # Make line plot absolute lifetime numbers
#         for ic in range(lt_class.size):
#             ax.plot(3*np.arange(len(lkf_lifetime[iyear])), np.array(np.cumsum(lt_abs,axis=1)[:,ic]),
#                     color = col[ic])

#         # Plot storms
#         cmap_storm = plt.get_cmap('inferno')
#         storm_stren_max = 25.
#         storm_stren_min = 0.
#         for it in range(len(lkf_lifetime[iyear])):
#             for isto in range(len(storms_year[it])):
#                 axt.plot(3*it,storms_year[it][isto][1],'.',
#                          color=cmap_storm((storms_year[it][isto][1]-storm_stren_min)/(storm_stren_max-storm_stren_min)))


# 	# Deformation plot

# 	deformation_filt = np.copy(lkf_deformation[iyear])
# 	num_class = 25
# 	def_class = np.linspace(0,2,num_class+1)
# 	def_class = np.concatenate([np.array([0]),np.logspace(-2,0.5,num_class)])	
# 	def_class = np.concatenate([np.logspace(-2,0.,num_class),np.array([np.inf])])

# 	lkf_def_abs = np.zeros((len(deformation_filt),num_class))

# 	for it,idef in enumerate(deformation_filt):
# 	    lkf_def_abs[it,:],bins  = np.histogram(np.sqrt(np.sum(np.array(idef)**2,axis=1)),def_class)#,weights=lkf_length[iyear][it])
#             if datatype=='rgps':
# 	        lkf_def_abs[it,:] /= float(np.sum(coverage[it,:,:]))

#         # Generate plot
#         style_label = 'seaborn-darkgrid'
#         with plt.style.context(style_label):
#             fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6), sharex='col',tight_layout=True,
#                                    gridspec_kw={'height_ratios':[1,2],'width_ratios':[25,1]})
# 	#fig = plt.figure()
# 	#ax = fig.add_subplot(111, projection='polar')
# 	#axt = ax.twinx()
# 	#rlim = 0.02

# 	# Make bar plot
#         bottoms = np.zeros((len(deformation_filt),))
#         wid=1*3
#         cmap_def = plt.get_cmap('inferno')
#         col = cmap_def(np.linspace(0,1,num_class))

#         for ic in range(num_class):
#             heights = np.array(lkf_def_abs[:,ic])
#             ax[1,0].bar(lkf_meta[iyear][:,0],#3*np.arange(len(deformation_filt)), 
# 		        heights, width = wid, bottom=bottoms,
#                         color = col[ic])
#             bottoms += heights

        
# 	# Plot storms
# 	cmap_storm = plt.get_cmap('YlOrRd')#inferno')
# 	storm_stren_max = 30.
# 	storm_stren_min = 0.
# 	for it in range(len(lkf_deformation[iyear])):
# 	    for isto in range(len(storms_year[it])):
# 	    	ax[0,0].plot(lkf_meta[iyear][it,0],#3*it,
# 			     storms_year[it][isto][1],'.',
# 	  		     color=cmap_storm((storms_year[it][isto][1]-storm_stren_min)/(storm_stren_max-storm_stren_min)))

#         import matplotlib.dates as mdates
#         months = mdates.MonthLocator(range(1, 13), bymonthday=1, interval=1)
#         monthsFmt = mdates.DateFormatter("%b")
        
#         ax[0,0].xaxis_date()
#         ax[1,0].xaxis_date()
        
#         ax[1,0].xaxis.set_major_locator(months)
#         ax[1,0].xaxis.set_major_formatter(monthsFmt)
#         ax[0,0].set_yticklabels([])
        
#         ax[0,0].set_ylabel('Storm stength')
#         ax[1,0].set_ylabel('No. of LKFs (normalized)')

#         # Plot storm colorbar
#         norm_sto = mpl.colors.Normalize(vmin=storm_stren_min, vmax=storm_stren_max)
#         norm_sto = mpl.colors.BoundaryNorm(boundaries=np.linspace(storm_stren_min,
#                                                                   storm_stren_max,13)
#                                            ,ncolors=256)
#         cbar_sto = mpl.colorbar.ColorbarBase(ax[0,1], cmap=cmap_storm,norm=norm_sto)
#         cbar_sto.set_label('Local Laplacian [mPa/km^2]')
#         cbar_sto.outline.set_visible(False)

#         # Plot deformation colorbar
#         norm_def = mpl.colors.BoundaryNorm(boundaries=def_class[:-1],ncolors=256)
#         cbar_def = mpl.colorbar.ColorbarBase(ax[1,1], cmap=cmap_def,norm=norm_def)
#         cbar_def.set_label('Total deformation [1/day]')
#         cbar_def.outline.set_visible(False)
#         ticks = []
#         for it in cbar_def.ax.yaxis.get_majorticklabels():
#             ticks.append('%.2f' %float(it.get_text()))
#         cbar_def.ax.yaxis.set_ticklabels(ticks)

#         fig.savefig(plot_output_path + 'deformation_linked_storms_year_%s.pdf' %years[iyear])

# # Link spatial with temporal statistics

# if intersection:
#     # Plot intersection angle statistics depending on deformation rate 
#     if link_interc_def & link_interc_lifetime:
#         # Compute mean deformation of intersecting partners
#         def_par = []; diff_def_par = [];
# 	life_par = []
# 	if link_interc_len:
# 	    len_par = []
#         for iyear in range(len(lkf_dataset)):
#             def_par_year = []; diff_def_par_year = [];
# 	    life_par_year = []
# 	    if link_interc_len:
# 		len_par_year = []
#             for iday in range(len(lkf_dataset[iyear])):
#                 def_par_day = np.array([np.sqrt(np.sum(np.array(lkf_deformation[iyear][iday])[np.stack(lkf_interc_par[iyear][iday])[:,0].astype('int')]**2,axis=1)),
#                                         np.sqrt(np.sum(np.array(lkf_deformation[iyear][iday])[np.stack(lkf_interc_par[iyear][iday])[:,1].astype('int')]**2,axis=1))])
#                 diff_def_par_day = np.abs(np.diff(def_par_day,axis=0))
# 		life_par_day = np.array([lkf_lifetime[iyear][iday][np.stack(lkf_interc_par[iyear][iday])[:,0].astype('int')],
# 					 lkf_lifetime[iyear][iday][np.stack(lkf_interc_par[iyear][iday])[:,1].astype('int')]])
# 		if link_interc_len:
# 		    len_par_day = np.array([np.array(lkf_length[iyear][iday])[np.stack(lkf_interc_par[iyear][iday])[:,0].astype('int')],
#                                             np.array(lkf_length[iyear][iday])[np.stack(lkf_interc_par[iyear][iday])[:,1].astype('int')]])

#                 def_par_year.append(def_par_day)
#                 diff_def_par_year.append(diff_def_par_day)
# 		life_par_year.append(life_par_day)
# 		if link_interc_len:
#  		    len_par_year.append(len_par_day)
#             def_par.append(def_par_year)
#             diff_def_par.append(diff_def_par_year)
# 	    life_par.append(life_par_year)
# 	    if link_interc_len:
#                 len_par.append(len_par_year)


#         # Plot histograms in different deformation rate classes
#         fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4.8))
#         if datatype=='rgps':
#             def_class = [0,0.03,0.05,2]
#         elif datatype == 'mitgcm_2km':
#             def_class = [0,0.05,0.2,10]
# 	len_thres = 8*12.5e3
#         nbins = 45
#         bins = np.linspace(0,90,nbins)
#         def_masked_class = [[],[],[]]
#         for iax,axi in enumerate(ax):
#             axi.set_title('Def. class: %.2f to %.2f [1/day]' %(def_class[iax],def_class[iax+1]))
#             axi.set_xlabel('Intersection angle')
#             axi.set_ylabel('PDF')
#             axi.set_xlim([0,90])

#             for iyear in range(len(lkf_interc)):
#                 mask_def = np.all([np.all(np.hstack(def_par[iyear])>=def_class[iax],axis=0),
#                                    np.all(np.hstack(def_par[iyear])<def_class[iax+1],axis=0)],
#                                   axis=0)
# 		mask_life = np.all(np.hstack(life_par[iyear])==1,axis=0)
# 		if link_interc_len:
# 		    mask_len = np.all(np.hstack(len_par[iyear])>=len_thres,axis=0)
# 		    mask_life = mask_len & mask_life
#                 pdf_interc, bins_interc = np.histogram(np.concatenate(lkf_interc[iyear])[mask_def & mask_life],
#                                                bins=bins, density=True)
#                 bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#                 axi.plot(bins_mean, pdf_interc,label=years[iyear],color='0.5',alpha=0.5)
#                 def_masked_class[iax].append(np.concatenate(lkf_interc[iyear])[mask_def & mask_life])
#             pdf_interc, bins_interc = np.histogram(np.concatenate(def_masked_class[iax]),
#                                                    bins=bins, density=True)
#             bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#             axi.plot(bins_mean, pdf_interc,label=years[iyear],color='k',alpha=1.0)
#             axi.text(axi.get_xlim()[0]+0.1*axi.get_xlim()[1],
#                      axi.get_ylim()[0]+0.9*axi.get_ylim()[1],
#                      'Tot. num: %i' %np.concatenate(def_masked_class[iax]).size)

#         fig.savefig(plot_output_path + 'interc_pdf_def_class_lifetime.pdf')

#         with plt.style.context(style_label):
#             fig,axi = plt.subplots(nrows=1,ncols=1, figsize=(6,5))
#         if datatype=='rgps':
#             def_class = [0,0.03,0.00,2]
#         elif datatype == 'mitgcm_2km':
#             def_class = [0,0.05,0.2,10]
#         len_thres = 10*12.5e3
#         nbins = 23
#         bins = np.linspace(0,90,nbins)
#         def_masked_class = [[],[],[]]
# 	iax = 2

#         #axi.set_title('Def. class: %.2f to %.2f [1/day]' %(def_class[iax],def_class[iax+1]))
#         axi.set_xlabel('Intersection angle')
#         axi.set_ylabel('PDF')
#         axi.set_xlim([0,90])

# 	pdf_year_save = []

#         for iyear in range(len(lkf_interc)):
#             mask_def = np.all([np.all(np.hstack(def_par[iyear])>=def_class[iax],axis=0),
#                                np.all(np.hstack(def_par[iyear])<def_class[iax+1],axis=0)],
#                               axis=0)
#             mask_life = np.all(np.hstack(life_par[iyear])==1,axis=0)
#             if link_interc_len:
#                 mask_len = np.all(np.hstack(len_par[iyear])>=len_thres,axis=0)
#                 mask_life = mask_len & mask_life
#             pdf_interc, bins_interc = np.histogram(np.concatenate(lkf_interc[iyear])[mask_def & mask_life],
#                                            bins=bins, density=True)
#             pdf_year_save.append(pdf_interc)
#             bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#             if iyear==0:
#                 axi.plot(bins_mean, pdf_interc,'.',label='single years',color='0.5',alpha=0.5)
#             else:
#                 axi.plot(bins_mean, pdf_interc,'.',color='0.5',alpha=0.5)
#             def_masked_class[iax].append(np.concatenate(lkf_interc[iyear])[mask_def & mask_life])
#         pdf_interc, bins_interc = np.histogram(np.concatenate(def_masked_class[iax]),
#                                                bins=bins, density=True)
#         bins_mean = 0.5*(bins_interc[1:]+bins_interc[:-1])
#         axi.plot(bins_mean, pdf_interc,label='all years',color=colors[0],alpha=1.0)
#         #axi.text(axi.get_xlim()[0]+0.1*axi.get_xlim()[1],
#         #         axi.get_ylim()[0]+0.9*axi.get_ylim()[1],
#         #         'Tot. num: %i' %np.concatenate(def_masked_class[iax]).size)
#         axi.plot([],[],' ',label='Tot. num: %i' %np.concatenate(def_masked_class[iax]).size)
# 	axi.legend()

# 	np.savez(int_mem_path + 'Plot_data_interc_len_lifetime.npz',pdf_years=pdf_year_save,pdf_all=pdf_interc)

#         fig.savefig(plot_output_path + 'interc_pdf_def_lifetime0_len%i.pdf' %len_thres)

