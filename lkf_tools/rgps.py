# -*- coding: utf-8 -*-

"""
All functions used to read and georeference RGPS data.
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
#from mpl_toolkits.basemap import Basemap # need to be replaced!
from pyproj import Proj


# --------------- 1. RGPS read and georeference functions --------------
# ----------------------------------------------------------------------



def read_RGPS(filename,land_fill=1e10,nodata_fill=1e20):
    RGPS_file = open(filename,'r',encoding= 'unicode_escape')
    
    # RGPS product header 
    dxg=0. #Size of x cell in product
    dyg=0. #Size of y cell in product
    xg0=0. #Map location of x lower left
    yg0=0. #Map location of y lower left
    xg1=0. #Map location of x higher right
    yg1=0. #Map location of y higher right
    nxcell=0 #x cells dimensional array
    nycell=0 #y cells dimensional array

    dxg,dyg,xg0,yg0,xg1,yg1 = RGPS_file.readline().strip().split()
    nxcell,nycell = RGPS_file.readline().strip().split()

    data = np.fromfile(RGPS_file,np.float32).reshape(int(nycell),int(nxcell))

    if sys.byteorder == 'little': data.byteswap(True)

    data[data==1e10] = land_fill
    data[data==1e20] = nodata_fill

    return data, float(xg0), float(xg1), float(yg0), float(yg1), int(nxcell), int(nycell)


def mSSMI():
    ''' Returns the SSMI grid projection used for RGPS data
        as Basemap class 
        ATTENION: for coordinate transform from RGPS coordinate
                  m(0,90) must be added, because in RGPS NP is the origin'''
    return Proj(proj='stere',lat_0=90, lat_ts=75, lon_0=-45, ellps='WGS84')#Basemap(projection='stere',lat_ts=70,lat_0=90,lon_0=-45,resolution='l',llcrnrlon=279.26-360,llcrnrlat=33.92,urcrnrlon=102.34,urcrnrlat=31.37,ellps='WGS84')



def get_latlon_RGPS(xg0,xg1,yg0,yg1,nxcell,nycell,m=mSSMI()):
    # Gives only rough estimate, better use SSM/I POLAR STEREOGRAPHIC PROJECTION
    x = np.linspace(xg0,xg1,nxcell+1); x = 0.5*(x[1:]+x[:-1])
    y = np.linspace(yg0,yg1,nycell+1); y = 0.5*(y[1:]+y[:-1])
    x,y = np.meshgrid(x,y)
    #xpol,ypol = m(0,90)
    #lon,lat = m(x*1e3 + xpol, y*1e3 + ypol,inverse=True)
    lon,lat = m(x*1e3, y*1e3,inverse=True)
    return lon, lat



# --------------- 1. lagranian RGPS read and interpolate functions -----
# ----------------------------------------------------------------------



def read_RGPS_lag_motion(filename):
    RGPS_file = open(filename,'r')
    
    # RGPS product identifier
    idf_id='123456789012345678901234'	
    # Description of this product
    prod_desc='1234567890123456789012345678901234567890'
    # Number of images used in the creation of this product 
    n_images = np.int16(0)
    # Number of trajectories in this product
    n_trajectories = np.int32(0)
    # Product Type
    prod_type='12345678'
    # Product creation year/time
    create_year = np.int16(0)
    create_time = np.float64(0)
    # Season start year/time
    season_start_year = np.int16(0)
    season_start_time = np.float64(0)
    # Season end year/time
    season_end_year = np.int16(0)
    season_end_time = np.float64(0)
    # Software version used to create this product
    sw_version = '123456789012'
    #Northwest Lat/Long of initial datatake 
    n_w_lat = np.float32(0) ; n_w_lon = np.float32(0)
    #Northeast Lat/Long of initial datatake
    n_e_lat = np.float32(0) ; n_e_lon = np.float32(0)
    #Southwest Lat/Long of initial datatake
    s_w_lat = np.float32(0) ; s_w_lon = np.float32(0)
    #Southeast Lat/Long of initial datatake
    s_e_lat = np.float32(0) ; s_e_lon = np.float32(0)

    #=======================================================
    # AREA CHANGE and ICE MOTION DERIVATIVES DATA
    #=======================================================
    # Cell identifier
    gpid = np.int32(0)
    # Birth  and Death year/time of gridpoint 
    birth_year = np.int16(0) ; birth_time = np.float64(0)
    death_year = np.int16(0) ; death_time = np.float64(0)
    # Number of observations of cell
    n_obs = np.int32(0)
    # Year/Time of observation
    obs_year = np.int16(0) ; obs_time = np.float64(0)
    # Map location of observation
    x_map = np.float64(0) ; y_map = np.float64(0)
    # Quality Flag of observation
    q_flag = np.int16(0)
    # Only the first 3 cells in this product will be printed out
    max_read_cell = 3

    # =======================================================
    # ASF image identifier
    image_id = '1234567890123456'
    # Image center year/time
    image_year = np.int16(0) ; image_time = np.float64(0)
    # Image center location
    map_x = np.float64(0) ; map_y = np.float64(0)


    para_val = [idf_id,prod_desc,n_images,n_trajectories,prod_type,
                create_year,create_time,
                season_start_year,season_start_time,
                season_end_year, season_end_time,
                sw_version,
                n_w_lat,n_w_lon, n_e_lat,n_e_lon,
                s_w_lat,s_w_lon, s_e_lat,s_e_lon]
    para_name = ['idf_id','prod_desc','n_images','n_trajectories','prod_type',
                 'create_year','create_time',
                 'season_start_year','season_start_time',
                 'season_end_year',' season_end_time',
                 'sw_version',
                 'n_w_lat','n_w_lon',' n_e_lat','n_e_lon',
                 's_w_lat','s_w_lon',' s_e_lat','s_e_lon']

    for ip in range(len(para_val)):
        if para_val[ip] == 0:
            para_val[ip] = np.fromfile(RGPS_file,np.dtype(para_val[ip]),1).byteswap(True)
        else:
            para_val[ip] = RGPS_file.read(len(para_val[ip]))

    # Read image data
    n_images = para_val[2]

    image_para_val_org = [image_id,image_year,image_time,map_x,map_y]

    image_para_val = [image_id,image_year,image_time,map_x,map_y]

    image_data = []

    for ii in range(n_images):
        image_para_val = []

        # Read header:
        for ip in range(len(image_para_val_org)):
            if image_para_val_org[ip] == 0:
                image_para_val.append(np.fromfile(RGPS_file,np.dtype(image_para_val_org[ip]),1).byteswap(True))
            else:
                image_para_val.append(RGPS_file.read(len(image_para_val_org[ip])))

        image_data.append(image_para_val)
            

    # Read ice motion data
    cell_para_val_org = [gpid,birth_year,birth_time,death_year,death_time,n_obs]

    cell_para_val = [gpid,birth_year,birth_time,death_year,death_time,n_obs]

    data_para_val_org = [obs_year,obs_time,
                         x_map,y_map,
                         q_flag]

    data_para_val = [obs_year,obs_time,
                     x_map,y_map,
                     q_flag]
    
    cell_data = []

    n_cells = para_val[3]
    
    for ic in range(n_cells):
        cell_para_val = np.copy(cell_para_val_org)
        
        # Read header:
        for ip in range(len(cell_para_val)):
            cell_para_val[ip] = np.fromfile(RGPS_file,np.dtype(cell_para_val_org[ip]),1).byteswap(True)


        # Read data
        n_obs = cell_para_val[-1]
        data_list = []
        for id in range(int(n_obs)):
            data_para_val = np.copy(data_para_val_org)
            for ip in range(len(data_para_val)):
                readout = np.fromfile(RGPS_file,np.dtype(data_para_val_org[ip]),1).byteswap(True)
                data_para_val[ip] = readout
                #data_para_val[ip] = np.fromfile(RGPS_file,np.dtype(data_para_val_org[ip]),1).byteswap(True)
            data_list.append(data_para_val)

        cell_data.append([cell_para_val, np.array(data_list)])
            
    return para_name, para_val, image_data, cell_data





def get_icemotion_RGPS(RGPS_path,stream='None'):
    ''' Function that reads in all RGPS files in directory (most probably month) and 
        saves them in gridded format'''
    
    if not RGPS_path.endswith('/'):
        RGPS_path += '/'
    
    if stream != 'None':
        icemotion_files = [f for f in os.listdir(RGPS_path) if f.endswith('.LP') and f.startswith('R1001'+stream)]
    else:
        icemotion_files = [f for f in os.listdir(RGPS_path) if f.endswith('.LP')]

    motion_data = []

    for iif in range(len(icemotion_files)):
        para_name, para_val, image_data, cell_data = read_RGPS_lag_motion(RGPS_path + icemotion_files[iif])

        motion_data += cell_data

    
    gid = np.zeros((len(motion_data),1)) # List of all grid cell IDs
    nobs = np.zeros((len(motion_data),1)) # List of number of observations at all grid cell IDs

    for it in range(len(motion_data)):
        gid[it] = motion_data[it][0][0]
        nobs[it] = motion_data[it][0][5]

    # Test for double mentioning of grid IDs
    if np.unique(gid).size != gid.size:
        gcids,n_gcids = np.unique(gid,return_index=True)
        print('ERROR: grid cell IDs: ' + str(gcids[n_gcids!=1]) + ' are more than once in the dataset')

    return motion_data



def get_icemotion_RGPS_season(season_path,stream='None'):
    ''' Function that reads in all RGPS files for one season (each month in 
        one directory and saves them in gridded format (GID,year,day,x,y,qflag)

        Note as RGPS positions are saved cummulative for month only last month
        of season is read, because it contains the entire season'''

    if not season_path.endswith('/'):
        season_path += '/'

    month_path = [f for f in os.listdir(season_path)]

    month_rgps = ['may', 'apr', 'mar', 'feb', 'jan', 'dec', 'nov']

    for im in range(len(month_rgps)):
        if np.any(np.array(month_path)==month_rgps[im]):
            imonth = np.where(np.array(month_path)==month_rgps[im])[0][0]
            break

    print('Read last month available for season: ' + month_path[imonth])
    motion_data = []
    if stream != 'None':
        motion_data += get_icemotion_RGPS(season_path + month_path[imonth],stream=stream)
    else:
        motion_data += get_icemotion_RGPS(season_path + month_path[imonth])
    
    gid_org = np.zeros((len(motion_data),1)) # List of all grid cell IDs
    nobs_org = np.zeros((len(motion_data),1)) # List of number of observations at all grid cell IDs

    for it in range(len(motion_data)):
        gid_org[it] = motion_data[it][0][0]
        nobs_org[it] = motion_data[it][0][5]

    gid, gid_ind = np.unique(gid_org,return_index=True)
    nobs = np.zeros(gid.shape)
    for it in range(gid.size):
        nobs[it] = np.sum(nobs_org[gid_org==gid[it]])

    icemotion_data = np.zeros((gid.size,np.int(nobs.max()),5))*np.nan # obs_year,obs_time,x_map,y_map,q_flag
    
    for it_id in range(gid.size):
        cur_ind = 0
        for it in np.where(gid_org==gid[it_id])[0]:
            icemotion_data[it_id,cur_ind:cur_ind+np.int(nobs_org[it])] = motion_data[it][1]
            cur_ind += np.int(nobs_org[it])

    return icemotion_data
    

