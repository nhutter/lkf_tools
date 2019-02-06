import numpy as np
import sys
from mpl_toolkits.basemap import Basemap
import os
import rw as rw



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
        print 'ERROR: grid cell IDs: ' + str(gcids[n_gcids!=1]) + ' are more than once in the dataset'

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

    print 'Read last month available for season: ' + month_path[imonth]
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
    
