# -*- coding: utf-8 -*-

"""
Tracking routines to follow detected LKFs in time using drift data.
"""


# Package Metadata
__version__ = 0.1
__author__ = "Nils Hutter"
__author_email__ = "nhutter@uw.edu"




import numpy as np
import matplotlib.pylab as plt

from .detection import *




# ------------------- 0. Helper Functions

def compute_MHD_segment(A,B,return_overlap=False,overlap_thres=2,angle_thres=45,return_overlaping_area=False):
    """ Function to compute Modified Hausdorff Distnce between two
    segments.
    Following: Marie-Pierre Dubuisson and Anil K. Jain: A Modified 
               Hausdorff Distance for Object Matching, 1994
    
    Input : A,B                 - two segments
            return_part_overlap - optinal: return number of pixels
                                  that partly overlap between segments
            overlap_thres       - threshold what defines overlap
    Output: mhd, pixel_overlap(optional)"""
    daB = np.array([np.min(np.sqrt(np.sum((B.T-a)**2,axis=1))) for a in A.T])
    dbA = np.array([np.min(np.sqrt(np.sum((A.T-b)**2,axis=1))) for b in B.T])
    
    if return_overlap:
        overlap = np.min([(daB <= overlap_thres).sum(),
                          (dbA <= overlap_thres).sum()])
        if overlap>1:
            A_o = A[:,daB <= overlap_thres]
            B_o = B[:,dbA <= overlap_thres]
            # Filter two large angles
            angle = angle_segs(A_o[:,[0,-1]],B_o[:,[0,-1]])
            if angle > 90: angle = 180-angle
            if angle >= angle_thres: overlap = 0
            if return_overlaping_area:
                return np.max([(daB.sum()/ A.shape[1]),(dbA.sum()/ B.shape[1])]), overlap, [A_o,B_o]
            else:
                return np.max([(daB.sum()/ A.shape[1]),(dbA.sum()/ B.shape[1])]), overlap
        else: 
            overlap = 0
            if return_overlaping_area:
                return np.max([(daB.sum()/ A.shape[1]),(dbA.sum()/ B.shape[1])]), overlap, [np.array([]),np.array([])]
            else:
                return np.max([(daB.sum()/ A.shape[1]),(dbA.sum()/ B.shape[1])]), overlap

    else:
        return np.max([(daB.sum()/ A.shape[1]),(dbA.sum()/ B.shape[1])])








# ------------------- 1. Tracking function 

def track_lkf(lkf0_d, lkf1, nx, ny, thres_frac=0.75, min_overlap=4,first_overlap=False,overlap_thres=1.5,angle_thres=25.,search_area_expansion=1):
    """Tracking function for LKFs

    Input: lkf0_d: advected detected LKF features
           lkf1:   detected LKF features as a list of arrays that contain to indices of all cell containing to one LKF
           
    Output: lkf_track_pairs: List with pairs of indexes to LKFs in lkf0 that are tracked in lkf1
    """

    # ----------------------- Define index grid -----------------------------

    xgi = np.linspace(1,nx,nx)-1
    ygi = np.linspace(1,ny,ny)-1
    XGi,YGi = np.meshgrid(xgi,ygi)
    

    # -------------- First rough estimate of drifted LKFs -------------------

    lkf_track_pairs = []
    #thres_frac = 0.75
    #min_overlap = 4

    for ilkf,iseg_d in enumerate(lkf0_d):

        if ~np.any(np.isnan(iseg_d)):
            # Define search area
            search_area = np.concatenate([np.floor(iseg_d[:,:2]),np.ceil(iseg_d[:,:2]),
                                          np.vstack([np.floor(iseg_d)[:,0],np.ceil(iseg_d)[:,1]]).T,
                                          np.vstack([np.ceil(iseg_d)[:,0],np.floor(iseg_d)[:,1]]).T],
                                         axis=0) # Floor and ceil broken indexes
            # Broadening of search area
            #search_area_expansion = 1 # Number of cell for which the search area is expanded to be consider differences in the morphological thinning
            for i in range(search_area_expansion):
                n_rows = search_area[:,0].size
                search_area = np.concatenate([search_area,
                                              search_area+np.concatenate([np.ones(n_rows).reshape((n_rows,1)),
                                                                          np.zeros(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([-np.ones(n_rows).reshape((n_rows,1)),
                                                                          np.zeros(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([np.zeros(n_rows).reshape((n_rows,1)),
                                                                          np.ones(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([np.zeros(n_rows).reshape((n_rows,1)),
                                                                          np.ones(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([np.ones(n_rows).reshape((n_rows,1)),
                                                                          np.ones(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([np.ones(n_rows).reshape((n_rows,1)),
                                                                          -np.ones(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([-np.ones(n_rows).reshape((n_rows,1)),
                                                                          np.ones(n_rows).reshape((n_rows,1))],axis=1),
                                              search_area+np.concatenate([-np.ones(n_rows).reshape((n_rows,1)),
                                                                          -np.ones(n_rows).reshape((n_rows,1))],axis=1)],axis=0)
    
            search_area = np.unique(search_area, axis=0)

            search_area = search_area[np.all(search_area>=0,axis=1),:]
            search_area = search_area[np.all([search_area[:,0]<nx,search_area[:,1]<ny],axis=0),:]

            if np.any(search_area<0):
                print('Attention negative index')

            # Replaces by new numpy version unique
            #search_area = np.vstack({tuple(row) for row in search_area})
            
            # Find area orthogonal to the feature
            ## Perform linear fit to feature
            A = np.transpose(np.array([iseg_d[:,0], np.ones(iseg_d[:,0].shape)]))
            b = np.reshape(iseg_d[:,1],(iseg_d[:,1].size,1))

            if np.linalg.matrix_rank(A)>=2:
                coeff = np.squeeze(np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b)))

                if coeff[0]!=0:
                    ## Define boundary lines at both endpoints
                    ### determine line through endpoint #1
                    x1 = iseg_d[0,0]; y1 = iseg_d[0,1]
                    a = -1/coeff[0]
                    b1 = y1 - a*x1
                    f1 = a*XGi + b1 - YGi
    
                    ### determine line through endpoint #2
                    x2 = iseg_d[-1,0]; y2 = iseg_d[-1,1]
                    b2 = y2 - a*x2
                    f2 = a*XGi + b2 -YGi

                    ## Define mask of orthogonal area
                    orth_area = ((f1>0) & (f2<0)) | ((f1<0) & (f2>0))
                elif coeff[0]==0: # LKF parallel to x axis
                    orth_area = ((XGi >= np.min([iseg_d[0,0],iseg_d[-1,0]])) & 
                                 (XGi <= np.max([iseg_d[0,0],iseg_d[-1,0]])))
            else: # LKF parallel to y axis
                orth_area = ((YGi >= np.min([iseg_d[0,1],iseg_d[-1,1]])) & 
                             (YGi <= np.max([iseg_d[0,1],iseg_d[-1,1]])))

            orth_area = np.concatenate([XGi[orth_area].reshape((XGi[orth_area].size,1)),
                                        YGi[orth_area].reshape((YGi[orth_area].size,1))],axis=1)

    
            # Ravel indeces to 1D index for faster comparison
            orth_area_ravel = []
            for io in range(orth_area.shape[0]):
                orth_area_ravel.append(np.ravel_multi_index(orth_area[io,:].astype('int'),
                                                            np.transpose(XGi).shape))
            search_area_ravel = []
            for io in range(search_area.shape[0]):
                search_area_ravel.append(np.ravel_multi_index(search_area[io,:].astype('int'),
                                                              np.transpose(XGi).shape))
            search_area_ravel = list(set(search_area_ravel).intersection(orth_area_ravel))
            search_area = np.zeros((len(search_area_ravel),2))
            for io in range(search_area.shape[0]):
                search_area[io,:] = np.unravel_index(search_area_ravel[io],np.transpose(XGi).shape)
        
    
            # Loop over all LKFs to check whether there is overlap with search area
            for i in range(lkf1.shape[0]):
                lkf_ravel = []
                for io in range(lkf1[i].shape[0]):
                    lkf_ravel.append(np.ravel_multi_index(lkf1[i][io,:].astype('int'),
                                                          np.transpose(XGi).shape))

                comb_seg_search_area, comb_seg_search_area_count = np.unique(search_area_ravel+lkf_ravel,
                                                                             return_counts=True)

                # Check for overlap
                if np.any(comb_seg_search_area_count > 1):
                    # LKF1[i] is overlapping with search area
                    num_points_overlap = np.sum(comb_seg_search_area_count>1)

                    if first_overlap:
                        if (num_points_overlap >= min_overlap):
                            # Test again with overlap:
                            A = iseg_d.T
                            B = np.stack(np.unravel_index(comb_seg_search_area[comb_seg_search_area_count>1],
                                                          np.transpose(XGi).shape))
                            mhdi, overlap_i = compute_MHD_segment(A,B,return_overlap=True,
                                                                  overlap_thres=overlap_thres,
                                                                  angle_thres=angle_thres)
                            if overlap_i>=min_overlap:
                                lkf_track_pairs.append(np.array([ilkf,i]))
                    else:
                        # Check in orthogonal area of LKF
                        comb_seg_orth_area, comb_seg_orth_area_count = np.unique(orth_area_ravel+lkf_ravel,
                                                                                 return_counts=True)
                        num_points_overlap_orth = np.sum(comb_seg_orth_area_count>1)
                        frac_search_to_orth = num_points_overlap/float(num_points_overlap_orth)

                        if (frac_search_to_orth > thres_frac) & (num_points_overlap >= min_overlap):
                            # Test again with overlap:
                            A = iseg_d.T
                            B = np.stack(np.unravel_index(comb_seg_orth_area[comb_seg_orth_area_count>1],
                                                          np.transpose(XGi).shape))
                            mhdi, overlap_i = compute_MHD_segment(A,B,return_overlap=True,
                                                                  overlap_thres=overlap_thres,
                                                                  angle_thres=angle_thres)
                            if overlap_i>=min_overlap:
                                lkf_track_pairs.append(np.array([ilkf,i]))

                        
    return lkf_track_pairs




# ------------------- 2. Drift functions

def drift_estimate_rgps(lkf0_path,drift_path,read_lkf0=None):
    """Function that computes the position of LKFs after a certain time
    considering the drift

    Input: lkf0_path  - filename of lkf0 including path
           drift_path - directory where drift data is stored including prefix

    Output: lkf0_d    - drifted LKFs from lkf0"""

    # Read in lkf0
    if read_lkf0 is None:
        lkf0 = np.load(lkf0_path)
    else:
        lkf0 = read_lkf0

    # Read in drift data
    drift = np.load(drift_path + lkf0_path[-19:])

    # Compute drift estimate
    t = 3*24.*3600.
    res = 12.5e3
    lkf0_d = []
    for ilkf,iseg in enumerate(lkf0):
        iseg_d = drift[iseg[:,0].astype('int'),iseg[:,1].astype('int'),:]*t/res + iseg[:,:2]
        lkf0_d.append(iseg_d)

    return lkf0_d


def drift_estimate(lkf0_path,ncfile,mask,index_x,index_y,red_fac,
                   dxu,dyu,read_lkf0=None,adv_time=3.*24.*3600.,
                   t=1.*24.*3600.,dt = 1.*24.*3600.):
    """Function that computes the position of LKFs after a certain time
    considering the drift

    Input: lkf0_path  - filename of lkf0 including path
           drift_path - directory where drift data is stored including prefix

    Output: lkf0_d    - drifted LKFs from lkf0"""

    # Read in lkf0
    if read_lkf0 is None:
        lkf0 = np.load(lkf0_path,allow_pickle=True)
    else:
        lkf0 = read_lkf0

    if len(lkf0)==1:
        lkf0 = [lkf0.squeeze()]
        lkf0_d = [lkf0[0]]
    else:
        lkf0_d = lkf0.copy()

    # Loop over days
    t_tot = adv_time
    #t = 1.*24.*3600.
    #dt = 1.*24.*3600.
    it0 = int(str(lkf0_path).split('/')[-1].split('.')[0].split("_")[-1])

    #lkf0_d = lkf0.copy()

    for i in range(int(t_tot/t)):
        it = int(it0 + i*t/dt)
        # Read in drift data
        drift = np.stack([np.array(ncfile.U[it,1:-1,1:-1]),
                          np.array(ncfile.V[it,1:-1,1:-1])])
        
        # Mask
        drift[:,~mask[1:-1,1:-1]] = np.nan
        drift = drift[:,max([0,index_y[0][0]-1]):index_y[0][-1]+2:red_fac,
                      max([0,index_x[0][0]-1]):index_x[0][-1]+2:red_fac]
        drift = np.rollaxis(drift,0,3)
        
        res = np.stack([dxu,dyu])[:,max([0,index_y[0][0]-1]):index_y[0][-1]+2:red_fac,
                                  max([0,index_x[0][0]-1]):index_x[0][-1]+2:red_fac]
        res = np.rollaxis(res,0,3)
        
        # Compute drift estimate
        for ilkf,iseg in enumerate(lkf0):
            iseg_d = (drift[iseg[:,0].astype('int'),iseg[:,1].astype('int'),:]/
                      res[iseg[:,0].astype('int'),iseg[:,1].astype('int'),:])/float(red_fac)*t + lkf0_d[ilkf][:,:2]
            mask_d = np.all(np.stack([np.all(np.isfinite(iseg_d),axis=1),
                                      np.all(iseg_d>=1,axis=1),
                                      (iseg_d[:,0]<res.shape[0]-2),
                                      (iseg_d[:,1]<res.shape[1]-2)]),axis=0)
            lkf0_d[ilkf] = iseg_d[mask_d,:]
            lkf0[ilkf] = iseg_d[mask_d,:].astype('int')

    return lkf0_d



# ------------------- 3. Generate tracking dataset
def gen_tracking_dataset_rgps(lkf_path,drift_path,output_path):
    """Function that generates tracking data set

    Input: lkf_path    - directory including all LKF files for season
           drift_path  - directory where drift data is stored including prefix
           output_path - directory where output is stored
    """
    
    nx = 264; ny = 248

    lkf_filelist = [i for i in os.listdir(lkf_path) if i.startswith('lkf') and i.endswith('.npy')]
    lkf_filelist.sort()
    
    for ilkf in range(len(lkf_filelist[:-1])):
        print("Track features in %s to %s" %(lkf_filelist[ilkf],
                                             lkf_filelist[ilkf+1]))
        # Open lkf0 and compute drift estimate
        lkf0_d = drift_estimate_rgps(lkf_path + lkf_filelist[ilkf],drift_path)

        # Read LKFs
        lkf1 = np.load(lkf_path + lkf_filelist[ilkf+1])
        # lkf1_l = []
        # for ilkf,iseg in enumerate(lkf1):
        #     lkf1_l.append(iseg[:,:2])
        lkf1_l = lkf1
        for ilkf1,iseg in enumerate(lkf1):
            lkf1_l[ilkf1] = iseg[:,:2]

        # Compute tracking
        tracked_pairs = track_lkf(lkf0_d, lkf1_l, nx, ny, thres_frac=0.75, min_overlap=4,overlap_thres=1.5,angle_thres=25)

        # Save tracked pairs
        np.save(output_path + 'lkf_tracked_pairs_%s_to_%s' %(lkf_filelist[ilkf][4:-4],
                                                             lkf_filelist[ilkf+1][4:-4]),
                tracked_pairs)



def gen_tracking_dataset_sirex(lkf_path,output_path,ncfile,dxu,dyu,
                               mask,index_x,index_y,red_fac,
                               adv_time=3.*24.*3600.,continue_comp=False):
    """Function that generates tracking data set

    Input: lkf_path    - directory including all LKF files for season
           output_path - directory where output is stored
           ncfile      - opened netcdf file that stores U and V
    """

    nx,ny = mask[max([0,index_y[0][0]-1]):index_y[0][-1]+2:red_fac,
                 max([0,index_x[0][0]-1]):index_x[0][-1]+2:red_fac].shape

    lkf_filelist = [i for i in os.listdir(lkf_path) if i.startswith('lkf') and i.endswith('.npy')]
    lkf_filelist.sort()

    if continue_comp:
        tracked_lkfs = [i for i in os.listdir(output_path) if i.startswith('lkf') and i.endswith('.npy')]
        tracked_lkfs.sort()
    else:
        tracked_lkfs = []

    for ilkf in range(len(tracked_lkfs),len(lkf_filelist[:-1])):
        print("Track features in %s to %s" %(lkf_filelist[ilkf],
                                             lkf_filelist[ilkf+1]))
        # Open lkf0 and compute drift estimate
        lkf0_d = drift_estimate_sirex(lkf_path + lkf_filelist[ilkf],ncfile,mask,
                                      index_x,index_y,red_fac,dxu,dyu,
                                      adv_time=adv_time)
        
        # Filter zero length LKFs due to NaN drift
        ind_f   = np.where(np.array([iseg.size for iseg in lkf0_d])>0)[0]
        lkf0_df = [iseg for iseg in lkf0_d if iseg.size>0]

        # Read LKFs
        lkf1 = np.load(lkf_path + lkf_filelist[ilkf+1])
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
        tracked_pairs = track_lkf(lkf0_df, lkf1_l, nx, ny, thres_frac=0.75, min_overlap=4,overlap_thres=1.5,angle_thres=25)
        
        if len(tracked_pairs)==0:
            tracked_pairs = np.array([[],[]])
        else:
            tracked_pairs = np.stack(tracked_pairs)
            tracked_pairs[:,0] = ind_f[np.stack(tracked_pairs)[:,0]]

        # Save tracked pairs
        np.save(output_path + 'lkf_tracked_pairs_%s_to_%s' %(lkf_filelist[ilkf][4:-4],
                                                             lkf_filelist[ilkf+1][4:-4]),
                tracked_pairs)
