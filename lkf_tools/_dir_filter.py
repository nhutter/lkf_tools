from scipy import signal
import sys
import numpy as np
import os
import scipy.ndimage as ndim
from netCDF4 import Dataset
import datetime
import skimage.morphology


# ----------------------- Filter routines -------------------------


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def line(x0, y0, x1, y1):
    """Bresenham's line algorithm
    Source: stackoverflow.com
    Returns index of line points between the two points (x0,y0) and (x1,y1)"""
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return points_in_line



def gen_dir_kernels(kernelsize=5):
    """Generates directional kernels:
    List of arrays of size (kernelsize,kernelsize) that mask all possible directions
    crossing the center of the array that can be discretized with the number of pixels."""
    kernels = []
    for i in range(kernelsize):
        kern = np.zeros((kernelsize,kernelsize))
        points=np.stack(line(i,0,kernelsize-i-1,kernelsize-1))
        kern[points[:,0],points[:,1]]+=1.
        points=np.stack(line(kernelsize-i-1,kernelsize-1,i,0))
        kern[points[:,0],points[:,1]]+=1.
        kernels.append(kern/np.sum(kern))
    for i in range(1,kernelsize-1):
        kern = np.zeros((kernelsize,kernelsize))
        points=np.stack(line(0,i,kernelsize-1,kernelsize-i-1))
        kern[points[:,0],points[:,1]]+=1.
        points=np.stack(line(kernelsize-1,kernelsize-i-1,0,i))
        kern[points[:,0],points[:,1]]+=1.
        kernels.append(kern/np.sum(kern))
    return kernels



def gen_dir_kernels_gaus(kernelsize=5,std=2.5):
    """Generates directional Gaussian kernels:
    List of arrays of size (kernelsize,kernelsize) that contain the magnitude of 1-D Gaussian kernels 
    with standard deviation std along all directions crossing the center of the array that can be
    discretized with the number of pixels."""
    kernels = []
    for i in range(kernelsize):
        mask = np.zeros((kernelsize,kernelsize),dtype='bool')
        points=np.stack(line(i,0,kernelsize-i-1,kernelsize-1))
        mask[points[:,0],points[:,1]]=True
        points=np.stack(line(kernelsize-i-1,kernelsize-1,i,0))
        mask[points[:,0],points[:,1]]=True
        kern = gkern(kernlen=kernelsize,std=std)
        kern[~mask] = 0
        kernels.append(kern/np.sum(kern))
    for i in range(1,kernelsize-1):
        mask = np.zeros((kernelsize,kernelsize),dtype='bool')
        points=np.stack(line(0,i,kernelsize-1,kernelsize-i-1))
        mask[points[:,0],points[:,1]]=True
        points=np.stack(line(kernelsize-1,kernelsize-i-1,0,i))
        mask[points[:,0],points[:,1]]=True
        kern = gkern(kernlen=kernelsize,std=std)
        kern[~mask] = 0
        kernels.append(kern/np.sum(kern))
    return kernels


def slicing_run(img, patch_shape):
    """Rolling window over array img of size (patch_shape,patch_shape)"""
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = (int(X-x+1), int(Y-y+1), int(x), int(y)) # number of patches, patch_shape
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)



def dir_filt(field,kernelsize=7):
    """Directional filter:
    For each pixel the local standard deviation along all possible directions is computed. The neighbourhood 
    for the computation of the standard deviation is given by the kernel size. The filter chooses for each 
    pixel the direction with lowest standard deviation and averages along this direction.
    
    Input:  field        - input array to be filtered
            kernelsize   - number of points used to describe the neighbourhood of a pixel
           
    Output: field filtered with directional filter"""
    
    kernels = gen_dir_kernels(kernelsize=kernelsize)
    sliced_field = slicing_run(field,(kernelsize,kernelsize))
    #inds = np.argmin([np.std(sliced_field[:,:,kernels[i]>0],axis=-1) for i in range(len(kernels))]),axis==0)
    #return np.stack([signal.convolve2d(field, kernels[i], boundary='symm', mode='same') for i in range(len(kernels))])[inds,:,:]
    inds = np.argmin([np.std(sliced_field[:,:,kernels[i]>0],axis=-1) for i in range(len(kernels))],axis=0)
    filt_all = np.stack([np.sum(sliced_field*kernels[i],axis=(-1,-2)) for i in range(len(kernels))])
    filt = np.zeros(inds.shape)*np.NaN
    for i in range(len(kernels)):
        filt[inds==i] = filt_all[i,inds==i]
    return filt

# def skeleton_along_max(field,detect,skeleton,kernelsize=7):
#     """
#     skeleton at maximum value
#     """
#     kernels = gen_dir_kernels(kernelsize=kernelsize)
#     sliced_skel = slicing_run(skeleton,(kernelsize,kernelsize))
#     sliced_field = slicing_run(field,(kernelsize,kernelsize))
#     # Finds direction perpendicular to LKF orientation
#     inds = np.argmin([np.sum(sliced_field[:,:,kernels[i]>0],axis=-1) for i in range(len(kernels))])
#     # Find maximum value in this direction for each pixel
#     max_ind = [[np.argmax(sliced_field[ix,iy,kernels[inds[ix,iy]]>0]) for iy in range(field.shape[1])] for ix in range(field.shape[0])]
    
#     return np.stack([signal.convolve2d(field, kernels[i], boundary='symm', mode='same') for i in range(len(kernels))])[inds,:,:]

def skeleton_along_max(field,detect,kernelsize=7):
    kernels = gen_dir_kernels(kernelsize=kernelsize)
    sliced_detect = slicing_run(detect,(kernelsize,kernelsize))
    sliced_field = slicing_run(field,(kernelsize,kernelsize))
    # Finds direction perpendicular to LKF orientation
    inds = np.argmin([np.sum(sliced_detect[:,:,kernels[i]>0],axis=-1) for i in range(len(kernels))],axis=0)
    # Find maximum value in this direction for each pixel
    nx,ny = field.shape; nx -= kernelsize-1; ny -= kernelsize-1
    max_ind = np.array([[np.argmax(sliced_field[ix,iy,kernels[inds[ix,iy]]>0]) for iy in range(ny)] for ix in range(nx)])
    # Convert from kernel indexing to x,y indexing
    km = int(kernelsize/2)
    ix = np.array([np.where(kernels[inds[ix,iy]])[0][max_ind[ix,iy]]+ix for ix,iy in list(zip(*np.where(detect[km:-km,km:-km]>=1)))],dtype='int')
    iy = np.array([np.where(kernels[inds[ix,iy]])[1][max_ind[ix,iy]]+iy for ix,iy in list(zip(*np.where(detect[km:-km,km:-km]>=1)))],dtype='int')
    #Generate new mask
    mask = np.zeros(field.shape)
    mask[ix,iy]=1
    return skimage.morphology.skeletonize(mask).astype('float')


# def DoG_leads(in_array,max_kern,min_kern):
#     """DoG: Difference of Gaussian Filters Combination as implemented in Linow & Dierking, 2017"""
    
#     res = np.zeros(in_array.shape)
#     c = np.arange(min_kern,max_kern+1)*0.5
    
#     for i in range(0,c.size-1):
#         gaus1 = ndim.gaussian_filter(in_array,c[i],truncate=2)
#         gaus2 = ndim.gaussian_filter(in_array,c[i+1],truncate=2)
#         res += (gaus1 - gaus2)
        
#     return res
  