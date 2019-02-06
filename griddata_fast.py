import numpy as np
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools

def interp_weights(x,y,xint,yint):
    """ Function to find vertices on x,y grid for
    each point on xint,yint and compute corresponding
    weight

    Input: x,y        Grid of interpolation data
           xint,yint  Grid to which data is interpolated

    Output: vtx       Vertices
            wts       Weights of each vertice"""

    vtx, wts = interp_weights_nd(np.array([x.flatten(),y.flatten()]).T,
                                 np.array([xint.flatten(),yint.flatten()]).T,
                                 d=2)
    return vtx, wts
    

def interp_weights_nd(xyz, uvw, d=2):
    # Check for right shape:
    if xyz.shape[1]!=d:
        xyz = xyz.T
    if uvw.shape[1]!=d:
        uvw = uvw.T
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

        
        

class griddata_fast(object):
    
    def __init__(self,x,y,xint,yint):
        """ Interpolation Class: In the initialisation vertices are computed around
        each point on the interpolation grid (xint,yint) of neighbouring grid
        points (x,y). In the next step weights for interpolation are computed.
        """
        # Test for right shapes
        if np.any([(x.shape != y.shape),(xint.shape != yint.shape)]):
            print "Input grid data does not have corresponding shape"

        self.x = x.flatten()
        self.y = y.flatten()
        self.xint = xint.flatten()
        self.yint = yint.flatten()
        self.intshape = xint.shape

        self.vtx, self.wts = interp_weights_nd(np.array([x.flatten(),y.flatten()]).T,
                                               np.array([xint.flatten(),yint.flatten()]).T,
                                               d=2)

        # Filter for points in xint,yint that lay outside of x,y
        self.wts[np.where(np.any(self.wts<0,axis=1)),:]=np.zeros((3,))*np.nan

    def interpolate(self,data):
        """ Interpolates data field with the same dimension as x,y to xint,yint
        """
        return np.einsum('nj,nj->n', np.take(data.flatten(), self.vtx), self.wts).reshape(self.intshape)

    def minimum_distance(self,distance):
        """ Discards vertices where the distance between the vertex exceed
        a minimum distance """
        delete_list = []
        for iv in range(self.vtx.shape[0]):
            # disv = np.sqrt((self.x[self.vtx[iv]]-np.roll(self.x[self.vtx[iv]],1))**2+
            #                (self.x[self.vtx[iv]]-np.roll(self.x[self.vtx[iv]],1))**2)
            disv = np.sqrt((self.x[self.vtx[iv]]-self.xint[iv])**2+
                           (self.y[self.vtx[iv]]-self.yint[iv])**2)
            
            if np.any(disv > distance):
                delete_list.append(iv)

        self.wts[delete_list,:]=np.zeros((3,))*np.nan
        
