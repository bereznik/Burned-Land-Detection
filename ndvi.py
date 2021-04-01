import numpy as np
np.seterr(divide='ignore', invalid = 'ignore')

def make_ndvi(before,after):

    red_before = before[:,:,0]
    nir_before = before[:,:,1]

    red_after = after[:,:,0]
    nir_after = after[:,:,1]

    ndvi_before = np.empty(red_before.shape,dtype = np.half)
    check_before = np.logical_or(red_before>0,nir_before>0)
    ndvi_before = np.where(check_before, (nir_before-red_before)/(nir_before+red_before),0).astype(np.half)
    
    ndvi_after = np.empty(red_after.shape, dtype = np.uint16)
    check_after = np.logical_or(red_after>0, nir_after>0)
    ndvi_after = np.where(check_after, (nir_after-red_after)/(nir_after+red_after),0).astype(np.half)
    
    return ndvi_before,ndvi_after

