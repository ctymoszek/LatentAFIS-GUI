import numpy as np
import scipy.misc
import scipy.ndimage.interpolation
import math
import matplotlib.pylab as plt
from skimage.filters import gaussian

def GetPatchIndexV(angleInc = 3,Hw=2,Hv=4):


    x, y = np.meshgrid(range(-Hw, Hw+1), range(-Hv, Hv+1))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    angleNum = np.around(180/angleInc)
    patchIndexX = np.zeros((2*Hv+1,2*Hw+1,angleNum))
    patchIndexY = np.zeros((2*Hv+1,2*Hw+1,angleNum))
    for i in range(angleNum):
        th = i*math.pi/angleNum
        u = x*np.cos(th)-y*np.sin(th)
        v = x*np.sin(th)+y*np.cos(th)
        patchIndexX[:,:,i] = np.around(u)
        patchIndexY[:,:,i] = np.around(v)
    patchIndexY = patchIndexY.astype(np.int32,copy=False)
    patchIndexX = patchIndexX.astype(np.int32,copy=False)
    return patchIndexX,patchIndexY

def binarization(img,dir_map,block_size=16, angle_inc=3, mask=None,Hw = 2, Hv = 15):

    # expand block size ridge flow to pixel wise ridge flow
    cos2Theta = np.cos(dir_map*2)
    sin2Theta = np.sin(dir_map * 2)
    blkH, blkW = dir_map.shape
    h = blkH*block_size
    w = blkW*block_size

    img = img[:h, :w]

    if mask is None:
        mask = np.ones((h,w),dtype=np.uint8)
    else:
        assert(mask.shape[0]>=img.shape[0] and mask.shape[1]>=img.shape[1] )

    img = gaussian(img, sigma=1, multichannel=False, mode='reflect')
    if block_size>1:
        cos2Theta = scipy.ndimage.interpolation.zoom(cos2Theta,block_size)
        sin2Theta = scipy.ndimage.interpolation.zoom(sin2Theta,block_size)

        angle = np.arctan2(sin2Theta,cos2Theta)*0.5
    else:
        angle = dir_map
    angle = angle/math.pi*180
    angle = angle.astype(int)
    angle[angle < 0] = angle[angle < 0] + 180
    angle[angle==180] = 0


    patchIndexX, patchIndexY = GetPatchIndexV(angleInc=angle_inc, Hw=Hw, Hv=Hv)

    angle_ind = angle//angle_inc
    bin_img = np.ones((h,w),dtype=np.uint8)

    #mh,mw = mask.shape
    for i in range(h): #600
        y0 = i
        for j in range(w): #50
            x0 = j
            # if mask[i,j] == 0:
            #     continue
            ind = angle_ind[i,j]
            X = patchIndexX[:,:,ind] + x0
            Y = patchIndexY[:,:,ind] + y0
            X[X < 0] = 0
            Y[Y < 0] = 0
            X[X > w-1] = w-1
            Y[Y > h - 1] = h - 1
            patch = img[Y,X]
            # plt.imshow(patch,cmap='gray')
            # plt.show()
            patch_row = np.sum(patch,axis=1)
            patch_mean = np.mean(patch_row)
            if patch_row[Hv] <patch_mean:
                bin_img[i,j] = 0

    bin_img[mask==0] = 1
    #plt.imshow(bin_img,cmap='gray')
    #plt.show()
    return bin_img


if __name__=='__main__':
    patchIndexX, patchIndexY = GetPatchIndexV(angleInc=3, Hw=2, Hv=4)