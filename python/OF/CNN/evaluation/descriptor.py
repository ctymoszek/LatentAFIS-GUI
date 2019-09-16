import Template
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def get_patch_location(patch_type=1):
    if patch_type == 1:
        x = np.array(xrange(40,120))
        y = list(xrange(40,120))
    elif patch_type == 2:
        x = np.array(xrange(32,128))
        y = np.array(xrange(32,128))
    elif patch_type == 3:
        x = np.array(xrange(24,136))
        y = np.array(xrange(24,136))
    elif patch_type == 4:
        x = np.array(xrange(16,144))
        y = np.array(xrange(16,144))
    elif patch_type == 5:
        x = np.array(xrange(8,152))
        y = np.array(xrange(8,152))
    elif patch_type == 6:
        x = np.array(xrange(0,160))
        y = np.array(xrange(0,160))
    elif patch_type == 7:
        x = np.array(xrange(0, 96))
        y = np.array(xrange(0, 96))
    elif patch_type == 8:
        x = np.array(xrange(32, 128))
        y = np.array(xrange(0, 96))
    elif patch_type == 9:
        x = np.array(xrange(64, 160))
        y = np.array(xrange(0, 96))
    elif patch_type == 10:
        x = np.array(xrange(64, 160))
        y = np.array(xrange(32, 128))
    elif patch_type == 11:
        x = np.array(xrange(64, 160))
        y = np.array(xrange(64, 160))
    elif patch_type == 12:
        x = np.array(xrange(32, 128))
        y = np.array(xrange(64, 160))
    elif patch_type == 13:
        x = np.array(xrange(1, 96))
        y = np.array(xrange(64, 160))
    elif patch_type == 14:
        x = np.array(xrange(1, 96))
        y = np.array(xrange(32, 128))

    xv, yv = np.meshgrid(x, y)
    return yv, xv

def get_patch_index(patchSize_L, patchSize_H, oriNum, isMinu=1):
    if isMinu == 1:
        PI2 = 2 * math.pi
    else:
        PI2 = math.pi
    x = list(xrange(-patchSize_L / 2 + 1,patchSize_L / 2+1))
    x = np.array(x)
    y = list(xrange(-patchSize_H / 2 + 1,patchSize_H / 2+1))
    y = np.array(y)
    xv, yv = np.meshgrid(x, y)
    #print xv
    # [x, y] = meshgrid(-patchSize_L / 2 + 1:patchSize_L / 2, -patchSize_H / 2 + 1:patchSize_H / 2);
    patchIndexV = {}
    patchIndexV['x'] = []
    patchIndexV['y'] = []
    for i in range(oriNum):

        th = i * PI2 / oriNum
        u = xv * np.cos(th) - yv * np.sin(th)
        v = xv * np.sin(th) + yv * np.cos(th)
        u = np.around(u)
        v = np.around(v)
        patchIndexV['x'].append(u)
        patchIndexV['y'].append(v)
    return patchIndexV
    # {i + 1}.x = round(u);
    # patchIndexV
    # {i + 1}.y = round(v);
def extract_patches(minutiae,img,patchIndexV,patch_type=1):
    assert(minutiae.shape[1]>0)
    num_minu = minutiae.shape[0]
    patchSize = patchIndexV['x'][0].shape[0]
    oriNum = len(patchIndexV['x'])

    ly, lx = get_patch_location(patch_type=patch_type)
    h,w,c = img.shape
    patches = np.zeros((num_minu, patchSize, patchSize,c), dtype=np.float32)
    for i in xrange(num_minu):
        x = minutiae[i, 0]
        y = minutiae[i, 1]
        ori = -minutiae[i, 2]
        if ori<0:
            ori += math.pi*2
        oriInd = round(ori/(math.pi*2)*oriNum)
        if oriInd >= oriNum:
            oriInd -= oriNum
        oriInd = np.int(oriInd)
        xv = patchIndexV['x'][oriInd]+x
        yv = patchIndexV['y'][oriInd]+y
        xv[xv < 0] = 0
        xv[xv >= w] = w-1
        yv[yv < 0] = 0
        yv[yv >= h] = h-1
        xv = xv.astype(int)
        yv = yv.astype(int)
        patch = img[yv,xv,:]
        patch = patch[ly,lx,:]
        if patch_type!=6:
            patch = cv2.resize(patch, (patchSize, patchSize))

        patches[i,:,:,:] = patch

    return patches
if __name__=='__main__':
    patchSize = 160
    oriNum = 64
    patchIndexV = get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

    fname = 'Data/Latent/001.dat'

    template = Template.Bin2Template_Byte(fname, isLatent=1)
    imgfile = '/home/kaicao/Dropbox/Research/LatentMatching/CodeForPaper/Evaluation/Code/time/latent/001.bmp'

    img = cv2.imread(imgfile) #cv2.IMREAD_GRAYSCALE
    h,w,c= img.shape

    print img
    # num_minu = len(template.minu_template[0].minutiae)
    # for i in xrange(num_minu):
    #     x = template.minu_template[0].minutiae[i,0]
    #     y = template.minu_template[0].minutiae[i,1]
    #     ori = template.minu_template[0].minutiae[i,2]
    #     ori = -ori
    #     if ori<0:
    #         ori += math.pi*2
    #     oriInd = round(ori/(math.pi*2)*oriNum)
    #     if oriInd >= oriNum:
    #         oriInd -= oriNum
    #     oriInd = np.int(oriInd)
    #     xv = patchIndexV['x'][oriInd]+x
    #     yv = patchIndexV['y'][oriInd]+y
    #     xv[xv < 0] = 0
    #     xv[xv >= w] = w-1
    #     yv[yv < 0] = 0
    #     yv[yv >= h] = h-1
    #     xv = xv.astype(int)
    #     yv = yv.astype(int)
    #     patch = np.zeros((patchSize,patchSize),dtype=np.float32)
    #     patch = img[yv,xv,:]
    #     plt.imshow(patch, cmap='gray')
    #     plt.show()
        #print patch.shape
    num_minu = len(template.minu_template[0].minutiae)
    patches = extract_patches(template.minu_template[0].minutiae, img, patchIndexV, patch_type=1)
    for i in range(len(patches)):
        patch = patches[i, :, :, 0]
        plt.imshow(patch, cmap='gray')
        plt.show()

    num_minu = len(template.texture_template[0].minutiae)
    patches = extract_patches(template.texture_template[0].minutiae, img, patchIndexV, patch_type=1)
    for i in range(len(patches)):
        patch = patches[i,:,:,0]
        plt.imshow(patch, cmap='gray')
        plt.show()