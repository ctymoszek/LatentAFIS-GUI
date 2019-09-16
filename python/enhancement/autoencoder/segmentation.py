import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import os

def Cropping(img,blur,thr = 30):
    # get the center of foreground
    h, w = img.shape
    x = []
    y = []
    for i in xrange(blksize // 2, h, blksize):
        for j in xrange(blksize // 2, w, blksize):
            if blur[i, j] > 0.5:
                x.append(j)
                y.append(i)

    x0 = np.uint(round(statistics.median(x)))
    y0 = np.uint(round(statistics.median(y)))
    col_sum = blur.sum(axis=0)
    row_sum = blur.sum(axis=1)
    minY = y0
    maxY = y0
    minX = x0
    maxX = x0
    for i in xrange(y0, 0, -1):
        if row_sum[i] < thr:
            minY = i
            break
    if minY == y0:
        minY = 0
    for i in xrange(y0, h):
        if row_sum[i] < thr:
            maxY = i
            break
    if maxY == y0:
        maxY = h
    for j in xrange(x0, 0, -1):
        if col_sum[j] < thr:
            minX = j
            break
    if minX == x0:
        minX = 0
    for j in xrange(x0, w):
        if col_sum[j] < thr:
            maxX = j
            break
    if maxX == x0:
        maxX = w
    crop_img = img[minY:maxY, minX:maxX]
    crop_blur = blur[minY:maxY, minX:maxX]
    return crop_img,crop_blur

pathname = '/future/Data/Rolled/NSITSD14/Image2_jpeg/'
t_pathname = '/future/Data/Rolled/NSITSD14/Image2_jpeg_crop/'
imgfiles = glob.glob(pathname+'*.jpeg')
imgfiles.sort()
blksize = 16
for n, imgfile in enumerate(imgfiles):
    if n<41:
        continue
    if n>10000:
        break
    #fig, ax = plt.subplots(1)
    img = cv2.imread(imgfile,cv2.CV_LOAD_IMAGE_GRAYSCALE)


    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.fabs(laplacian)
    blur = cv2.blur(laplacian, (15, 15))
    blur[blur < 5.] = 0
    blur[blur >= 5.] = 1
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(blur, cmap='gray')
    # plt.show()




    crop_img, crop_blur = Cropping(img,blur,thr = 200)


    # plt.subplot(1, 2, 1)
    # plt.imshow(crop_img, cmap='gray')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(crop_blur, cmap='gray')
    # plt.show()


    crop_img, crop_blur = Cropping(crop_img, crop_blur, thr=100)


    # plt.subplot(1, 2, 1)
    # plt.imshow(crop_img, cmap='gray')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(crop_blur, cmap='gray')
    # plt.show()
    # #
    if crop_img.shape[0]<128 or crop_img.shape[1]<128:
        continue
    cv2.imwrite(t_pathname+os.path.basename(imgfile),crop_img)
    print imgfile

