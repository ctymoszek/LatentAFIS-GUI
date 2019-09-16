import matplotlib as mpl
from skimage import io
import math
import numpy as np
from matplotlib.patches import Circle
import cv2
from skimage import measure
import glob


def show_mask(mask,img=None,fname=None,block=False):

    contours = measure.find_contours(mask, 0.8)

    # Display the image and plot all contours found
    fig, ax = mpl.pyplot.subplots(1)
    ax.set_aspect('equal')
    mpl.pyplot.gca().set_axis_off()
    mpl.pyplot.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
    mpl.pyplot.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
    mpl.pyplot.margins(0,0)
    mpl.pyplot.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)

    if img is not None:
        ax.imshow(img, interpolation='nearest', cmap=mpl.pyplot.cm.gray)
    else:
        ax.imshow(mask, interpolation='nearest', cmap=mpl.pyplot.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    #ax.axis('ROI')
    ax.set_xticks([])
    ax.set_yticks([])

    #mpl.pyplot.show(block=block)
    if fname is not None:
        fig.savefig(fname,dpi = 600, bbox_inches='tight', pad_inches=0.0)
        mpl.pyplot.close()
    else:
        mpl.pyplot.close()

def show_image(img,mask=None,block = True, fname = None):
    # Display the image and plot all contours found
    img = img.copy()
    img = (img - np.min(img))/(np.max(img)-np.min(img))*255
    if mask is not None:
        mh,mw = mask.shape
        h,w = img.shape[:2]
        assert(h==mh and w==mw)
        for i in range(h):
            for j in range(w):
                if mask[i,j] == 0:
                    img[i,j] = 127

    fig, ax = mpl.pyplot.subplots()
    ax.imshow(img, interpolation='nearest', cmap=mpl.pyplot.cm.gray)
    #ax.axis('ROI')
    ax.set_xticks([])
    ax.set_yticks([])
    mpl.pyplot.show(block=block)
    if fname is not None:
        fig.savefig(fname,dpi = 600)
    mpl.pyplot.close()

def show_minutiae(img,minutiae,mask=None,fname=None,block = True):
    # for the latent or the low quality rolled print
    fig, ax = mpl.pyplot.subplots(1)

    #img = img.copy()
    ax.set_aspect('equal')
    mpl.pyplot.gca().set_axis_off()
    mpl.pyplot.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
    mpl.pyplot.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
    mpl.pyplot.margins(0,0)
    mpl.pyplot.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)

    R = 10
    arrow_len = 15
    if mask is not None:
        h,w = mask.shape
        for i in range(h):
            for j in range(w):
                if mask[i,j] == 0:
                    img[i,j] = 127

    ax.imshow(img, cmap='gray')
    minu_num = len(minutiae)
    for i in range(0, minu_num):
        xx = minutiae[i][0]
        yy = minutiae[i][1]
        circ = Circle((xx, yy), R, color='r', fill=False, alpha=0.7)
        ax.add_patch(circ)

        ori = -minutiae[i][2]
        dx = math.cos(ori) * arrow_len
        dy = math.sin(ori) * arrow_len
        ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r', alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    #mpl.pyplot.show(block=block)
    if fname is not None:
        fig.savefig(fname,dpi = 600, bbox_inches='tight', pad_inches=0.0)
        mpl.pyplot.close()

def show_minutiae_sets(img,minutiae_sets,ROI=None,fname=None,block = True):
    # for the latent or the low quality rolled print
    fig, ax = mpl.pyplot.subplots(1)
    ax.set_aspect('equal')

    arrow_len = 15
    if ROI is not None:
        h,w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i,j] == 0:
                    img[i,j] = 255

    ax.imshow(img, cmap='gray')
    color = ['r','b']
    R = [10,8,6]
    for k in range(len(minutiae_sets)):
        minutiae = minutiae_sets[k]
        #minutiae = np.asarray(minutiae)
        minu_num = len(minutiae)
        for i in range(0, minu_num):
            xx = minutiae[i,0]
            yy = minutiae[i,1]
            circ = Circle((xx, yy), R[k], color=color[k], fill=False)
            ax.add_patch(circ)

            ori = -minutiae[i,2]
            dx = math.cos(ori) * arrow_len
            dy = math.sin(ori) * arrow_len
            ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc=color[k], ec=color[k])

    mpl.pyplot.show(block=block)
    if fname is not None:
        fig.savefig(fname,dpi = 600)
        mpl.pyplot.close()


#show the orientation field
def show_orientation_field(img,dir_map,mask=None,fname=None):
    h,w = img.shape[:2]

    if mask is None:
        mask = np.ones((h,w),dtype=np.uint8)
    blkH, blkW = dir_map.shape

    blk_size = h/blkH

    R = blk_size/2*0.8
    fig, ax = mpl.pyplot.subplots(1)
    mpl.pyplot.gca().set_axis_off()
    mpl.pyplot.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
    mpl.pyplot.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
    mpl.pyplot.margins(0,0)
    mpl.pyplot.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    ax.imshow(img, cmap='gray')
    for i in range(blkH):
        y0 = i*blk_size + blk_size/2
        y0 = int(y0)
        for j in range(blkW):
            x0 = j*blk_size + blk_size/2
            x0 = int(x0)
            ori = dir_map[i,j]
            if mask[y0,x0] == 0:
                continue
            if ori<-9:
                continue
            x1 = x0 - R * math.cos(ori)
            x2 = x0 + R * math.cos(ori)
            y1 = y0 - R * math.sin(ori)
            y2 = y0 + R * math.sin(ori)
            mpl.pyplot.plot([x1, x2], [y1, y2], 'r-', lw=2, alpha=0.6)
    #mpl.pyplot.show(block=True)
    mpl.pyplot.close()
    if fname is not None:
        fig.savefig(fname,dpi = 600, bbox_inches='tight', pad_inches=0.0)
        mpl.pyplot.close()

if __name__ == '__main__':
     
     img_path = '../../../Data/Latent/NISTSD27/image/'
     mask_path =  '../../../Data/Latent/NISTSD27/maskNIST27/'
     img_files = glob.glob(img_path+'*.bmp')
     #img_files = glob.glob('../../../*.*')
     print img_files

     img_files.sort()
     mask_files = glob.glob(mask_path+'*.bmp')
     mask_files.sort()
     for img_file, mask_file in zip(img_files,mask_files):
         img = io.imread(img_file)
         mask = io.imread(mask_file)
         show_mask(mask,img)
