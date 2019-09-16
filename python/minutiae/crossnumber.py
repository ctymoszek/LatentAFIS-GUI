import numpy as np
import matplotlib.pylab as plt
import matplotlib
from skimage import io
import math
from matplotlib.patches import Circle
import scipy.spatial.distance
matplotlib.interactive(False)

x_idx= [-1, -1,  0,  1,  1,  1,  0, -1, -1, -1,  0,  1,  1,  1,  0, -1 ]   # starts from the left
y_idx = [0, -1, -1, -1,  0,  1,  1,  1,  0, -1, -1, -1,  0,  1,  1,  1]

x_dir= [-1, -0.7071,  0,  0.7071,  1,  0.7071,  0, -0.7071, -1, -0.7071,  0,  0.7071,  1,  0.7071,  0, -0.7071 ]   # starts from the left-top
y_dir= [0, -0.7071, -1, -0.7071,  0,  0.7071,  1,  0.7071,  0, -0.7071, -1, -0.7071,  0,  0.7071,  1,  0.7071]



def extract_minutiae(thin_img,mask = None, R = 5):
    # input
    # 1) thin_img: skeleton is represented by black values
    # 2) mask: white represent the forground, black represent background
    h,w = thin_img.shape
    if mask is None:
        mask = np.ones((h,w),dtype=np.uint8)
    else:
        assert (thin_img.shape[0] <= mask.shape[0] and thin_img.shape[1] <= mask.shape[1])

    thin_img0 = thin_img.copy()
    rawMinu = []  # only record the coordinates for further processing

    trace_len = 12
    spur_len_thr = 7
    connected_len_thr = 9

    debug_extraction = False
    arrow_len = 15
    # using the cross number to detect raw minutiae
    for i in range(R, h-R):
        for j in range(R,w-R):
            if thin_img[i, j] >0:
                continue
            if mask[i,j] == 0 or  mask[i-R,j] == 0 or mask[i-R,j-R] == 0 or mask[i+R,j] == 0 or mask[i+R,j+R] == 0 or mask[i,j+R] == 0 or mask[i, j-R] == 0:
                continue
            ndiff = 0
            for n in range(8):
                #print thin_img[i + y_idx[n], j + x_idx[n]] - thin_img[i + y_idx[n + 1], j + x_idx[n + 1]]
                ndiff = ndiff + abs(thin_img[i + y_idx[n], j + x_idx[n]] != thin_img[i + y_idx[n + 1], j + x_idx[n + 1]])
            #print ndiff
            if ndiff == 2: # ridge ending,
                dir, ending_type, xArr, yArr = ridge_ending_tracing(thin_img, j, i, trace_len)
                if dir is None:
                    continue
                if debug_extraction:
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 2, 1)
                    plt.imshow(thin_img0, cmap='gray')
                    # ax.title('enhanced image'), plt.xticks([]), plt.yticks([])
                    circ = Circle((j, i), 10, color='r', fill=False)
                    ax.add_patch(circ)
                    ori = -dir
                    dx = math.cos(ori) * arrow_len
                    dy = math.sin(ori) * arrow_len
                    ax.arrow(j, i, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')

                ep = len(xArr)
                removed = False
                if ending_type == 2 and len(xArr) < spur_len_thr:
                    ep -= 1
                    removed = True
                elif ending_type == 1 and len(xArr) < spur_len_thr:
                    removed = True
                if removed:
                    for k in range(len(xArr)):
                        x = xArr[k]
                        y = yArr[k]
                        thin_img[y, x] = 1
                    if debug_extraction:
                        ax = fig.add_subplot(1, 2, 2)
                        plt.imshow(thin_img, cmap='gray')
                        # ax.title('enhanced image'), plt.xticks([]), plt.yticks([])
                        circ = Circle((j, i), 10, color='r', fill=False)
                        ax.add_patch(circ)
                        ori = -dir
                        dx = math.cos(ori) * arrow_len
                        dy = math.sin(ori) * arrow_len
                        ax.arrow(j, i, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')

                        plt.show(block=True)
                        plt.close()
                else:
                    rawMinu.append([j, i, -dir, 1])
                    #ending.append(ending_type)
            elif ndiff == 6:
                dir, ending_type, xArr,yArr = ridge_birfurcate_tracing(thin_img, j, i, trace_len)
                if dir is None:
                    continue
                fake = False
                if debug_extraction:
                    fig = plt.figure()
                    ax = fig.add_subplot(1,2,1)
                    plt.imshow(thin_img, cmap='gray')
                    #ax.title('enhanced image'), plt.xticks([]), plt.yticks([])
                    circ = Circle((j, i), 10, color='r', fill=False)
                    ori = -dir
                    dx = math.cos(ori) * arrow_len
                    dy = math.sin(ori) * arrow_len
                    ax.arrow(j, i, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
                    ax.add_patch(circ)

                for p in range(len(ending_type)):
                    removed = False
                    ep = len(xArr[p])
                    if (ending_type[p] == 2 and len(xArr[p]) < connected_len_thr) :
                        removed = True
                        fake = True
                        ep -= 1
                    elif (ending_type[p] == 1 and len(xArr[p]) < spur_len_thr):
                        removed = True
                        fake = True

                    if not removed:
                        continue
                    for k in range(1, ep):
                        x = xArr[p][k]
                        y = yArr[p][k]
                        thin_img[y, x] = 1
                if debug_extraction:
                    ax = fig.add_subplot(1, 2, 2)
                    plt.imshow(thin_img, cmap='gray')
                    # ax.title('enhanced image'), plt.xticks([]), plt.yticks([])
                    circ = Circle((j, i), 10, color='r', fill=False)
                    ax.add_patch(circ)
                    ori = -dir
                    dx = math.cos(ori) * arrow_len
                    dy = math.sin(ori) * arrow_len
                    ax.arrow(j, i, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
                    plt.show(block=True)
                    plt.close()
                if not fake:
                    rawMinu.append([j, i, -dir, 2])

    rawMinu = np.asarray(rawMinu)

    rawMinu = remove_crowded_minutiae(rawMinu)
    return rawMinu,thin_img

def ridge_ending_tracing(thin_img, x, y, trance_len=12):

    #l = (pImgThin[y - 1,x - 1] == BLACK) ? 0: 4;
    if thin_img[y, x - 1] == 0:
        p = 0
    else:
        p = 4
    flag = 1
    h,w = thin_img.shape


    xArr = []
    yArr = []
    xArr.append(x)
    yArr.append(y)
    n = 1
    cx = x
    cy = y
    ending_type = None
    while flag == 1:
        prex = cx
        prey = cy
        k = (p + 5)
        while k >= 7:
            k = k - 7 # *integer;
        isfind = 0
        for p in range(k, k + 7):
            cy = prey + y_idx[p]
            cx = prex + x_idx[p]
            if cy >= 0 and cy < h and cx >= 0 and cx < w and thin_img[cy, cx] == 0:
                isfind = 1
                break
        if isfind == 1:
            # sum(sum(skel(cy - 1:cy + 1, cx - 1:cx + 1)))
            if cy <= 0 or cx <= 0 or cx >= w-1 or cy >= h-1:
                flag = 0
                ending_type = 3 #% boundary
                continue
            localSum = np.sum(thin_img[cy - 1: cy + 2, cx - 1:cx + 2])
            if localSum == 6:
                n = n + 1
                xArr.append(cx)
                yArr.append(cy)
                if n >= trance_len:
                    break
            elif localSum == 7:
                flag = 0
                ending_type = 1  # ridge ending
            elif localSum < 6:
                flag = 0
                ending_type = 2
            else:
                flag = 0
        else:
            flag = 0

    dir = None
    if len(xArr)>=12:
        ep = 11
        dir = np.arctan2(yArr[ep] - y, xArr[ep] - x)
    return dir, ending_type, xArr, yArr


def ridge_birfurcate_tracing(thin_img, x, y,trace_len):
    dir = None
    xArr = None
    yArr = None
    ending_type = None
    h, w = thin_img.shape
    rgnX = np.zeros((3, ),dtype=np.int)
    rgnY = np.zeros((3, ),dtype=np.int)
    ridgeNum = 0
    thin_img[y - 3:y + 4, x- 3:x + 4]
    for n in range(8):
        if thin_img[y + y_idx[n], x + x_idx[n]] == 0:
            if ridgeNum>=3:
                ridgeNum = 4
                break
            rgnX[ridgeNum] = x + x_idx[n]
            rgnY[ridgeNum] = y + y_idx[n]
            ridgeNum = ridgeNum + 1
    if ridgeNum>3:
        return dir, ending_type, xArr, yArr

    #calculate the orientation of three ridges according to the orientations to arrange its position

    xArr = [[], [], []]
    yArr = [[], [], []]
    ending_type = np.zeros((ridgeNum, ),dtype=np.int)
    mdir =  np.zeros((ridgeNum, ))
    for m in range(ridgeNum):
        for n in range(ridgeNum):
            thin_img[rgnY[n], rgnX[n]] = 1

        thin_img[rgnY[m], rgnX[m]] = 0
        if thin_img[y, x - 1] == 0:
            p = 0
        else:
            p = 4

        flag = 1

        xArr[m].append(x)
        yArr[m].append(y)
        cx = x
        cy = y
        while flag == 1:
            prex = cx
            prey = cy
            k = (p + 5)
            while k > 7:
                k = k - 8
            isfind = 0
            for p in range(k,k + 7):
                cy = prey + y_idx[p]
                cx = prex + x_idx[p]
                if cy >= 0 and cy <= h-1 and cx >= 0 and cx <= w-1 and thin_img[cy, cx] == 0:
                    isfind = 1
                    break
            if isfind == 1:
                if cy <= 0 or cx <= 0 or cx >= w-1 or cy >= h-1:
                    flag = 0
                    ending_type[m] = 3 #% boundary
                    continue
                # sum(sum(skel(cy - 1:cy + 1, cx - 1:cx + 1)))
                localSum = np.sum(thin_img[cy - 1:cy + 2, cx - 1:cx + 2])
                xArr[m].append(cx)
                yArr[m].append(cy)
                if localSum == 6:
                    flag = 1
                elif localSum == 7:
                    flag = 0
                    ending_type[m] = 1 # ridge ending
                elif localSum<6:
                    flag = 0
                    ending_type[m] = 2
            else:
                flag = 0

            if len(xArr[m]) > trace_len:
                break
        ep = len(yArr[m])
        if ep > 12:
            ep = 12
            #point  for orientation calculation
        mdir[m] = np.arctan2(yArr[m][ep-1] - y, xArr[m][ep-1] - x )
        # angleDiff(m) = abs(angleDiff(m) - theta);
        # if angleDiff(m) > pi
        # angleDiff(m) = angleDiff(m);
    tdir =np.concatenate((np.array([mdir[-1]]), mdir, np.array([mdir[0]])))
    angle_diff = np.zeros((3,))
    for i in range(3):
        angle_diff[i] = min(np.abs(tdir[i + 1] - tdir[i]), 2 * math.pi - np.abs(tdir[i + 1] - tdir[i])) + \
                               min(np.abs(tdir[i + 1] - tdir[i + 2]), 2 * math.pi - np.abs(tdir[i + 1] - tdir[i + 2]))

    ind = np.argsort(angle_diff)[::-1]
    cosTheta = np.cos(mdir[ind[1:3]])
    sinTheta = np.sin(mdir[ind[1:3]])
    dir = np.arctan2(np.sum(sinTheta), np.sum(cosTheta))
    xArr[0], xArr[1], xArr[2] = xArr[ind[0]], xArr[ind[1]], xArr[ind[2]]
    yArr[0], yArr[1], yArr[2] = yArr[ind[0]], yArr[ind[1]], yArr[ind[2]]
    ending_type[0], ending_type[1], ending_type[2] = ending_type[ind[0]], ending_type[ind[1]], ending_type[ind[2]]
    # min_len = np.min(len(xArr[ind[1]]),len(xArr[ind[2]]))
    # for i in range(min_len):
    #     x = (xArr[ind[1]][i] + xArr[ind[2]][i])/2
    #     y = (yArr[ind[1]][i] + yArr[ind[2]][i])/2
    #     xArr_ord.append(x)
    #     yArr_ord.append(y)
    # xArr_ord = np.asarray(xArr_ord)
    # yArr_ord = np.asarray(yArr_ord)

    for n in range(ridgeNum):
        thin_img[rgnY[n], rgnX[n]] = 0
    return dir, ending_type, xArr, yArr

def remove_crowded_minutiae(rawMinu):
    if type(rawMinu) == 'list':
        rawMinu = np.asarray(rawMinu)
    dists = scipy.spatial.distance.cdist(rawMinu[:,:2],rawMinu[:,:2], 'euclidean')
    minu_num = rawMinu.shape[0]

    flag = np.ones((minu_num,),np.bool)
    neighor_num = 3
    neighor_thr = 12

    neighor_num2 = 5
    neighor_thr2 = 25
    if minu_num<neighor_num:
        return rawMinu
    for i in range(minu_num):
        # if two points are two close, both are removed
        ind = np.argsort(dists[i, :])
        if dists[i,ind[1]]<5:
            flag[i]=False
            flag[ind[1]] = False
            continue
        if np.mean(dists[i,ind[1:neighor_num+1]]) <neighor_thr:
            flag[i] = False
        if minu_num>neighor_num2 and np.mean(dists[i,ind[1:neighor_num2+1]]) <neighor_thr2:
            flag[i] = False
    rawMinu = rawMinu[flag,:]
    return rawMinu


if __name__=="__main__":
    thin_img = io.imread('thin_img.bmp')
    maskfile = '../../../Data/Latent/NISTSD27/maskNIST27/roi018.bmp'
    mask = io.imread(maskfile)
    thin_img[thin_img>0] = 1
    minutiae , thin_img2= extract_minutiae(1-thin_img,mask = mask)
    show_minutiae(thin_img2, minutiae)
    print len(minutiae)