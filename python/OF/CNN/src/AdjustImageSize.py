import cv2
import os
import numpy as np
import time
def Cropping(path,tpath,img_size):
    path_exp = os.path.expanduser(path)
    tpath_exp = os.path.expanduser(tpath)
    classes = os.listdir(path_exp)


    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        print i
        class_name = classes[i]
        #print class_name
        fingerdir = os.path.join(path_exp, class_name)
        target_fingerdir = os.path.join(tpath_exp,class_name)
        os.mkdir(target_fingerdir)
        if os.path.isdir(fingerdir):
            imagelist = os.listdir(fingerdir)
            #image_paths = [os.path.join(fingerdir, img) for img in images]
            for i,imgname in enumerate(imagelist):
                #print i, imgname
                image_path = os.path.join(fingerdir, imgname)
                target_image_path = os.path.join(target_fingerdir, imgname)
                img = cv2.imread(image_path)
                #cv2.imshow('image',img)

                #img2 =img[50:400,80:430]
                #img2 =cv2.resize(img,None,fx=0.65,fy=0.65,interpolation=cv2.INTER_CUBIC)
                #if img.shape[0]~=350
                img2 = img
                if img2.shape[0]<img_size or img2.shape[1]<img_size:

                    #print img2.shape
                    imgtmp = np.zeros((img_size,img_size,3))
                    nrows = np.minimum(img2.shape[0],img_size)
                    ncols = np.minimum(img2.shape[1],img_size)
                    imgtmp[:nrows,:ncols,:] = img2[:nrows,:ncols,:]
                    img2 = imgtmp
                    #print img2.shape
                    #cv2.waitKey(0)
                    #continue
                img2 = img2[0:img_size,0:img_size]
                cv2.imwrite(target_image_path,img2)

                #cv2.imshow('image2', img2)
                #cv2.waitKey(0)
def Cropping_NIST(path,tpath,img_size):
    path_exp = os.path.expanduser(path)
    tpath_exp = os.path.expanduser(tpath)
    imagelist = os.listdir(path_exp)



    #image_paths = [os.path.join(fingerdir, img) for img in images]
    for i,imgname in enumerate(imagelist):
                #print i, imgname
                image_path = os.path.join(path_exp, imgname)
                target_image_path = os.path.join(tpath_exp, imgname)
                img = cv2.imread(image_path)
                #cv2.imshow('image',img)

                #img2 =img[50:400,80:430]
                img2 =cv2.resize(img,None,fx=0.65,fy=0.65,interpolation=cv2.INTER_CUBIC)
                #if img.shape[0]~=350
                if img2.shape[0]<img_size or img2.shape[1]<img_size:
                    #print img2.shape
                    imgtmp = np.zeros((img_size,img_size,3))
                    nrows = np.minimum(img2.shape[0],img_size)
                    ncols = np.minimum(img2.shape[1],img_size)
                    imgtmp[:nrows,:ncols,:] = img2[:nrows,:ncols,:]
                    img2 = imgtmp
                    #print img2.shape
                    #cv2.waitKey(0)
                    #continue
                img2 = img2[0:img_size,0:img_size]
                cv2.imwrite(target_image_path,img2)

                #cv2.imshow('image2', img2)

if __name__ == '__main__':
    #pathname = '/home/kaicao/Research/Data/Rolled/aligned_image_rearranged/'
    #pathname = '/media/kaicao/db6b99cc-3796-4fed-a5a0-5656bce8442c/Data/Rolled/Longitudinal/aligned_image_enhanced_rearranged/'
    pathname ='/research/prip-kaicao/Data/Rolled/aligned_image_rearranged/'

    tpathname =  '/research/prip-kaicao/Data/Rolled/aligned_image_rearranged_512/'
    img_size = int(np.round(512))
    #print os.stat(tpathname)
    try:
        os.stat(tpathname)
    except:
        os.mkdir(tpathname) 

    Cropping(pathname, tpathname,img_size)
