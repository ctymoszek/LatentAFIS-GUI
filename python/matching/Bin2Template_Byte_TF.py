import struct
import numpy as np
from numpy import linalg as LA

class MinuTemplate():
    def __init__(self,h = 0, w = 0, blkH = 0, blkW = 0, minutiae = None,
                 des = None,oimg = None, mask=None):
        self.h = h
        self.w = w
        self.blkH = blkH
        self.blkW = blkW
        self.minutiae = minutiae
        self.des = des
        self.oimg = oimg
        self.mask = mask

class TextureTemplate():
    def __init__(self,h = 0, w = 0, minutiae = None,des = None,mask=None):
        self.h = h
        self.w = w
        self.minutiae = minutiae
        self.des = des
        self.mask = mask

class Template():
    def __init__(self,minu_template=None,texture_template=None):
        self.minu_template = [] if minu_template is None else minu_template
        self.texture_template = [] if texture_template is None else texture_template
        #return
    def add_minu_template(self,minu_template):
        self.minu_template.append(minu_template)

    def add_texture_template(self,texture_template):
        self.texture_template.append(texture_template)

def run_length_decoding(run_mask,h,w):
    mask = np.zeros((h*w,),dtype=int)
    run_mask = np.cumsum(run_mask)
    for i in range(1,len(run_mask),2):
        mask[run_mask[i-1]:run_mask[i]] = 1
    mask = np.reshape(mask, (w, h))
    mask = mask.transpose()

    return mask

def Bin2Template_Byte_TF(fname,isLatent=True):

    template = Template()
    with open(fname,'rb') as file:
        tmp = struct.unpack('H'*2, file.read(4))
        h = tmp[0]
        w = tmp[1]
        if w<=0 or h<=0:
            return None
        tmp = struct.unpack('H' * 2, file.read(4))
        blkH = tmp[0]
        blkW = tmp[1]
        tmp = struct.unpack('B', file.read(1))
        # number of minutiae templates
        num_template = tmp[0]
        for i in range(num_template):
            tmp = struct.unpack('H', file.read(2))
            minu_num = tmp[0]
            if minu_num<=0:
                minutiae = None
                des = None
                continue
            minutiae = np.zeros((minu_num,3 + isLatent),dtype=float)

            # x location
            tmp = struct.unpack('H'*minu_num, file.read(2*minu_num))
            minutiae[:,0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))
            
            # reliability
            #tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
            #minutiae[:, 3] = np.array(list(tmp))

            tmp = struct.unpack('H', file.read(2))
            des_num = tmp[0]
            tmp = struct.unpack('H', file.read(2))
            des_len = tmp[0]
            des = []
            for j in range(des_num):
                tmp = struct.unpack('f' * des_len * minu_num, file.read(4*des_len*minu_num))
                single_des = np.array(list(tmp))
                single_des = np.reshape(single_des,(minu_num,des_len))
                # different from the matlab code. Each row of single_des is a
                # descriptor for a minutia
                single_des = np.float32(single_des)
                for k in range(minu_num):
                    single_des[k] = single_des[k]*1.0/(LA.norm(single_des[k])+0.000001)
                des.append(single_des)

            # get the orientation field
            tmp = struct.unpack('f' * blkH*blkW, file.read(4 *blkH*blkW))
            oimg = np.array(list(tmp))
            oimg = np.reshape(oimg,(blkW,blkH))
            oimg = oimg.transpose()

            tmp = struct.unpack('H', file.read(2))
            run_mask_num = tmp[0]
            tmp = struct.unpack('I'*run_mask_num, file.read(4*run_mask_num))
            tmp_list = list(tmp)
            mask = run_length_decoding(tmp_list, h, w)
            minu_template = MinuTemplate(h=h,w=w,blkH=blkH,blkW=blkW,
                                         minutiae=minutiae,des=des,oimg=oimg,
                                         mask=mask)
            template.add_minu_template(minu_template)
        # load texture template

        tmp = struct.unpack('H', file.read(2))
        minu_num = tmp[0]
        if isLatent == 1:
            minutiae = np.zeros((minu_num, 4), dtype=float)
        else:
            minutiae = np.zeros((minu_num, 3), dtype=float)

        if minu_num <= 0:
            minutiae = []
            des = []
            texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae,
                                               des=des, mask=mask)
            template.add_texture_template(texture_template)
            return template

        # x location
        tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
        minutiae[:, 0] = np.array(list(tmp))
        # y location
        tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
        minutiae[:, 1] = np.array(list(tmp))

        # minutiae orientation
        tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
        minutiae[:, 2] = np.array(list(tmp))

        # distance to the border
        if isLatent:
            tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

        tmp = struct.unpack('H', file.read(2))
        des_num = tmp[0]
        tmp = struct.unpack('H', file.read(2))
        des_len = tmp[0]
        des = []
        for j in range(des_num):
            tmp = struct.unpack('f'*des_len*minu_num, file.read(4*des_len*minu_num))
            single_des = np.array(list(tmp))
            single_des = np.reshape(single_des, (minu_num, des_len))
            # different from the matlab code. Each row of single_des is a
            #descriptor for a minutia
            single_des = np.float32(single_des)
            for k in range(minu_num):
                single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
            des.append(single_des)
        texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae,
                                           des=des, mask=mask)
        template.add_texture_template(texture_template)
    return template
