import os
import numpy as np
import RunLengthEncoding

def Bin2Template_Byte_TF(fname,isLatent):
    print("fname: " + fname)
    template = []
    
    if not os.path.isfile(fname):
        return
    try:
        f = open(fname,"rb")
    except OSError as e:
        print("Could not open file %s: %s",fname, e)
        
    tmp = np.fromfile(f,dtype=np.uint16,count=2)
    h = tmp[0]
    w = tmp[1]
    
    if h==0 or w==0:
        f.close()
        return
    
    tmp = np.fromfile(f,dtype=np.uint16,count=2)
    blkH = tmp[0]
    blkW = tmp[1]
    
    #minutiae points
    num_template = np.fromfile(f,dtype=np.uint8,count=1)[0]
    minu_template = np.empty((num_template,1))
    for n in range(num_template):
        minu_num = np.fromfile(f,dtype=np.uint16,count=1)[0]
        if isLatent:
            minutiae = np.zeros((minu_num,4))
        else:
            minutiae = np.zeros((minu_num,3))
        
        if minu_num > 0:
            x = np.fromfile(f,dtype=np.uint16,count=minu_num) # x
            minutiae[:,1] = x
            y = np.fromfile(f,dtype=np.uint16,count=minu_num) # y
            minutiae[:,2] = y
            ori = np.fromfile(f,dtype=np.float32,count=minu_num) # ori
            minutiae[:,3] = ori
            des_num = np.fromfile(f,dtype=np.uint16,count=1)[0]
            des_len = np.fromfile(f,dtype=np.uint16,count=1)[0]
            Des = []
            for i in range(des_num):
                tmp = np.fromfile(f,dtype=np.float32,count=des_len*minu_num) # descriptor
                tmp = np.reshape(tmp,(des_len,minu_num))
                for j in range(minu_num):
                    tmp[:,j] = tmp[:,j]/(np.linalg.norm(tmp[:,j])+0.0001)
                Des.append(tmp)
        else:
            Des = [];

        oimg = np.fromfile(f,dtype=np.float32,count=blkH*blkW)
        oimg = np.reshape(oimg,(blkH,blkW))
    
        run_mask_num = np.fromfile(f,dtype=np.uint16,count=1)[0]
        run_mask = np.fromfile(f,dtype=np.uint32,count=run_mask_num)
        mask = RunLengthEncoding.RunLengthEncoding(run_mask,h,w)
        minu_template[n].minutiae = minutiae
        minu_template[n].Des = Des
        minu_template[n].oimg = oimg
        minu_template[n].mask = mask
            
    # texture template
    # minutiae points
    minu_num = np.fromfile(f,dtype=np.uint16,count=1)
    minutiae = np.zeros(minu_num,4)
    if minu_num > 0:
         x = np.fromfile(f,dtype=np.uint16,count=minu_num) #x 
         minutiae[:,1] = x
         y = np.fromfile(f,dtype=np.uint16,count=minu_num) # y
         minutiae[:,2] = y
         ori = np.fromfile(f,dtype=np.float32,count=minu_num) #ori
         minutiae[:,3] = ori
         if isLatent == 1:
             D = np.fromfile(f,dtype=np.float32,count=minu_num) #something
             minutiae[:,4] = D
         des_num = np.fromfile(f,dtype=np.uint16,count=1)
         des_len = np.fromfile(f,dtype=np.uint16,count=1)
         Des = np.empty((des_num,1))
         for i in  range(des_num):
             tmp = np.fromfile(f,dtype=np.float32,count=des_len*minu_num) # descriptor
             tmp = np.reshape(tmp,des_len,minu_num)
             for j in range(minu_num):
                 tmp[:,j] = tmp[:,j]/(np.linalg.norm(tmp[:,j])+0.0001)
             Des[i] = tmp
    else:
        minutiae = []
        Des = []
        mask = []
    texture_template = {}
    texture_template['minutiae'] = minutiae
    texture_template['Des'] = Des
    texture_template['mask'] = minu_template[1].mask
    
    template.minu_template = minu_template
    template.texture_template = texture_template
    
    f.close()
    
    return