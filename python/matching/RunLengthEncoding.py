import numpy as np

def RunLengthEncoding(mask,h,w):
    if np.ndim(mask)==1 and len(locals())==3:
        print("Length: ", len(mask))
        #from run length coding to image
        run_mask = np.zeros((h*w,1))
        mask = np.cumsum(mask)
        
        for i in range(0,len(mask),2):
            run_mask[mask[i-1]:mask[i]] = 1
        run_mask = np.reshape(run_mask,(h,w))
    else:
        #from image to run length
        mask = mask>0
        mask[1,1] = 0
        mask = mask[:]
     
        J = np.nonzero(np.diff([np.logical_not(mask[1]),[mask]]))
        run_mask = np.diff([J, np.size(mask)+1])
        run_mask = run_mask[:]
    
    return