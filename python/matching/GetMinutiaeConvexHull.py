from scipy.spatial import ConvexHull
import numpy as np
from math import pi, atan2, sin, cos
import matplotlib.pyplot as plt

def GetMinutiaeConvexHull(minutiae,R):
    ind = ConvexHull(np.transpose([minutiae[:,0],minutiae[:,1]]))
#    plt.plot(minutiae[:,0], minutiae[:,1], 'o')
#    for simplex in ind.simplices:
#        plt.plot(minutiae[simplex, 0], minutiae[simplex, 1], 'k-')
    
    vertices = np.insert(ind.vertices, 0, ind.vertices[len(ind.vertices)-1])
    vertices = np.append(vertices, ind.vertices[0])
    xv_t = minutiae[vertices,0]
    yv_t = minutiae[vertices,1]
    
    xv = np.zeros(len(xv_t)-2)
    yv = np.zeros(len(yv_t)-2)
    
    for i in range(1, len(vertices)-1):
        angle = atan2(yv_t[i+1] - yv_t[i-1], xv_t[i+1] - xv_t[i-1])
        if (yv_t[i] - yv_t[i-1])*(xv_t[i+1] - xv_t[i-1]) - \
            (xv_t[i]-xv_t[i-1])*(yv_t[i+1]-yv_t[i-1]) < 0:
            angle = angle - pi/2
        else:
            angle = angle + pi/2
        xv[i-1] = xv_t[i] + R*cos(angle)
        yv[i-1] = yv_t[i] + R*sin(angle)
    xv = np.append(xv, xv[0])
    yv = np.append(yv, yv[0])
    
    return xv, yv