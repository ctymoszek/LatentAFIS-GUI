from numpy import pi

def dphi(t1, t2):
    t = t1 - t2
    if t < -pi:
        t = t + 2*pi
    elif t >= pi:
        t = t - 2*pi
    
    return t