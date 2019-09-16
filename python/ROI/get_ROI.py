import numpy as np


gy = np.gradient(img, axis=0)
gx = np.gradient(img, axis=1)

mag = np.abs(gx) + np.abs(gy)
sigma = 5
mag = gaussian(mag, sigma, multichannel=False, mode='reflect')