import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import os
from PIL import Image
import tifffile as tif

from pixel_lib import *

#%%

img = []

for file in os.listdir("./"):
    if file.endswith(".tif"):
        print(file)
        img.append( tif.imread( file ) )
        img[-1] = np.mean( img[-1][:,:,0:3], axis = 2) # convert to greyscale
        img[-1] = img[-1]*hann2d(img[-1].shape[1], img[-1].shape[0])
        plt.figure()
        plt.imshow(img[-1])
del(file)

#%%

DX = []
DY = []

for i in range(0, len(img)-1):
    C = normxcorr2(img[i],img[i+1], 'same')
    plt.figure()
    plt.imshow(C)
    dx, dy = drift(C)
    print('dx = ', dx)
    print('dy = ', dy)
    DX.append(dx)
    DY.append(dy)
    
DX = np.asarray(DX)
DY = np.asarray(DY)
    
d = 100 # um

#%%

px_size = np.abs( np. mean( d/DY ) )
err = np.std( d/DY ) / np.sqrt( len(DY) )

print('pixel size =', px_size, "\u00B1", err)

#%%

real_px_size = 3.6 # um

print('real pixel size =', real_px_size)

Magnification = real_px_size / px_size

print('Magnification =', Magnification)


plt.show()