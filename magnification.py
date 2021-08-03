import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile as tif
import pandas as pd
from natsort import natsorted

from pixel_lib import *

plt.rcParams['pdf.fonttype'] = 42

#%% theoretical results

def mag_d3(f_1, f_2, f_3, f_4, f_5, d_3):

    m = ( f_2*f_3*f_5 ) / ( f_1*f_4*( d_3 - f_2 + f_3 ) )
    return m


def D4(f_2, f_3, f_4, d_3):

    d_4 = (-d_3*f_3 + d_3*f_4 + f_2*f_3 - f_2*f_4 + f_3*f_4)/(d_3 - f_2 + f_3)
    return d_4

f_1 = 50
f_2 = 100
f_3 = 75
f_4 = 75
f_5 = 50

#%%

d = 100 # um
real_px_size = 3.6 # um

#%%

magnifications = []
errors = []

for folder in natsorted( os.listdir("./") ):
    if os.path.isdir(folder) and ( folder[0] != '_' ):
        print(folder)
        img = []
        for file in os.listdir(folder):
            if file.endswith(".tif"):
                print(file)
                img.append( tif.imread( folder + '/' + file ) )
                img[-1] = np.mean( img[-1][:,:,0:3], axis = 2) # convert to greyscale
                img[-1] = img[-1]*hann2d( img[-1].shape )
        
        DX = []
        DY = []
        
        for i in range(0, len(img)-1):
            C = normxcorr2(img[i],img[i+1], 'same')
            # plt.figure()
            # plt.imshow(C)
            dx, dy = drift(C)
            # print('dx = ', dx)
            # print('dy = ', dy)
            DX.append(dx)
            DY.append(dy)
            
        DX = np.asarray(DX)
        DY = np.asarray(DY)
        
        DX = DX[ DX != 0 ]
        DY = DY[ DY != 0 ]
        
        px_size = np.abs( np. mean( d/DY ) )
        err = np.std( d/DY ) / np.sqrt( len(DY) )
        
        print('pixel size =', px_size, "\u00B1", err)
    
        Magnification = real_px_size / px_size
        err_mag = real_px_size * err / px_size**2
        
        print('Magnification =', Magnification, "\u00B1", err_mag)
        
        magnifications.append( Magnification )
        errors.append( err_mag )

#%% experiments - read data

for file in os.listdir():
    if file.endswith(".xlsx"):
        # print(file)
        data = pd.read_excel( file )

d_3 = data['d3 [mm]'].to_numpy()
d_4 = data['d4 [mm]'].to_numpy()

magnifications = np.asarray( magnifications )
errors = np.asarray( errors )

#%% theory - create data

d_3_t = np.arange(d_3[0], d_3[-1] + 1, step = 1)

theoretical_mag = mag_d3(f_1, f_2, f_3, f_4, f_5, d_3_t)
theoretical_d4 = D4(f_2, f_3, f_4, d_3_t)

#%% generate plots

plt.figure(1)

plt.plot(d_3_t, theoretical_mag)
plt.errorbar(d_3, magnifications, errors, fmt = 'o')

plt.legend(['Theoretical model', 'Experimental measure'])
plt.xlabel( '$d_3$ (mm)' )
plt.ylabel( 'Magnification' )

plt.figure(2)

plt.plot(d_3_t, theoretical_d4)
plt.plot(d_3, d_4, 'o')

plt.legend(['Theoretical model', 'Experimental measure'])
plt.xlabel( '$d_3$ (mm)' )
plt.ylabel( '$d_4$ (mm)' )