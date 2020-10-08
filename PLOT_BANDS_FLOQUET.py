# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:15:37 2018

@author: toppgabr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:00:25 2018

@author: toppgabr
"""

import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['font.size'] = 16  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['figure.figsize'] = [10.,3]


ii = 4                                                                        # cell index
nn = 4*(ii**2+(ii+1)*ii+(ii+1)**2)                                             # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

num_GK = 32                                                                  # number of k-point per hgh symmetry line
num_KM = 31   

RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 
GREY = '#bdbdbd'

file_BANDS = open('Data/mu.dat','r')
mu = np.loadtxt(file_BANDS)
file_BANDS.close()

for mm in range(1):
    print(mm)
    file_BANDS = open('Data/bands_floquet.dat','r')
    MAT_BANDS_FLOQ = np.loadtxt(file_BANDS)-mu
    file_BANDS.close()
    
    file_BANDS = open('Data/bands_strob.dat','r')
    MAT_BANDS_FLOQ_STROB = np.loadtxt(file_BANDS)-mu
    file_BANDS.close()
    
    file_BANDS = open('Data/overlap_floquet.dat','r')
    MAT_OVERLAP = np.loadtxt(file_BANDS)
    file_BANDS.close()
    
    file_BANDS = open('Data/FLOQUET_BC_LOOP_PATH.dat','r')
    FLOQEUT_BC_LOOP_PATH = np.loadtxt(file_BANDS)-mu
    file_BANDS.close()
    
    
    omega=  2.228  
    n_max = 2 
    band_max = 20
    
    above = MAT_BANDS_FLOQ[63,:][MAT_BANDS_FLOQ[63,:] > 0.0].min()
    below = MAT_BANDS_FLOQ[63,:][MAT_BANDS_FLOQ[63,:] < 0.0].max()
    mu_update = 0.5*(below+above)
           
    fig1 = plt.figure(1)
    gs1 = gridspec.GridSpec(1, 1)
    
    ax12 = fig1.add_subplot(gs1[0,0])
    ax12.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
    ax12.set_xticks([0 , 32, 95,  128])
    ax12.set_xticklabels(['', r'$\mathrm{K1}$',  r'$\mathrm{K2}$', ''])
    
    kk = np.linspace(0,127,128)
    colormap = plt.cm.get_cmap('bwr', 100)
    MIN = -100000
    MAX = 100000

    ax12.plot(MAT_BANDS_FLOQ[:,0], 'b.', markersize=10.0, mew=0.1)
    ax12.plot(MAT_BANDS_FLOQ_STROB[:,:], 'kx', markersize=5.0, mew=0.1,)
    #for i in range(int(nn/2-band_max)*(2*n_max+1),int(nn/2+band_max)*(2*n_max+1)):
    #    sc1 = ax11.scatter(kk, MAT_ZERO[:,i], color=GREY, linewidth=0.0)
    for i in range(int(nn/2-band_max)*(2*n_max+1),int(nn/2+band_max)*(2*n_max+1)):    
        size = MAT_OVERLAP[:,i]*10
        sc1 = ax12.scatter(kk, MAT_BANDS_FLOQ[:,i]-mu_update, c=FLOQEUT_BC_LOOP_PATH[:,i], cmap=colormap, vmin=MIN, vmax=MAX, s=size, linewidth=0.1, edgecolor='k')
    ax12.vlines(x=63.5, ymin=-0.15, ymax = +0.15, color="k", linestyle="-", linewidth=1.0)
    plt.colorbar(sc1)
    ax12.set_ylim(-0.05*2.0,0.06*2.0)
    plt.legend(loc="upper right")
    plt.show()


