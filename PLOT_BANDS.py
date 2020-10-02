# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:00:25 2018

@author: toppgabr
"""

import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 20  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['figure.figsize'] = [10.,8]

#m = 1

RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 

ii = 4                                                                # cell index
nn = int(4*(ii**2+(ii+1)*ii+(ii+1)**2))                                             # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

num_GK = 32                                                                  # number of k-point per hgh symmetry line
num_KM = 31     

file_BANDS = open('mu.dat','r')
mu = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('bands.dat','r')
MAT_BANDS = np.loadtxt(file_BANDS)-mu
file_BANDS.close()

file = open('EQ_BC_LOOP_PATH.dat','r')
EQ_BC_LOOP_PATH = np.loadtxt(file)
file.close()

print(np.shape(MAT_BANDS))


above = MAT_BANDS[0,:][MAT_BANDS[0,:] > 0.0].min()
below = MAT_BANDS[0,:][MAT_BANDS[0,:] < 0.0].max()
GAP_GAMMA = above-below   

aboveM = MAT_BANDS[127,:][MAT_BANDS[127,:] > 0.0].min()
belowM = MAT_BANDS[127,:][MAT_BANDS[127,:] < 0.0].max()
GAP_M = aboveM-belowM
  
file = open("../GAPS.txt", "a")
file.write(str(GAP_GAMMA) + " " + str(GAP_M) + "\n")
file.close()

fig1 = plt.figure(1)
gs1 = gridspec.GridSpec(2, 1)
ax12 = fig1.add_subplot(gs1[0,0])

ax12.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
#ax12.set_xticks([0 , num_GK, num_GK+num_KM])
#ax12.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{K1}$' , r'$\mathrm{M}$'])
ax12.set_xticks([num_GK, num_GK+2*num_KM+1])
ax12.set_xticklabels([r'$\mathrm{K1}$',  r'$\mathrm{K2}$'])
ax12.plot([0]*np.size(MAT_BANDS[:,0]), 'k--', linewidth=1.0)
ax12.plot(MAT_BANDS[:,:], 'k', linewidth=2.0)
ax12.plot(0, above, 'ro', linewidth=2.0)
ax12.plot(0, below, 'ro', linewidth=2.0)
ax12.plot(127, aboveM, 'bo', linewidth=2.0)
ax12.plot(127, belowM, 'bo', linewidth=2.0)


ax12.plot(MAT_BANDS[:,int(nn/2)] , 'k-') 
ax12.plot(MAT_BANDS[:,int(nn/2+1)] , 'k-')
ax12.plot(MAT_BANDS[:,int(nn/2-2)] , '-', color=GREEN) 
ax12.plot(MAT_BANDS[:,int(nn/2-1)] , '--', color=VIOLETT)  
ax12.plot(MAT_BANDS[:,0], 'k', linewidth=2.0, label=r"$\Delta_\Gamma =$"+str(np.round(GAP_GAMMA,3))+" $\mathrm{eV}$")
ax12.plot(MAT_BANDS[:,0], 'k', linewidth=2.0, label=r"$\Delta_M =$"+str(np.round(GAP_M,3))+" $\mathrm{eV}$")
#plt.legend(loc="upper right",handlelength=0, handletextpad=0, fancybox=True)
#ax12.set_ylim(below*1.2,above*1.2)  
ax12.text(0.99, 0.99, r'(b)', fontsize=20, horizontalalignment='right', verticalalignment='top', transform=ax12.transAxes)
ax12.text(0.06, 0.6, r'$\mathrm{\Delta_\Gamma}$', color='r', fontsize=15, horizontalalignment='left', verticalalignment='top', transform=ax12.transAxes)
ax12.annotate("", xy=(0.0, below), xytext=(0.0, above), arrowprops=dict(arrowstyle="<->", color='r'))
ax12.text(0.94, 0.6, r'$\mathrm{\Delta_M}$', color='b', fontsize=15, horizontalalignment='right', verticalalignment='top', transform=ax12.transAxes)
ax12.annotate("", xy=(127, belowM), xytext=(127, aboveM), arrowprops=dict(arrowstyle="<->", color='b'))
#plt.subplots_adjust(top=0.90)

ax21 = fig1.add_subplot(gs1[1,0])
ax21.set_xticks([num_GK, num_GK+2*num_KM+1])
ax21.set_xticklabels([])
ax21.set_ylabel(r'$\mathrm{Curvature}$ ($\mathrm{\AA^2}$)')
ax21.plot(EQ_BC_LOOP_PATH[:,int(nn/2-2)] , '-', color=GREEN) 
ax21.plot(EQ_BC_LOOP_PATH[:,int(nn/2-1)] , '--', color=VIOLETT)  
ax21.set_xticks([num_GK, num_GK+2*num_KM+1])
ax21.set_xticklabels([r'$\mathrm{K1}$',  r'$\mathrm{K2}$'])

plt.tight_layout()

plt.show()
