import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import assoc_laguerre
sys.path.insert(0, '../src')
from init import *
import wf
import utilities

#psifunc = wf.get_WfHydrogen_radial(2,1)
#psi_eval = psifunc(grid)
#plt.plot(grid,psi_eval)
#plt.show()


psiArr = wf.initWfs(name='hydrogen')


print(psiArr[0].shape)
for q in range(2):
    for n in range(nmax+1):
        for l in range(lmax+1):
            for s in range(len(spin)):
                plt.plot(grid,psiArr[0][q][n][l][s])
plt.show()


