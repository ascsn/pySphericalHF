import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import assoc_laguerre
sys.path.insert(0, '../src')
import wf
import densities
import utilities
from init import *

psi_array,energies  = wf.initWfs(name='HO')
print(psi_array.shape)
plt.plot(grid,psi_array[0][0][0][1])
plt.title('a wavefunction')
plt.show()


rhoArr = densities.rho(psi_array)
print(np.linalg.norm(rhoArr[0]))
npro,nneu = utilities.getNZ(rhoArr)
print(f'Total number of nucleons: {nt}')
print(f'Integrated number of nucleons: {npro+nneu}')
print(f'Total number of protons: {Z}')
print(f'Integrated number of protons: {npro}')
print(f'Total number of neutrons: {N}')
print(f'Integrated number of Neutrons: {nneu}')
#Rp,Rn,Rch =  utilities.getRadi(rhoArr)

#print(f'Rp = {Rp}')
#print(f'Rn = {Rn}')
#print(f'Rch = {Rch}')
plt.plot(grid,rhoArr[0])
plt.title('total rho')
plt.show()

plt.title('proton rho')
plt.plot(grid,rhoArr[1])
plt.show()

plt.title('neutron rho')
plt.plot(grid,rhoArr[2])
plt.show()
