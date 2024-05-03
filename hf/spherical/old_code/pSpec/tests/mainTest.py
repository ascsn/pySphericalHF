import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
from init import *
import densities 
import fields 
import solvers 
import wf
'''
Essentially just tests the loops by solving the HO SE for each loop. each wavefunction is decoupled
'''


### Initialize the single particle wavefunctions
print(f'N: {N}, Z: {Z}')
print(f'Initilizaing Wavefunctions with {initial_wf}')
psi_array,energies = wf.initWfs(N,Z,name='test')
print('psi_array shape:', psi_array.shape)
print('shape labels: (q,n,l,s)')

'''
### Initialize the Densities (\rho)
print(f'Initilizaing Wavefunctions with {initial_wf}')
rhoArr = densities.rho(psi_array)
print('rhoArr shape: ',rhoArr[0].shape)

### Initialize the fields 
V_yuk = np.zeros(len(grid))
V_c = np.zeros(len(grid))

for i,r in enumerate(grid):
    V_yuk[i] = yuk(rhoArr[0],r)
    V_c[i] = fields.coulomb(rhoArr[1],r)
'''

alpha = .5
E0 = .2
dE = 10**-2
VArr = fields.harm(grid,alpha)

for q in range(0,2):
    for n in range(nmax):
        for l in range(lmax+1):
            for s in range(len(spin)):
                j = l + spin[s]
                E_0,psi_test = solvers.solve_Numerov(psi_array[q][n][l][s],E0,dE,VArr)
                psi_array[q][n][l][s] = psi_test
                energies[q][n][l][s] = E_0
                print('Numerical E_0: ',E_0)
print(energies)
plt.plot(grid,psi_array[0][0][0][1])
plt.show()

