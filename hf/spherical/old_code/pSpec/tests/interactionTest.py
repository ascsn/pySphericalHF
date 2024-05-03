import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import wf
import densities
import fields 
import functionals
from init import *


psi_array,energies = wf.initWfs(name='HO')
#print('psi_array shape:', psi_array.shape)
#print('shape labels: (q,n,l,s)')

### Initialize the Densities (\rho)
#print(f'Initilizaing Wavefunctions with {initial_wf}')
rhoArr = densities.rho(psi_array)


### Initialize the coulomb field
V_c = fields.coulombArr(rhoArr[1])


### Intialize the mean field Hamiltonian h with initial wavefunctions 
#hArr = np.zeros(len(grid)) # the hamiltonian
yuk = fields.yukArr4(rhoArr[0])
hArr = .75*t0*rhoArr[0] + 0.1875*t3*rhoArr[0]**2  + yuk 
fArr = hArr + V_c

plt.plot(grid,fArr,label='fArr')
plt.plot(grid,hArr,label='hArr')
plt.plot(grid,V_c,label='coulomb')
plt.legend()
plt.show()