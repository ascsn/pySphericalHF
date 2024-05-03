import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from init import *
import wf
import densities
import fields 
import functionals
#step_size = .01
#grid = np.arange(-10,10+step_size,step_size)
#box = len(grid)
#N , Z = 2,2
def getNumerov_matricies():
    off_diag = np.zeros(nbox)
    off_diag[1] = 1
    A = (-2*np.identity(nbox) + toeplitz(off_diag))/step_size**2
    B = (10*np.identity(nbox) + toeplitz(off_diag))/12
    return A,B
'''
def V_HO(x,alpha):
    return(alpha*x**2)

def construct_H(V,grid,mass,alpha):
    dim = len(grid)
    off_diag = np.zeros(dim)
    off_diag[1] = 1
    H = -1*(-2*np.identity(dim) + toeplitz(off_diag))/(mass*step_size**2) + np.diag(V(grid,alpha))
    return H
def solve(H,grid,h):
    evals,evects = np.linalg.eigh(H)
    evects = evects.T
    for i,evect in enumerate(evects):
        #norm = np.sqrt(1/sci.integrate.simpson(evect*evect,grid))
        norm = 1/(np.linalg.norm(np.dot(evect,evect)))
        evects[i] = evects[i]*norm
    return evals,evects
'''

l = 0
alpha = .5
#mass = 1

#H = construct_H(V_HO, grid, mass, alpha)

psi_array,energies = wf.initWfs(N,Z,name='HO')
plt.plot(grid,psi_array[0][0][0][1])
plt.title('rho')
plt.show()
### Initialize the Densities (\rho)
rhoArr = densities.rho(psi_array)


### Initialize the coulomb field
V_c = fields.coulombArr(rhoArr[1])
plt.plot(grid,V_c)
plt.title('fArr')
plt.show()

### Intialize the mean field Hamiltonian h with initial wavefunctions 
hArr = np.zeros(len(grid)) # the hamiltonian
hArr = functionals.h_BKN(rhoArr[0])
fArr = hArr + V_c + l*(l+1)/grid**2
#fArr = V_HO(grid,alpha)
plt.plot(grid,fArr)
plt.title('fArr')
plt.show()

V_matrix = np.diag(fArr)

dim = len(grid)
off_diag = np.zeros(dim)
off_diag[1] = 1
#A = (-2*np.identity(dim) + toeplitz(off_diag))/step_size**2
#B = (10*np.identity(dim) + toeplitz(off_diag))/12
A,B = getNumerov_matricies()
B_inv = np.linalg.inv(B)
H = np.matmul(B_inv,A) - hb2m0*V_matrix
#H = -1*(-2*np.identity(dim) + toeplitz(off_diag))/(mass*step_size**2) + V_matrix




#B_inv = np.linalg.inv(B)
#D = -A/(2.0*mass) + V_matrix

evals, evects = np.linalg.eigh(H)
#evects = evects.T
idx = np.argsort(evals)
evals = evals[idx]
evects = evects[:,idx]
E_0 = evals[-1]
psi = evects[:,-1]

print(-E_0/hb2m0)
plt.plot(grid,psi)
plt.show()
