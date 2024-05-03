import numpy as np
import matplotlib.pyplot as plt
import sys
import matrix

sys.path.insert(0, '../solvers')
import spec
import utilities
'''
This script uses a pseudo-spectral Fourier routine to solve the 1-d TISE.
Note that in the Fourier basis, the problem needs to be mapped to the interval
[0,2pi] and then inverse mapped back to the original domain. Note the Gibbs phenomena 
in the ground state solution. This means some kind of windowing function needs
to be applied to make sure the basis smoothly goes to 0 as it approaches the boundary
'''
###############################################################################
##### Solver parameters
lb = 10**(-3)## left boundary of box in fm
rb = 20 ## right boundary of box in fm
N = 301 ## number of collocation points
currentInterval = (0,2*np.pi) # interval the derivatives are defined on.
targetInterval = (lb,rb) ## interval on real space to solve on

###############################################################################
##### model parameters
hb2m0 = 20.735530 # expression for hbar^2 /2mp

coulomb = False # option for to turn on coulomb force on proton
Z = 20
Nneu = 20

s = .5
nmax = 0 # max quantum number n to solve for
lmax = 0 # max orbital angular momentum to solve for

Vls = 22 - 14*(Nneu - Z)/(Nneu + Z)# spin orbit strength from Bohr and Mottelson
V0 = -51 + 33*(Nneu - Z)/(Nneu + Z) # MeV WS strength From Bohr and Mottelson

r0 = 1.27 # fm , radius parameter
a = .67 # fm , WS parameter
R = r0*(Nneu + Z)**(1/3) # fm , WS parameter

r_cutoff = .84 # approximate Rp (proton radius) in fm
e2 = 1.4399784 # e^2 charge in MeV fm

kappa = 1.0
###############################################################################


params = {'V0': V0,'R':R,'r0':r0,'a':a,'e2':e2,'r_cutoff': r_cutoff,'Vls': Vls,\
          'Z':Z,'Nneu':Nneu,'hb2m0':hb2m0,'kappa':kappa}


# Define Fourier points in the interval [0,2pi]
GL_func = spec.GaussLobatto()
CPnts,weights = GL_func.fourier(N)


# Define derivative matrix at the new Cpnts
## First transform the interval of points in real space to [0,2pi]
coord_trans = spec.coord_transforms(currentInterval,targetInterval)
CPnts_mapped,dgdy_affine  = coord_trans.inv_affine_gen(CPnts)
# define derivative matrix on the mapped interval
D_1 = spec.DerMatrix().getFourier(CPnts)/dgdy_affine

# define integration weights
int_weights = spec.GaussLobatto().getDx(CPnts)*dgdy_affine


H_func = matrix.spec_H_func(N, CPnts_mapped,D_1,params)

step_size = 10**(-2)
grid = np.arange(lb,rb,step_size)


# benchmark against finite difference
H_func_FD = matrix.FD_H_func(grid,step_size,params)

H_FD = H_func_FD.spherical_ws(0,0,coulomb=False,BC=True)

engs_FD,evects_FD = np.linalg.eigh(H_FD)

idx = engs_FD.argsort()
engs_FD = engs_FD[idx]
evects_FD = evects_FD[:,idx]
evects_FD = evects_FD.T

sol_FD = evects_FD[0]
FD_weights = np.full(grid.shape,step_size)

norm = utilities.normalize(sol_FD,FD_weights)
sol_FD = sol_FD/norm


print(f'FD Energy (0,0,0.5): {engs_FD[0]}')
nArr = np.arange(0,nmax+1,1)
lArr = np.arange(0,lmax+1,1)
for l in lArr:
    if l == 0:
        jArr = [l+s]
    else:
        jArr = [l-s,l+s]
    for j in jArr:
        # generate the potential for every j
        Vws = matrix.potentials.ws(CPnts_mapped,params)
        Vc = matrix.potentials.coulomb(CPnts_mapped,params)
        Vcent = matrix.potentials.centrifugal(CPnts_mapped,l)
        Vso = matrix.potentials.spin_orbit(CPnts_mapped,j,l,params)
        Vtot = params['hb2m0']*Vcent + Vws + Vso
        
        #plot the potential
        plt.plot(CPnts_mapped,Vtot)
        plt.title(f'V(r),  l = {l}, j = {j}')
        plt.ylim([-52,6])
        plt.xlabel('r')
        plt.ylabel('V(r)')
        plt.show()

        #construct the Hamiltonian
        H = H_func.spherical_ws(j,l,params,coulomb=coulomb,BC=True)
        # solve for evals and evects
        engs,evects = np.linalg.eigh(H)
        #sorting them from lowest to highest
        idx = engs.argsort()
        engs = engs[idx]
        evects = evects[:,idx]
        evects = evects.T
        sols = np.zeros((len(evects),N+1))
        for i,evect in enumerate(evects):
            # reattaching 0 to the wavefunction because we assumed Dirichlet BCs
            # in the derivative matrix
            sols[i] = np.concatenate([[0],np.real(evect),[0]])
        sol = sols[0]
        norm_spec = utilities.normalize(sol,int_weights)
        sol = sol/norm_spec


        plt.plot(CPnts_mapped,sol)
        plt.plot(grid,sol_FD)
        plt.title(f'Wavefunction l = {l}, j = {j}')
        plt.xlabel('r')
        plt.ylabel(r'$\psi(r)$')
        plt.show()
        for n in nArr:
            print(f'Fourier Energy ({n},{l},{j}) = {engs[n]}')

