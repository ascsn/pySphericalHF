import scipy as sci
import numpy as np
import potentials
import sys
#from init import *
sys.path.insert(0, '../solvers')
import spec

class FD_H_func:
    def __init__(self,grid,params):
        self.grid = grid
        self.params = params
        
    def HO(self,psi,psiStar,params):
        '''
        Uses 2nd order finite difference scheme to construct a discretized differential
        H operator for the GP potential.

        Parameters
        ----------
        psi : ndArray
            wavefunction.
        psiStar : ndArray
            wavefunction conjugate dual.
        mass : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        '''
        dim = len(self.grid)
        off_diag = np.zeros(dim)
        off_diag[1] = 1
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(params['step_size']**2) + \
        np.diag(potentials.V_HO(self.grid,self.params)) \
            +np.diag(potentials.V_rho(psi,psiStar,self.params))
        return H
    def quartic(self,psi,psiStar,params):
        '''
        Uses 2nd order finite difference scheme to construct a discretized differential
        H operator for the GP with quartic potential.

        Parameters
        ----------
        psi : ndArray
            wavefunction.
        psiStar : ndArray
            wavefunction conjugate dual.
        mass : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        '''
        dim = len(self.grid)
        off_diag = np.zeros(dim)
        off_diag[1] = 1
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(self.params['mass']*self.params['step_size']**2) + \
        np.diag(potentials.V_quartic(self.grid,self.params)) \
            + np.diag(potentials.V_rho(psi,psiStar,self.params))
        H = np.array(H,dtype='complex')
        return H
    def gaussian(self,psi,psiStar,params):
        '''
        Uses 2nd order finite difference scheme to construct a discretized differential
        H operator for the GP potential with gaussian potential.

        Parameters
        ----------
        psi : ndArray
            wavefunction.
        psiStar : ndArray
            wavefunction conjugate dual.
        mass : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        '''
        dim = len(self.grid)
        off_diag = np.zeros(dim)
        off_diag[1] = 1
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(self.params['mass']*self.params['step_size']**2) +\
        np.diag(potentials.V_gaussian(self.grid)) \
            + np.diag(potentials.V_rho(psi,psiStar,self.params))
        return H
class spec_H_func:
    def __init__(self, N,CPnts,D_1,params):
        self.N = N
        self.CPnts = CPnts
        self.D_1 = D_1
        self.D_2 = np.matmul(D_1,D_1)
        self.params = params
    def HO(self,psi,psiStar,BC=True):
        '''
        Constructs a harmonic oscillator hamiltonian given a differentiation matrix
        operator. Note: all the constants are assumed to be smashed into alpha.

        Parameters
        ----------
        psi : ndArray
            wavefunction.
        psiStar : ndArray
            wavefunction conjugate dual.
        mass : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        '''
        H = -1.0*self.D_2 + \
        np.diag(potentials.V_HO(self.CPnts,self.params['alpha']))
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    def GP_HO(self,psi,psiStar,BC=True):
        H = -1.0*self.D_2 + \
        np.diag(potentials.V_HO(self.CPnts,self.params['alpha'])) \
            + np.diag(potentials.V_rho(psi,psiStar,self.params['q']))
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    
