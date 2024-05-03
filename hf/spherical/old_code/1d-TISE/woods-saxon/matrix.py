import scipy as sci
import numpy as np
import sys
sys.path.insert(0, '../solvers')
import spec
class potentials:
    def HO(r,params):
        return params['kappa']*r**2
    def ws(r,params):
        result = params['V0']/(1+np.exp((r-params['R'])/params['a']))
        return(result)
    def coulomb(r,params):
        '''
        Uses the classical expression for the coulomb force with a cutoff radius
        (typically taken to the the proton radius Rp)

        V(r) = Z e^2 /2Rp  * ( 3 - r/Rp) for r <= Rp
             = Z e^2 /r for r > Rp

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        '''

        result = np.zeros(r.shape)
        for i,rVal in enumerate(r):
            if rVal < params['r_cutoff']:
                result[i] = params['Z']*params['e2']/(2*params['r_cutoff']) * ( 3.0 - (r[i]/params['r_cutoff'])**(2) )
            else:
                result[i] = params['Z']*params['e2']/r[i]
        return result
    def spin_orbit(r,j,l,params):
        '''
        Uses phenomenlogical expression for the spin-orbit potential:

            V_{so}(r) = V_{ls} (j(j+1) - l(l+1) - 3/4) r_{0}^2 dF(r)/dr * 1/r

        where V_[ls} and r_{0} are parameters, s = 1/2, and F(r)is the fermi functions

            F(r) = 1/(1 + exp((r - R)/a))
        The derivative of the  fermi function is done analytically and subbed in here.

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        l : TYPE
            DESCRIPTION.
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        '''

        R,a,Vls,r0 = params['R'], params['a'],params['Vls'],params['r0']
        dfdr =  - np.exp((r - R)/a)/(a*(1+ np.exp((r-R)/a))**2)
        #check_zero = np.where(r < params['r_cutoff'])
        result = Vls*r0**(2) * (j*(j+1) - l*(l+1) - .75)*dfdr/r
        #result[check_zero] = 0
        return result
    def centrifugal(r,l):
        #check_zero = np.where(r < params['r_cutoff'])
        result = l*(l+1)/r**2
        #result[check_zero] = 0
        return result

class spec_H_func:
    def __init__(self, N,CPnts,D_1,params):
        self.N = N
        self.CPnts = CPnts
        self.D_1 = D_1
        self.D_2 = np.matmul(D_1,D_1)
        self.params = params
    def spherical_ws(self,j,l,params,coulomb=True,BC=True):
        '''
        Uses 2nd order finite difference scheme to construct a discretized differential
        H operator for the WS + Vcent + Vcoulmn
        Parameters
        ----------
        mass : float
            DESCRIPTION.
        l: integer
            orbital angular momentum
        Returns
        -------
        H : function
            DESCRIPTION.

        '''
        Vr = np.diag(potentials.ws(self.CPnts,self.params))
        Vcent = np.diag(potentials.centrifugal(self.CPnts,l))
        Vso = np.diag(potentials.spin_orbit(self.CPnts,j,l,params))
        if coulomb == True:
            Vc = np.diag(potentials.coulomb(self.CPnts,params))
            H = params['hb2m0']*(-1.0*self.D_2 + Vcent) + Vr + Vc + Vso
        else:
            H = params['hb2m0']*(-1.0*self.D_2 + Vcent) + Vr + Vso

        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    def inf_well(self,params,BC=True):
        H = -1.0*params['hb2m0']*self.D_2
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    def HO(self,params,BC=True):
        Vho = np.diag(potentials.HO(self.CPnts,params))
        H = -1.0*params['hb2m0']*self.D_2 + Vho
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H


class FD_H_func:
    def __init__(self,grid,step_size,params):
        self.grid = grid
        self.dim = len(grid)
        self.step_size = step_size
        self.params = params
    def spherical_ws(self,j,l,coulomb=True,BC=True):
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
        Vr = np.diag(potentials.ws(self.grid,self.params))
        Vcent = np.diag(potentials.centrifugal(self.grid,l))
        Vso = np.diag(potentials.spin_orbit(self.grid,j,l,self.params))

        off_diag = np.zeros(self.dim)
        off_diag[1] = 1
        D2 = (-2*np.identity(self.dim) + sci.linalg.toeplitz(off_diag))/(self.step_size**2)

        if coulomb == True:
            Vc = np.diag(potentials.coulomb(self.CPnts,self.params))
            H = self.params['hb2m0']*(-1.0*D2 + Vcent) + Vr + Vc + Vso
        else:
            H = self.params['hb2m0']*(-1.0*D2 + Vcent) + Vr + Vso
        return H

    def HO(self,alpha,q):
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
        off_diag = np.zeros(self.dim)
        off_diag[1] = 1
        H = -1*(-2*np.identity(self.dim) + sci.linalg.toeplitz(off_diag))/(self.step_size**2)
        return H
    def quartic(self,psi,psiStar,mass,alpha,q):
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
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(mass*self.step_size**2) + np.diag(potentials.V_quartic(self.grid,alpha)) \
            + np.diag(potentials.V_rho(psi,psiStar,q))
        H = np.array(H,dtype='complex')
        return H
    def gaussian(self,psi,psiStar,mass,alpha,q):
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
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(mass*self.step_size**2) + np.diag(potentials.V_gaussian(self.grid)) \
            + np.diag(potentials.V_rho(psi,psiStar,q))
        return H
