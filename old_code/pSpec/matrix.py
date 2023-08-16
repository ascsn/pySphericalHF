import numpy as np
import potentials
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
