import numpy as np
import utils

class EDFParams:
    def inm_to_isospin(rhoc,Ms,Mv,Knm,Enm,anm,Lnm):
        h2m = utils.GlobalVariables.h2m
        Ck = 3/5*(3*np.pi**2/2)**(2/3)

        tc = Ck*rhoc**(2/3)
        
        g = (utils.GlobalVariables.h2m * (4*Ms-3)*tc - Knm - 9*Enm) / (h2m*(6*Ms-9)*tc + 9*Enm)
        
        C00pp = 1/(3*g*rhoc)* (h2m *((2-3*g)*Ms-3)*tc + 3*(1+g)*Enm)
        C0Dpp = 1/(3*g*rhoc**(1+g)) * (h2m*(3-2*Ms)*tc - 3*Enm)
        C0pt = h2m*(Ms-1)/rhoc
        C1pt = C0pt - h2m*(Mv-1)/rhoc
        C10pp = 1/(27*g*rhoc) * (27*(1+g)*anm - 9*Lnm + 5*tc*(2-3*g)*(C0pt+3*C1pt)*rhoc -\
                                5*tc*(1+3*g)*h2m)
        C1Dpp = 1/(27*g*rhoc**(g+1)) * (-27*anm + 9*Lnm + 5*(h2m-2*rhoc*(C0pt+3*C1pt))*tc)
        
        return g, C00pp, C0Dpp, C0pt, C1pt, C10pp, C1Dpp

    def isospin_to_pn(C00pp,C10pp,C0pt,C1pt,C0pdp,C1pdp,C0Dpp,C1Dpp,
                    C0pdJ,C1pdJ):
        return C00pp-C10pp, 2*C10pp, C0pt-C1pt, 2*C1pt, C0pdp-C1pdp, 2*C1pdp,\
            C0Dpp-C1Dpp, 2*C1Dpp, C0pdJ-C1pdJ, 2*C1pdJ

def localization(rho,dr_rho,dz_rho,tau):
    """
    Note that this is not the same definition as in various papers.
    Instead, it uses densities that are averaged over the spin states.
    """
    rho = rho/2
    dr_rho = dr_rho/2
    dz_rho = dz_rho/2
    tau = tau/2

    tauTF = 3/5*(6*np.pi**2)**(2/3) * rho**(5/3)
    ret = tau*rho - 1/4*(dr_rho**2 + dz_rho**2)
    ret /= (rho*tauTF)
    return 1/(1+ret**2)