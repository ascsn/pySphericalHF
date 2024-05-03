import numpy as np
import potentials
import matrix
import sys
sys.path.insert(0, './solvers')
import utilities

def woodSaxon(CPnts_mapped,D_1,params):
    N = params['N'] #number of collocation points
    nmax_neu,lmax_neu = utilities.getMax_n_l(params['N_neu']) # max quantum number n to solve for
    nmax_pro,lmax_pro = utilities.getMax_n_l(params['Z_pro'])
    H_func = matrix.spec_H_func(N, CPnts_mapped,D_1,params)

    '''
    Vws = potentials.ws(CPnts_mapped,params)
    Vc = potentials.coulomb(CPnts_mapped,params)
    Vcent = potentials.centrifugal(CPnts_mapped,l)
    Vso = potentials.spin_orbit(CPnts_mapped,j,l,params)
    Vtot = params['hb2m0']*Vcent + Vws + Vso
    '''
    #nArr_neu = np.arange(0,nmax_neu+1,1)
    lArr_neu = np.arange(0,lmax_neu+1,1)

    #nArr_pro = np.arange(0,nmax_pro+1,1)
    lArr_pro = np.arange(0,lmax_pro+1,1)
    # First compute the wavefunctions for the neutrons
    sols_neu = []
    engs_neu = []
    sols_pro = []
    engs_pro = []
    qNum_list_neu = []
    qNum_list_pro = []
    for l in lArr_neu:
        if l == 0:
            jArr = [l+0.5]
        else:
            jArr = [l-0.5,l+0.5]
        for j in jArr:
            # BC is enforcing Dirchlet BCs to find bound states
            H = H_func.spherical_ws(j,l,params,coulomb=False,BC=True)
            engs,evects = np.linalg.eig(H)
            #sorting them from lowest to highest
            idx = engs.argsort()
            engs = engs[idx]
            evects = evects[:,idx]
            evects = evects.T
            bound_state_idx = np.where(engs < 0)[0]
            if len(bound_state_idx) != 0:
                for i in range(len(bound_state_idx)):
                    #First, store the quantum numbers corresponding to the bound states
                    # that is, all the states with negative
                    # since the evects are ordered least to greatest, the bound state
                    # index should be ascending.
                    qNum = [bound_state_idx[i],l,j]
                    qNum_list_neu.append(qNum)
                    sols_neu.append(np.real(evects[bound_state_idx[i]]))
                    engs_neu.append(engs[bound_state_idx[i]])
            else: continue
    engs_neu = np.array(engs_neu)
    qNum_list_neu = np.array(qNum_list_neu)
    sols_neu = np.array(sols_neu)

    idx = engs_neu.argsort()
    print(engs_neu[idx])
    sols_neu = sols_neu[idx]
    qNum_list_neu = qNum_list_neu[idx]
    print(qNum_list_neu)
    # Next solve for protons. Check if we want the coulomb interaction to be turned
    # on for the protons. If it is set to false, just copy the neutron Wfs
    if params['coulomb'] == False:
        sols_pro = sols_neu.copy()
    else:
        # if we want coulomb, then solve the same problem with coulomb turned on.
        for l in lArr_pro:
            if l == 0:
                jArr = [l+0.5]
            else:
                jArr = [l-0.5,l+0.5]
            for j in jArr:
                # BC is enforcing Dirchlet BCs to find bound states
                H = H_func.spherical_ws(j,l,params,coulomb=params['coulomb'],BC=True)
                engs,evects = np.linalg.eig(H)
                #sorting them from lowest to highest
                idx = engs.argsort()
                engs = engs[idx]
                #print(engs)
                evects = evects[:,idx]
                evects = evects.T
                bound_state_idx = np.where(engs < 0)[0]
                if len(bound_state_idx) != 0:
                    for i in range(len(bound_state_idx)):
                        #First, store the quantum numbers corresponding to the bound states
                        # that is, all the states with negative
                        # since the evects are ordered least to greatest, the bound state
                        # index should be ascending.
                        qNum = [bound_state_idx[i],l,j]
                        qNum_list_pro.append(qNum)
                        sols_pro.append(np.real(evects[bound_state_idx[i]]))
                        engs_pro.append(engs[i])
                else: continue
        engs_pro = np.array(engs_pro)
        qNum_list_pro = np.array(qNum_list_pro)
        sols_pro = np.array(sols_pro)

        idx = engs_pro.argsort()
        sols_pro = sols_pro[idx]
        qNum_list_pro = qNum_list_pro[idx]

        ## Now, just return the occupied states, for neutron and proton
        particle_count = 0
        state_count = 0
        while particle_count < params['N_neu']:
            print('neutron state', qNum_list_neu[state_count])
            particle_count += 2*qNum_list_neu[state_count][-1] + 1
            state_count += 1
            print('neutron count: ',particle_count)
        sols_neu = sols_neu[:state_count]
        qNum_list_neu= qNum_list_neu[:state_count]

        particle_count = 0
        state_count = 0
        while particle_count < params['Z_pro']:
            print('proton state', qNum_list_pro[state_count])
            particle_count += 2*qNum_list_pro[state_count][-1] + 1
            state_count += 1
        print('proton count: ',particle_count)
        sols_pro= sols_pro[:state_count]
        qNum_list_pro= qNum_list_pro[:state_count]
    return sols_neu, qNum_list_neu, sols_pro, qNum_list_pro

def HO():
    return