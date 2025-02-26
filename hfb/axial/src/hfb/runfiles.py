"""
A whole bunch of classes for typical running of my HFB code,
similar to things like ReducedHFB in utils.py
"""
import sys, os
import utils, solvers

import numpy as np
import multiprocessing as mp

class Main:
    def __init__(self,edf,basis,nParticles,r,bz,bp,
                 xi,eta,wz,wr,
                 densitiesToConstruct=utils.VariableNames.defaultReconDensities):
        self.edf = edf
        self.basis = basis
        self.pairRegObjs = {pn:solvers.PairingRegularization(basis) for pn in ['n','p']}
        self.nParticles = nParticles
        
        self.bz = bz
        self.bp = bp

        self.xi, self.eta = xi, eta
        self.wz, self.wr = wz, wr
        
        self.rGridRepeated = np.repeat(r[None,:,None],basis.psi[0].shape[-1],axis=-1)
        
        self.densitiesToConstruct = densitiesToConstruct
    
    def _atomic_hfb_matrix(self,fields,chemPot):
        arrObj = solvers.HFBMatrix(self.basis,self.xi,self.eta,self.wz,self.wr)

        phFields = {}
        ppFields = {}
        for (key,arr) in fields.items():
            if key in utils.VariableNames.phKeys:
                phFields[key] = arr
            elif key in utils.VariableNames.ppKeys:
                ppFields[key] = arr

        return arrObj.make_hfb_matrix(phFields,ppFields,self.bz,self.bp,chemPot)

    @utils.timer
    def make_hfb_matrix(self,fields,chemPot):
        args = [[fields['p'],chemPot['p']],
                [fields['n'],chemPot['n']]]
        with mp.Pool(2) as pool:
            res = pool.starmap(self._atomic_hfb_matrix,args)
        
        hfbArrDict = {'p':res[0],'n':res[1]}
        
        return hfbArrDict
    
    # @utils.timer
    # def make_hfb_matrix(self,fields,chemPot):
    #     hfbArrDict = {'p':[],'n':[]}
    #     for pn in ['n','p']:
    #         arrObj = solvers.HFBMatrix(self.basis,self.xi,self.eta,self.wz,self.wr)

    #         phFields = {}
    #         ppFields = {}
    #         for (key,arr) in fields[pn].items():
    #             if key in utils.VariableNames.phKeys:
    #                 phFields[key] = arr
    #             elif key in utils.VariableNames.ppKeys:
    #                 ppFields[key] = arr

    #         hfbArrDict[pn] = arrObj.make_hfb_matrix(phFields,ppFields,self.bz,self.bp,chemPot[pn])
    #     return hfbArrDict

    @utils.timer
    def diagonalize_hfb_matrix(self,hfbArrDict):
        Udict = {}
        Vdict = {}
        eqpDict = {}
    
        for pn in ['n','p']:
            Udict[pn], Vdict[pn], eqpDict[pn] = solvers.diagonalize_hfb_matrix(hfbArrDict[pn])
        
        return Udict, Vdict, eqpDict
    
    @utils.timer
    def pair_reg(self,V,eqp,chemPot):
        activeStates = {pn:self.pairRegObjs[pn].get_active_states(V[pn],eqp[pn],chemPot[pn])
                           for pn in ['n','p']}
        newChemPot = {pn:self.pairRegObjs[pn].adjust_fermi_energy(activeStates[pn],
                                                                  self.nParticles[pn],chemPot[pn])
                                                                  for pn in ['n','p']}
            
        return activeStates, newChemPot

    @utils.timer
    def reconstruct_densities(self,activeStates,U,V,returnReconObj=False):
        densities = {pn:{} for pn in ['n','p']}

        if returnReconObj:
            objs = {}

        for pn in ['n','p']:
            reconstruction = solvers.Reconstruction(self.basis,
                                                    self.rGridRepeated)
            for key in self.densitiesToConstruct:
                densities[pn][key] = getattr(reconstruction,key)(activeStates[pn],U[pn],V[pn])
                
            if returnReconObj:
                objs[pn] = reconstruction

        if returnReconObj:
            return densities, objs
        else:
            return densities

    @utils.timer
    def reconstruct_fields(self,densities):
        newFields = {pn:{key:np.zeros(utils.GlobalVariables.meshShape) for key in self.densitiesToConstruct} 
                     for pn in ['n','p']}
        
        for var in self.densitiesToConstruct:
            for (key,obj) in self.edf.terms.items():
                integArgs = [densities[tup[0]][tup[1]] for tup in self.edf.integArgs[key]]
                pField, nField = getattr(obj,var)(integArgs,self.bz,self.bp)
                
                newFields['p'][var] += pField
                newFields['n'][var] += nField
        
        return newFields
    
    @utils.timer
    def single_iter(self,fields,chemPot):
        hfbArrDict = self.make_hfb_matrix(fields,chemPot)
        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)
        densities = self.reconstruct_densities(activeStates,Udict,Vdict)
        fields = self.reconstruct_fields(densities)
        
        return fields, newChemPot, densities
    
class LipkinNogami(Main):
    def __init__(self, edf, basis, nParticles, r, bz, bp, xi, eta, wz, wr, densitiesToConstruct=utils.VariableNames.defaultReconDensities):
        super().__init__(edf, basis, nParticles, r, bz, bp, xi, eta, wz, wr, densitiesToConstruct)

    @utils.timer
    def add_LN_to_hfb_matrix(self,hfbArrDict,rhoQP,lmd2):
        if rhoQP is None:
            for pn in ['n','p']:
                for k in range(self.basis.nBlocks):
                    nStates = len(self.basis.quantNumbersByBlock[k])
                    # toAdd = 2*lmd2[pn]*(np.identity(nStates)-0.1*np.ones((nStates,nStates)))
                    toAdd = 2*lmd2[pn]*np.identity(nStates)
                    hfbArrDict[pn][k][:nStates,:nStates] -= toAdd
                    hfbArrDict[pn][k][nStates:,nStates:] += toAdd
        else:
            for pn in ['n','p']:
                for k in range(self.basis.nBlocks):
                    nStates = len(self.basis.quantNumbersByBlock[k])
                    toAdd = 2*lmd2[pn]*(np.identity(nStates)-rhoQP[pn][k])
                    hfbArrDict[pn][k][:nStates,:nStates] -= toAdd
                    hfbArrDict[pn][k][nStates:,nStates:] += toAdd
        return hfbArrDict

    @utils.timer
    def adjust_lipkin_nogami(self,activeStates,U,V,hfbArrDict):
        newLmd2 = {}
        rhoQP = {}
        GeffDict = {}
        for pn in ['n','p']:
            lnObj = solvers.LipkinNogami(self.basis)
            
            hTilde = []
            for k in range(self.basis.nBlocks):
                nStates = len(self.basis.quantNumbersByBlock[k])
                hTilde.append(hfbArrDict[pn][k][:nStates,nStates:])

            Geff, rhoQP[pn] = lnObj.get_Geff(activeStates[pn],U[pn],V[pn],hTilde)
            GeffDict[pn] = Geff
            
            newLmd2[pn] = lnObj.get_lmd2(Geff,rhoQP[pn])
        
        return newLmd2, rhoQP
    
    @utils.timer
    def startup(self,fields,chemPot,lmd2,rhoQP):
        hfbArrDict = self.make_hfb_matrix(fields,chemPot)
        hfbArrDict = self.add_LN_to_hfb_matrix(hfbArrDict,rhoQP,lmd2)
        return hfbArrDict
    
    @utils.timer
    def single_iter(self,hfbArrDict,chemPot,lmd2,rhoQP):
        # for pn in ['n','p']:
        #     chemPot[pn] += 2*lmd2[pn]

        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        activeStates, chemPot = self.pair_reg(Vdict,eqpDict,chemPot)

        lmd2, rhoQP = self.adjust_lipkin_nogami(activeStates,Udict,Vdict,hfbArrDict)
        densities = self.reconstruct_densities(activeStates,Udict,Vdict)
        fields = self.reconstruct_fields(densities)

        hfbArrDict = self.make_hfb_matrix(fields,chemPot)
        hfbArrDict = self.add_LN_to_hfb_matrix(hfbArrDict,rhoQP,lmd2)

        self.fields = fields
        
        return hfbArrDict, chemPot, densities, lmd2, rhoQP

    # @utils.timer
    # def single_iter(self,fields,chemPot,lmd2,rhoQP):
    #     hfbArrDict = self.make_hfb_matrix(fields,chemPot)
    #     hfbArrDict = self.add_LN_to_hfb_matrix(hfbArrDict,rhoQP,lmd2)

    #     Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
    #     activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)

    #     lmd2, rhoQP = self.adjust_lipkin_nogami(activeStates,Udict,Vdict,hfbArrDict)
    #     densities = self.reconstruct_densities(activeStates,Udict,Vdict)
    #     fields = self.reconstruct_fields(densities)

    #     self.hfbArrDict = hfbArrDict
        
    #     return fields, newChemPot, densities, lmd2, rhoQP