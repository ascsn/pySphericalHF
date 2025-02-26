import numpy as np
import matplotlib.pyplot as plt
import h5py

import sys, os
import pandas as pd
import time

import logging
hfbPath = '../src'
sys.path.insert(0,hfbPath)
import hfb

datDir = 'data'

#Initialization
bz, bp, nShells = hfb.data_processing.get_oscillator_config(datDir,datDir)
nParticles = {'n':154,'p':100}
chempotDict = hfb.data_processing.read_chempot(datDir)

#HO basis
eta, wr, xi, wz = hfb.solvers.ho_weights_and_nodes()

hoBasis = hfb.solvers.HarmonicOscillatorBasis(bp,bz)
hoBasis.get_quantum_numbers(nShells)

r = np.sqrt(eta)*bp
z = xi*bz

hoBasis.split_by_block(r,z)

#Train-test inds
indsObj = hfb.data_processing.TrainTestInds(datDir)

#%% Reading data
fieldsDict = hfb.data_processing.read_fields(datDir)
        
edfParamsDf = pd.read_csv(os.path.join(datDir,'edf-params.txt'),sep='\s+',skiprows=[1])
realEnegs = pd.read_csv(os.path.join(datDir,'moments.dat'),sep='\s+')
realEnegs = realEnegs['EHFB']

def main(testIdx,logName):
    solverTol = 10*thresh
        
    print("Test index:",testIdx)
    
    fields = {pn:{key:arr[testIdx] for (key, arr) in fieldsDict[pn].items()}
              for pn in ['n','p']}
    chemPot = {pn:arr[testIdx] for (pn,arr) in chempotDict.items()}

    #%% Preparing EDF
    directCoulomb = hfb.solvers.CoulombDirect_Gaussian(xi,eta,wz,wr,Iarr=Iarr)
    skyrme = hfb.solvers.SkyrmeEDF(edfParamsDf.iloc[testIdx],xi,eta,wz,wr,
                                   directCoulomb)
    
    mainCls = hfb.runfiles.Main(skyrme,hoBasis,nParticles,r,bz,bp,xi,eta,wz,wr)
    
    #%% Running
    flattenCls = hfb.mixing.VariableFlattening(fields)
    flatCoeffs = flattenCls.flatten(fields)

    chemPotHistDict = {pn:np.zeros(maxIter+1) for pn in ['n','p']}

    for pn in ['n','p']:
        chemPotHistDict[pn][0] = chemPot[pn]

    mixCls = hfb.mixing.Broyden(flatCoeffs,maxIter,0.1)
    
    paramsDict = {'edfParams':edfParamsDf.iloc[testIdx].to_dict(),
                  'solverParams':mixCls.paramsDict | {'tol':solverTol}, #New thing I just learned
                  'eiParams':{'svdThresh':thresh,'reconstructionCoordInds':'all'}}
    logger = hfb.utils.Logging(logName,paramsDict)

    densities = {pn:None for pn in ['p','n']}
    timingArr = np.zeros(maxIter)
    
    t00 = time.time()
    print('i\tdiff\t\ttime\t\tlmdn\t\tlmdp\t\tEHFB')
    for i in range(maxIter):
        t0 = time.time()
        newFields, newPot, densities = mainCls.single_iter(fields,chemPot)
        e = skyrme.hfb_energy_wrapper(densities,bz,bp,display=False)
        
        #Mixing and tracking
        flatFields = flattenCls.flatten(newFields)
        mixedFields, mixType = mixCls(flattenCls.flatten(fields),flatFields)
        
        fields = flattenCls.unflatten(mixedFields)
        
        for pn in ['n','p']:
            chemPot[pn] = newPot[pn]
            chemPotHistDict[pn][i+1] = chemPot[pn]
        
        t1 = time.time()
        timingArr[i] = t1 - t0
        
        diff = mixCls.resMax[i]
        print('%d%s\t%.3e\t%.3e\t%.6f\t%.6f\t%.6f'%
              (i,mixType,diff,t1-t0,chemPot['n'],chemPot['p'],e['net']))

        if diff <= solverTol:
            break
        
    t11 = time.time()

    print(50*'=')
    print('Total runtime: %.3e s'%(t11-t00))
    print('Niters: ',i)
    print('Runtime per iteration: %.3e s'%((t11-t00)/(i+1)))

    e = skyrme.hfb_energy_wrapper(densities,bz,bp,display=False)

    finalIter = i

    logger.write_nested_dict_as_arr('histDict',
                                    {'alpha':hfb.utils.recursive_trim_dict(mixCls.alphaArr,finalIter),
                                     'resMax':hfb.utils.recursive_trim_dict(mixCls.resMax,finalIter),
                                     'chemPot':hfb.utils.recursive_trim_dict(chemPotHistDict,finalIter),
                                     'runtime':hfb.utils.recursive_trim_dict(timingArr,finalIter),})
    logger.write_nested_dict_as_arr('densities',densities)
    
    logger.write_dict_as_attrs('hfbEneg',e)
    
    print('Final energy: ',e['net'])
    print('Real energy: ',float(realEnegs.iloc[testIdx]))
    print('Energy difference: %.3e MeV'%(e['net'] - float(realEnegs.iloc[testIdx])))
    return

def main_wrapper(args):
    idx, datFile, logFile = args

    with hfb.utils.OutputManager(logFile):
        try:
            main(idx,datFile)
        except Exception as e:
            logging.exception(e)
            pass

thresh = 10.**(-4)
maxIter = 50

dummyCoul = hfb.solvers.CoulombDirect_Gaussian(xi,eta,wz,wr)
Iarr = dummyCoul.get_Iarr(bz,bp)

outputDir = os.path.join('results')
os.makedirs(outputDir,exist_ok=True)
outputNames = [os.path.join(outputDir,str(i).zfill(6)+'.h5') for i in indsObj.valInds]

logDir = os.path.join(outputDir,'logs')
os.makedirs(logDir,exist_ok=True)
logNames = [os.path.join(logDir,str(i).zfill(6)+'.dat') for i in indsObj.valInds]

args = list(zip(indsObj.valInds,outputNames,logNames))

main_wrapper(args[0])
