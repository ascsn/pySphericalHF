import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import wf
import densities
import fields 
from init import *


def test_rho(rArr):
    rho = np.zeros(len(grid))
    for i,r in enumerate(rArr):
        if r <= 2:
            rho[i] = 1.0
        else: 
            rho[i] = 0
    return rho


psi_array,energies = wf.initWfs(name='HO')


rho = densities.rho(psi_array)

#V_yuk = fields.yukArr(rho[0])
#V_yuk = fields.yukArr2(rho[0])
#V_yuk = fields.yukArr3(rho[0])
V_yuk = fields.yukArr4(rho[0])
V_c = fields.coulombArr(rho[1])



plt.plot(grid,V_c,label='coulomb')
plt.legend()
plt.show()

plt.plot(grid,V_c+V_yuk,label='coulomb+yukawa')
plt.legend()
plt.show()

