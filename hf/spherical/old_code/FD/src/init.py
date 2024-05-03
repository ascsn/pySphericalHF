import numpy as np
import sys
from inputs import *

sys.path.insert(0, '../../src/methods')
import spec
### initilization file for arrays that are used throughout the package

###############################################################################
## Define grid on [-1,1] using Chebychev GL points.
interval = (lb,rb)
CPnts,GL_weights = spec.GaussLobatto().chebyshev(N_Col)



transform = spec.coord_transforms(interval=interval)
CPnts_shift,Jacb_arc = transform.arcTransform(CPnts,beta)
CPnts_mapped,Jacb_affine = transform.inv_affine(CPnts_shift)

int_weights = spec.GaussLobatto().getDx(CPnts_shift)*Jacb_affine



nbox = len(CPnts)
