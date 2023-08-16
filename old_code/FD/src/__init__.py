import os, sys
#Taken from https://stackoverflow.com/a/49375740
dirName = os.path.dirname(os.path.realpath(__file__))
if dirName not in sys.path:
    sys.path.append(dirName)

from densities import *
from fields import *
from functionals import *
from io import *
from solvers import *
from utilities import *
from wf import *
