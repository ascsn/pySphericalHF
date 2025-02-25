import os, sys
#Taken from https://stackoverflow.com/a/49375740
dirName = os.path.dirname(os.path.realpath(__file__))
if dirName not in sys.path:
    sys.path.append(dirName)

import data_processing
import misc_physics
import mixing
import reduced_hfb
import runfiles
import solvers
import utils