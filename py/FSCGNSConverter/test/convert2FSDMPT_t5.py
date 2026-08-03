# - convert2FSDMPT (pyTree) -
import numpy as np
import Converter.PyTree as C
import Converter.Internal as Internal
import Generator.PyTree as G
from KCore.test import getLocal

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSPlugins.test import testH5

LOCAL = getLocal()

# Testing datasets - BE
N = 5

def fFunc(x, y, z): return 3.*np.cos(x) + 2.*np.sin(y) - np.tan(z)
def gFunc(x, y, z): return np.sqrt(1.4*x*y/(1. + z))
def hFunc(x, y, z): return 0.2*x, 0.4*y, -0.7*z

def _addFlowSolutions(t):
    C._initVars(
        t, 'centers:F', fFunc,
        ['centers:CoordinateX', 'centers:CoordinateY', 'centers:CoordinateZ'],
        isVectorized=True
    )
    C._initVars(
        t, 'centers:G', gFunc,
        ['centers:CoordinateX', 'centers:CoordinateY', 'centers:CoordinateZ'],
        isVectorized=True
    )
    C._initVars(
        t, 'F', fFunc,
        ['CoordinateX', 'CoordinateY', 'CoordinateZ'],
        isVectorized=True
    )
    C._initVars(
        t, ['VelocityX', 'VelocityY', 'VelocityZ'], hFunc,
        ['CoordinateX', 'CoordinateY', 'CoordinateZ'],
        isVectorized=True
    )
    return None
    
def _addBCsBySubzone(t, nxblocks=1, is3D=True, quadFaces=True):
    Nz = N if is3D else 2
    Lz = 1. if is3D else 0.1
    cartFunc = G.cartHexa if quadFaces else G.cartTetra
    subzone = cartFunc((0., 0., 0.), (0., 1., Lz), (1, N, Nz))
    C._addBC2Zone(t, 'inlet', 'BCInflow', subzone=subzone)
    subzone = cartFunc((0., 0., 0.), (1., 1., 0.), (nxblocks*(N - 1) + 1, N, 1))
    C._addBC2Zone(t, 'wall', 'BCWallInviscid', subzone=subzone)
    subzone = cartFunc((0., 0., Nz-1.), (1., 1., 0.), (nxblocks*(N - 1) + 1, N, 1))
    C._addBC2Zone(t, 'farfield', 'BCFarfield', subzone=subzone)
    subzone = cartFunc(((N-1.)*nxblocks, 0., 0.), (0., 1., Lz), (1, N, Nz))
    C._addBC2Zone(t, 'outlet', 'BCOutflow', subzone=subzone)
    return None
    

# --- Without BCs --- #
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
_addFlowSolutions(t)

# - no datasets, option 1
convObj = FSCGNSConverter(pyTree=t, datasets=None)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 1)

# - no datasets, option 2
convObj = FSCGNSConverter(pyTree=t, datasets=[])
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 2)

# - dataset selection by name
convObj = FSCGNSConverter(pyTree=t, datasets=['nodes:VelocityX'])
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 3)

convObj = FSCGNSConverter(pyTree=t, datasets=['F'])
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 4)

convObj = FSCGNSConverter(pyTree=t, datasets=['VelocityX', 'VelocityZ', 'centers:G'])
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 5)

convObj = FSCGNSConverter(pyTree=t, datasets=Internal.__FlowSolutionNodes__)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 6)

convObj = FSCGNSConverter(pyTree=t, datasets=Internal.__FlowSolutionCenters__)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 7)

# - all datasets (default)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 8)


# --- With BCs --- #
t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
_addBCsBySubzone(t, quadFaces=False)
_addFlowSolutions(t)
exit()
# add BCDatasets
C.convertFile2PyTree(t, 'out.cgns')

# - no datasets
convObj = FSCGNSConverter(pyTree=t, datasets=[])
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 11)

# - dataset selection by name
convObj = FSCGNSConverter(pyTree=t, datasets=Internal.__FlowSolutionCenters__)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 17)

# - all datasets (default)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 18)





