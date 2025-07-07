# - convert2CGNSPT (pyTree) -
import os
import Converter.PyTree as C
import Generator.PyTree as G
import KCore.test as test

from FSDataManager import FSClac, FSMesh
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

LOCAL = test.getLocal()

# Simple box, Elements, from FSClac and FSMesh
N = 5
meshName = os.path.join(LOCAL, "out.h5")

# Hexa BE
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
C.convertPyTree2File(t, meshName)
clac = FSClac()
fsmesh = FSMesh(clac)
fsmesh.ImportMeshHDF5(Filename=meshName)
convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh)
convObj.convert()
test.testT(convObj.pyTree, 1)

os.remove(meshName)
