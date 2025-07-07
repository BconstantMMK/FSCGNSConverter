# - convert2FSDMPT (pyTree) -
import os
import Converter.PyTree as C
import Generator.PyTree as G
from KCore.test import getLocal

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSPlugins.test import testH5

LOCAL = getLocal()

# Simple box, Elements, from filename
N = 5
meshName = os.path.join(LOCAL, "out.cgns")

# BE
t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 1)

# ME
a = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartPenta((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
c = G.cartPyra((2.*(N-1.), 0., 0.), (1., 1., 1.), (N, N, N))
d = G.cartHexa((3.*(N-1.), 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity([a, b, c, d], boundary=0)
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 2)

os.remove(meshName)
