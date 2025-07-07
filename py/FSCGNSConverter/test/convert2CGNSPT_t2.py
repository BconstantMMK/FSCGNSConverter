# - convert2CGNSPT (pyTree) -
import os
import Converter.PyTree as C
import Generator.PyTree as G
import KCore.test as test

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

LOCAL = test.getLocal()

# Simple box, Elements, from filename
N = 5
meshName = os.path.join(LOCAL, "out.h5")

# --- 2D --- #
# - BE -
# Tri BE
t = G.cartTetra((0., 0., 0.), (1., 1., 0.1), (N, N, 2))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 1)

# Quad BE
t = G.cartHexa((0., 0., 0.), (1., 1., 0.1), (N, N, 2))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 2)

# - ME -
a = G.cartTetra((0., 0., 0.), (1., 1., 0.1), (N, N, 2))
b = G.cartHexa((N-1., 0., 0.), (1., 1., 0.1), (N, N, 2))
t = C.mergeConnectivity(a, b, boundary=0)
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 3)

# --- 3D --- #
# - BE -
# Tetra BE
t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 4)

# Penta BE
t = G.cartPenta((0., 0., 0.), (1., 1., 1.), (N, N, N))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 5)

# Pyra BE
t = G.cartPyra((0., 0., 0.), (1., 1., 1.), (N, N, N))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 6)

# Hexa BE
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 7)

# - ME -
# Hexas
a = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartHexa((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity(a, b, boundary=0)
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 8)

# All BE
a = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartPenta((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
c = G.cartPyra((2.*(N-1.), 0., 0.), (1., 1., 1.), (N, N, N))
d = G.cartHexa((3.*(N-1.), 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity([a, b, c, d], boundary=0)
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName)
convObj.convert()
test.testT(convObj.pyTree, 9)

os.remove(meshName)
