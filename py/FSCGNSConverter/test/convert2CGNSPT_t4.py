# - convert2CGNSPT (pyTree) -
import os
import Converter.PyTree as C
import Converter.Internal as Internal
import Generator.PyTree as G
import KCore.test as test

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

LOCAL = test.getLocal()

# Testing various input arguments
N = 5
meshName = os.path.join(LOCAL, "out.h5")

# - flipYZAxes
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, 2*N, N))
C.convertPyTree2File(t, meshName)
convObj = FSCGNSConverter(meshName=meshName, flipYZAxes=True)
convObj.convert()
test.testT(convObj.pyTree, 1)

# - coordsName
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
n_coords = Internal.getNodeFromType(t, "GridCoordinates_t")
n_coords[0] = "Deformed" + n_coords[0]
C.convertPyTree2File(t, meshName)  # TODO crashes because of "Deformed"
convObj = FSCGNSConverter(meshName=meshName, coordsName="DeformedCoordinates")
convObj.convert()
test.testT(convObj.pyTree, 2)

os.remove(meshName)
