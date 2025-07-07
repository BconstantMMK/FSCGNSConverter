# - convert2FSDMPT (pyTree) -
import Converter.Internal as Internal
import Generator.PyTree as G

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSPlugins.test import testH5


# Testing various input arguments
N = 5

# - flipYZAxes
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, 2*N, N))
convObj = FSCGNSConverter(pyTree=t, flipYZAxes=True)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 1)

# - coordsName
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
n_coords = Internal.getNodeFromType(t, "GridCoordinates_t")
n_coords[0] = "Deformed" + n_coords[0]
convObj = FSCGNSConverter(pyTree=t, coordsName="DeformedCoordinates")
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 2, coordsName=convObj.coordsName)
