# - convert2FSDMPT (pyTree) -
import Converter.PyTree as C
import Generator.PyTree as G

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSPlugins.test import testH5

# Testing datasets - ME
a = G.cartHexa((0.,0.,0.), (0.1,0.1,0.1), (10, 10, 2))
b = G.cartPenta((0.9,0.,0.), (0.1,0.1,0.1), (10, 10, 2))
c = G.cartPyra((1.8,0.,0.), (0.1,0.1,0.1), (10, 10, 2))
d = G.cartTetra((2.7,0.,0.), (0.1,0.1,0.1), (10, 10, 2))
t = C.mergeConnectivity([a, b, c, d], None)

C._initVars(t, '{centers:f}={centers:CoordinateX}')
C._initVars(t, '{centers:g}={centers:CoordinateY}')

convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 1)

