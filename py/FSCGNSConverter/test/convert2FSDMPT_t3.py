# - convert2FSDMPT (pyTree) -
import Converter.PyTree as C
import Generator.PyTree as G

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSPlugins.test import testH5


# Simple box, Elements, with BCs
N = 5

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

def _addBCsByVertexPL(t, is3D=True):
    Nz = N if is3D else 1
    C._addBC2Zone(t, 'inlet', 'BCInflow', pointList=[(j*N**2 + i*N + 1) for j in range(Nz) for i in range(N)])
    C._addBC2Zone(t, 'wall', 'BCWallInviscid', pointList=[(j*N**2 + i + 1) for j in range(Nz) for i in range(N)])
    C._addBC2Zone(t, 'farfield', 'BCFarfield', pointList=[(j*N**2 + (N-1)*N + i + 1) for j in range(Nz) for i in range(N)])
    C._addBC2Zone(t, 'outlet', 'BCOutflow', pointList=[(j*N**2 + (N-1) + i*N + 1) for j in range(Nz) for i in range(N)])
    print([(j*N**2 + i*N + 1) for j in range(Nz) for i in range(N)])
    print([(j*N**2 + i + 1) for j in range(Nz) for i in range(N)])
    print([(j*N**2 + (N-1)*N + i + 1) for j in range(Nz) for i in range(N)])
    print([(j*N**2 + (N-1) + i*N + 1) for j in range(Nz) for i in range(N)])
    return None


# --- Without bcDict defined (ignoring BCs during conversion) --- #

# Tetra BE, by subzone
t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
_addBCsBySubzone(t, quadFaces=False)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 1)

# Hexa BE, by subzone
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
_addBCsBySubzone(t, quadFaces=True)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 2)

# ME, by subzone
a = G.cartPyra((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartHexa((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity(a, b, boundary=0)
_addBCsBySubzone(t, nxblocks=2, quadFaces=True)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 3)

# # Tetra BE, by vertex PL
# t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
# _addBCsByVertexPL(t)
# convObj = FSCGNSConverter(pyTree=t)
# convObj.convert()
# testH5(convObj.clac, convObj.fsmesh, 4)

# # Hexa BE, by vertex PL
# t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
# _addBCsByVertexPL(t)
# convObj = FSCGNSConverter(pyTree=t)
# convObj.convert()
# testH5(convObj.clac, convObj.fsmesh, 5)

# # ME, by vertex PL
# a = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
# b = G.cartHexa((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
# t = C.mergeConnectivity(a, b, boundary=0)
# C._addBC2Zone(t, 'inlet', 'BCInflow', pointList=[(j*N**2 + i*N + 1) for j in range(N) for i in range(N)])
# C._addBC2Zone(t, 'outlet', 'BCOutflow', pointList=[(N*N**2 + j*(N-1)*N + (N-2) + i*(N-1) + 1) for j in range(N) for i in range(N)])
# convObj = FSCGNSConverter(pyTree=t)
# convObj.convert()
# testH5(convObj.clac, convObj.fsmesh, 6)


# --- With bcDict defined --- #
bcDict = {
    1: "BCInflow",
    2: "BCWallInviscid",
    3: "BCFarfield",
    4: "BCOutflow"
}

# Tetra BE, by subzone
t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
_addBCsBySubzone(t, quadFaces=False)
convObj = FSCGNSConverter(pyTree=t, bcDict=bcDict)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 11)

# Hexa BE, by subzone
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
_addBCsBySubzone(t, quadFaces=True)
convObj = FSCGNSConverter(pyTree=t, bcDict=bcDict)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 12)

# ME, by subzone
a = G.cartPyra((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartHexa((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity(a, b, boundary=0)
_addBCsBySubzone(t, nxblocks=2, quadFaces=True)
convObj = FSCGNSConverter(pyTree=t, bcDict=bcDict)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 13)

# # Hexa BE, by vertex PL
# t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
# _addBCsByVertexPL(t)
# convObj = FSCGNSConverter(pyTree=t, bcDict=bcDict)
# convObj.convert()
# testH5(convObj.clac, convObj.fsmesh, 13)

# # ME
# bcDict = {
#     1: "BCInflow",
#     4: "BCOutflow"
# }

# a = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
# b = G.cartHexa((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
# t = C.mergeConnectivity(a, b, boundary=0)
# C._addBC2Zone(t, 'inlet', 'BCInflow', pointList=[(j*N**2 + i*N + 1) for j in range(N) for i in range(N)])
# C._addBC2Zone(t, 'outlet', 'BCOutflow', pointList=[(N*N**2 + j*(N-1)*N + (N-2) + i*(N-1) + 1) for j in range(N) for i in range(N)])
# print([(j*N**2 + i*N + 1) for j in range(N) for i in range(N)])
# print([(N*N**2 + j*(N-1)*N + (N-2) + i*(N-1) + 1) for j in range(N) for i in range(N)])
# convObj = FSCGNSConverter(pyTree=t, bcDict=bcDict)
# convObj.convert()
# testH5(convObj.clac, convObj.fsmesh, 14)
