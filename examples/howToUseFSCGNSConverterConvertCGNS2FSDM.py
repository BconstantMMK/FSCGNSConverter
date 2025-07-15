import os
import sys

import Converter.PyTree as C
import Generator.PyTree as G

from FSDataManager import FSClac, FSLog, FSError, FSMesh

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

clac = FSClac()

if clac.GetNProcs() > 1:
    FSLog(clac, 0, "Must be run with only 1 MPI process!")
    sys.exit(os.EX_USAGE)

# origin
xo = 0.0
yo = 0.0
zo = 0.0

# distance between nodes
hi = 1.0
hj = 1.0
hk = 1.0

# number of nodes in each direction (not cells)
ni = 5
nj = 6
nk = 2

t = G.cart((xo, yo, zo), (hi, hj, hk), (ni, nj, nk))

t = C.addBC2Zone(t, 'wall1', 'BCWall', 'imin')
t = C.addBC2Zone(t, 'wall2', 'BCWall', 'imax')
t = C.addBC2Zone(t, 'wall3', 'BCWall', 'jmin')
t = C.addBC2Zone(t, 'wall4', 'BCWall', 'jmax')
t = C.addBC2Zone(t, 'wall5', 'BCWall', 'kmin')
t = C.addBC2Zone(t, 'wall6', 'BCWall', 'kmax')

# initialize arbitrary data that can be checked in future
def F(x, y):
    return x * y
C._initVars(t, 'DataValues', F, ['CoordinateX','CoordinateY'])
C._initVars(t, 'centers:DataValues', F, ['centers:CoordinateX','centers:CoordinateY'])

C.convertPyTree2File(t, 'mesh.cgns')

FSLog(clac, 0, "------ Conversion CGNS -> FSDM -------")

convObj = FSCGNSConverter(meshName="mesh.cgns")
convObj.convert()
convObj.export(filename="mesh.h5")

FSLog(clac, 0, "------ Conversion done -------")

fsmeshConv = FSMesh(clac)
fsmeshConv.ImportMeshHDF5(Filename="mesh.h5") or FSError.PrintAndExit()
fsmeshConv.PrintInfo()
fsmeshConv.ExportMeshTECPLOT(Filename="mesh.dat") or FSError.PrintAndExit()
