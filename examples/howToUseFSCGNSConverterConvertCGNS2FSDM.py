import os
import sys
import Converter.PyTree as C
import Generator.PyTree as G
from FSDataManager import FSClac
from FSDataManager import FSLog
from FSDataManager import FSError
from FSDataManager import FSMesh
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter


def main():
    clac = FSClac()
    if clac.GetNProcs() > 1:
        FSLog(clac, 0, "Must be run with only 1 MPI process!")
        sys.exit(os.EX_USAGE)

    t = G.cart((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (5, 6, 2))
    t = C.addBC2Zone(t, 'wall1', 'BCWall', 'imin')
    t = C.addBC2Zone(t, 'wall2', 'BCWall', 'imax')
    t = C.addBC2Zone(t, 'wall3', 'BCWall', 'jmin')
    t = C.addBC2Zone(t, 'wall4', 'BCWall', 'jmax')
    t = C.addBC2Zone(t, 'wall5', 'BCWall', 'kmin')
    t = C.addBC2Zone(t, 'wall6', 'BCWall', 'kmax')

    # initialize arbitrary data that can be checked in future
    def _F(x, y):
        return x * y
    C._initVars(t, 'DataValues', _F, ['CoordinateX','CoordinateY'])
    C._initVars(
        t,
        'centers:DataValues',
        _F,
        ['centers:CoordinateX','centers:CoordinateY']
    )
    C.convertPyTree2File(t, 'mesh.cgns')

    FSLog(clac, 0, "------ Conversion CGNS -> FSDM -------")
    convObj = FSCGNSConverter(meshName="mesh.cgns")
    convObj.convert()
    convObj.export(filename="mesh.h5")
    FSLog(clac, 0, "------ Conversion done -------")

    fsmeshConv = FSMesh(clac)
    if not fsmeshConv.ImportMeshHDF5(Filename="mesh.h5"):
        FSError.PrintAndExit()
    fsmeshConv.PrintInfo()
    if not fsmeshConv.ExportMeshTECPLOT(Filename="mesh.dat"):
        FSError.PrintAndExit()


if __name__ == "__main__":
   main()

