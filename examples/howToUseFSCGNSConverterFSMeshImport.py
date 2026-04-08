import os
import sys
from FSDataManager import FSClac
from FSDataManager import FSLog
from FSDataManager import FSError
from FSDataManager import FSMesh
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter


# Boundary marker info are required
bcDict = {
    1: "Marker1",
    2: "Marker2",
    3: "Marker3",
    4: "Marker4",
    5: "Marker5"
}


def main():
    clac = FSClac()
    if clac.GetNProcs() > 1:
        FSLog(clac, 0, "Must be run with only 1 MPI process!")
        sys.exit(os.EX_USAGE)

    fsmeshOrig = FSMesh(clac)
    if not fsmeshOrig.ImportMeshTAU(Filename="./tau.grid"):
        FSError.PrintAndExit()
    if not fsmeshOrig.RepartitionMeshRCB():
        FSError.PrintAndExit()
    if not fsmeshOrig.ExportMeshTAU(Filename="mesh.grid"):
        FSError.PrintAndExit()

    FSLog(clac, 0, "------ Conversion FSDM -> CGNS -------")
    convObj = FSCGNSConverter(meshName="mesh.grid", bcDict=bcDict)
    convObj.convert()
    convObj.export(filename="mesh.cgns")
    FSLog(clac, 0, "------ Conversion CGNS -> FSDM -------")
    convObj = FSCGNSConverter(meshName="mesh.cgns")
    convObj.convert()
    convObj.export(filename="mesh.h5")
    FSLog(clac, 0, "------ Conversions done -------")

    fsmeshConv = FSMesh(clac)
    if not fsmeshConv.ImportMeshHDF5(Filename="mesh.h5"):
        FSError.PrintAndExit()


if __name__ == "__main__":
   main()

