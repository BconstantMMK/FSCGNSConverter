import os
import sys

import Converter as C

from FSDataManager import FSClac, FSLog, FSError, FSMesh

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

# This script can be run also in parallel
bcDict = {
    1: "BCWallViscous",
    2: "BCWallViscous",
    3: "BCWallViscous",
    4: "BCSymmetryPlane",
    5: "BCFarfield"
}

clac = FSClac()
fsmeshOrig = FSMesh(clac)

fsmeshOrig.ImportMeshTAU(Filename="./tau.grid") or FSError.PrintAndExit()
fsmeshOrig.RepartitionMeshRCB() or FSError.PrintAndExit()

FSLog(clac, 0, "------ Conversion FSDM -> CGNS NGon -------")

# Unlike this example, you will need to keep FlowSolutions for use in FFD
convObj = FSCGNSConverter(meshName="mesh.grid", bcDict=bcDict, datasets=[])
convObj.convert(forFFDX=True)
convObj.export(filename="mesh_ngon.cgns")

FSLog(clac, 0, "------ Conversions done -------")
