import os, sys

# import the stuff we need from FSDM
import FSDM
from FSDataManager import FSClac, FSLog, FSError, FSMesh, FSMeshEnums, FSString, FSIntArray, FSFloatArray, FSStringArray
from FSDataManager import FSUnstructCellTypes

import FSMeshPyTreeConversion
from FSMeshPyTreeConversion.FSDM_CGNS import Converter_FSDM_CGNS


from utils import CompareMeshes

clac = FSClac()

if clac.GetNProcs() > 1:
    FSLog(clac, 0, "Must be run with only 1 MPI process!")
    sys.exit(os.EX_USAGE)


fsmeshOrig = FSMesh(clac)

fsmeshOrig.ImportMeshTAU(Filename="$(FLOWSIM_PATH)/FSDemoData/TAU/tau.grid") or FSError.PrintAndExit()
fsmeshOrig.RepartitionMeshRCB() or FSError.PrintAndExit()
fsmeshOrig.ExportMeshTAU(Filename="mesh.grid") or FSError.PrintAndExit()

print("------ Conversion FSDM -> CGNS -------")

fsdm2cgns = Converter_FSDM_CGNS("mesh.grid")
fsdm2cgns.convertFSDM2CGNS()

print("------ Conversion CGNS -> FSDM -------")

cgns2fsdm = Converter_FSDM_CGNS("mesh.cgns")
cgns2fsdm.convertCGNS2FSDM()

print("------ Conversions done -------")

fsmeshConv = FSMesh(clac)
fsmeshConv.ImportMeshHDF5(Filename="mesh.h5") or FSError.PrintAndExit()

fsmeshConv.PrintInfo()

if not CompareMeshes(fsmeshOrig, fsmeshConv):
    FSError.Print()
    exit(-1)
