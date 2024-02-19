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

# provide boundary marker information, as it is required at the moment
dict_BCs = {1:"Marker1",
            2:"Marker2",
            3:"Marker3",
            4:"Marker4",
            5:"Marker5"}

FSLog(clac, 0, "------ Conversion FSDM -> CGNS -------")

fsdm2cgns = Converter_FSDM_CGNS("mesh.grid", dict_BCs=dict_BCs)
fsdm2cgns.convertFSDM2CGNS()

FSLog(clac, 0, "------ Conversion CGNS -> FSDM -------")

cgns2fsdm = Converter_FSDM_CGNS("mesh.cgns")
cgns2fsdm.convertCGNS2FSDM()

FSLog(clac, 0, "------ Conversions done -------")

fsmeshConv = FSMesh(clac)
fsmeshConv.ImportMeshHDF5(Filename="mesh.h5") or FSError.PrintAndExit()

if not CompareMeshes(fsmeshOrig, fsmeshConv, checkDatasets=True, checkCellAttributes=False):
    FSError.Print()
    fsmeshOrig.PrintInfo()
    fsmeshConv.PrintInfo()
    exit(-1)
