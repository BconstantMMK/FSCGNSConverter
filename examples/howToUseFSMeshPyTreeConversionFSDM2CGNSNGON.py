import os, sys

# import the stuff we need from FSDM
import FSDM
from FSDataManager import FSClac, FSLog, FSError, FSMesh
import Converter as C
import Converter.Mpi as Cmpi
import Converter.Internal as Internal

import FSMeshPyTreeConversion
from FSMeshPyTreeConversion.FSDM_CGNS import Converter_FSDM_CGNS

# This script can be run also in parallel

dict_BCs = {1:"BCWallViscous",
            2:"BCWallViscous",
            3:"BCWallViscous",
            4:"BCSymmetryPlane",
            5:"BCFarfield"}

clac = FSClac()
fsmeshOrig = FSMesh(clac)

fsmeshOrig.ImportMeshTAU(Filename="$(FLOWSIM_PATH)/FSDemoData/TAU/tau.grid") or FSError.PrintAndExit()
fsmeshOrig.RepartitionMeshRCB() or FSError.PrintAndExit()

FSLog(clac, 0, "------ Conversion FSDM -> CGNS -------")

 #for FFD you will need keepFlowSolution=True
fsdm2cgns = Converter_FSDM_CGNS("mesh.grid",keepFlowSolution=False,dict_BCs=dict_BCs,inmemory=True)
fsdm2cgns.convertFSDM2CGNS()

FSLog(clac, 0, "------ Conversion CGNS multielement -> CGNS Ngon -------")

# Attention when using mergeOnProc0=True because the "allGather" behind this option may require too much memory and make the conversion crash.
# mergeOnProc=False is advised, the output will be one pytree, with one zone per processor (however not yet supported by FFD).

fsdm2cgns.convertMonozoneME2Ngon4FFD(reorient=True,mergeOnProc0=False)

Cmpi.convertPyTree2File(fsdm2cgns.pytree,"mesh_ngon.cgns")

FSLog(clac, 0, "------ Conversions done -------")
