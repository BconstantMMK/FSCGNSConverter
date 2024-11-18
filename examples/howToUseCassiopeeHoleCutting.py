#!/usr/bin/env python

#-----------------------------------------------------------------------------
#  Project: FSOversetBlanking
#
#  File:    howToUseCassiopeeHoleCutting.py
#
#  Level:   Expert
#
#  Author:  Francesca Basile
#
#  Date:    2024/10/18 (created)
#-----------------------------------------------------------------------------

import FSDM
import os, math, sys
from FSDataManager import FSClac, FSLog, FSError, FSMesh, FSMeshEnums, FSQuantityDesc, FSUtil, FSFloatArray, FSString, FSStringArray, FSIntArray
from FSDataManager import FSDataName, FSDataSpec, FSDataSpecArray, FSDatasetInfo, FSDataValue, FSUnstructVolumeCellTypes, FSCellAverageValuesComputation
from FSDataManager import FSMeshSelection, FSMeshSelectionOpReferencedCells, FSMeshSelectionOpQuantityValueGT
from FSDataManager import FSDataManager

import FSMeshPyTreeConversion
from CassiopeeHoleCutting.ControlLayerOverset import OversetBlankingMeshInterfaceCGNS, HoleMesh, BackgroundMesh

offsets = [0.3]
BCsCODA_bgmesh = [
{'treatment type': 'BCFarfield','boundary markers': [21,22,25,26]},
{'treatment type': 'BCSymmetryPlane','boundary markers': [23,24],},
]

wallBoundaryMarkers_childmesh = [3]
BCsCODA_childmesh = [
{'treatment type': 'BCWallViscousAdiabatic', 'boundary markers': wallBoundaryMarkers_childmesh},
{'treatment type': 'BCOverset','boundary markers': [2]},
{'treatment type': 'BCSymmetryPlane','boundary markers': [1]},
]

# --- FS stuff ---
globalClac = FSClac() # by dafault, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    print("Must be run with at least 2 MPI processes!")
    sys.exit(os.EX_USAGE)

nGlobalProcsHalf = nGlobalProcs / 2
globalProcID = globalClac.GetProcID()
meshID = 0 if globalProcID < nGlobalProcsHalf else 1

meshLocalClac = FSClac()
globalClac.DivideIntoGroups(meshID, meshLocalClac)

# --- meshKeys to get meshes from FSDataManager in the FSPlugins ---
meshKeyOriginal = "invalid"
meshKeyActive = "invalid"
discParaDict = {}

if meshID == 0:
    meshKeyOriginal = "back_orig" # the original background mesh
    meshKeyActive = "back_active" # the active part of the background mesh
    discParaDict['boundary treatments'] = BCsCODA_bgmesh
    wallBoundaryMarkers = []
else:
    meshKeyOriginal = "child" # the original child mesh
    meshKeyActive = "child"
    discParaDict['boundary treatments'] = BCsCODA_childmesh
    wallBoundaryMarkers = wallBoundaryMarkers_childmesh

meshFilename = "naca_background.h5" if meshID == 0 else "naca_curvi.h5"

dm = FSDataManager(globalClac)
fsmeshOriginal = dm.GetMesh(meshKeyOriginal, meshLocalClac, True)
meshOps = (("ImportMeshHDF5", {"MeshFilename" : meshFilename}),
       "RepartitionMeshRCB",
       "CreateLocalNumbering",
       ("RepartitionMeshZOLTAN", { "PreserveCellStacks" : True,
                                   "LineSectionsExtractionParameters" :
                                   { "ActiveNodesSelection" : { "CellTypes" : ("Prisms", "Hexahedra","Tetrahedra","Pyramids","Quadrilaterals","Triangles") },
                                     "StartNodesSelection"  : { "CellAttribute" : "CADGroupID",
                                                                },
                                   },
                                   "GraphExtraction"  : {"GraphType": "CellBased"},
                                    "Approach" : "CoordinateGraphMultilevel",
                                 }),
       "CreateLocalNumbering",
       )

if not fsmeshOriginal.DoOps(meshOps):
  FSError.PrintAndExit()
# --- Get mesh active (not used here but useful for CODA computations where the active part is extracted) ---
fsmeshActive = dm.GetMesh(meshKeyActive, meshLocalClac, True)

print("meshID, globalProcID, meshKeyOriginal")
print(meshID, globalProcID, meshKeyOriginal)
fsmeshOriginal.PrintInfo()

pytreeHole = HoleMesh(meshLocalClac,fsmeshOriginal,discParaDict,wallBoundaryMarkers,offsets,meshID,offsetFromBC="BCWall") #offsetFromBC="BCWall" or "BCOverset" (by default)
pytreeBG = BackgroundMesh(meshLocalClac,fsmeshOriginal,meshID)
blankingObject = OversetBlankingMeshInterfaceCGNS(meshLocalClac,fsmeshOriginal,pytreeBG)


# This could be a time step loop...
for i in range(0, 1):

    # Maybe transform computational meshes and hole definition meshes first...
    blankingObject.ComputeBlankingCassiopee(pytreeHole)

    fsmeshOriginal.ExportMeshTECPLOT(Filename="blanking_mesh_%i.plt" % (meshID), FileFormat = "ASCII", PrefixDatasetName=True) or FSError.PrintAndExit()

    # In case of CODA: extraction of active mesh parts, computation, solution transfer to original complete meshes...

