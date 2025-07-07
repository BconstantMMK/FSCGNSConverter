import os
import sys

from FSDataManager import (
    FSClac, FSLog, FSError, FSMesh, FSMeshEnums,
    FSString, FSIntArray, FSFloatArray, FSStringArray,
    FSUnstructCellTypes, FSDMIterator, FSCellTypeSet
)

def CompareMeshes(meshRef, meshCmp, checkDatasets=False, checkCellAttributes=False):
    clac = meshRef.GetClac()

    FSLog(clac, 0, "Comparing meshes")

    celltypesRef =  meshRef.GetCellTypeArray()
    celltypesCmp =  meshCmp.GetCellTypeArray()

    if celltypesRef.Size() != celltypesCmp.Size():
        FSError("Comparing meshes: Number of celltype differ")
        return False

    for ct in FSDMIterator(celltypesRef):
        if not meshCmp.HasCellType(ct):
            FSError("Comparing meshes: celltypes differ")
            return False

        if meshRef.GetNGlobalCells(ct) != meshCmp.GetNGlobalCells(ct):
            FSError("Comparing meshes: Number of global cells in a cell pool differ")
            return False
        if meshRef.GetNOwnedCells(ct) != meshCmp.GetNOwnedCells(ct):
            FSError("Comparing meshes: Number of owned cells in a cell pool differ")
            return False

    if checkDatasets:
        datasetNamesRef = meshRef.GetUnstructDatasetNames()
        datasetNamesCmp = meshCmp.GetUnstructDatasetNames()

        if datasetNamesRef.Size() != datasetNamesCmp.Size():
            FSError("Comparing meshes: Number of datasets differ")
            return False

        for name in FSDMIterator(datasetNamesRef):
            if not meshCmp.HasUnstructDataset(name):
                FSError("Comparing meshes: Names of datasets differ")
                return False

            if meshRef.GetUnstructDataset(name).GetDatasetInfo() != meshCmp.GetUnstructDataset(name).GetDatasetInfo():
                FSError("Comparing meshes: Dataset info differs")
                return False

            if meshRef.GetUnstructDataset(name).GetValues().Size(0) != meshCmp.GetUnstructDataset(name).GetValues().Size(0):
                FSError("Comparing meshes: Dataset values size0 differs")
                return False
            if meshRef.GetUnstructDataset(name).GetValues().Size(1) != meshCmp.GetUnstructDataset(name).GetValues().Size(1):
                FSError("Comparing meshes: Dataset values size1 differs")

    if checkCellAttributes:
        celltypesRef =  meshRef.GetCellTypeArray()
        for ct in FSDMIterator(celltypesRef):
            attribNamesRef = meshRef.GetCellAttributes(FSMeshEnums.Int2CellType(ct))
            attribNamesCmp = meshCmp.GetCellAttributes(FSMeshEnums.Int2CellType(ct))

            if attribNamesRef.Size() != attribNamesCmp.Size():
                FSError("Comparing meshes: Number of cell attributes in a cell pool differ")
                return False

            for name in FSDMIterator(attribNamesRef):
                if not meshCmp.HasCellAttribute(ctname):
                    FSError("Comparing meshes: Names of cell attributes in a cell pool differ")
                    return False

    FSLog(clac, 0, "Successful")
    return True
