# Simple script to generate cartesian background meshes.
# The mesh features an inner region of specified size with specified cell spacing.
# From that inner region on the cells grow by a specified factor up to the specified outer extent.

import os, sys, gc

# import the stuff we need from FSDM
import FSDM
from FSDataManager import FSClac, FSMesh, FSMeshEnums, FSDataName, FSString, FS_AT_CADGroupID, FSDataValue, FSIntArray, FSFloatArray, FSStringArray, FSDataSpecArray, FSDatasetInfo, FSTimer, FSEnums

def generateCoordComponentArrayPerAxis(origin, spacingInner, innerMin, innerMax, growthRate, outerMin, outerMax) :
    coord = FSFloatArray()

    if outerMin < outerMax:
        coord.Append(origin)

    currentCoord = origin
    currentFactor = 1.0

    while currentCoord > innerMin:
        currentCoord -= spacingInner
        coord.Append(currentCoord)

    while currentCoord > outerMin:
        currentFactor *= growthRate
        currentCoord -= spacingInner * currentFactor
        coord.Append(currentCoord)

    currentCoord = origin
    currentFactor = 1.0

    while currentCoord < innerMax:
        currentCoord += spacingInner
        coord.Append(currentCoord)

    while currentCoord < outerMax:
        currentFactor *= growthRate
        currentCoord += spacingInner * currentFactor
        coord.Append(currentCoord)

    coord.Sort()
    return coord

def generateCoords(coordX, coordY, coordZ):
    nPointsX = coordX.Size()
    nPointsY = coordY.Size()
    nPointsZ = coordZ.Size()
    nPoints = nPointsX * nPointsY * nPointsZ

    print(nPointsX)
    print(nPointsY)
    print(nPointsZ)
    print(nPoints)

    coords = FSFloatArray(nPoints, 3)
    print(coords.Size())

    for k in range(nPointsZ):
        for j in range(nPointsY):
            for i in range(nPointsX):
                idx = k * nPointsY * nPointsX + j * nPointsX + i
                coords[idx * 3 + 0] = coordX[i]
                coords[idx * 3 + 1] = coordY[j]
                coords[idx * 3 + 2] = coordZ[k]
    return coords


def generateHexas(nPointsX, nPointsY, nPointsZ):
    nCellsX = nPointsX - 1
    nCellsY = nPointsY - 1
    nCellsZ = nPointsZ - 1
    nCells = nCellsX * nCellsY * nCellsZ
    hexa2Node = FSIntArray(nCells, 8)

    offset = 0
    for k in range(nCellsZ):
        for j in range(nCellsY):
            for i in range(nCellsX):
                hexa2Node[offset + 0] =  k      * nPointsX * nPointsY +  j      * nPointsX +  i
                hexa2Node[offset + 1] =  k      * nPointsX * nPointsY +  j      * nPointsX + (i + 1)
                hexa2Node[offset + 2] =  k      * nPointsX * nPointsY + (j + 1) * nPointsX + (i + 1)
                hexa2Node[offset + 3] =  k      * nPointsX * nPointsY + (j + 1) * nPointsX +  i
                hexa2Node[offset + 4] = (k + 1) * nPointsX * nPointsY +  j      * nPointsX +  i
                hexa2Node[offset + 5] = (k + 1) * nPointsX * nPointsY +  j      * nPointsX + (i + 1)
                hexa2Node[offset + 6] = (k + 1) * nPointsX * nPointsY + (j + 1) * nPointsX + (i + 1)
                hexa2Node[offset + 7] = (k + 1) * nPointsX * nPointsY + (j + 1) * nPointsX +  i
                offset += 8

    return hexa2Node


def generateFaces(nPointsX, nPointsY, nPointsZ, markerList):
    nCellsX = nPointsX - 1
    nCellsY = nPointsY - 1
    nCellsZ = nPointsZ - 1
    nQuads = 2 * nCellsX * nCellsY + 2 * nCellsX * nCellsZ + 2 * nCellsY * nCellsZ
    quad2Node = FSIntArray(nQuads, 4)
    boundaryMarker = FSIntArray(nQuads)

    idx = 0
    offset = 0
    for k in range(nPointsZ - 1):
        for j in range(nPointsY - 1):
            quad2Node[offset + 0] =  k      * nPointsX * nPointsY +  j      * nPointsX + 0
            quad2Node[offset + 1] = (k + 1) * nPointsX * nPointsY +  j      * nPointsX + 0
            quad2Node[offset + 2] = (k + 1) * nPointsX * nPointsY + (j + 1) * nPointsX + 0
            quad2Node[offset + 3] =  k      * nPointsX * nPointsY + (j + 1) * nPointsX + 0
            boundaryMarker[idx] = markerList[0]
            offset += 4
            idx += 1

    for k in range(nPointsZ - 1):
        for j in range(nPointsY - 1):
            quad2Node[offset + 0] =  k      * nPointsX * nPointsY +  j      * nPointsX + (nPointsX - 1)
            quad2Node[offset + 1] =  k      * nPointsX * nPointsY + (j + 1) * nPointsX + (nPointsX - 1)
            quad2Node[offset + 2] = (k + 1) * nPointsX * nPointsY + (j + 1) * nPointsX + (nPointsX - 1)
            quad2Node[offset + 3] = (k + 1) * nPointsX * nPointsY +  j      * nPointsX + (nPointsX - 1)
            boundaryMarker[idx] = markerList[1]
            offset += 4
            idx += 1

    for i in range(nPointsX - 1):
        for k in range(nPointsZ - 1):
            quad2Node[offset + 0] =  k      * nPointsX * nPointsY + 0 * nPointsX +  i
            quad2Node[offset + 1] =  k      * nPointsX * nPointsY + 0 * nPointsX + (i + 1)
            quad2Node[offset + 2] = (k + 1) * nPointsX * nPointsY + 0 * nPointsX + (i + 1)
            quad2Node[offset + 3] = (k + 1) * nPointsX * nPointsY + 0 * nPointsX +  i
            boundaryMarker[idx] = markerList[2]
            offset += 4
            idx += 1

    for i in range(nPointsX - 1):
        for k in range(nPointsZ - 1):
            quad2Node[offset + 0] =  k      * nPointsX * nPointsY + (nPointsY - 1) * nPointsX +  i
            quad2Node[offset + 1] = (k + 1) * nPointsX * nPointsY + (nPointsY - 1) * nPointsX +  i
            quad2Node[offset + 2] = (k + 1) * nPointsX * nPointsY + (nPointsY - 1) * nPointsX + (i + 1)
            quad2Node[offset + 3] =  k      * nPointsX * nPointsY + (nPointsY - 1) * nPointsX + (i + 1)
            boundaryMarker[idx] = markerList[3]
            offset += 4
            idx += 1

    for j in range(nPointsY - 1):
        for i in range(nPointsX - 1):
            quad2Node[offset + 0] = 0 * nPointsX * nPointsY +  j      * nPointsX +  i
            quad2Node[offset + 1] = 0 * nPointsX * nPointsY + (j + 1) * nPointsX +  i
            quad2Node[offset + 2] = 0 * nPointsX * nPointsY + (j + 1) * nPointsX + (i + 1)
            quad2Node[offset + 3] = 0 * nPointsX * nPointsY +  j      * nPointsX + (i + 1)
            boundaryMarker[idx] = markerList[4]
            offset += 4
            idx += 1

    for j in range(nPointsY - 1):
        for i in range(nPointsX - 1):
            quad2Node[offset + 0] = (nPointsZ - 1) * nPointsX * nPointsY +  j      * nPointsX +  i
            quad2Node[offset + 1] = (nPointsZ - 1) * nPointsX * nPointsY +  j      * nPointsX + (i + 1)
            quad2Node[offset + 2] = (nPointsZ - 1) * nPointsX * nPointsY + (j + 1) * nPointsX + (i + 1)
            quad2Node[offset + 3] = (nPointsZ - 1) * nPointsX * nPointsY + (j + 1) * nPointsX +  i
            boundaryMarker[idx] = markerList[5]
            offset += 4
            idx += 1

    return quad2Node, boundaryMarker



def generateGrid(clac, spacingInnerX, spacingInnerY, spacingInnerZ, growthRateX, growthRateY, growthRateZ,
    xInnerMin, xInnerMax, yInnerMin, yInnerMax, zInnerMin, zInnerMax,
    xOuterMin, xOuterMax, yOuterMin, yOuterMax, zOuterMin, zOuterMax,
    markerList, bcList) :

    timer = FSTimer(clac)
    timer.Start()
    fsmesh = FSMesh(clac)
    procID = clac.GetProcID()

    coords = FSFloatArray(0, 3)
    quad2Node = FSIntArray(0, 4)
    hexa2Node = FSIntArray(0, 8)
    marker = FSIntArray(0)
    print("1")
    if procID == 0:
        coordX = generateCoordComponentArrayPerAxis(0.0, spacingInnerX, xInnerMin, xInnerMax, growthRateX, xOuterMin, xOuterMax)
        coordY = generateCoordComponentArrayPerAxis(0.0, spacingInnerY, yInnerMin, yInnerMax, growthRateY, yOuterMin, yOuterMax)
        coordZ = generateCoordComponentArrayPerAxis(0.0, spacingInnerZ, zInnerMin, zInnerMax, growthRateZ, zOuterMin, zOuterMax)
        print("2")
        coords = generateCoords(coordX, coordY, coordZ)
        print("3")
        nPointsX = coordX.Size()
        nPointsY = coordY.Size()
        nPointsZ = coordZ.Size()
        hexa2Node = generateHexas(nPointsX, nPointsY, nPointsZ)
        print("4")
        quad2Node, marker = generateFaces(nPointsX, nPointsY, nPointsZ, markerList)
        print("5")
    fsmesh.BeginInitialization()
    print("6")
    # nodes and coords
    fsmesh.InitUnstructNodes(coords.Size(0))
    # quads
    print("7")
    fsmesh.InitUnstructCells(FSMeshEnums.CT_Quad4, quad2Node)
    # hexa
    print("8")
    fsmesh.InitUnstructCells(FSMeshEnums.CT_Hexa8, hexa2Node)

    fsmesh.EndInitialization()

    coordNames = FSStringArray(3)
    coordNames[0] = FSDataName.Coordinate().X()
    coordNames[1] = FSDataName.Coordinate().Y()
    coordNames[2] = FSDataName.Coordinate().Z()
    coordSpecs = FSDataSpecArray(3)
    coordSpecs[0].Length()
    coordSpecs[1].Length()
    coordSpecs[2].Length()
    print("9")
    fsmesh.InitUnstructDataset(FSDataName.Coordinates(), FSDatasetInfo(coordNames, coordSpecs, FSMeshEnums.CT_Node), coords)
    fsmesh.InitCellAttribute(FS_AT_CADGroupID, FSMeshEnums.CT_Quad4, marker)

    print("10")
    if len(markerList) == len(bcList):
        for i in range(len(markerList)):
            fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, markerList[i], bcList[i])

    timer.Stop()
    timer.Print(0, "    Generated grid in ")

    fsmesh.PrintInfo()
    return fsmesh


clac = FSClac()

#clac, spacingInnerX, spacingInnerY, spacingInnerZ, growthRateX, growthRateY, growthRateZ,
#    xInnerMin, xInnerMax, yInnerMin, yInnerMax, zInnerMin, zInnerMax,
#    xOuterMin, xOuterMax, yOuterMin, yOuterMax, zOuterMin, zOuterMax,
#    markerList, bcList

fsmesh = generateGrid(clac, 0.02, 0.01, 0.02, 1.2, 1.2, 1.2,
                       -0.5, 2.0, 0.0, 0.01, -0.5, 0.5,
                       -15.0, 16.0, 0.0, 0.01, -15.0, 15.0,
                       [21, 22, 23, 24, 25, 26], ['BCFarfield', 'BCFarfield', 'BCSymmetryPlane', 'BCSymmetryPlane', 'BCFarfield', 'BCFarfield'])

fsmesh.ExportMeshHDF5(Filename="naca_background.h5")
fsmesh.ExportMeshTECPLOT(Filename="naca_background.plt")

