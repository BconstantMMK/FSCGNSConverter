import Converter.PyTree as C
import Post.PyTree as P
import Geom.PyTree as D
import Transform.PyTree as T
import Generator.PyTree as G
import Converter.Internal as Internal
import sys, numpy
import FSDM
from FSDataManager import FSUnstructVolumeCellTypes, FSUnstructSurfaceCellTypes, FSMeshEnums, FS_AT_CADGroupID
import FSMeshPyTreeConversion
from FSMeshPyTreeConversion.FSDM_CGNS import Converter_FSDM_CGNS, _addBC2ZoneLoc, E_NpyInt

dz = 0.01
xmin, ymin, zmin, xmax, ymax, zmax = [-0.5,-0.5,0,1.5,0.5,dz]
print("refinement box", xmin,ymin,zmin,xmax,ymax,zmax)

mesh_name = "naca_curvi"
size = 0.01
L = 1
N = 200
N_airfoil = int(L/size*2)

tbox = G.cartHexa((xmin,ymin,zmin),(xmax-xmin,ymax-ymin,zmin),(2,2,1))
ff = P.exteriorFaces(tbox)
ff = D.uniformize(ff, N)

airfoil = D.naca(12., N=N_airfoil, sharpte=True)
airfoil = C.convertArray2Tetra(airfoil)

borders = T.join(airfoil, ff)

m2 = G.T3mesher2D(borders, triangulateOnly=0, grading=1.1, metricInterpType=0)

T._addkplane(m2)
T._contract(m2, (0,0,0), (1,0,0), (0,1,0), dz)

Internal.printTree(m2)
zones = Internal.getZones(m2)
tol = 1e-8
extFaces = P.exteriorFaces(zones[0])
extFaces = T.breakElements(extFaces)
Internal.printTree(extFaces)
bbo = G.bbox(m2)
xMin, yMin, zMin, xMax, yMax, zMax = bbo

list_bcs = ["BCFarfield","BCSymmetryPlane","BCWall"]
for zone_extFaces in Internal.getZones(extFaces):
    print("Creating boundary condition.. ")
    GE = Internal.getNodeFromName(zone_extFaces, 'GridElements')
    EC = Internal.getNodeFromName(zone_extFaces, 'ElementConnectivity')[1]
    ER = Internal.getNodeFromName(zone_extFaces, 'ElementRange')[1]
    xCoord = Internal.getNodeFromName(zone_extFaces, 'CoordinateX')[1]
    yCoord = Internal.getNodeFromName(zone_extFaces, 'CoordinateY')[1]
    zCoord = Internal.getNodeFromName(zone_extFaces, 'CoordinateZ')[1]

    nb_cell_boundary = zone_extFaces[1][0][1]
    etype = GE[1][0]
    if etype == 5: n_vertices = 3
    elif etype == 7: n_vertices = 4

    EC_reshape = EC.reshape((nb_cell_boundary,n_vertices))
    dictionary_bcs_idx = {k:[] for k in list_bcs}

    for idx in range(nb_cell_boundary):
        CODABCType = None
        if numpy.all(zCoord[EC_reshape[idx]-1]-tol<zMin) or numpy.all(zCoord[EC_reshape[idx]-1]+tol>zMax):
            CODABCType='BCSymmetryPlane'
        elif (numpy.all(xCoord[EC_reshape[idx]-1]+tol>xMax) or numpy.all(xCoord[EC_reshape[idx]-1]-tol<xMin) or numpy.all(yCoord[EC_reshape[idx]-1]+tol>yMax) or numpy.all(yCoord[EC_reshape[idx]-1]-tol<yMin)):
          CODABCType='BCFarfield'
        elif not(numpy.all(zCoord[EC_reshape[idx]-1]-tol<zMin) or numpy.all(zCoord[EC_reshape[idx]-1]+tol>zMax) or numpy.all(xCoord[EC_reshape[idx]-1]+tol>xMax) or numpy.all(xCoord[EC_reshape[idx]-1]-tol<xMin) or numpy.all(yCoord[EC_reshape[idx]-1]+tol>yMax) or numpy.all(yCoord[EC_reshape[idx]-1]-tol<yMin)):
          CODABCType="BCWall"
        dictionary_bcs_idx[CODABCType].append(idx)

    for CODABCType in list_bcs:
      print(CODABCType)
      if len(dictionary_bcs_idx[CODABCType])>0:
        zf = T.subzone(zone_extFaces,dictionary_bcs_idx[CODABCType], type='elements')
        _addBC2ZoneLoc(zones[0], CODABCType, CODABCType, zf)

C_FC = Converter_FSDM_CGNS(m2,invertPlanesYZ=True,inmemory=True)
C_FC.convertCGNS2FSDM()

C_FC = Converter_FSDM_CGNS(m2,invertPlanesYZ=True,inmemory=True)
C_FC.convertCGNS2FSDM()
volumeCellTypes = tuple(FSMeshEnums.CellTypeToString(x) for x in FSUnstructVolumeCellTypes)
C_FC.fsmesh.ExportMeshTECPLOT(Filename=mesh_name+"_vol.plt", PrefixDatasetName=True,  ExportCellTypes=volumeCellTypes) or FSError.PrintAndExit()

surfaceCellTypes = tuple(FSMeshEnums.CellTypeToString(x) for x in FSUnstructSurfaceCellTypes)
C_FC.fsmesh.ExportMeshTECPLOT(Filename=mesh_name+"_surf.plt", PrefixDatasetName=True, ZonePerCellAttributeValue=True,CellAttribute=FS_AT_CADGroupID, UseCellAttributeValueName=True,ExportCellTypes=surfaceCellTypes) or FSError.PrintAndExit()

C_FC.fsmesh.ExportMeshHDF5(Filename=mesh_name+".h5") or FSError.PrintAndExit()



