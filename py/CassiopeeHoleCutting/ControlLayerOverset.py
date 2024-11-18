# import the stuff we need from FSDM
import FSDM
from FSDataManager import FSClac, FSLog, FSError, FSMesh, FSMeshEnums, FSDataName, FSString, FS_AT_CADGroupID, FSCellTypeSet
from FSDataManager import FSTimer, FSDataLog, FSUnstructSurfaceCellTypes, FSQuantityDesc, FSDataValue, FSIntArray, FSFloatArray, FSStringArray, FSDMIterator
from FSDataManager import FSDataManager, FSUnstructVolumeCellTypes, FSDataSpecArray, FSDatasetInfo
from FSDataManager import FSDataManagerOpParams

import Converter.PyTree as C
import Connector.PyTree as X
import Converter.Internal as Internal
import Converter.Mpi as Cmpi
import Generator.PyTree as G
import Transform.PyTree as T
import Geom.PyTree as D
import Geom.Offset as Offset
import Dist2Walls.PyTree as DTW

import FSMeshPyTreeConversion
from FSMeshPyTreeConversion.FSDM_CGNS import Converter_FSDM_CGNS


import copy, numpy


class OversetBlankingMeshInterfaceCGNS:

    def __init__(self, clac, fsmesh, pytree):
        self.clac = clac
        self.fsmesh = fsmesh
        self.tBG = pytree

    def ComputeBlankingCassiopee(self, tb2, blankingType="center_in"): #blankingType can be also "cell_intersect"

      if self.tBG != None:
        C._deleteEmptyZones(self.tBG)
        for meshid in tb2.keys():
          bodies = [[tb2[meshid]]]
          self.tBG = X.blankCellsTri(self.tBG, bodies, [], blankingType=blankingType) #center_in or node_in or cell_intersect

        # Fill the dataset cellN with the values obtained with Cassiopee
        cellN_list = Internal.getNodesFromName(self.tBG,"cellN")#[1]
        cellN = numpy.ascontiguousarray(numpy.empty(0,dtype=numpy.float64))
        for el in cellN_list:
          cellN = numpy.concatenate((cellN,el[1]))
        Internal._rmNodesFromType(self.tBG,"FlowSolution_t")
        # Creation of the FSDM dataset cellN
        quantity_name = "cellN"
        if not self.fsmesh.HasUnstructDataset(quantity_name):
          fs_volume_cell_types = FSIntArray(0)
          for cell_type in FSUnstructVolumeCellTypes:
            if self.fsmesh.HasCellType(cell_type) :
              fs_volume_cell_types.Append(cell_type)
          quantityNames = FSStringArray(1)
          quantityNames[0] = quantity_name
          quantitySpecs = FSDataSpecArray(1)
          self.fsmesh.InitUnstructDataset(quantity_name, FSDatasetInfo(quantityNames, quantitySpecs, fs_volume_cell_types))
        Var = self.fsmesh.GetUnstructDataset(quantity_name).GetValues()

        for elemIndex in range(Var.Size(0)):
          Var[elemIndex] = int(cellN[elemIndex])

      return None

def HoleMesh(clac,fsmesh,paraDict,wallBoundaryMarkers,offsets,meshID,offsetFromBC="BCOverset"):
  tb2 = None
  # Conversion of the curvilinear mesh of the body ("standard" conversion)
  if meshID>0:
    tmp_dict_BCs = {}
    for dictio in paraDict["boundary treatments"]:
      if dictio["treatment type"] in tmp_dict_BCs:
        for value in dictio["boundary markers"]:
          tmp_dict_BCs[dictio["treatment type"]].append(value)
      else:
        tmp_dict_BCs[dictio["treatment type"]] = dictio["boundary markers"]
    dict_BCs = {}
    for bcname, markers in tmp_dict_BCs.items():
       for marker in markers:
          dict_BCs[marker] = bcname
    coords_name = "UndeformedCoordinates" if fsmesh.HasUnstructDataset("UndeformedCoordinates") else "Coordinates"
    C_FC = Converter_FSDM_CGNS(clac=clac,fsmesh=fsmesh,inmemory=True,dict_BCs=dict_BCs,coords_name=coords_name)
    C_FC.convertFSDM2CGNS()

    z_body = Internal.getZones(C_FC.pytree)[0]
    z_body[0] = z_body[0]+"."+str(Cmpi.rank)

    # Extract only the wall for the blanking (surface mesh)
    wall = C.extractBCOfType(z_body,offsetFromBC)
    C._deleteFlowSolutions__(wall)
    elts = Internal.getNodesFromType(wall,"Elements_t")
    for elt in elts:
        if elt[0].startswith("GridElements"):
            Internal._rmNode(wall,elt)
    tb2 = C.convertArray2Tetra(wall)
    if len(tb2) != 0:
      param_solver = Internal.getNodeFromType(tb2,"UserDefinedData_t")
      proc_safe = Internal.getNodeFromName(param_solver,"proc")[1][0][0]
      tb2 = T.join(tb2)
      tb2 = Internal.getZones(tb2)[0]
      param_solver = Internal.newUserDefinedData("param",parent=tb2)
      Internal.newDataArray(".Solver#Param",parent=param_solver,value=proc_safe)
      Internal.newDataArray("meshID",parent=param_solver,value=meshID)

  meshIDs = Cmpi.allgather(meshID)
  tb2 = Cmpi.allgatherZones(tb2)
  if Cmpi.rank == 0: C.convertPyTree2File(tb2, "wall.plt")
  bodies = {}
  for zone in Internal.getZones(tb2):
    meshid = Internal.getNodeFromName(zone,"meshID")[1][0]
    #meshid = meshID
    if meshid not in bodies.keys():
      bodies[meshid] = zone
    else:
      bodies[meshid] = T.join(bodies[meshid],zone)
      bodies[meshid] = G.close(bodies[meshid])
  bodies_offset = bodies.copy()
  sign_offset = 1. if offsetFromBC=="BCWall" else -1.
  for meshid in bodies_offset.keys():

      BB = G.bbox(bodies_offset[meshid])
      xmin = BB[0]; ymin = BB[1]; zmin = BB[2]
      xmax = BB[3]; ymax = BB[4]; zmax = BB[5]
      dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
      ppul = 100./dmax
      print("Points per unit lenght=",ppul)
      bodies_offset[meshid] = D.offsetSurface(bodies_offset[meshid], offset=sign_offset*offsets[meshid-1], pointsPerUnitLength=ppul, algo=0, dim=3)[0]
      if Cmpi.rank == 0: C.convertPyTree2File(bodies_offset[meshid], "wall_offset_%s.plt" %meshid)
      bodies_offset[meshid] = C.convertArray2Tetra(bodies_offset[meshid])
      bodies_offset[meshid] = G.close(bodies_offset[meshid])

  #offset_tb2 = C.newPyTree(["bodies",list(bodies.values())])
  return bodies_offset

def SurfaceBackgroundMesh(clac,fsmesh,paraDict,wallBoundaryMarkers,offsets,meshID):
  tb2 = None

  Cmpi.barrier()
  # Conversion of the curvilinear mesh of the body ("standard" conversion)
  if meshID==0:
    tmp_dict_BCs = {}
    for dictio in paraDict["boundary treatments"]:
      if dictio["treatment type"] in tmp_dict_BCs:
        for value in dictio["boundary markers"]:
          tmp_dict_BCs[dictio["treatment type"]].append(value)
      else:
        tmp_dict_BCs[dictio["treatment type"]] = dictio["boundary markers"]
    dict_BCs = {}
    for bcname, markers in tmp_dict_BCs.items():
       for marker in markers:
          dict_BCs[marker] = bcname

    C_FC = Converter_FSDM_CGNS(clac=clac,fsmesh=fsmesh,inmemory=True,dict_BCs=dict_BCs)
    C_FC.convertFSDM2CGNS()
    z_body = Internal.getZones(C_FC.pytree)[0]
    z_body[0] = z_body[0]+"."+str(Cmpi.rank)

    # Extract only the wall for the blanking (surface mesh)

    print(Cmpi.rank,"zonevalue",Internal.getValue(z_body))
    ER = Internal.getNodeFromName(z_body, "ElementRange")[1]

    if (ER[1] - ER[0] + 1) > 0:
        extFaces = z_body
        Internal._rmNodesFromType(extFaces,"ZoneBC_t")
        elts_extFaces = Internal.getNodesFromType(extFaces,"Elements_t")
        for elt in elts_extFaces:
            if elt[0].startswith("GridElements"):
                Internal._rmNode(extFaces,elt)
        zones = []
        elts_extFaces = Internal.getNodesFromType(extFaces,"Elements_t")

        coords_x = Internal.getNodeFromName(extFaces,"CoordinateX")[1]
        coords_y = Internal.getNodeFromName(extFaces,"CoordinateY")[1]
        coords_z = Internal.getNodeFromName(extFaces,"CoordinateZ")[1]

        zones = []
        for elt_eF in elts_extFaces:
          extFaces_select = C.selectConnectivity(extFaces,elt_eF[0])
          extFaces_select = C.convertArray2Tetra(extFaces_select)
          zones.append(extFaces_select)

        tb2 = T.join(zones)

  print("before all gather ")
  meshIDs = Cmpi.allgather(meshID)
  tb2 = Cmpi.allgatherZones(tb2)
  bodies = {}
  for zone in Internal.getZones(tb2):
    #meshid = Internal.getNodeFromName(zone,"meshID")[1][0]
    meshid = meshID
    if meshid not in bodies.keys():
      bodies[meshid] = zone
    else:
      bodies[meshid] = T.join(bodies[meshid],zone)
      bodies[meshid] = G.close(bodies[meshid])
  bodies_offset = bodies.copy()
  for meshid in bodies_offset.keys():
      BB = G.bbox(bodies_offset[meshid])
      xmin = BB[0]; ymin = BB[1]; zmin = BB[2]
      xmax = BB[3]; ymax = BB[4]; zmax = BB[5]
      dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
      ppul = 100./dmax
      print("Points per unit lenght=",ppul)
      bodies_offset[meshid] = D.offsetSurface(bodies_offset[meshid], offset=-offsets[meshid-1], pointsPerUnitLength=ppul, algo=0, dim=3)[0]
      C.convertPyTree2File(bodies_offset[meshid], "wall_OVERSET_%s.plt" %meshid)
      bodies_offset[meshid] = C.convertArray2Tetra(bodies_offset[meshid])
      bodies_offset[meshid] = G.close(bodies_offset[meshid])
  #offset_tb2 = C.newPyTree(["bodies",list(bodies.values())])
  return bodies, bodies_offset

def BackgroundMesh(clac,fsmesh,meshID):
  if meshID == 0:
    # Conversion of the background mesh ("light" conversion -> only the volume element types)
    C_FC2 = Converter_FSDM_CGNS(clac=clac,fsmesh=fsmesh,inmemory=True)
    C_FC2.convertFSDM2CGNSforOverset()
    zBG = Internal.getZones(C_FC2.pytree)[0]
    zBG = C.breakConnectivity(zBG)
    #zBG = C.convertArray2NGon(zBG)
    #zBG = T.join(zBG)
    #zBG[0] = zBG[0]+"."+str(Cmpi.rank)
    tBG = C.newPyTree(['Base',zBG])
  else: tBG = None
  return tBG

def ChildMesh(clac,fsmesh,meshID,meshIDChildMeshToBlank):
  if meshID == meshIDChildMeshToBlank:
    # Conversion of the background mesh ("light" conversion -> only the volume element types)
    C_FC2 = Converter_FSDM_CGNS(clac=clac,fsmesh=fsmesh,inmemory=True)
    C_FC2.convertFSDM2CGNSforOverset()
    zBG = Internal.getZones(C_FC2.pytree)[0]
    zBG = C.breakConnectivity(zBG)
    #zBG = C.convertArray2NGon(zBG)
    #zBG = T.join(zBG)
    #zBG[0] = zBG[0]+"."+str(Cmpi.rank)
    tBG = C.newPyTree(['Base',zBG])
  else: tBG = None
  return tBG
