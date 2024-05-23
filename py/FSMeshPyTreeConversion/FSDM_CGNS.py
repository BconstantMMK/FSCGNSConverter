import Converter.PyTree as C
import Converter.converter as converter
import Converter.Mpi as Cmpi
import Transform.PyTree as T
import Intersector.PyTree as XOR
import Generator.PyTree as G
import Post.PyTree as P
import Converter.Internal as Internal


#FSDM imports
import FSDM
from FSDataManager import FSClac, FSMesh, FSError, FSFloatArray, FSIntArray, FSStringArray, FSDataName, FSDataSpecArray, FSDatasetInfo, FSMeshEnums, FSUnstructVolumeCellTypes, FSUnstructSurfaceCellTypes, FS_AT_CADGroupID, FSCellInfo
import FSZoltan
import numpy
from functools import cmp_to_key

try:
  from FSDMPyUtils import ArrayOps
except:
  print("No FSDMPyUtils found. Continuing..")

import os, sys,time

class Converter_FSDM_CGNS:

  def __init__(self,mesh_name="mesh",clac=FSClac(),fsmesh=None,dimPb=2,invertPlanesYZ=False,conformal=True,IBM=False,IBM_parameters={},keepFlowSolution=False,whichDatasets=[],dict_BCs={},inmemory=False):

    #Parameters passed as arguments
    self.mesh_name = mesh_name
    self.conformal = conformal
    self.dimPb = dimPb
    self.IBM = IBM
    self.IBM_parameters = IBM_parameters
    self.invertPlanesYZ = invertPlanesYZ
    self.keepFlowSolution = keepFlowSolution
    self.whichDatasets = whichDatasets
    self.inmemory = inmemory
    #Variables defined inside the class

    self.clac = clac
    self.fsmesh = fsmesh
    self.pytree = Internal.newCGNSTree()
    self.meshType = "Unstructured"
    self.nb_vertices = 0
    self.nb_cells_volume = 0
    self.nb_cells_surface = 0
    self.fs_surface_cell_types = []
    self.fs_volume_cell_types = []
    self.fs_cell_types = []
    self.fs_cell_types_BCs = []
    self.coordinatesX = numpy.empty([])
    self.coordinatesY = numpy.empty([])
    self.coordinatesZ = numpy.empty([])
    self.numpy_cell2node = {}
    self.numpy_cell2node_volume = []
    self.numpy_cell2node_surface = []
    self.numpy_range = []
    self.indices_per_boundary = []
    self.list_names_BCs = []
    self.fs_markers = []
    self.boundary_marker_to_bc_name = {}
    self.boundary_marker_to_point_list = {}
    self.dict_bcs = dict_BCs

    self.MPI=False
    self.cell2Proc = {}

  def CellTypesFS2Cassiopee(self,idx):
    #Manuel entry of the keys
    CELLTYPE_FS_TO_CASSIOPEE = {}
    CELLTYPE_FS_TO_CASSIOPEE['Tri3'] = 'TRI'
    CELLTYPE_FS_TO_CASSIOPEE['Quad4'] = 'QUAD'
    CELLTYPE_FS_TO_CASSIOPEE['Tetra4'] = 'TETRA'
    CELLTYPE_FS_TO_CASSIOPEE['Pyra5'] = 'PYRA'
    CELLTYPE_FS_TO_CASSIOPEE['Prism6'] = 'PENTA'
    CELLTYPE_FS_TO_CASSIOPEE['Hexa8'] = 'HEXA'

    #We extend the dictionnary to access cassiopee strings with integers
    for key in list(CELLTYPE_FS_TO_CASSIOPEE) :
      CELLTYPE_FS_TO_CASSIOPEE[FSMeshEnums.StringToCellType(key)] = CELLTYPE_FS_TO_CASSIOPEE[key]
    return CELLTYPE_FS_TO_CASSIOPEE[idx]


  def CellTypesCassiopee2FS(self,idx):
    #now we build the inverse dictionnary
    CELLTYPE_CASSIOPEE_TO_FS = {}
    CELLTYPE_CASSIOPEE_TO_FS[5] = 'Tri3'
    CELLTYPE_CASSIOPEE_TO_FS[7] = 'Quad4'
    CELLTYPE_CASSIOPEE_TO_FS[10] = 'Tetra4'
    CELLTYPE_CASSIOPEE_TO_FS[12] = 'Pyra5'
    CELLTYPE_CASSIOPEE_TO_FS[14] = 'Prism6'
    CELLTYPE_CASSIOPEE_TO_FS[17] = 'Hexa8'
    return CELLTYPE_CASSIOPEE_TO_FS[idx]

  def recoverConnectivitySurf(self, zbcs, noelt):
    len_bcs = len(zbcs)
    connectivity_surf = numpy.array([])
    elts_t = Internal.getNodesFromType(self.pytree,'Elements_t')
    for elt in elts_t:
      if elt[1][0] == noelt and elt[0]!="NonConformalFaces":
        zone_surf = Internal.getNodeFromName(elt, elt[0])
        EC = Internal.getNodeFromName(zone_surf,'ElementConnectivity')[1]
        len_array_conn = len(connectivity_surf)
        connectivity_surf = numpy.concatenate((connectivity_surf,EC))

    return connectivity_surf

  def mergeQuadConnUnstructured(self,zbcs):

    elements_nodes = Internal.getNodesFromType(self.pytree,"Elements_t")

    connectivity_surf = []
    for fs_cell_type in self.fs_surface_cell_types:
      cell_type_CGNS = Internal.eltName2EltNo(self.CellTypesFS2Cassiopee(fs_cell_type))[0]
      connectivity_surf.append(self.recoverConnectivitySurf(zbcs, cell_type_CGNS))
    node_first_surf = None
    counter_cells = 0
    for i, fs_cell_type in enumerate(self.fs_surface_cell_types):
      cell_type_CGNS = Internal.eltName2EltNo(self.CellTypesFS2Cassiopee(fs_cell_type))[0]
      firstFound = False

      for node in elements_nodes:

        val = Internal.getValue(node)[0]
        name = Internal.getName(node)
        if val == cell_type_CGNS and name!="NonConformalFaces":
            if firstFound == True:
              Internal._rmNode(self.pytree,node)
            if firstFound == False:
              node_first_surf = node
              firstFound = True
        print("val",val,cell_type_CGNS)
        if i==0 and FSMeshEnums.StringToCellType(self.CellTypesCassiopee2FS(val)) in FSUnstructSurfaceCellTypes:
            self.fs_cell_types_BCs.append(val)

      nb_vertex_per_cell    = Internal.eltNo2EltName(cell_type_CGNS)[1]
      nb_current_cells_surf = connectivity_surf[i].shape[0]
      rangemin = self.nb_cells_volume + counter_cells + 1
      rangemax = rangemin + nb_current_cells_surf//nb_vertex_per_cell

      if node_first_surf != None:
        Internal.getNodeFromName(node_first_surf,'ElementRange')[1][0] = rangemin
        Internal.getNodeFromName(node_first_surf,'ElementRange')[1][1] = rangemax
        Internal.getNodeFromName(node_first_surf,'ElementConnectivity')[1] = connectivity_surf[i]
        #
        node_first_surf[0]    = self.CellTypesFS2Cassiopee(fs_cell_type)
        node_first_surf[1][1] = 0
        counter_cells += nb_current_cells_surf
    return

  def createBCZonePerSurfaceElementType(self):
    print("\n\n\n\n")
    bcs_node = Internal.getNodesFromType(self.pytree,"BC_t")
    point_list = numpy.array([])
    len_point_list = 0
    len_range = 0

    elements_nodes = Internal.getNodesFromType(self.pytree,"Elements_t")
    for node in elements_nodes:
      val = Internal.getValue(node)[0]
      name = Internal.getName(node)
    for idx,bc in enumerate(bcs_node):
      bcname = Internal.getName(bc)
      if bcname != "NonConformalFaces":
        #if bcname[-5:] != ".QUAD" and bcname[-4:] != ".TRI":
        if not ".QUAD" in bc[0] and not "TRI" in bc[0]:
          bc[0]=self.list_names_BCs[idx]
        print("bc[0]=", bcname)
        if Internal.getNodeFromType(bc, "IndexArray_t")!=None:
          if Internal.getNodeFromType(bc, "IndexArray_t")[0] == "PointList" :
            len_point_list += len(point_list)
            point_list = Internal.getNodeFromName(bc, "PointList")[1][0] + self.nb_cells_volume + len_point_list
            Internal.getNodeFromName(bc, "PointList")[1][0] = point_list
        elif Internal.getNodeFromType(bc, "IndexRange_t")!=None:
          if Internal.getNodeFromType(bc, "IndexRange_t")[0] == "ElementRange" :
            element_range_old = Internal.getNodeFromName(bc, "ElementRange")[1][0]
            element_range = numpy.array([[self.nb_cells_volume + len_point_list +1, self.nb_cells_volume + len_point_list + element_range_old[1]-element_range_old[0]+1]])
            len_point_list += (element_range_old[1] - element_range_old[0] +1)
            Internal.getNodeFromName(bc, "ElementRange")[1][0] = element_range
    return

  def prepareDatasetOfNonConformalFaces(self):
    bc_names = [bc[0] for bc in Internal.getNodesFromType(self.pytree,'BC_t')]
    rm = Internal.getNodeFromName(self.pytree,'QuadNQuad')
    old_name_hf = rm[0]
    rm[0] = "NonConformalFaces"
    Internal._renameNode(self.pytree,old_name_hf,"NonConformalFaces")
    NCF = Internal.getNodeFromName(self.pytree,"NonConformalFaces")
    if Internal.getNodeFromName(NCF,"PointList")!= None:
      len_NCF = Internal.getNodeFromName(NCF,"PointList")[1][0].shape[0]
    elif Internal.getNodeFromName(NCF,"ElementRange")!= None:
      if len(Internal.getNodeFromName(NCF,"ElementRange")[1])==1:
        ERmin = Internal.getNodeFromName(NCF,"ElementRange")[1][0][0]
        ERmax = Internal.getNodeFromName(NCF,"ElementRange")[1][0][1]
      elif len(Internal.getNodeFromName(NCF,"ElementRange")[1])==2:
        ERmin = Internal.getNodeFromName(NCF,"ElementRange")[1][0]
        ERmax = Internal.getNodeFromName(NCF,"ElementRange")[1][1]
      else: raise ValueError("Problem in the Element Range of non conformal interfaces.")
      len_NCF = ERmax-ERmin+1
    self.nb_cells_surface -= len_NCF
    return

  def createZoneOfNonConformalFaces(self):

    xCoord = Internal.getNodeFromName(self.pytree,"CoordinateX")[1]
    yCoord = Internal.getNodeFromName(self.pytree,"CoordinateY")[1]
    zCoord = Internal.getNodeFromName(self.pytree,"CoordinateZ")[1]

    octree_faces_node = Internal.getNodesFromName(self.pytree,"NonConformalFaces")
    octree_faces_EC_global = Internal.getNodeFromName(octree_faces_node,"ElementConnectivity")[1]
    len_NCF = len(octree_faces_EC_global)//4

    rm = Internal.getNodesFromName(self.pytree,'NonConformalFaces')
    for i in rm:
      Internal._rmNode(self.pytree,i)

    _,idx = numpy.unique(octree_faces_EC_global,return_index=True)
    octree_faces_idx_nodes = octree_faces_EC_global[numpy.sort(idx)] - 1 #indices loc2glob
    len_nodes_NCF = len(octree_faces_idx_nodes)
    if len_nodes_NCF>0:
      xCoord_nodes_NCF = xCoord[octree_faces_idx_nodes]
      yCoord_nodes_NCF = yCoord[octree_faces_idx_nodes]
      zCoord_nodes_NCF = zCoord[octree_faces_idx_nodes]

      nonconformal_faces_global = numpy.reshape(octree_faces_EC_global-1,(len_NCF,4))
      indices1 = numpy.linspace(0,len_nodes_NCF-1,len_nodes_NCF,dtype=int)
      glob2loc = numpy.zeros(self.nb_vertices,dtype=int)
      glob2loc[octree_faces_idx_nodes] = indices1
      nonconformal_faces_local = glob2loc[nonconformal_faces_global]

      z_octree_faces = Internal.newZone(name = "NonConformalFaces",zsize=[[len(xCoord_nodes_NCF),len(octree_faces_idx_nodes)]],ztype="Unstructured")
      gc = Internal.newGridCoordinates(parent = z_octree_faces)
      Internal.newDataArray('CoordinateX', value = xCoord_nodes_NCF, parent = gc)
      Internal.newDataArray('CoordinateY', value = yCoord_nodes_NCF, parent = gc)
      Internal.newDataArray('CoordinateZ', value = zCoord_nodes_NCF, parent = gc)
      Internal.newElements(name = "NonconformalFaces", etype = 7, econnectivity = numpy.ravel(nonconformal_faces_local+1), erange = [1, len_NCF], eboundary = 0, parent = z_octree_faces)
    else:
      z_octree_faces = None #Internal.newZone(name = "NonConformalFaces",zsize=[[0,0]],ztype="Unstructured")
    return z_octree_faces

  def modifyConnectivityUnstructured(self):

    elements_nodes = Internal.getNodesFromType(self.pytree,"Elements_t")
    if self.meshType=="Structured":
      C._rmBCOfType(self.pytree,"BCMatch")
      C._rmBCOfType(self.pytree,"BCDegeneratedLine")
    elif self.meshType=="Unstructured":
      list_types_elements = []
      for en in elements_nodes:
        list_types_elements.append(Internal.getValue(en)[0])

    if self.MPI or self.meshType=="Structured" or (len(set(list_types_elements)) < len(list_types_elements)):

      if Cmpi.rank==0:print("Merging several QUAD-CGNS zones into one zone.\n")
      novol = len(self.fs_volume_cell_types)
      nobc = len(elements_nodes)-novol
      initial_zones_BC = elements_nodes[novol:nobc+novol+1]

      zones = Internal.getZones(self.pytree)

      z = zones[0]
      for noz in range(novol,len(zones)):
        z = T.join([z,zones[noz]])
      #if self.IBM==True: self.list_names_BCs.append("IBMWall")
      zbcs=[]; bctypes=[]; bcs=[];
      for bc in Internal.getNodesFromType(z,'BC_t'):
        #if bc[0][-5:] != ".QUAD" and bc[0][-4:] != ".TRI":
        if not ".QUAD" in bc[0] and not "TRI" in bc[0]:
          self.list_names_BCs.append(bc[0].replace('.', ''))
        else:
          self.list_names_BCs.append(bc[0].split(".")[0])

        bctype = Internal.getValue(bc)
        if bctype not in bctypes:
          bctypes.append(bctype)
          bcs.append(bc)

      for bctype in bctypes:
          zbc = C.extractBCOfType(self.pytree,bctype)
          if self.meshType == "Structured":
              zbc = C.convertArray2Hexa(zbc)
              zbc = T.join(zbc)
              self.nb_cells_surface += Internal.getValue(zbc)[0][1]
          zbcs.append(zbc)

      if self.meshType == "Structured":
         z = C.convertArray2Hexa(z)
         z = G.close(z)
         self.nb_vertices = int(Internal.getValue(z)[0][0])
         len_bcs = len(bctypes)

         for i in range(len_bcs):
           self._addBC2ZoneLoc(z,bctypes[i],bctypes[i], zbcs[i])
           print("bctypes[i]",bctypes[i])
         self.pytree = C.newPyTree(["Unstructured",z])
      self.mergeQuadConnUnstructured(zbcs)
      self.createBCZonePerSurfaceElementType()

    return None

  def recoverInfoMeshCGNS(self):
    #Import mesh
    if Cmpi.size == 1:
        self.MPI=False
        self.pytree = C.convertFile2PyTree(self.mesh_name)
    elif Cmpi.size > 1:
        self.MPI=True
        self.pytree = Cmpi.convertFile2PyTree(self.mesh_name,proc=Cmpi.rank)

    base = Internal.getBases(self.pytree)[0]

    if Cmpi.rank==0:print("Retrieving mesh information from Cassiopee...")

    #Get mesh information
    zones = Internal.getZones(base)
    if zones != []:
      zone = zones[0]
      zoneDim = Internal.getZoneDim(zone)
      self.meshType = zoneDim[0]

      if self.meshType == "Unstructured":
        self.nb_vertices = zoneDim[1]
        self.nb_cells_volume = zoneDim[2]

        pytree_element_nodes = Internal.getNodesFromType(zone, "Elements_t")
        for element_node in pytree_element_nodes:
          element_node_value = Internal.getValue(element_node)
          element_node_range = Internal.getNodeFromName(element_node,"ElementRange")[1]
          fs_cell_type = self.CellTypesCassiopee2FS(element_node_value[0])
          fs_cell_type = FSMeshEnums.StringToCellType(fs_cell_type)
          if fs_cell_type in FSUnstructSurfaceCellTypes:
            nb_current_cells = element_node_range[1]-element_node_range[0]+1
            #if nb_current_cells == 0:
            #  raise ValueError("ERROR: Wrong number of surface cells.")
            self.nb_cells_surface += nb_current_cells
            if fs_cell_type not in self.fs_surface_cell_types:
              self.fs_surface_cell_types.append(fs_cell_type)
          elif fs_cell_type in FSUnstructVolumeCellTypes :
            self.fs_volume_cell_types.append(fs_cell_type)

          if fs_cell_type not in self.fs_cell_types:
            self.fs_cell_types.append(fs_cell_type)

      elif self.meshType == "Structured":
        index_i, index_j, index_k = zoneDim[1], zoneDim[2], zoneDim[3]
        self.nb_vertices = index_i * index_j * index_k
        self.nb_cells_volume = (index_i-1) * (index_j-1) * (index_k-1)

        if self.nb_cells_volume != 0:
          self.fs_volume_cell_types.append(8) #Hexa
          self.fs_surface_cell_types.append(4) #Quad
        self.fs_cell_types = self.fs_volume_cell_types + self.fs_surface_cell_types
    else:
        self.nb_vertices = 0
        self.nb_cells_volume = 0
        self.fs_cell_types = []
        self.fs_volume_cell_types = []
        self.fs_surface_cell_types = []
    Cmpi.barrier()
    gath_fs_cell_types = Cmpi.allgather(self.fs_cell_types)
    gath_fs_cell_types = list({x for v in gath_fs_cell_types for x in v})
    if zones == []:
        self.fs_cell_types = gath_fs_cell_types
    return

  def recoverCoordinatesCGNS(self):
    #Create node coordinates dataset
    if Cmpi.rank==0:print("Retrieving node coordinates from Cassiopee...")
    pytree_coordinates_node = Internal.getNodeFromType(self.pytree, "GridCoordinates_t")
    check_x, check_y, check_z = False, False, False
    coordinate_data_nodes = Internal.getNodesFromType(pytree_coordinates_node, "DataArray_t")
    for data_node in coordinate_data_nodes:
      node_name = data_node[0]
      if node_name == "CoordinateX":
        self.coordinatesX = data_node[1]
        check_x = True
      elif node_name == "CoordinateY":
        self.coordinatesY = data_node[1]
        check_y = True
      elif node_name == "CoordinateZ":
        self.coordinatesZ = data_node[1]
        check_z = True
    if not check_x or not check_y or not check_z :
      raise ValueError("ERROR: Some coordinates were not found in the Coordinate node, exiting...")
    return

  def initializeCell2Proc(self,t,newcell2Proc=0):
     nProcs = self.clac.NProcs()
     myID = Cmpi.rank
     myIDs_gathered = ArrayOps.AllGather(myID,self.clac)

     newcell2ProcGathered = ArrayOps.AllGather(newcell2Proc,self.clac)

     CORRECT_newcell2ProcGathered = FSIntArray(nProcs+1)
     CORRECT_newcell2ProcGathered.Fill(int(0))
     for i in range(nProcs):
       CORRECT_newcell2ProcGathered[i+1] = int(CORRECT_newcell2ProcGathered[i]+newcell2ProcGathered[i])
     self.cell2Proc[t]   = CORRECT_newcell2ProcGathered

     return

  def recoverCGNSConnectivity(self):
    pytree_element_nodes = Internal.getNodesFromType(self.pytree, "Elements_t")
    #Loop on pytree element nodes
    for i,element_node in enumerate(pytree_element_nodes):
      element_node_value = Internal.getValue(element_node)
      #We retrieve the cell 2 node connectivity and the element range
      element_connectivity_node = Internal.getNodeFromName(element_node, "ElementConnectivity")

      if element_connectivity_node == None :
        raise ValueError('ERROR: element connectivity is not defined in one of the element nodes')
      else :
        nb_vertex_per_cell = Internal.eltNo2EltName(element_node_value[0])[1]

        nb_current_cell = int(element_connectivity_node[1].size / nb_vertex_per_cell)
        fs_cell_type = FSMeshEnums.StringToCellType(self.CellTypesCassiopee2FS(element_node_value[0]))
        self.numpy_cell2node[fs_cell_type] = numpy.int_(element_connectivity_node[1]-1).reshape(nb_current_cell, nb_vertex_per_cell)
    return

  def initUnstructCellsOfType(self,cell_type):
      has_cells = cell_type in self.numpy_cell2node.keys()

      if has_cells:
        np_cell2node = self.numpy_cell2node[cell_type]
        fs_cell2node = FSIntArray(np_cell2node.shape[0],np_cell2node.shape[1])
      else:
        fs_cell2node = FSIntArray(0,FSCellInfo.NNodes(cell_type))

      if self.MPI:
        if has_cells:
          nNodesPrevious = self.cell2Proc[FSMeshEnums.CT_Node][Cmpi.rank]
          numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), np_cell2node + nNodesPrevious, casting='unsafe')
        self.fsmesh.InitUnstructCells(cell_type, self.cell2Proc[cell_type], fs_cell2node, True)
      else:
        if has_cells:
          numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), np_cell2node, casting='unsafe')
        self.fsmesh.InitUnstructCells(cell_type, fs_cell2node, True)
      return

  def initializeFSMesh(self,z_NCfaces=None):
    #Init mesh number of node

    if self.fsmesh==None:
      self.fsmesh = FSMesh(self.clac)

    self.fsmesh.BeginInitialization()
    if self.MPI==True:
      self.initializeCell2Proc(FSMeshEnums.CT_Node,self.nb_vertices)
      self.fsmesh.InitUnstructNodes(self.cell2Proc[FSMeshEnums.CT_Node])

      for fs_cell_type in self.fs_cell_types:
        if fs_cell_type in self.numpy_cell2node.keys():
          self.initializeCell2Proc(fs_cell_type,self.numpy_cell2node[fs_cell_type].shape[0])
        else:
          self.initializeCell2Proc(fs_cell_type,0)

        self.initUnstructCellsOfType(fs_cell_type)

    else:
      self.fsmesh.InitUnstructNodes(self.nb_vertices)

      for fs_cell_type in self.fs_cell_types:
        self.initUnstructCellsOfType(fs_cell_type)

    if self.conformal==False:
      if self.MPI==True:
        self.initializePseudoCell_QuadNQuad_MPI(z_NCfaces)
      else:
        self.initializePseudoCell_QuadNQuad(z_NCfaces)

    self.fsmesh.EndInitialization()
    self.initEmptyDatasetOfCoordinatesBC([FSDataName.Coordinates()],FSMeshEnums.CT_Node)

    if self.nb_vertices != 0:
      #We put all the coordinates together to fill the FSFloatArray
      if self.invertPlanesYZ==False:
        np_coordinates = numpy.append(self.coordinatesX.reshape(self.nb_vertices, 1), self.coordinatesY.reshape(self.nb_vertices, 1),axis=1)
        np_coordinates = numpy.append(np_coordinates, self.coordinatesZ.reshape(self.nb_vertices, 1),axis=1)
      elif self.invertPlanesYZ==True:
        np_coordinates = numpy.append(self.coordinatesX.reshape(self.nb_vertices, 1), self.coordinatesZ.reshape(self.nb_vertices, 1),axis=1)
        np_coordinates = numpy.append(np_coordinates, -self.coordinatesY.reshape(self.nb_vertices, 1),axis=1)
      Var = self.fsmesh.GetUnstructDataset(FSDataName.Coordinates()).GetValues()
      for nodeIndex in range(self.nb_vertices):
          VarIndex = Var.MapIndex(nodeIndex,0)
          Var[VarIndex] = np_coordinates[nodeIndex,0]
          VarIndex = Var.MapIndex(nodeIndex,1)
          Var[VarIndex] = np_coordinates[nodeIndex,1]
          VarIndex = Var.MapIndex(nodeIndex,2)
          Var[VarIndex] = np_coordinates[nodeIndex,2]

    return

  def initIBMDatasets(self):
    #Flis wall distance initialization
    spatial_discretization = self.IBM_parameters["spatial discretization"]["type"]
    if spatial_discretization == "FV":
      N_IP_per_element = 1
    else:
      degree = self.IBM_parameters["spatial discretization"]["degree"]
      if spatial_discretization == "DG":
        integrationDegree = 2*degree+1
        quadratureType = "GaussLegendre"
      elif spatial_discretization == "DGSEM":
        integrationDegree = 2*degree-1
        quadratureType = "GaussLobatto"
      import QuadratureDG as Q
      N_IP_per_element = Q.GetReferencePointsHexa(integrationDegree, quadratureType)[0]

    list_suffix_datasets = [""]
    list_suffix_datasets.extend(range(1, N_IP_per_element))
    for i in range(N_IP_per_element):
      flis_node = Internal.getNodeFromName(self.pytree,"FlisWallDistance"+str(list_suffix_datasets[i]))
      flis_distance = Internal.getNodeFromName(flis_node,"TurbulentDistance")[1]
      quantity_name = "FlisWallDistance"+str(list_suffix_datasets[i])
      quantityNames = FSStringArray(1)
      quantityNames[0] = quantity_name
      quantitySpecs = FSDataSpecArray(1)

      fsarray_volume_cell_types = FSIntArray(len(self.fs_volume_cell_types))
      numpy.copyto(numpy.array(fsarray_volume_cell_types.Buffer(), copy=False), self.fs_volume_cell_types, casting='unsafe')
      self.fsmesh.InitUnstructDataset(quantity_name, FSDatasetInfo(quantityNames, quantitySpecs, fsarray_volume_cell_types))
      Var = self.fsmesh.GetUnstructDataset(quantity_name).GetValues()
      for elemIndex in range(self.nb_cells_volume):
              VarIndex =    Var.MapIndex(elemIndex, 0)
              Var[VarIndex] = flis_distance[elemIndex]

    return

  def recoverPointList2BoundaryMarkers(self):

    pytree_zonebc_node = Internal.getNodeFromType(self.pytree, "ZoneBC_t")
    pytree_bc_nodes = Internal.getNodesFromType(pytree_zonebc_node, "BC_t")

    current_automatic_marker = 1
    self.dict_bc_elts = {}
    self.dict_bc_elts["TRI"] = []
    self.dict_bc_elts["QUAD"] = []

    if self.IBM:
      IBM_BC_coords_x = {}
      IBM_BC_coords_y = {}
      IBM_BC_coords_z = {}
      IBM_BC_names = []
      if self.IBM_parameters["IBM type"]["type"]=="local":
        wall_boundary_markers = self.IBM_parameters["IBM type"]["wall boundary markers"]
        BC_wall_coords_x = []
        BC_wall_coords_y = []
        BC_wall_coords_z = []
        BC_wall_names = []

    #Loop on BCs
    for count,bc_node in enumerate(pytree_bc_nodes) :

      bc_name = bc_node[0]

      #Get boundary marker : generate one if it does not exist otherwise we expect it to be in a user defined node named "BoundaryMarker"
      bc_boundary_marker = current_automatic_marker
      current_automatic_marker += 1
      len_point_list = 0

      #Get list of points on the BC : based on name to make the difference between point list and point range ;
      if (Internal.getNodeFromType(bc_node, "IndexArray_t")!=None):
        pointlist_node = Internal.getNodeFromType(bc_node, "IndexArray_t")
        point_list = numpy.ravel(pointlist_node[1]) - 1

      elif (Internal.getNodeFromType(bc_node, "IndexRange_t")!=None):
        pointlist_node = Internal.getNodeFromType(bc_node, "IndexRange_t")
        if len(pointlist_node[1])==1:
          point_list = numpy.arange(pointlist_node[1][0][0]-1, pointlist_node[1][0][1])
        elif len(pointlist_node[1])==2:
          point_list = numpy.arange(pointlist_node[1][0]-1, pointlist_node[1][1])
      len_point_list += len(point_list)
      #Now we can fill our dictionnaries
      self.boundary_marker_to_bc_name[bc_boundary_marker] = bc_name
      self.boundary_marker_to_point_list[bc_boundary_marker] = point_list

      if len(bc_name.split("."))>1 and bc_name.split(".")[1].startswith("TRI"):
        self.dict_bc_elts["TRI"].append(bc_boundary_marker)
      elif len(bc_name.split("."))>1 and bc_name.split(".")[1].startswith("QUAD"):
        self.dict_bc_elts["QUAD"].append(bc_boundary_marker)

      bc_dataset_node = Internal.getNodeFromType(bc_node, "BCDataSet_t")
      if bc_dataset_node != None:
        #raise ValueError("recoverPointList2BoundaryMarkers: empty BCDataset for IBM.")
        bc_dataset_grid_location = Internal.getNodeFromType(bc_dataset_node, "GridLocation_t")
        bc_data_nodes = Internal.getNodesFromType(bc_dataset_node, "BCData_t")
        for data_node in bc_data_nodes :
          fs_bc_dataset_name = data_node[0]
          if self.IBM and bc_name.startswith("IBMWall"):
            if bc_name not in IBM_BC_coords_x.keys():
              IBM_BC_coords_x[bc_name] = []
              IBM_BC_coords_y[bc_name] = []
              IBM_BC_coords_z[bc_name] = []
            IBM_BC_names.append(fs_bc_dataset_name)
            if self.invertPlanesYZ==False:
              IBM_BC_coords_x[bc_name].append(data_node[2][0][1])
              IBM_BC_coords_y[bc_name].append(data_node[2][1][1])
              IBM_BC_coords_z[bc_name].append(data_node[2][2][1])
            elif self.invertPlanesYZ==True:
              IBM_BC_coords_x[bc_name].append(data_node[2][0][1])
              IBM_BC_coords_y[bc_name].append(data_node[2][2][1])
              IBM_BC_coords_z[bc_name].append(-data_node[2][1][1])
          elif self.IBM and self.IBM_parameters["IBM type"]["type"] == "local" and bc_boundary_marker in wall_boundary_markers:
            BC_wall_names.append(fs_bc_dataset_name)
            if self.invertPlanesYZ==False:
              BC_wall_coords_x.append(data_node[2][0][1])
              BC_wall_coords_y.append(data_node[2][1][1])
              BC_wall_coords_z.append(data_node[2][2][1])
            elif self.invertPlanesYZ==True:
              BC_wall_coords_x.append(data_node[2][0][1])
              BC_wall_coords_y.append(data_node[2][2][1])
              BC_wall_coords_z.append(-data_node[2][1][1])
    if self.IBM == False: return
    elif self.IBM_parameters["IBM type"]["type"]=="global"  : return [IBM_BC_names, IBM_BC_coords_x, IBM_BC_coords_y, IBM_BC_coords_z]
    elif self.IBM_parameters["IBM type"]["type"]=="local": return [IBM_BC_names, IBM_BC_coords_x, IBM_BC_coords_y, IBM_BC_coords_z, BC_wall_names,BC_wall_coords_x, BC_wall_coords_y,BC_wall_coords_z]

  def initBCsFSDMmesh(self,IBMDatasets=[]):
    #We now have our point list for each marker so we can init the cell attribute in the fsmesh
    marker_array = numpy.zeros(self.nb_cells_surface,dtype=int)
    for marker in self.boundary_marker_to_bc_name :
        point_list = self.boundary_marker_to_point_list[marker]-self.nb_cells_volume
        marker_array[point_list] = marker #
    #Loop on surface cell types in the mesh and slice the array above to get the data we need
    for cell_type in self.fs_surface_cell_types :
      if self.dict_bc_elts["QUAD"]!=[] and self.dict_bc_elts["TRI"]!=[]:
        string_cassiopee_celltype = self.CellTypesFS2Cassiopee(cell_type)
        temp = set(self.dict_bc_elts[string_cassiopee_celltype])
        res = [i for i, val in enumerate(marker_array) if val in temp]
        np_marker_array_cell_type = marker_array[res]
      else:
        np_marker_array_cell_type = marker_array
      fs_marker_array_cell_type = FSIntArray(np_marker_array_cell_type.shape[0])
      numpy.copyto(numpy.array(fs_marker_array_cell_type.Buffer(), copy=False), np_marker_array_cell_type, casting='unsafe')

      self.fsmesh.InitCellAttribute(FS_AT_CADGroupID, cell_type, fs_marker_array_cell_type)
    #Then we attach our boundary marker to their name in the fsmesh
    if self.IBM ==True:
        IBM_boundary_markers = []
        IBM_names = []
    for marker in self.boundary_marker_to_bc_name.keys() :
      self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, self.boundary_marker_to_bc_name[marker])
      if self.IBM == True and self.boundary_marker_to_bc_name[marker].startswith("IBMWall"):
        IBM_boundary_markers.append(marker)
        IBM_names.append(self.boundary_marker_to_bc_name[marker])


    if self.IBM:
      IBMDataset1 = numpy.empty((len(IBMDatasets[1][IBM_names[0]]),0))
      IBMDataset2 = numpy.empty((len(IBMDatasets[1][IBM_names[0]]),0))
      IBMDataset3 = numpy.empty((len(IBMDatasets[1][IBM_names[0]]),0))
      pointlistIBC = numpy.empty(0,dtype=int)
      for IBM_boundary_marker,IBM_name in zip(IBM_boundary_markers,IBM_names):
        IBMDataset1 = numpy.concatenate([IBMDataset1, numpy.array(IBMDatasets[1][IBM_name])],axis=1)
        IBMDataset2 = numpy.concatenate([IBMDataset2, numpy.array(IBMDatasets[2][IBM_name])],axis=1)
        IBMDataset3 = numpy.concatenate([IBMDataset3, numpy.array(IBMDatasets[3][IBM_name])],axis=1)
        pointlistIBC = numpy.concatenate([pointlistIBC, self.boundary_marker_to_point_list[IBM_boundary_marker]-self.nb_cells_volume])

      self.createDatasetOfCoordinatesBC(self.fsmesh,IBMDataset1, IBMDataset2, IBMDataset3,IBMDatasets[0],self.nb_cells_surface,pointlistIBC)

      if self.IBM_parameters["IBM type"]["type"]=="local":
        wall_boundary_markers = self.IBM_parameters["IBM type"]["wall boundary markers"]
        self.createDatasetOfCoordinatesBC(self.fsmesh,IBMDatasets[5], IBMDatasets[6], IBMDatasets[7],IBMDatasets[4],self.nb_cells_surface,self.boundary_marker_to_point_list[wall_boundary_markers[0]]-self.nb_cells_volume)

    return

  def initBCsFSDMmesh_MPI(self,IBMDatasets=[]):
    marker_array = numpy.zeros(self.nb_cells_surface,dtype=int)

    keys = list(self.boundary_marker_to_bc_name.keys())
    values = list(self.boundary_marker_to_bc_name.values())
    keys = Cmpi.allgather(keys)
    values = Cmpi.allgather(values)
    bc_names_all = sorted(set([item for row in values for item in row]))
    bc_markers_all = list(range(1,len(bc_names_all)+1))
    boundary_marker_to_bc_name2 = {}
    boundary_marker_to_point_list2 = {}
    for (marker,name) in zip(bc_markers_all,bc_names_all):
      if name in self.boundary_marker_to_bc_name.values():
        idx =  list(self.boundary_marker_to_bc_name.keys())[list(self.boundary_marker_to_bc_name.values()).index(name)]
        boundary_marker_to_bc_name2[marker] = self.boundary_marker_to_bc_name[idx]
        boundary_marker_to_point_list2[marker] = self.boundary_marker_to_point_list[idx]
    for marker in boundary_marker_to_bc_name2 :
        point_list = boundary_marker_to_point_list2[marker]-self.nb_cells_volume
        marker_array[point_list] = marker #
    #Loop on surface cell types in the mesh and slice the array above to get the data we need
    for cell_type in self.fs_surface_cell_types :
      if self.dict_bc_elts["QUAD"]!=[] and self.dict_bc_elts["TRI"]!=[]:
        string_cassiopee_celltype = self.CellTypesFS2Cassiopee(cell_type)
        temp = set(self.dict_bc_elts[string_cassiopee_celltype])
        res = [i for i, val in enumerate(marker_array) if val in temp]
        np_marker_array_cell_type = marker_array[res]
      else:
        np_marker_array_cell_type = marker_array
      fs_marker_array_cell_type = FSIntArray(np_marker_array_cell_type.shape[0])
      numpy.copyto(numpy.array(fs_marker_array_cell_type.Buffer(), copy=False), np_marker_array_cell_type, casting='unsafe')
      self.fsmesh.InitCellAttribute(FS_AT_CADGroupID, cell_type, fs_marker_array_cell_type)

    #Then we attach our boundary marker to their name in the fsmesh
    #for marker in boundary_marker_to_bc_name2.keys() :
    if self.IBM ==True:
      IBM_boundary_markers = []
      IBM_names = []
      fsdatanames = ["WallPointCoordinates","DonorPointCoordinates"]
      fsarray_surface_cell_types = FSIntArray(len(self.fs_surface_cell_types))
      numpy.copyto(numpy.array(fsarray_surface_cell_types.Buffer(), copy=False), self.fs_surface_cell_types, casting='unsafe')
      self.initEmptyDatasetOfCoordinatesBC(fsdatanames,fsarray_surface_cell_types)

    for (marker,name) in zip(bc_markers_all,bc_names_all) :
      self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, name)
      if self.IBM == True and marker in  self.boundary_marker_to_bc_name.keys() and self.boundary_marker_to_bc_name[marker].startswith("IBMWall"):
        IBM_boundary_markers.append(marker)
        IBM_names.append(self.boundary_marker_to_bc_name[marker])

    if self.IBM and IBM_names!=[]:
      IBMDataset1 = numpy.empty((len(IBMDatasets[1][IBM_names[0]]),0))
      IBMDataset2 = numpy.empty((len(IBMDatasets[1][IBM_names[0]]),0))
      IBMDataset3 = numpy.empty((len(IBMDatasets[1][IBM_names[0]]),0))
      pointlistIBC = numpy.empty(0,dtype=int)
      for IBM_boundary_marker,IBM_name in zip(IBM_boundary_markers,IBM_names):
        IBMDataset1 = numpy.concatenate([IBMDataset1, numpy.array(IBMDatasets[1][IBM_name])],axis=1)
        IBMDataset2 = numpy.concatenate([IBMDataset2, numpy.array(IBMDatasets[2][IBM_name])],axis=1)
        IBMDataset3 = numpy.concatenate([IBMDataset3, numpy.array(IBMDatasets[3][IBM_name])],axis=1)
        pointlistIBC = numpy.concatenate([pointlistIBC, self.boundary_marker_to_point_list[IBM_boundary_marker]-self.nb_cells_volume])

      self.createDatasetOfCoordinatesBC(self.fsmesh,IBMDataset1, IBMDataset2, IBMDataset3,IBMDatasets[0],self.nb_cells_surface,pointlistIBC)

    return

  def initEmptyDatasetOfCoordinatesBC(self,BC_names,cell_type):
      coordNames = FSStringArray(3)
      coordNames[0] = FSDataName.Coordinate().X()
      coordNames[1] = FSDataName.Coordinate().Y()
      coordNames[2] = FSDataName.Coordinate().Z()

      coordSpecs = FSDataSpecArray(3)
      for fsdataname in BC_names:
        self.fsmesh.InitUnstructDataset(fsdataname, FSDatasetInfo(coordNames, coordSpecs, cell_type))

      return


  def createDatasetOfCoordinatesBC(self,fsmesh,coords_x, coords_y, coords_z,BC_names,nb_cell_surf,point_list):
    if not self.MPI:
      fsarray_surface_cell_types = FSIntArray(len(self.fs_surface_cell_types))
      numpy.copyto(numpy.array(fsarray_surface_cell_types.Buffer(), copy=False), self.fs_surface_cell_types, casting='unsafe')
      self.initEmptyDatasetOfCoordinatesBC(BC_names,fsarray_surface_cell_types)
    #for i in range(coords_x.shape[0]):
    for i in range(len(coords_x)):

       coord_x = coords_x[i]
       coord_y = coords_y[i]
       coord_z = coords_z[i]

       nb_points = coord_x.size
       np_coordinates = numpy.append(coord_x.reshape(nb_points, 1), coord_y.reshape(nb_points, 1),axis=1)
       np_coordinates = numpy.append(np_coordinates, coord_z.reshape(nb_points, 1),axis=1)
       fs_coordinates = FSFloatArray(nb_points,3)
       numpy.copyto(numpy.array(fs_coordinates.Buffer(), copy=False), np_coordinates, casting='unsafe')

       dataset = numpy.zeros((nb_cell_surf,3))
       for j in range(nb_points):
         dataset[point_list[j]][0] = np_coordinates[j][0]
         dataset[point_list[j]][1] = np_coordinates[j][1]
         dataset[point_list[j]][2] = np_coordinates[j][2]
       fsdataname = FSDataName(BC_names[i])

       Var = self.fsmesh.GetUnstructDataset(fsdataname).GetValues()
       Var.Fill(0.0)
       for elemIndex in range(nb_cell_surf):
              VarIndex =    Var.MapIndex(elemIndex, 0)
              Var[VarIndex] = dataset[elemIndex][0]
              VarIndex =    Var.MapIndex(elemIndex, 1)
              Var[VarIndex] = dataset[elemIndex][1]
              VarIndex =    Var.MapIndex(elemIndex, 2)
              Var[VarIndex] = dataset[elemIndex][2]
    return

  def checkAndExportFSDMmesh(self):
    isMeshOK = self.fsmesh.Check()
    if isMeshOK :
      print("FSMESH successfully created")
    else:
      print("WARNING : An error was found in the fsmesh check....")
    volumeCellTypes = tuple(FSMeshEnums.CellTypeToString(x) for x in FSUnstructVolumeCellTypes)
    self.fsmesh.ExportMeshTECPLOT(Filename=self.mesh_name.split('.')[0]+"_vol.tp", PrefixDatasetName=True,  ExportCellTypes=volumeCellTypes) or FSError.PrintAndExit()

    surfaceCellTypes = tuple(FSMeshEnums.CellTypeToString(x) for x in FSUnstructSurfaceCellTypes)
    self.fsmesh.ExportMeshTECPLOT(Filename=self.mesh_name.split('.')[0]+"_surf.tp", PrefixDatasetName=True, ZonePerCellAttributeValue=True,
            CellAttribute=FS_AT_CADGroupID, UseCellAttributeValueName=True,
            ExportCellTypes=surfaceCellTypes) or FSError.PrintAndExit()

    self.fsmesh.ExportMeshHDF5(Filename=self.mesh_name.split('.')[0]+".h5") or FSError.PrintAndExit()

    return None

  def convertCGNS2FSDM(self):

    self.recoverInfoMeshCGNS()
    z_NCfaces = None
    if self.nb_vertices != 0:
      if self.conformal==False:
        self.prepareDatasetOfNonConformalFaces()
        z_NCfaces = self.createZoneOfNonConformalFaces()
      self.modifyConnectivityUnstructured()
      self.recoverCoordinatesCGNS()
      self.recoverCGNSConnectivity()

    self.initializeFSMesh(z_NCfaces)

    IBMDatasets = None
    if self.nb_vertices != 0:
      if self.IBM==True: self.initIBMDatasets()
      IBMDatasets = self.recoverPointList2BoundaryMarkers()

    if self.MPI==True:
        self.initBCsFSDMmesh_MPI(IBMDatasets)
        self.serialDeduplicateNodesFSMesh()
    else:
        self.initBCsFSDMmesh(IBMDatasets)
        #self.testDeduplicateNodesFSMesh()

    if self.inmemory:
        return self.fsmesh, self.clac
    else:
        self.checkAndExportFSDMmesh()
    return
  ############## other way around ###############

  def recoverInfoMeshFSDM(self,ghostCells=False):
    if self.fsmesh==None:
      self.fsmesh = FSMesh(self.clac)
      #Import mesh
      if self.mesh_name.split('.')[-1] == 'h5' :
        command = "ImportMeshHDF5"
      elif self.mesh_name.split('.')[-1] == 'grid' or self.mesh_name.split('.')[-1] == 'cdf':
        command = "ImportMeshTAU"

      meshOps = ((command, {"MeshFilename" : self.mesh_name}),
             "PrintInfo",
             # --- create a reasonable partitioning for ZOLTAN (avoids memory bottlenecks) ---
             "RepartitionMeshRCB",
             # ----create local numbering,
             "CreateLocalNumbering",
             # --- main task ---
             ("RepartitionMeshZOLTAN", { "PreserveCellStacks" : True,
                                         "LineSectionsExtractionParameters" :
                                         { "ActiveNodesSelection" : { "CellTypes" : ("Prisms", "Hexahedra","Tetrahedra","Pyramids","Quadrilaterals","Triangles") },
                                           "StartNodesSelection"  : { "CellAttribute" : "CADGroupID",
                                                                      },
                                         },
                                         "GraphExtraction"  : {"GraphType": "CellBased"},
                                          "Approach" : "CoordinateGraphMultilevel",
                                       }),
             "PrintInfo",
             )

      if not self.fsmesh.DoOps(meshOps):
        FSError.PrintAndExit()


    self.nb_vertices = self.fsmesh.GetNOwnedCells(FSMeshEnums.CT_Node)

    #Get information regarding surface cells
    for cell_type in FSUnstructSurfaceCellTypes:
      if self.fsmesh.HasCellType(cell_type) :
        self.fs_surface_cell_types.append(cell_type)
        if ghostCells == False: self.nb_cells_surface += self.fsmesh.GetNOwnedCells(cell_type)
        else: self.nb_cells_surface += self.fsmesh.GetNCells(cell_type)


    #Get information regarding volume cells
    for cell_type in FSUnstructVolumeCellTypes:
      if self.fsmesh.HasCellType(cell_type) :
        self.fs_volume_cell_types.append(cell_type)
        if ghostCells == False: self.nb_cells_volume += self.fsmesh.GetNOwnedCells(cell_type)
        else: self.nb_cells_volume += self.fsmesh.GetNCells(cell_type)
    return

  def recoverCoordinatesFSDM(self):
    node_coordinates = self.fsmesh.GetUnstructDataset("Coordinates").GetValues()
    node_coordinates_numpy = numpy.array(node_coordinates.Buffer(), copy=True)
    self.coordinatesX = numpy.ravel(node_coordinates_numpy[:,0])
    self.coordinatesY = numpy.ravel(node_coordinates_numpy[:,1])
    self.coordinatesZ = numpy.ravel(node_coordinates_numpy[:,2])
    return

  def initializeCGNSCoordinates(self):

    base = Internal.newCGNSBase('Base', 3, 3, parent=self.pytree)
    print('Creating main zone and elements...')
    pytree_zone = Internal.newZone(name = 'Zone1', zsize = [[self.nb_vertices,self.nb_cells_volume,0]], ztype = self.meshType, family = None, parent = base)
    #Create coordinate node
    pytree_coordinates_node = Internal.newGridCoordinates(parent = pytree_zone)

    #Init mesh number of node
    Internal.newDataArray('CoordinateX', value = self.coordinatesX, parent = pytree_coordinates_node)
    if self.invertPlanesYZ==False:
      Internal.newDataArray('CoordinateY', value = self.coordinatesY, parent = pytree_coordinates_node)
      Internal.newDataArray('CoordinateZ', value = self.coordinatesZ, parent = pytree_coordinates_node)
    elif self.invertPlanesYZ==True:
      Internal.newDataArray('CoordinateY', value = self.coordinatesZ, parent = pytree_coordinates_node)
      Internal.newDataArray('CoordinateZ', value = -self.coordinatesY, parent = pytree_coordinates_node)

    return

  def recoverFSDMConnectivity(self,ghostCells=False,surf=True):
    for cell_type in self.fs_volume_cell_types:
      n_cell_owned = self.fsmesh.GetNOwnedCells(cell_type)
      fs_cell2Node = self.fsmesh.GetCell2Node(cell_type)
      numpy_cell2node_not_raveled = numpy.array(fs_cell2Node.Buffer(), copy=True) + 1
      if ghostCells == False: numpy_cell2node_not_raveled = numpy_cell2node_not_raveled[:n_cell_owned]
      self.numpy_cell2node_volume.append(numpy.ravel(numpy_cell2node_not_raveled))

    if surf == True:

      #Get marker list
      fs_boundary_marker_list = self.fsmesh.GetCellAttributeValuesWithNames("CADGroupID")
      np_boundary_marker_list = numpy.array(fs_boundary_marker_list.Buffer(),copy=True)
      unique_markers = {}
      shared_markers = []
      for i,cell_type in enumerate(self.fs_surface_cell_types):
        n_cell_owned = self.fsmesh.GetNOwnedCells(cell_type)
        fs_boundary_markers_celltype = self.fsmesh.GetCellAttribute("CADGroupID",cell_type)
        np_boundary_markers_celltype = numpy.array(fs_boundary_markers_celltype.Buffer(), copy=True)
        if ghostCells == False: np_boundary_markers_celltype = np_boundary_markers_celltype[:n_cell_owned]
        unique_markers[cell_type] =  numpy.unique(np_boundary_markers_celltype)
        if i>0:
            shared_markers = list(set(unique_markers[cell_type]) & set(unique_markers[self.fs_surface_cell_types[i-1]])  )

      np_boundary_markers_celltype_dict = {}
      if shared_markers !=[]:
        print("markers associated with different surface element types:",shared_markers)
        val = 0
        for i,cell_type in enumerate(self.fs_surface_cell_types):
          n_cell_owned = self.fsmesh.GetNOwnedCells(cell_type)
          fs_boundary_markers_celltype = self.fsmesh.GetCellAttribute("CADGroupID",cell_type)
          np_boundary_markers_celltype = numpy.array(fs_boundary_markers_celltype.Buffer(), copy=True)
          if ghostCells == False: np_boundary_markers_celltype = np_boundary_markers_celltype[:n_cell_owned]
          for shared_marker in shared_markers:
            np_boundary_markers_celltype = numpy.where(np_boundary_markers_celltype==shared_marker, shared_marker+val,np_boundary_markers_celltype)
            if (shared_marker+val) not in np_boundary_marker_list:
                np_boundary_marker_list = numpy.append(np_boundary_marker_list,shared_marker+val)
            self.dict_bcs[shared_marker+val] = self.dict_bcs[shared_marker]
          np_boundary_markers_celltype_dict[cell_type] = np_boundary_markers_celltype
          val+=0.1
      else:
        for i,cell_type in enumerate(self.fs_surface_cell_types):
          n_cell_owned = self.fsmesh.GetNOwnedCells(cell_type)
          fs_boundary_markers_celltype = self.fsmesh.GetCellAttribute("CADGroupID",cell_type)
          np_boundary_markers_celltype_dict[cell_type] = numpy.array(fs_boundary_markers_celltype.Buffer(), copy=True)[:n_cell_owned]
      print("\n\n\n\n\n\n")
      #for marker in fs_boundary_marker_list:
      for marker in np_boundary_marker_list:
        indices_vector = []
        offset = 0
        for cell_type in self.fs_surface_cell_types:
          n_cell_owned = self.fsmesh.GetNOwnedCells(cell_type)
          n_cell = self.fsmesh.GetNCells(cell_type)
          fs_cell2Node = self.fsmesh.GetCell2Node(cell_type)
          numpy_cell2node_not_raveled = (numpy.array(fs_cell2Node.Buffer(), copy=True) + 1)[:n_cell_owned]
          np_boundary_markers_celltype = np_boundary_markers_celltype_dict[cell_type]
          if len(numpy.ravel(numpy.argwhere(np_boundary_markers_celltype==marker)))>0:
            self.list_names_BCs.append(str(self.fsmesh.GetCellAttributeValueName("CADGroupID", int(marker))))
            indices_vector = numpy.ravel(numpy.argwhere(np_boundary_markers_celltype==marker))
            offset_fsdm = self.fsmesh.GetCellOffset(cell_type) - self.nb_vertices
            print(offset_fsdm,offset)
            self.indices_per_boundary.append(indices_vector+offset)
            self.numpy_cell2node_surface.append(numpy.ravel(numpy_cell2node_not_raveled[indices_vector]))
            self.fs_cell_types_BCs.append(cell_type)
            self.fs_markers.append(marker)
          offset = n_cell_owned
    return

  def buildCGNSConnectivity(self, surf=True):
    pytree_zone = Internal.getZones(self.pytree)[0]

    counter_cells = 1
    for i,fs_cell_type in enumerate(self.fs_volume_cell_types):
      nb_vertex_per_cell = int(FSMeshEnums.CellTypeToString(fs_cell_type)[-1])
      nb_cell_current_elt = self.numpy_cell2node_volume[i].shape[0]//nb_vertex_per_cell
      Internal.newElements(name = "GridElements_"+self.CellTypesFS2Cassiopee(fs_cell_type), etype = self.CellTypesFS2Cassiopee(fs_cell_type), econnectivity = self.numpy_cell2node_volume[i], erange = [counter_cells, counter_cells+nb_cell_current_elt-1], eboundary = 0, parent = pytree_zone)
      counter_cells += nb_cell_current_elt
    if surf == True:
      counter_cells = self.nb_cells_volume+1
      for idx,fs_cell_type in enumerate(self.fs_cell_types_BCs):

        nb_vertex_per_cell = int(FSMeshEnums.CellTypeToString(fs_cell_type)[-1])

        nb_cell_current_boundary = len(self.numpy_cell2node_surface[idx])//nb_vertex_per_cell

        bcname = self.list_names_BCs[idx].split(".")[0]+"."+self.CellTypesFS2Cassiopee(fs_cell_type)+"_"+str(int(self.fs_markers[idx]))
        ELT = Internal.newElements(name = bcname, etype = self.CellTypesFS2Cassiopee(fs_cell_type),erange =[counter_cells, counter_cells+nb_cell_current_boundary-1], econnectivity = self.numpy_cell2node_surface[idx],  eboundary = nb_cell_current_boundary, parent = pytree_zone)

        bctype = self.dict_bcs[self.fs_markers[idx]]
        C._addBC2Zone(pytree_zone,bcname,bctype, elementRange=[counter_cells,counter_cells+nb_cell_current_boundary-1])
        zone_bc =  Internal.getNodeFromType(pytree_zone,"ZoneBC_t")
        lastbcname = C.getLastBCName(bcname)
        node_bc = Internal.getNodeFromName(zone_bc,lastbcname)
        node_bc[0] = bcname
        boundarystatedataset=Internal.createNode('BCDataSet','BCDataSet_t',parent=node_bc,value='Null')
        boundarystate = Internal.createNode("Boundary",'BCData_t',parent=boundarystatedataset)
        boundarystate[2].append(["BoundaryMarker",int(self.fs_markers[idx])*numpy.ones(nb_cell_current_boundary), [], 'DataArray_t'])
        counter_cells += nb_cell_current_boundary
    return

  def recoverFlowSolution(self):
    if self.whichDatasets == []:
        print("All the available datasets will be converted in the cgns pytree.")

    for datasetName in self.fsmesh.GetUnstructDatasetNames():
      nameAugState = str(datasetName)
      if nameAugState != "Coordinates" and (self.whichDatasets==[] or (nameAugState in self.whichDatasets)):
          print("Dataset:",nameAugState)
          unstructDataset = self.fsmesh.GetUnstructDataset(nameAugState)
          flow_solution_values = unstructDataset.GetValues()
          flow_solution_names = unstructDataset.GetNames()
          types  = unstructDataset.GetCellTypes()
          types_numpy = numpy.array(types.Buffer(),copy=True)

          indices_GC = numpy.empty(0,dtype=numpy.int32)
          totalNCells = 0
          for idx,cell_type in enumerate(types):
             NOwnedCells = self.fsmesh.GetNOwnedCells(cell_type)
             NGhostCells = self.fsmesh.GetNGhostCells(cell_type)
             NCells = self.fsmesh.GetNCells(cell_type)
             indices_GC = numpy.concatenate([indices_GC,numpy.arange(totalNCells+NOwnedCells,totalNCells+NCells)])
             totalNCells += NCells
          flow_solution_values_numpy = numpy.array(flow_solution_values.Buffer(),copy=True)
          flow_solution_values_numpy = numpy.delete(flow_solution_values_numpy, indices_GC,axis=0)

          flow_solution_names_string = []
          for name in flow_solution_names:
            flow_solution_names_string.append(str(name))

          if cell_type in self.fs_volume_cell_types:
              zone = Internal.getZones(self.pytree)[0]
              FS = Internal.newFlowSolution(name='FlowSolution#Centers', gridLocation='CellCenter', parent=zone)
              for i,augState_name in enumerate(flow_solution_names_string):
                Internal.newDataArray(nameAugState+"."+augState_name, value = flow_solution_values_numpy[:,i], parent = FS)
          elif cell_type in self.fs_surface_cell_types:
              zone_bc = Internal.getNodeFromType(self.pytree,"ZoneBC_t")
              if zone_bc != None:
                nodes_bcs = Internal.getNodesFromType(zone_bc,"BC_t")
                for i,node_bc in enumerate(nodes_bcs):
                    boundarystatedataset=Internal.getNodeFromType(node_bc,'BCDataSet_t')
                    boundarystate = Internal.getNodeFromType(boundarystatedataset,'BCData_t')
                    for j,boundary_values_name in enumerate(flow_solution_names_string):
                        Internal.newDataArray(boundary_values_name, value = flow_solution_values_numpy[:,j][self.indices_per_boundary[i]], parent = boundarystate)

    return

  def exportCGNSmesh(self):
    #Done : write cgns file
    C.convertPyTree2File(self.pytree, self.mesh_name.split('.')[0]+".cgns")
    C.convertPyTree2File(self.pytree, self.mesh_name.split('.')[0]+".plt")
    Internal.printTree(self.pytree)
    return None

  def convertFSDM2CGNS(self):
    self.recoverInfoMeshFSDM(ghostCells=False)
    self.recoverCoordinatesFSDM()
    self.initializeCGNSCoordinates()
    self.recoverFSDMConnectivity(ghostCells=False,surf=True)
    self.buildCGNSConnectivity(surf=True)
    if self.keepFlowSolution:
      self.recoverFlowSolution()
    if Cmpi.size>1:
      Cmpi._setProc(self.pytree, Cmpi.rank)
      zones = Internal.getZones(self.pytree)
      for z in zones:
        z[0] = z[0]+str(Cmpi.rank)
    if self.inmemory: return None
    else: self.exportCGNSmesh()
    return None

  def convertFSDM2CGNSforOverset(self):
    self.recoverInfoMeshFSDM(ghostCells=True)
    self.recoverCoordinatesFSDM()
    self.initializeCGNSCoordinates()
    self.recoverFSDMConnectivity(ghostCells=True,surf=False)
    self.buildCGNSConnectivity(surf=False)
    if self.inmemory: return None
    else: self.exportCGNSmesh()

  def initializePseudoCell_QuadNQuad(self,z_NCfaces):
    ##### CREATE QUAD2QUAD CONNECTIVITY for OCTREE meshes #########

    nonconformal_faces_local = Internal.getNodeFromName(z_NCfaces,"ElementConnectivity")[1]-1
    len_NCF = len(nonconformal_faces_local)//4

    nonconformal_faces_nodes_x = Internal.getNodeFromName(z_NCfaces,"CoordinateX")[1]
    nonconformal_faces_nodes_y = Internal.getNodeFromName(z_NCfaces,"CoordinateY")[1]
    nonconformal_faces_nodes_z = Internal.getNodeFromName(z_NCfaces,"CoordinateZ")[1]

    len_nodes_NCF = len(nonconformal_faces_nodes_x)

    nonconformal_faces_nodes = numpy.hstack([nonconformal_faces_nodes_x.reshape((len_nodes_NCF,1)),nonconformal_faces_nodes_y.reshape((len_nodes_NCF,1)),nonconformal_faces_nodes_z.reshape((len_nodes_NCF,1))])
    nonconformal_faces_ctr_x,nonconformal_faces_ctr_y,nonconformal_faces_ctr_z = computeCellCenters_Quads(nonconformal_faces_nodes_x, nonconformal_faces_nodes_y, nonconformal_faces_nodes_z, nonconformal_faces_local)

    nonconformal_faces_ctr = numpy.hstack([nonconformal_faces_ctr_x.reshape((len_NCF,1)),nonconformal_faces_ctr_y.reshape((len_NCF,1)),nonconformal_faces_ctr_z.reshape((len_NCF,1))])


    nonconformal_faces_local = numpy.reshape(nonconformal_faces_local,(len_NCF,4))
    tic = time.perf_counter()
    if self.dimPb == 2:   listQuadNQuad_local, rest_faces = create_Quad2Quad(nonconformal_faces_nodes,nonconformal_faces_local,nonconformal_faces_ctr)#plane,tol
    elif self.dimPb == 3: listQuadNQuad_local, rest_faces = create_Quad4Quad(nonconformal_faces_nodes,nonconformal_faces_local,nonconformal_faces_ctr)#plane,tol
    toc = time.perf_counter()
    print("time for hanging nodes search: ", toc-tic)
    Internal._rmNodesFromType(self.pytree,"Elements_t")
    hook = C.createHook(self.pytree, 'nodes')
    ids = C.identifyNodes(hook, z_NCfaces)
    ids = ids[ids!=-1]-1
    listQuadNQuad = ids[listQuadNQuad_local]

    fs_cell2node = FSIntArray(listQuadNQuad.shape[0],listQuadNQuad.shape[1])
    numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), listQuadNQuad, casting='unsafe')
    if self.dimPb == 2:   self.fsmesh.InitUnstructCells(FSMeshEnums.PCT_Quad2Quad, fs_cell2node, False)
    elif self.dimPb == 3: self.fsmesh.InitUnstructCells(FSMeshEnums.PCT_Quad4Quad, fs_cell2node, False)
    return

  def initializePseudoCell_QuadNQuad_MPI(self,z_NCfaces):

    myID = self.clac.ProcID()
    if z_NCfaces != None:
      z_NCfaces[0] = z_NCfaces[0]+str(Cmpi.rank)
      nonconformal_faces_nodes_x = Internal.getNodeFromName(z_NCfaces,"CoordinateX")[1]
      nonconformal_faces_nodes_y = Internal.getNodeFromName(z_NCfaces,"CoordinateY")[1]
      nonconformal_faces_nodes_z = Internal.getNodeFromName(z_NCfaces,"CoordinateZ")[1]
    else:
      nonconformal_faces_nodes_x = numpy.empty(0)
      nonconformal_faces_nodes_y = numpy.empty(0)
      nonconformal_faces_nodes_z = numpy.empty(0)


    allgathered_x = Cmpi.gather(nonconformal_faces_nodes_x,0)
    allgathered_y = Cmpi.gather(nonconformal_faces_nodes_y,0)
    allgathered_z = Cmpi.gather(nonconformal_faces_nodes_z,0)

    len_NCF = len(nonconformal_faces_nodes_x)

    del nonconformal_faces_nodes_x; del nonconformal_faces_nodes_y; del nonconformal_faces_nodes_z

    if self.dimPb==2:  fs_cell_type = FSMeshEnums.PCT_Quad2Quad
    elif self.dimPb==3: fs_cell_type = FSMeshEnums.PCT_Quad4Quad

    self.initializeCell2Proc(fs_cell_type,len_NCF)
    if z_NCfaces != None:
      nonconformal_faces_local = Internal.getNodeFromName(z_NCfaces,"ElementConnectivity")[1]-1 + self.cell2Proc[fs_cell_type][myID]
    else:
      nonconformal_faces_local = numpy.empty(0,dtype=numpy.int64)

    allgathered_nonconformal_faces_local = Cmpi.gather(nonconformal_faces_local,0)
    del nonconformal_faces_local;

    listQuadNQuad_local = []

    if Cmpi.rank==0:

      allgathered_x = numpy.concatenate(allgathered_x)
      allgathered_y = numpy.concatenate(allgathered_y)
      allgathered_z = numpy.concatenate(allgathered_z)
      allgathered_nonconformal_faces_local = numpy.concatenate(allgathered_nonconformal_faces_local)
      allgathered_nodes = numpy.hstack([allgathered_x.reshape(len(allgathered_x),1), allgathered_y.reshape(len(allgathered_x),1), allgathered_z.reshape(len(allgathered_x),1)])

      if self.dimPb==2 and self.MPI==True:

          cmpIdx = lambda a, b : cmp(allgathered_nodes[a], allgathered_nodes[b])
          idx_sorted = sorted(range(len(allgathered_nodes)), key=cmp_to_key(cmpIdx))

          nnodes_old = len(allgathered_nodes)

          node_coordinates_numpy_sorted = allgathered_nodes[idx_sorted]


          unique_coords = numpy.empty((nnodes_old,3))
          dup2dedup = numpy.empty((nnodes_old))
          dedup2dup = numpy.empty((nnodes_old),dtype=int)
          previous = None
          j = -1

          for i in range(nnodes_old):
              if i==0 or (abs(previous-node_coordinates_numpy_sorted[i])>(10**(-10))).any():
                  j=j+1
                  unique_coords[j] = node_coordinates_numpy_sorted[i]
                  previous = node_coordinates_numpy_sorted[i]
                  dup2dedup[idx_sorted[i]] = j
                  dedup2dup[j] = idx_sorted[i]
              else:
                  dup2dedup[idx_sorted[i]] = j

          for i in range(len(allgathered_nonconformal_faces_local)):
            allgathered_nonconformal_faces_local[i] = dup2dedup[allgathered_nonconformal_faces_local[i]]

          unique_coords.resize((j+1,3))

          allgathered_nodes = unique_coords

          del unique_coords

      len_NCF = len(allgathered_nonconformal_faces_local)//4

      len_nodes_allgathered = len(allgathered_nodes)

      nonconformal_faces_ctr_x,nonconformal_faces_ctr_y,nonconformal_faces_ctr_z = computeCellCenters_Quads(allgathered_nodes[:,0], allgathered_nodes[:,1], allgathered_nodes[:,2], allgathered_nonconformal_faces_local)

      nonconformal_faces_ctr = numpy.hstack([nonconformal_faces_ctr_x.reshape((len_NCF,1)),nonconformal_faces_ctr_y.reshape((len_NCF,1)),nonconformal_faces_ctr_z.reshape((len_NCF,1))])

      del nonconformal_faces_ctr_x; del nonconformal_faces_ctr_y; del nonconformal_faces_ctr_z

      nonconformal_faces_local = numpy.reshape(allgathered_nonconformal_faces_local,(len_NCF,4))

      del allgathered_nonconformal_faces_local

      print(Cmpi.rank, "nonconformal_faces_local",nonconformal_faces_local.shape[0])
      tic = time.perf_counter()

      if self.dimPb == 2:   listQuadNQuad_local, rest_faces = create_Quad2Quad_MPI(allgathered_nodes,nonconformal_faces_local,nonconformal_faces_ctr)
      elif self.dimPb == 3: listQuadNQuad_local, rest_faces = create_Quad4Quad(allgathered_nodes,nonconformal_faces_local,nonconformal_faces_ctr)
      toc = time.perf_counter()
      print(Cmpi.rank,"time for hanging nodes search: ", toc-tic)
      print(Cmpi.rank, "size listQuadNQuad",listQuadNQuad_local.shape[0])

      if self.dimPb==2 and self.MPI==True:
          listQuadNQuad_local = dedup2dup[listQuadNQuad_local]

    print("before initializecell2proc")
    if self.MPI==True: self.initializeCell2Proc(fs_cell_type,len(listQuadNQuad_local))
    print("before hook")
    Internal._rmNodesFromType(self.pytree,"Elements_t")
    if z_NCfaces!=None:
      hook = C.createHook(self.pytree, 'nodes')
      ids = C.identifyNodes(hook, z_NCfaces)
    else:
      ids = numpy.empty(0,dtype=numpy.int64)
    if self.MPI==True:
      ids = ids[ids>-1]-1 + self.cell2Proc[1][myID]
      ids_gathered = numpy.concatenate(Cmpi.allgather(ids))
    else:
        ids_gathered = ids[ids>-1]-1
    if self.MPI==True:
      if Cmpi.rank==0:

        listQuadNQuad = ids_gathered[listQuadNQuad_local]
        fs_cell2node = FSIntArray(listQuadNQuad.shape[0],listQuadNQuad.shape[1])
        numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), listQuadNQuad, casting='unsafe')
      else:
        print('continuing on procs!=0')
        fs_cell2node = FSIntArray(0,FSCellInfo.NNodes(fs_cell_type))

      self.fsmesh.InitUnstructCells(fs_cell_type,self.cell2Proc[fs_cell_type], fs_cell2node, False)
    else:
      listQuadNQuad = ids_gathered[listQuadNQuad_local]
      fs_cell2node = FSIntArray(listQuadNQuad.shape[0],listQuadNQuad.shape[1])
      numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), listQuadNQuad, casting='unsafe')
      self.fsmesh.InitUnstructCells(fs_cell_type, fs_cell2node, False)

    return None

  def _addBC2ZoneLoc(self,z, bndName, bndType, zbc, loc='FaceCenter', zdnrName=None):
    s = bndType.split(':')
    bndType1 = s[0]
    if len(s) > 1: bndType2 = s[1]
    else: bndType2 = ''

    # Analyse zone zbc
    dims = Internal.getZoneDim(zbc)
    neb = dims[2] # nbre d'elts de zbc

    eltType, nf = Internal.eltName2EltNo(dims[3]) # type d'elements de zbc
    # On cherche l'element max dans les connectivites de z
    maxElt = 0
    connects = Internal.getNodesFromType(z, 'Elements_t')
    for cn in connects:
      r = Internal.getNodeFromName1(cn, 'ElementRange')
      m = r[1][1]
      maxElt = max(maxElt, m)

    # on cree un nouveau noeud connectivite dans z1 (avec le nom de la zone z2)
    nebb = neb
    node = Internal.createUniqueChild(z, zbc[0], 'Elements_t', value=[eltType,nebb])
    Internal.createUniqueChild(node, 'ElementRange', 'IndexRange_t',
                             value=[maxElt+1,maxElt+neb])
    oldc = Internal.getNodeFromName2(zbc, 'ElementConnectivity')[1]
    newc = numpy.copy(oldc)
    hook = C.createHook(z, 'nodes')
    ids = C.identifyNodes(hook, zbc)
    newc[:] = ids[oldc[:]-1]
    faceList = [i for i in range(1,neb+1)]
    Internal.createUniqueChild(node, 'ElementConnectivity', 'DataArray_t', value=newc)

    zoneBC = Internal.createUniqueChild(z, 'ZoneBC', 'ZoneBC_t')
    if len(s)==1:
      info = Internal.createChild(zoneBC, bndName, 'BC_t', value=bndType)
    else: # familyspecified
      info = Internal.createChild(zoneBC, bndName, 'BC_t', value=bndType1)
      Internal.createUniqueChild(info, 'FamilyName', 'FamilyName_t',
                                 value=bndType2)

    Internal.createUniqueChild(info, 'GridLocation', 'GridLocation_t',
                              value='FaceCenter')
    if isinstance(faceList, numpy.ndarray): r = faceList
    else: r = numpy.array(faceList, dtype=numpy.int32)
    r = r.reshape((1,r.size), order='F')
    info[2].append([Internal.__FACELIST__, r, [], 'IndexArray_t'])
    if bndType == 'Abutting1to1':
      info[2].append(["UserDefined", zdnrName, [], 'UserDefinedData_t'])
    return None

  def testDeduplicateNodesFSMesh(self):

      if self.dimPb==2:
        quadNQuad = 15
      else:
        quadNQuad = 16

      node_coordinates = self.fsmesh.GetUnstructDataset("Coordinates").GetValues()
      node_coordinates_numpy = numpy.array(node_coordinates.Buffer(), copy=True)
      myID = self.clac.GetProcID()

      unique_coords = numpy.empty((0,3))
      dup2dedup = numpy.empty((0),dtype=int)
      npSizePerProc = numpy.empty((0), dtype=numpy.dtype('int'))

      if myID == 0:
        print("Sorting coordinates..")
        cmpIdx = lambda a, b : cmp(node_coordinates_numpy[a], node_coordinates_numpy[b])
        idx_sorted = sorted(range(len(node_coordinates_numpy)), key=cmp_to_key(cmpIdx))

        nnodes_old = len(node_coordinates_numpy)

        node_coordinates_numpy_sorted = node_coordinates_numpy[idx_sorted]
        unique_coords = numpy.empty((nnodes_old,3))
        dup2dedup = numpy.empty((nnodes_old),dtype=int)
        previous = None
        j = -1

        print("Deduplicating coordinates..")
        for i in range(nnodes_old):
            if i==0 or (abs(previous-node_coordinates_numpy_sorted[i])>1e-9).any():
                j=j+1
                unique_coords[j] = node_coordinates_numpy_sorted[i]
                previous = node_coordinates_numpy_sorted[i]
                dup2dedup[idx_sorted[i]] = j
            else:
                dup2dedup[idx_sorted[i]] = j

        unique_coords.resize((j+1,3))

        nnodes_new = j+1
        npSizePerProc = numpy.zeros(self.clac.NProcs(), dtype=numpy.dtype('int'))
        for i in range(self.clac.NProcs()):
          npSizePerProc[i] = len(unique_coords)//self.clac.NProcs()
      dup2dedup = ArrayOps.Broadcast(dup2dedup,self.clac)
      fs_cell_types_NC = self.fs_cell_types if self.conformal else self.fs_cell_types + [quadNQuad]
      cell2NodeDict = {}
      for t in fs_cell_types_NC:
        cell2Node = self.fsmesh.GetCell2Node(t)
        cell2Node_numpy = numpy.array(cell2Node.Buffer(), copy=True)
        cell2Node_numpy = ArrayOps.Gather(cell2Node_numpy,self.clac)
        if myID==0:
          for i in range(len(cell2Node_numpy)):
            cell2Node_numpy[i] = dup2dedup[cell2Node_numpy[i]]
          cell2NodeDict[t] = cell2Node_numpy
        else: cell2NodeDict[t] = numpy.empty((0,FSCellInfo.NNodes(t)))

      fs_boundary_marker_list = self.fsmesh.GetCellAttributeValuesWithNames("CADGroupID")
      np_boundary_marker_list = numpy.array(fs_boundary_marker_list.Buffer(),copy=True)
      np_markers_celltype_dict = {}
      if self.nb_vertices > 0:
        for cell_type in self.fs_surface_cell_types:
          fs_markers_array_cell_type = self.fsmesh.GetCellAttribute("CADGroupID",cell_type)
          np_markers_array_cell_type = numpy.array(fs_markers_array_cell_type.Buffer(),copy=True)
          np_markers_celltype_dict[cell_type] = np_markers_array_cell_type
      else:
        np_markers_array_cell_type = numpy.empty(0)
      np_boundary_marker_list = ArrayOps.Gather(np_boundary_marker_list,self.clac)
      np_markers_array_cell_type = ArrayOps.Gather(np_markers_array_cell_type,self.clac)
      np_markers_celltype_dict_gath = Cmpi.gather(np_markers_celltype_dict)

      if myID==0:
        np_markers_celltype_dict = {}
        for proc_dict in np_markers_celltype_dict_gath:
            for key in proc_dict.keys():
                if key in np_markers_celltype_dict:
                    np_markers_celltype_dict[key] = numpy.concatenate([np_markers_celltype_dict[key],proc_dict[key]])
                else:
                    np_markers_celltype_dict[key] = proc_dict[key]

        fs_markers_celltype_dict = {}
        for cell_type in self.fs_surface_cell_types:
          fs_markers_array_cell_type = FSIntArray(len(np_markers_array_cell_type))
          for i in range(len(np_markers_array_cell_type)):
              fs_markers_array_cell_type[i] = int(np_markers_array_cell_type[i])
          fs_markers_celltype_dict[cell_type] = fs_markers_array_cell_type

        fs_boundary_marker_list = FSIntArray(len(np_boundary_marker_list))
        for i in range(len(np_boundary_marker_list)):
            fs_boundary_marker_list[i] = int(np_boundary_marker_list[i])

        names = []
        for marker in fs_boundary_marker_list:
          names.append(self.fsmesh.GetCellAttributeValueName("CADGroupID", marker))

      if self.IBM:
        if self.nb_vertices > 0:
            flis_distance = self.fsmesh.GetUnstructDataset("FlisWallDistance").GetValues()
            np_flis_distance = numpy.array(flis_distance.Buffer(), copy=True)
        else:
            np_flis_distance = numpy.empty(0)
        np_flis_distance = ArrayOps.Gather(np_flis_distance,self.clac)

        fsdatanames = ["WallPointCoordinates","DonorPointCoordinates"]

        datasets = []
        for fsdataname in fsdatanames:
            if self.nb_vertices > 0:
              dataset = numpy.array(self.fsmesh.GetUnstructDataset(fsdataname).GetValues().Buffer(),copy=True)
            else:
              dataset = numpy.empty(0)

            dataset = ArrayOps.Gather(dataset,self.clac)
            if myID==0: datasets.append(dataset)

      self.fsmesh.Reset()
      self.fsmesh.BeginInitialization()
      self.fsmesh.InitUnstructNodes(self.nb_vertices)
      for t in fs_cell_types_NC:
          if Cmpi.rank==0:
            fs_cell2node = FSIntArray(cell2NodeDict[t].shape[0],cell2NodeDict[t].shape[1])
          else: fs_cell2node = FSIntArray(0,FSCellInfo.NNodes(t))

          numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), cell2NodeDict[t], casting='unsafe')
          self.fsmesh.InitUnstructCells(t,fs_cell2node, True)

      self.fsmesh.EndInitialization()
      self.initEmptyDatasetOfCoordinatesBC([FSDataName.Coordinates()],FSMeshEnums.CT_Node)
      Var = self.fsmesh.GetUnstructDataset(FSDataName.Coordinates()).GetValues()
      for nodeIndex in range(len(unique_coords)):
          VarIndex = Var.MapIndex(nodeIndex,0)
          Var[VarIndex] = unique_coords[nodeIndex][0]
          VarIndex = Var.MapIndex(nodeIndex,1)
          Var[VarIndex] = unique_coords[nodeIndex][1]
          VarIndex = Var.MapIndex(nodeIndex,2)
          Var[VarIndex] = unique_coords[nodeIndex][2]

      self.nb_cells_surface = 0
      self.nb_cells_volume = 0
      self.fs_volume_cell_types = []
      self.fs_surface_cell_types = []
      self.recoverInfoMeshFSDM(ghostCells=False)

      if myID==0:
        for (name,marker) in zip(names,fs_boundary_marker_list):
          self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, name)
        for cell_type in self.fs_surface_cell_types:
          self.fsmesh.InitCellAttribute(FS_AT_CADGroupID, cell_type, fs_markers_celltype_dict[cell_type])
        if self.IBM:
            flis_distance = FSFloatArray(self.nb_cells_volume)
            for i in range(self.nb_cells_volume):
               flis_distance[i] = float(np_flis_distance[i])
            quantity_name = "FlisWallDistance"
            quantityNames = FSStringArray(1)
            quantityNames[0] = quantity_name
            quantitySpecs = FSDataSpecArray(1)
            fsarray_volume_cell_types = FSIntArray(len(self.fs_volume_cell_types))
            numpy.copyto(numpy.array(fsarray_volume_cell_types.Buffer(), copy=False), self.fs_volume_cell_types, casting='unsafe')

            self.fsmesh.InitUnstructDataset(quantity_name, FSDatasetInfo(quantityNames, quantitySpecs, fsarray_volume_cell_types))
            Var = self.fsmesh.GetUnstructDataset(quantity_name).GetValues()
            for elemIndex in range(self.nb_cells_volume):
                    VarIndex =  Var.MapIndex(elemIndex, 0)
                    Var[VarIndex] = flis_distance[elemIndex]
            fsarray_surface_cell_types = FSIntArray(len(self.fs_surface_cell_types))
            numpy.copyto(numpy.array(fsarray_surface_cell_types.Buffer(), copy=False), self.fs_surface_cell_types, casting='unsafe')

            self.initEmptyDatasetOfCoordinatesBC(fsdatanames,fsarray_surface_cell_types)
            for (fsdataname,dataset) in zip(fsdatanames,datasets):
              Var = self.fsmesh.GetUnstructDataset(fsdataname).GetValues()
              for elemIndex in range(self.nb_cells_surface):
                     VarIndex =    Var.MapIndex(elemIndex, 0)
                     Var[VarIndex] = dataset[elemIndex][0]
                     VarIndex =    Var.MapIndex(elemIndex, 1)
                     Var[VarIndex] = dataset[elemIndex][1]
                     VarIndex =    Var.MapIndex(elemIndex, 2)
                     Var[VarIndex] = dataset[elemIndex][2]

      return

  def serialDeduplicateNodesFSMesh(self):

      if self.dimPb==2:
        quadNQuad = 15
      else:
        quadNQuad = 16

      node_coordinates = self.fsmesh.GetUnstructDataset("Coordinates").GetValues()
      node_coordinates_numpy = numpy.array(node_coordinates.Buffer(), copy=True)
      myID = self.clac.GetProcID()

      node_coordinates_numpy = ArrayOps.Gather(node_coordinates_numpy,self.clac)

      unique_coords = numpy.empty((0,3))
      dup2dedup = numpy.empty((0),dtype=int)
      npSizePerProc = numpy.empty((0), dtype=numpy.dtype('int'))

      if myID == 0:
        print("Sorting coordinates..")
        cmpIdx = lambda a, b : cmp(node_coordinates_numpy[a], node_coordinates_numpy[b])
        idx_sorted = sorted(range(len(node_coordinates_numpy)), key=cmp_to_key(cmpIdx))

        nnodes_old = len(node_coordinates_numpy)

        node_coordinates_numpy_sorted = node_coordinates_numpy[idx_sorted]
        unique_coords = numpy.empty((nnodes_old,3))
        dup2dedup = numpy.empty((nnodes_old),dtype=int)
        previous = None
        j = -1

        print("Deduplicating coordinates..")
        for i in range(nnodes_old):
            if i==0 or (abs(previous-node_coordinates_numpy_sorted[i])>1e-9).any():
                j=j+1
                unique_coords[j] = node_coordinates_numpy_sorted[i]
                previous = node_coordinates_numpy_sorted[i]
                dup2dedup[idx_sorted[i]] = j
            else:
                dup2dedup[idx_sorted[i]] = j

        unique_coords.resize((j+1,3))

        nnodes_new = j+1
        npSizePerProc = numpy.zeros(self.clac.NProcs(), dtype=numpy.dtype('int'))
        for i in range(self.clac.NProcs()):
          npSizePerProc[i] = len(unique_coords)//self.clac.NProcs()
      cell2Proc_nodes = initializeCell2ProcOutsideClass(self.clac,FSMeshEnums.CT_Node,len(unique_coords))
      dup2dedup = ArrayOps.Broadcast(dup2dedup,self.clac)
      fs_cell_types_NC = self.fs_cell_types if self.conformal else self.fs_cell_types + [quadNQuad]
      cell2NodeDict = {}
      for t in fs_cell_types_NC:
        cell2Node = self.fsmesh.GetCell2Node(t)
        cell2Node_numpy = numpy.array(cell2Node.Buffer(), copy=True)
        cell2Node_numpy = ArrayOps.Gather(cell2Node_numpy,self.clac)
        if myID==0:
          for i in range(len(cell2Node_numpy)):
            cell2Node_numpy[i] = dup2dedup[cell2Node_numpy[i]]
          cell2NodeDict[t] = cell2Node_numpy
        else: cell2NodeDict[t] = numpy.empty((0,FSCellInfo.NNodes(t)))

      fs_boundary_marker_list = self.fsmesh.GetCellAttributeValuesWithNames("CADGroupID")
      np_boundary_marker_list = numpy.array(fs_boundary_marker_list.Buffer(),copy=True)
      np_markers_celltype_dict = {}
      if self.nb_vertices > 0:
        for cell_type in self.fs_surface_cell_types:
          fs_markers_array_cell_type = self.fsmesh.GetCellAttribute("CADGroupID",cell_type)
          np_markers_array_cell_type = numpy.array(fs_markers_array_cell_type.Buffer(),copy=True)
          np_markers_celltype_dict[cell_type] = np_markers_array_cell_type
      else:
        np_markers_array_cell_type = numpy.empty(0)
      np_boundary_marker_list = ArrayOps.Gather(np_boundary_marker_list,self.clac)
      np_markers_array_cell_type = ArrayOps.Gather(np_markers_array_cell_type,self.clac)
      np_markers_celltype_dict_gath = Cmpi.gather(np_markers_celltype_dict)

      if myID==0:
        np_markers_celltype_dict = {}
        for proc_dict in np_markers_celltype_dict_gath:
            for key in proc_dict.keys():
                if key in np_markers_celltype_dict:
                    np_markers_celltype_dict[key] = numpy.concatenate([np_markers_celltype_dict[key],proc_dict[key]])
                else:
                    np_markers_celltype_dict[key] = proc_dict[key]

        fs_markers_celltype_dict = {}
        for cell_type in self.fs_surface_cell_types:
          fs_markers_array_cell_type = FSIntArray(len(np_markers_array_cell_type))
          for i in range(len(np_markers_array_cell_type)):
              fs_markers_array_cell_type[i] = int(np_markers_array_cell_type[i])
          fs_markers_celltype_dict[cell_type] = fs_markers_array_cell_type

        fs_boundary_marker_list = FSIntArray(len(np_boundary_marker_list))
        for i in range(len(np_boundary_marker_list)):
            fs_boundary_marker_list[i] = int(np_boundary_marker_list[i])

        names = []
        for marker in fs_boundary_marker_list:
          names.append(self.fsmesh.GetCellAttributeValueName("CADGroupID", marker))

      if self.IBM:
        if self.nb_vertices > 0:
            flis_distance = self.fsmesh.GetUnstructDataset("FlisWallDistance").GetValues()
            np_flis_distance = numpy.array(flis_distance.Buffer(), copy=True)
        else:
            np_flis_distance = numpy.empty(0)
        np_flis_distance = ArrayOps.Gather(np_flis_distance,self.clac)

        fsdatanames = ["WallPointCoordinates","DonorPointCoordinates"]

        datasets = []
        for fsdataname in fsdatanames:
            if self.nb_vertices > 0:
              dataset = numpy.array(self.fsmesh.GetUnstructDataset(fsdataname).GetValues().Buffer(),copy=True)
            else:
              dataset = numpy.empty(0)

            dataset = ArrayOps.Gather(dataset,self.clac)
            if myID==0: datasets.append(dataset)

      self.fsmesh.Reset()
      self.fsmesh.BeginInitialization()
      self.fsmesh.InitUnstructNodes(cell2Proc_nodes)
      for t in fs_cell_types_NC:
          if Cmpi.rank==0:
            fs_cell2node = FSIntArray(cell2NodeDict[t].shape[0],cell2NodeDict[t].shape[1])
          else: fs_cell2node = FSIntArray(0,FSCellInfo.NNodes(t))

          cell2Proc = initializeCell2ProcOutsideClass(self.clac,t,len(cell2NodeDict[t]))
          numpy.copyto(numpy.array(fs_cell2node.Buffer(), copy=False), cell2NodeDict[t], casting='unsafe')
          self.fsmesh.InitUnstructCells(t,cell2Proc,fs_cell2node, True)

      self.fsmesh.EndInitialization()
      self.initEmptyDatasetOfCoordinatesBC([FSDataName.Coordinates()],FSMeshEnums.CT_Node)
      Var = self.fsmesh.GetUnstructDataset(FSDataName.Coordinates()).GetValues()
      for nodeIndex in range(len(unique_coords)):
          VarIndex = Var.MapIndex(nodeIndex,0)
          Var[VarIndex] = unique_coords[nodeIndex][0]
          VarIndex = Var.MapIndex(nodeIndex,1)
          Var[VarIndex] = unique_coords[nodeIndex][1]
          VarIndex = Var.MapIndex(nodeIndex,2)
          Var[VarIndex] = unique_coords[nodeIndex][2]

      self.nb_cells_surface = 0
      self.nb_cells_volume = 0
      self.fs_volume_cell_types = []
      self.fs_surface_cell_types = []
      self.recoverInfoMeshFSDM(ghostCells=False)

      if myID==0:
        for (name,marker) in zip(names,fs_boundary_marker_list):
          self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, name)
        for cell_type in self.fs_surface_cell_types:
          self.fsmesh.InitCellAttribute(FS_AT_CADGroupID, cell_type, fs_markers_celltype_dict[cell_type])
        if self.IBM:
            flis_distance = FSFloatArray(self.nb_cells_volume)
            for i in range(self.nb_cells_volume):
               flis_distance[i] = float(np_flis_distance[i])
            quantity_name = "FlisWallDistance"
            quantityNames = FSStringArray(1)
            quantityNames[0] = quantity_name
            quantitySpecs = FSDataSpecArray(1)
            fsarray_volume_cell_types = FSIntArray(len(self.fs_volume_cell_types))
            numpy.copyto(numpy.array(fsarray_volume_cell_types.Buffer(), copy=False), self.fs_volume_cell_types, casting='unsafe')

            self.fsmesh.InitUnstructDataset(quantity_name, FSDatasetInfo(quantityNames, quantitySpecs, fsarray_volume_cell_types))
            Var = self.fsmesh.GetUnstructDataset(quantity_name).GetValues()
            for elemIndex in range(self.nb_cells_volume):
                    VarIndex =  Var.MapIndex(elemIndex, 0)
                    Var[VarIndex] = flis_distance[elemIndex]
            fsarray_surface_cell_types = FSIntArray(len(self.fs_surface_cell_types))
            numpy.copyto(numpy.array(fsarray_surface_cell_types.Buffer(), copy=False), self.fs_surface_cell_types, casting='unsafe')

            self.initEmptyDatasetOfCoordinatesBC(fsdatanames,fsarray_surface_cell_types)
            for (fsdataname,dataset) in zip(fsdatanames,datasets):
              Var = self.fsmesh.GetUnstructDataset(fsdataname).GetValues()
              for elemIndex in range(self.nb_cells_surface):
                     VarIndex =    Var.MapIndex(elemIndex, 0)
                     Var[VarIndex] = dataset[elemIndex][0]
                     VarIndex =    Var.MapIndex(elemIndex, 1)
                     Var[VarIndex] = dataset[elemIndex][1]
                     VarIndex =    Var.MapIndex(elemIndex, 2)
                     Var[VarIndex] = dataset[elemIndex][2]

      return


  def deleteUselessVariables(self):
    del self.conformal, self.dimPb, self.IBM, self.IBM_parameters, self.invertPlanesYZ
    del self.meshType, self.nb_vertices, self.fs_surface_cell_types, self.fs_volume_cell_types, self.fs_cell_types, self.fs_cell_types_BCs
    del self.coordinatesX, self.coordinatesY, self.coordinatesZ, self.numpy_cell2node, self.numpy_cell2node_volume, self.numpy_cell2node_surface, self.numpy_range, self.indices_per_boundary
    del self.fsmesh, self.list_names_BCs,  self.fs_markers, self.boundary_marker_to_bc_name, self.boundary_marker_to_point_list, self.dict_bcs

  def convertMonozoneME2Ngon4FFD(self,reorient=True,mergeOnProc0=False,tol=1e-6):

    if Internal.getZones(self.pytree) == []:
      if Cmpi.size==1:
        print(self.mesh_name.split(".")[0]+".cgns")
        self.pytree = C.convertFile2PyTree(self.mesh_name.split(".")[0]+".cgns")
      else:
        self.pytree = Cmpi.convertFile2PyTree(self.mesh_name.split(".")[0]+".cgns",proc=Cmpi.rank)
    else:
      self.deleteUselessVariables()
    # breaking in one zone per type of volume element
    print("Breaking connectivity..")
    t3 = C.breakConnectivity(self.pytree)

    # Attention: bug solved in Cassiopee 4.0 -> lines from 1130 to 1138 must be deleted with the new release
    zones = Internal.getZones(t3)
    for zone in zones:
        elts = Internal.getNodesFromType(zone,"Elements_t")
        bcs = Internal.getNodesFromType(zone,"BC_t")
        for n_bc,(elt,bc) in enumerate(zip(elts[1:],bcs)):
          ER_el = Internal.getNodeFromName(elt,"ElementRange")
          ER_bc = Internal.getNodeFromName(bc,"ElementRange")
          if (ER_el[1] != ER_bc[1][0]).all():
              ER_bc[1][0] = ER_el[1]
    C._deleteEmptyZones(t3)

    print("Converting array 2 NGon..")
    # convert multielement in Ngon
    self.pytree = C.convertArray2NGon(t3,recoverBC=False)
    #self.pytree = G.close(self.pytree)
    C._deleteFlowSolutions__(t3)

    # save the BCs in the correct format for the recoverBC at the end of the function
    print("Getting BCs..")
    (BCs,BCNames,BCTypes) = C.getBCs(t3)
    true_len_BCs = len(BCs)//len(Internal.getZones(t3))

    del t3
    BCs = BCs[:true_len_BCs]
    BCNames = BCNames[:true_len_BCs]
    BCTypes = BCTypes[:true_len_BCs]
    for BC in BCs:
      Elts = Internal.getNodesFromType(BC,"Elements_t")
      if len(Elts)>1 and Elts[0][0].startswith("GridElements"):
        Internal._rmNodesByName(BC,Elts[0][0])

    # Multizone NGON getting rid of useless nodes
    zones = Internal.getZones(self.pytree)
    for z in zones:
      Elts = Internal.getNodesFromType(z,"Elements_t")
      for Elt in Elts:
        if Elt[0]!="NGonElements" and Elt[0]!="NFaceElements":
          Internal._rmNodesByName(z,Elt[0])
    #merging in one single zone
    print("Merging in one single zone..")
    self.pytree = T.merge(self.pytree)
    #changing the names of the zone
    z = Internal.getZones(self.pytree)[0]
    z[0] = "zone."+str(Cmpi.rank)

    #recover BCs
    print("Recovering BCs..")
    if Cmpi.size > 1:
      list_BCs, list_BCNames, list_BCTypes = _recoverBCsC(self.pytree,(BCs,BCNames,BCTypes))
      list_BCs = Cmpi.allgather(list_BCs)
      list_BCNames = Cmpi.allgather(list_BCNames)
      list_BCTypes = Cmpi.allgather(list_BCTypes)
      removeBC = False
    else:
      list_BCs = [BCs]
      list_BCNames = [BCNames]
      list_BCTypes = [BCTypes]
      removeBC = True

    for (BCs_h,BCNames_h,BCTypes_h) in zip(list_BCs,list_BCNames,list_BCTypes):
      _recoverBCs(self.pytree,(BCs_h,BCNames_h,BCTypes_h),tol=tol,removeBC=removeBC)

    n_assigned_bcs = 0
    bcs = Internal.getNodesFromType(self.pytree,"BC_t")
    for bc in bcs:
        PL = Internal.getNodeFromName(bc,"PointList")[1][0]
        n_assigned_bcs += len(PL)
    n_assigned_bcs_total = sum(Cmpi.allgather(n_assigned_bcs))
    n_cells_surface_total = sum(Cmpi.allgather(self.nb_cells_surface))
    n_cells_volume_total = sum(Cmpi.allgather(self.nb_cells_volume))

    if n_assigned_bcs_total != n_cells_surface_total:
        raise ValueError("Attention! Some BCs are missing. %d/%d surface elements have no BC assigned." %((n_cells_surface_total-n_assigned_bcs_total), n_cells_surface_total))
    else:
     if Cmpi.rank==0:
       print("Recovered all the BCs. %d/%d surface elements have a BC." %(n_assigned_bcs_total,n_cells_surface_total))

    print("Reorienting..")
    if reorient:
      XOR._reorient(self.pytree)

    if mergeOnProc0:
      print("Merging on proc 0..")
      self.pytree = Cmpi.gatherZones(self.pytree,root=0)
      self.pytree = C.newPyTree(['Base',self.pytree])
      self.pytree = T.merge(self.pytree)
      BCs_gathered = Cmpi.gather(BCs,root=0)
      BCNames_gathered = Cmpi.gather(BCNames,root=0)
      BCTypes_gathered = Cmpi.gather(BCTypes,root=0)

      if Cmpi.rank == 0:
        for (BCs,BCNames,BCTypes) in zip( BCs_gathered,BCNames_gathered,BCTypes_gathered):
          C._recoverBCs(self.pytree,(BCs,BCNames,BCTypes),removeBC=False)
        BCnodes = Internal.getNodesFromType(self.pytree,"BC_t")
        Internal._rmNodesFromType(self.pytree,"ZoneBC_t")
        dictBCs = {}
        dictBCsTypes = {}
        for bcnode in BCnodes:
            bcname = bcnode[0]
            bctype = Internal.getValue(bcnode)
            name_2 = bcname.split(".")[0]+"."+bcname.split(".")[1]
            if not name_2 in dictBCs.keys():
              dictBCs[name_2] = [bcnode]
              dictBCsTypes[name_2] = bctype
            else:
              dictBCs[name_2].append(bcnode)
        for key in dictBCs.keys():
          PLs = Internal.getNodesFromName(dictBCs[key],"PointList")
          new_PL = numpy.empty(0,dtype=int)
          for PL in PLs:
              new_PL = numpy.concatenate([new_PL,PL[1][0]])
          #C._addBC2Zone(self.pytree,key,"FamilySpecified:"+key,faceList=new_PL)
          C._addBC2Zone(self.pytree,key,dictBCsTypes[key],faceList=new_PL)
          dictFS = {}
          fscs = Internal.getNodesFromType(dictBCs[key], "BCDataSet_t")

          if fscs != []:
            for fsc in fscs:
              fsc_arrays = Internal.getNodesFromType(fsc,"DataArray_t")
              for fsc_array in fsc_arrays:
                if not fsc_array[0] in dictFS.keys():
                  dictFS[fsc_array[0]] = fsc_array[1]
                else:
                  dictFS[fsc_array[0]] = numpy.concatenate([dictFS[fsc_array[0]],fsc_array[1]])

            newNameOfBC = C.getLastBCName(key)
            bcz = Internal.getNodeFromName(self.pytree, newNameOfBC)
            ds = Internal.newBCDataSet(name='BCDataSet', value='UserDefined',
                                   gridLocation='FaceCenter', parent=bcz)
            d = Internal.newBCData('NeumannData', parent=ds)
            for key_FS in dictFS.keys():
              Internal._createUniqueChild(d, key_FS, "DataArray_t",value=dictFS[key_FS])
      else:
          self.pytree = Internal.newZone(name = "empty",zsize=[[0,0]],ztype="Unstructured")

      self.pytree = C.newPyTree(['Base', self.pytree])
    else:
      self.pytree = C.newPyTree(['Base', self.pytree])

    print("Fixing Flow Solution..")
    if self.keepFlowSolution: _fixNodesForFlowSolution(self.pytree)

    return None

  def merge_BCs_std(self,tol=1e-11):
      print("Merging BCs: one BC per boundary marker")
      t = self.pytree
      BCnodes = Internal.getNodesFromType(t, 'BC_t')
      family_names = []
      family_types = []
      markers = []

      for bcnode in BCnodes:
        name = Internal.getName(bcnode)
        bctype = Internal.getValue(bcnode)
        family_name = name.split('.')[0]
        marker = (name.split('.')[1]).split('_')[1]
        family_name_marker = family_name+"_BoundaryMarker_"+str(marker)
        if marker not in markers:
          markers.append(marker)
          family_names.append(family_name_marker)
          family_types.append(bctype)
        Internal.createChild(bcnode, 'FamilyName', 'FamilyName_t', value=family_name_marker, pos=0)
        Internal.setValue(bcnode, 'FamilySpecified')

      ############
      zbcs = []
      FS = Internal.getNodesFromType(t,"FlowSolution_t")
      Internal._rmNodesByType(t,"FlowSolution_t")
      for family_name in family_names:
        zbc = C.extractBCOfType(t,"FamilySpecified:"+family_name)
        zbc = T.join(zbc)
        zbcs.append(zbc)

      _recoverBCs(t,(zbcs,family_names,family_types),tol=tol,removeBC=True)
      zone = Internal.getZones(t)
      for FS_node in FS:
        Internal._addChild(zone[0], FS_node, pos=-1) # at the end

      return None

  def merge_BCs_FamilySpecified(self):
      print("Specifying FamilySpecified BCs: one FamilyName per boundary marker")
      t = self.pytree
      BCs = []
      BCnodes = Internal.getNodesFromType(t, 'BC_t')

      for bcnode in BCnodes:
          name = Internal.getName(bcnode)
          bctype = Internal.getValue(bcnode)
          family_name = name.split('.')[0]
          BCs.append((name, bctype, family_name))
          Internal.createChild(bcnode, 'FamilyName', 'FamilyName_t', value=family_name, pos=0)
          Internal.setValue(bcnode, 'FamilySpecified')

      base = Internal.getNodeFromType(t, 'CGNSBase_t')

      for bc in BCs:
          if Internal.getNodesFromNameAndType(t, bc[2], 'Family_t') == []:
              family_node = Internal.createNode(bc[2], 'Family_t', parent=base)
              Internal.createChild(family_node, 'FamilyBC', 'FamilyBC_t', value=bc[1], pos=0)
      return None

def _fixNodesForFlowSolution(t):

  FS_C = Internal.getNodeFromName(t,"FlowSolution#Centers")
  flow_solution_names = []
  array_names = []
  dictio = {}
  for node in FS_C[2][1:]:
    first = node[0].split(".")[0]
    second = node[0].split(".")[1]
    if first not in dictio:
        dictio[first] = []
    dictio[first].append(second)

  zone = Internal.getZones(t)
  for flow_solution_name,array_names in dictio.items():
    FS_new = Internal.newFlowSolution(name='FlowSolution#'+flow_solution_name, gridLocation='CellCenter', parent=zone[0])
    for array_name in array_names:
      node = Internal.getNodeFromName(FS_C,flow_solution_name+"."+array_name)
      Internal.newDataArray(array_name, value = node[1], parent = FS_new)
  Internal._rmNodesByName(t,"FlowSolution#Centers")
  return None


def _recoverBCsC(a, T, tol=1.e-11):
  """Recover given BCs on a tree.
  Usage: _recoverBCs(a, (BCs, BCNames, BCTypes), tol)"""
  try:import Post.PyTree as P
  except: raise ImportError("_recoverBCs: requires Post module.")
  C._deleteZoneBC__(a)
  zones = Internal.getZones(a)
  (BCs, BCNames, BCTypes) = T
  for z in zones:
    indicesF = []
    try: f = P.exteriorFaces(z, indices=indicesF)
    except: continue
    indicesF = indicesF[0]
    hook = C.createHook(f, 'elementCenters')
    list_BCs = []
    list_BCNames = []
    list_BCTypes = []
    for c in range(len(BCs)):
      b = BCs[c]

      if b == []:
        raise ValueError("_recoverBCs: boundary is probably ill-defined.")
      # Break BC connectivity si necessaire
      elts = Internal.getElementNodes(b)
      size = 0
      for e in elts:
        erange = Internal.getNodeFromName1(e, 'ElementRange')[1]
        size += erange[1]-erange[0]+1
      n = len(elts)
      if n == 1:
        ids = C.identifyElements(hook, b, tol)
      else:
        bb = C.breakConnectivity(b)
        ids = numpy.array([], dtype=Internal.E_NpyInt)
        for bc in bb:
          ids = numpy.concatenate([ids, C.identifyElements(hook, bc, tol)])

      # Cree les BCs
      ids0 = ids # keep ids for bcdata
      ids  = ids[ids > -1]
      sizebc = ids.size
      if len(ids) < len(ids0):
            list_BCs.append(b)
            list_BCNames.append(BCNames[c])
            list_BCTypes.append(BCTypes[c])
      else:
        id2 = numpy.empty(sizebc, dtype=Internal.E_NpyInt)
        id2[:] = indicesF[ids[:]-1]
        C._addBC2Zone(z, BCNames[c], BCTypes[c], faceList=id2)

        # Recupere BCDataSets
        fsc = Internal.getNodeFromName(b, Internal.__FlowSolutionCenters__)

        if fsc is not None:
          newNameOfBC = C.getLastBCName(BCNames[c])
          bcz = Internal.getNodeFromNameAndType(z, newNameOfBC, 'BC_t')

          ds = Internal.newBCDataSet(name='BCDataSet', value='UserDefined',
                                   gridLocation='FaceCenter', parent=bcz)
          d = Internal.newBCData('NeumannData', parent=ds)

          for node in Internal.getChildren(fsc):
            if Internal.isType(node, 'DataArray_t'):
              val0 = Internal.getValue(node)
              if isinstance(val0,numpy.ndarray):
                val0 = numpy.reshape(val0, val0.size, order='F')
              else:
                val0 = numpy.reshape([val0], 1, order='F')

              val1 = val0[ids0>-1]
              Internal._createUniqueChild(d, node[0], 'DataArray_t', value=val1)

    C.freeHook(hook)

  return list_BCs, list_BCNames, list_BCTypes

# -- recoverBCs
# Identifie des subzones comme BC Faces
# IN: liste de geometries de BC, liste de leur nom, liste de leur type
# OUT: a modifie avec des BCs ajoutees en BCFaces

def _recoverBCs(t, BCInfo, tol=1.e-11, removeBC=True):
  if removeBC: return _recoverBCs1(t, BCInfo, tol)
  else: return _recoverBCs2(t, BCInfo, tol)

# N'efface pas les matchs et bc deja existantes
def _recoverBCs2(t, BCInfo, tol):
  try: import Post.PyTree as P
  except: raise ImportError("_recoverBCs: requires Post module.")
  try: import Transform.PyTree as T
  except: raise ImportError("_recoverBCs: requires Transform module.")
  try: import Generator.PyTree as G
  except: raise ImportError("_recoverBCs: requires Generator module.")
  (BCs, BCNames, BCTypes) = BCInfo
  for z in Internal.getZones(t):
      indicesF = []
      zf = P.exteriorFaces(z, indices=indicesF)
      indicesF = indicesF[0]
      # BC classique
      bnds = Internal.getNodesFromType2(z, 'BC_t')
      # BC Match
      bnds += Internal.getNodesFromType2(z, 'GridConnectivity1to1_t')
      # BC Overlap/NearMatch/NoMatch
      bnds += Internal.getNodesFromType2(z, 'GridConnectivity_t')
      indicesBC = []
      for b in bnds:
          f = Internal.getNodeFromName1(b, 'PointList')
          indicesBC.append(f[1])

      undefBC = False
      if indicesBC != []:
          indicesBC = numpy.concatenate(indicesBC, axis=1)
          nfacesExt = indicesF.shape[0]
          nfacesDef = indicesBC.shape[1]
          if nfacesExt < nfacesDef:
              print('Warning: zone %s: number of faces defined by BCs is greater than the number of external faces. Try to reduce the matching tolerance.'%(z[0]))
          elif nfacesExt > nfacesDef:
              indicesBC = indicesBC.reshape( (indicesBC.size) )
              indicesE = converter.diffIndex(indicesF, indicesBC)
              undefBC = True
      else:
          undefBC = True
          indicesE = indicesF
      if undefBC:
          zf = T.subzone(z, indicesE, type='faces')
          hook = C.createHook(zf, 'elementCenters')
          for c in range(len(BCs)):
              if BCs[c] == []: raise ValueError("_recoverBCs: boundary is probably ill-defined.")
              for b in BCs[c]:
                if G.bboxIntersection(zf, b):
                    # Break BC connectivity si necessaire
                    elts = Internal.getElementNodes(b)
                    size = 0
                    for e in elts:
                        erange = Internal.getNodeFromName1(e, 'ElementRange')[1]
                        size += erange[1]-erange[0]+1
                    n = len(elts)
                    if n == 1:
                        ids = C.identifyElements(hook, b, tol)
                    else:
                        bb = breakConnectivity(b)
                        ids = numpy.array([], dtype=Internal.E_NpyInt)
                        for bc in bb:
                            ids = numpy.concatenate([ids, C.identifyElements(hook, bc, tol)])

                    # Cree les BCs
                    ids0 = ids # keep ids for bcdata
                    ids = ids[ids > -1]
                    sizebc = ids.size
                    if sizebc > 0:
                        id2 = numpy.empty(sizebc, dtype=Internal.E_NpyInt)
                        id2[:] = indicesE[ids[:]-1]
                        C._addBC2Zone(z, BCNames[c], BCTypes[c], faceList=id2)

                        # Recupere BCDataSets
                        fsc = Internal.getNodeFromName(b, Internal.__FlowSolutionCenters__)

                        if fsc is not None:
                          newNameOfBC = C.getLastBCName(BCNames[c])
                          bcz = Internal.getNodeFromNameAndType(z, newNameOfBC, 'BC_t')

                          ds = Internal.newBCDataSet(name='BCDataSet', value='UserDefined',
                                                     gridLocation='FaceCenter', parent=bcz)
                          d = Internal.newBCData('NeumannData', parent=ds)

                          for node in Internal.getChildren(fsc):
                            if Internal.isType(node, 'DataArray_t'):
                              val0 = Internal.getValue(node)
                              if isinstance(val0,numpy.ndarray):
                                  val0 = numpy.reshape(val0, val0.size, order='F')
                              else:
                                  val0 = numpy.array([val0])
                              val1 = val0[ids0>-1]
                              Internal._createUniqueChild(d, node[0], 'DataArray_t', value=val1)
          C.freeHook(hook)
  return None

def _recoverBCs1(a, T, tol=1.e-11):
  """Recover given BCs on a tree.
  Usage: _recoverBCs(a, (BCs, BCNames, BCTypes), tol)"""
  try:import Post.PyTree as P
  except: raise ImportError("_recoverBCs: requires Post module.")
  C._deleteZoneBC__(a)
  zones = Internal.getZones(a)
  (BCs, BCNames, BCTypes) = T
  for z in zones:
    indicesF = []
    try: f = P.exteriorFaces(z, indices=indicesF)
    except: continue
    indicesF = indicesF[0]
    hook = C.createHook(f, 'elementCenters')

    for c in range(len(BCs)):
      b = BCs[c]

      if b == []:
        raise ValueError("_recoverBCs: boundary is probably ill-defined.")
      # Break BC connectivity si necessaire
      elts = Internal.getElementNodes(b)
      size = 0
      for e in elts:
        erange = Internal.getNodeFromName1(e, 'ElementRange')[1]
        size += erange[1]-erange[0]+1
      n = len(elts)
      if n == 1:
        ids = C.identifyElements(hook, b, tol)
      else:
        bb = C.breakConnectivity(b)
        ids = numpy.array([], dtype=Internal.E_NpyInt)
        for bc in bb:
          ids = numpy.concatenate([ids, C.identifyElements(hook, bc, tol)])

      # Cree les BCs
      ids0 = ids # keep ids for bcdata
      ids  = ids[ids > -1]
      sizebc = ids.size
      if sizebc > 0:
        id2 = numpy.empty(sizebc, dtype=Internal.E_NpyInt)
        id2[:] = indicesF[ids[:]-1]
        C._addBC2Zone(z, BCNames[c], BCTypes[c], faceList=id2)

        # Recupere BCDataSets
        fsc = Internal.getNodeFromName(b, Internal.__FlowSolutionCenters__)

        if fsc is not None:
          newNameOfBC = C.getLastBCName(BCNames[c])
          bcz = Internal.getNodeFromNameAndType(z, newNameOfBC, 'BC_t')

          ds = Internal.newBCDataSet(name='BCDataSet', value='UserDefined',
                                   gridLocation='FaceCenter', parent=bcz)
          d = Internal.newBCData('NeumannData', parent=ds)

          for node in Internal.getChildren(fsc):
            if Internal.isType(node, 'DataArray_t'):
              val0 = Internal.getValue(node)
              if isinstance(val0,numpy.ndarray):
                  val0 = numpy.reshape(val0, val0.size, order='F')
              else:
                  val0 = numpy.array([val0])
              val1 = val0[ids0>-1]
              Internal._createUniqueChild(d, node[0], 'DataArray_t', value=val1)

    C.freeHook(hook)

  return None

def create_Quad2Quad(coordinates, nonconformal_faces, nonconformal_faces_ctr,plane="xy",tol=1e-6):

  if plane=='xy':
      planedir = [0,1]
      ndir = 2
  elif plane=='xz':
      planedir = [0,2]
      ndir = 1
  else:
      raise Exception("plane in create_Quad2Quad must be 'xy' or 'xz' instead of {}".format(plane))
  nfaces = len(nonconformal_faces)
  print("\nQuad2Quad: {} original non conformal faces ({} potential Quad2Quad)".format(nfaces, nfaces/3))
  print("Looking for non conformal faces in 2D mesh, plane {}".format(plane))

  listQuad2Quad = []
  len_NCF = len(nonconformal_faces)

  node2cell_list = ComputeNode2CellList(nonconformal_faces, len_NCF, len(coordinates[:,0]))
  lengths = numpy.array([len(x) for x in node2cell_list])
  points45_init = numpy.where(lengths==3)[0]

  node2cell_list = numpy.array(node2cell_list,dtype=object)
  node2cell_list_shr = numpy.vstack(node2cell_list[points45_init])

  indexes = numpy.lexsort(numpy.vstack([node2cell_list_shr[:,1], node2cell_list_shr[:,2]]))

  node2cell_list_shr_sorted = node2cell_list_shr[indexes]

  remove_node = []
  for i in range(0,len(node2cell_list_shr_sorted[:,0]),2):
    [idx1, iface1, iface2] = node2cell_list_shr_sorted[i]
    idx2 = node2cell_list_shr_sorted[i+1][0]
    x1_ctr = nonconformal_faces_ctr[iface1][0]
    y1_ctr = nonconformal_faces_ctr[iface1][1]
    z1_ctr = nonconformal_faces_ctr[iface1][2]

    x2_ctr = nonconformal_faces_ctr[iface2][0]
    y2_ctr = nonconformal_faces_ctr[iface2][1]
    z2_ctr = nonconformal_faces_ctr[iface2][2]

    x1 = coordinates[idx1][0]
    y1 = coordinates[idx1][1]
    z1 = coordinates[idx1][2]

    x2 = coordinates[idx2][0]
    y2 = coordinates[idx2][1]
    z2 = coordinates[idx2][2]


    vector1 = numpy.array([(y1-y1_ctr)*(z2-z1)-(y2-y1)*(z1-z1_ctr), (x1-x1_ctr)*(z2-z1)-(z1-z1_ctr)*(x2-x1), (x1-x1_ctr)*(y2-y1)-(y1-y1_ctr)*(x2-x1)])
    vector1norm = numpy.linalg.norm(vector1)
    vector1 = vector1/vector1norm
    vector2 = numpy.array([(y1-y2_ctr)*(z2-z1)-(y2-y1)*(z1-z2_ctr), (x1-x2_ctr)*(z2-z1)-(z1-z2_ctr)*(x2-x1), (x1-x2_ctr)*(y2-y1)-(y1-y2_ctr)*(x2-x1)])
    vector2norm = numpy.linalg.norm(vector2)
    vector2 = vector2/vector2norm
    scalar_product = numpy.dot(vector1,vector2)
    if scalar_product > 0.0:
      remove_node.append(node2cell_list_shr_sorted[i][0])
      remove_node.append(node2cell_list_shr_sorted[i+1][0])

  for i in remove_node:
    points45_init = points45_init[points45_init!=i]

    node2cell_list_shr = node2cell_list_shr[node2cell_list_shr[:,0]!=i]
    node2cell_list_shr_sorted = node2cell_list_shr_sorted[node2cell_list_shr_sorted[:,0]!=i]

  big_face = []
  hns = []
  for conn in range(0,len(node2cell_list_shr_sorted[:,0]),2):
    nd1 = node2cell_list_shr_sorted[conn][0]
    nd2 = node2cell_list_shr_sorted[conn+1][0]

    el1 = node2cell_list_shr_sorted[conn][1]
    el2 = node2cell_list_shr_sorted[conn][2]

    big_face_concatenated = numpy.concatenate([nonconformal_faces[el1],nonconformal_faces[el2]])
    big_face_concatenated = big_face_concatenated[big_face_concatenated!=nd1]
    big_face_concatenated = big_face_concatenated[big_face_concatenated!=nd2]

    hns.append([nd1,nd2])
    big_face.append(big_face_concatenated)

  for idx,nodes in enumerate(big_face):
    point4 = hns[idx][0]
    point5 = hns[idx][1]

    i01 = abs(coordinates[nodes,ndir]-coordinates[point4,ndir])<tol

    point01 = nodes[i01]
    point0 = point01[0]
    point1 = point01[1]
    point23 = nodes[~i01]
    point2 = point23[0]
    point3 = point23[1]

    r02 = coordinates[point2] - coordinates[point0]
    r03 = coordinates[point3] - coordinates[point0]
    r04 = coordinates[point4] - coordinates[point0]
    r05 = coordinates[point5] - coordinates[point0]

    if numpy.cross(r02,r03).dot(numpy.cross(r04,r05)) < 0:
        point2 = point23[1]
        point3 = point23[0]
    thisQuad2Quad = [ point0, point1, point2, point3, point4, point5 ]
    # Append to list
    listQuad2Quad.append(thisQuad2Quad)
  listQuad2Quad = numpy.array(listQuad2Quad)
  if listQuad2Quad.shape[0] != nfaces/3:
      raise ValueError("Problem on non conformal faces: only %d out of %d have been matched." %(listQuad2Quad.shape[0],nfaces//3))

  return listQuad2Quad, 0

def create_Quad2Quad_MPI(coordinates, nonconformal_faces, nonconformal_faces_ctr,tol=1e-6):

  print("num centers before", len(nonconformal_faces_ctr))
  t = 8
  tol = 10**(-t)

  cmpIdx = lambda a, b : cmp(nonconformal_faces_ctr[a], nonconformal_faces_ctr[b])
  idx_sorted = sorted(range(len(nonconformal_faces_ctr)), key=cmp_to_key(cmpIdx))


  nnodes_old = len(nonconformal_faces_ctr)

  nonconformal_faces_ctr_sorted = nonconformal_faces_ctr[idx_sorted]
  nonconformal_faces_sorted = nonconformal_faces[idx_sorted]

  previous = nonconformal_faces_ctr_sorted[0]
  indices = [0]
  j = 0
  flag = True
  for i in range(1,nnodes_old):
    if (abs(previous-nonconformal_faces_ctr_sorted[i])>(10**(-t))).any(): #se sono diversi
        j=j+1
        indices.append(i)
        flag = True
    elif not (abs(previous-nonconformal_faces_ctr_sorted[i])>(10**(-t))).any() and flag==True:
      del indices[-1]
      flag = False
    previous = nonconformal_faces_ctr_sorted[i]

  nonconformal_faces = nonconformal_faces_sorted[indices]
  nonconformal_faces_ctr = nonconformal_faces_ctr_sorted[indices]

  if len(numpy.unique(nonconformal_faces_ctr[:,2]))==1:
      planedir = [0,1]
      ndir = 2
      plane = "xy"
  elif len(numpy.unique(nonconformal_faces_ctr[:,1]))==1:
      planedir = [0,2]
      ndir = 1
      plane = "xz"
  else:
      raise Exception("plane in create_Quad2Quad must be 'xy' or 'xz'.")
  if len(nonconformal_faces) != len(nonconformal_faces_ctr):
      raise Exception("len(nonconformal_faces) != len(non_conformal_faces_ctr)")

  nfaces = len(nonconformal_faces)
  print("\nQuad2Quad: {} original non conformal faces ({} potential Quad2Quad)".format(nfaces, nfaces/3))
  print("Looking for non conformal faces in 2D mesh, plane {}".format(plane))

  listQuad2Quad = []
  len_NCF = len(nonconformal_faces)

  node2cell_list = ComputeNode2CellList(nonconformal_faces, len_NCF, len(coordinates[:,0]))
  lengths = numpy.array([len(x) for x in node2cell_list])

  points45_init = numpy.where(lengths==3)[0]
  node2cell_list = numpy.array(node2cell_list,dtype=object)
  node2cell_list_shr = numpy.vstack(node2cell_list[points45_init])

  indexes = numpy.lexsort(numpy.vstack([node2cell_list_shr[:,1], node2cell_list_shr[:,2]]))

  node2cell_list_shr_sorted = node2cell_list_shr[indexes]

  remove_node = []
  for i in range(0,len(node2cell_list_shr_sorted[:,0]),2):
    [idx1, iface1, iface2] = node2cell_list_shr_sorted[i]
    idx2 = node2cell_list_shr_sorted[i+1][0]
    x1_ctr = nonconformal_faces_ctr[iface1][0]
    y1_ctr = nonconformal_faces_ctr[iface1][1]
    z1_ctr = nonconformal_faces_ctr[iface1][2]

    x2_ctr = nonconformal_faces_ctr[iface2][0]
    y2_ctr = nonconformal_faces_ctr[iface2][1]
    z2_ctr = nonconformal_faces_ctr[iface2][2]

    x1 = coordinates[idx1][0]
    y1 = coordinates[idx1][1]
    z1 = coordinates[idx1][2]

    x2 = coordinates[idx2][0]
    y2 = coordinates[idx2][1]
    z2 = coordinates[idx2][2]


    vector1 = numpy.array([(y1-y1_ctr)*(z2-z1)-(y2-y1)*(z1-z1_ctr), (x1-x1_ctr)*(z2-z1)-(z1-z1_ctr)*(x2-x1), (x1-x1_ctr)*(y2-y1)-(y1-y1_ctr)*(x2-x1)])
    vector1norm = numpy.linalg.norm(vector1)
    vector1 = vector1/vector1norm
    vector2 = numpy.array([(y1-y2_ctr)*(z2-z1)-(y2-y1)*(z1-z2_ctr), (x1-x2_ctr)*(z2-z1)-(z1-z2_ctr)*(x2-x1), (x1-x2_ctr)*(y2-y1)-(y1-y2_ctr)*(x2-x1)])
    vector2norm = numpy.linalg.norm(vector2)
    vector2 = vector2/vector2norm
    scalar_product = numpy.dot(vector1,vector2)
    if scalar_product > 0.0:
      remove_node.append(node2cell_list_shr_sorted[i][0])
      remove_node.append(node2cell_list_shr_sorted[i+1][0])

  for i in remove_node:
    points45_init = points45_init[points45_init!=i]

    node2cell_list_shr = node2cell_list_shr[node2cell_list_shr[:,0]!=i]
    node2cell_list_shr_sorted = node2cell_list_shr_sorted[node2cell_list_shr_sorted[:,0]!=i]

  print("len(remove node)",len(remove_node))
  print("len(node2cell_list_shr_sorted)",len(node2cell_list_shr_sorted))
  big_face = []
  hns = []
  for conn in range(0,len(node2cell_list_shr_sorted[:,0]),2):
    nd1 = node2cell_list_shr_sorted[conn][0]
    nd2 = node2cell_list_shr_sorted[conn+1][0]

    el1 = node2cell_list_shr_sorted[conn][1]
    el2 = node2cell_list_shr_sorted[conn][2]

    big_face_concatenated = numpy.concatenate([nonconformal_faces[el1],nonconformal_faces[el2]])
    big_face_concatenated = big_face_concatenated[big_face_concatenated!=nd1]
    big_face_concatenated = big_face_concatenated[big_face_concatenated!=nd2]

    hns.append([nd1,nd2])
    big_face.append(big_face_concatenated)
  print("len(big_face)",len(big_face))
  for idx,nodes in enumerate(big_face):
    point4 = hns[idx][0]
    point5 = hns[idx][1]

    i01 = abs(coordinates[nodes,ndir]-coordinates[point4,ndir])<tol

    point01 = nodes[i01]
    point0 = point01[0]
    point1 = point01[1]
    point23 = nodes[~i01]
    point2 = point23[0]
    point3 = point23[1]

    r02 = coordinates[point2] - coordinates[point0]
    r03 = coordinates[point3] - coordinates[point0]
    r04 = coordinates[point4] - coordinates[point0]
    r05 = coordinates[point5] - coordinates[point0]

    if numpy.cross(r02,r03).dot(numpy.cross(r04,r05)) < 0:
        point2 = point23[1]
        point3 = point23[0]
    thisQuad2Quad = [ point0, point1, point2, point3, point4, point5 ]
    # Append to list
    listQuad2Quad.append(thisQuad2Quad)
  listQuad2Quad = numpy.array(listQuad2Quad)

  #if listQuad2Quad.shape[0] != nfaces/3:
  if listQuad2Quad.shape[0] < nfaces/3:
      raise ValueError("Problem on non conformal faces: only %d out of %d have been matched." %(listQuad2Quad.shape[0],nfaces//3))

  return listQuad2Quad, 0


def create_Quad4Quad(coordinates, nonconformal_faces, nonconformal_faces_ctr,tol=1e-6):
  nb_vertices = coordinates.shape[0]
  len_NCF = len(nonconformal_faces)

  z = Internal.newZone(name="Zone",zsize=[[nb_vertices,len_NCF]],ztype="Unstructured")
  gc = Internal.newGridCoordinates(parent = z)
  Internal.newDataArray('CoordinateX', value = coordinates[:,0], parent = gc)
  Internal.newDataArray('CoordinateY', value = coordinates[:,1], parent = gc)
  Internal.newDataArray('CoordinateZ', value = coordinates[:,2], parent = gc)

  zC = Internal.newZone(name="ZoneCenters",zsize=[[len_NCF,len_NCF]],ztype="Unstructured")
  gcC = Internal.newGridCoordinates(parent = zC)
  Internal.newDataArray('CoordinateX', nonconformal_faces_ctr[:,0], parent = gcC)
  Internal.newDataArray('CoordinateY', nonconformal_faces_ctr[:,1], parent = gcC)
  Internal.newDataArray('CoordinateZ', nonconformal_faces_ctr[:,2], parent = gcC)

  hook = C.createHook(z, 'nodes')
  ids = C.identifyNodes(hook, zC)
  ids_points8 = ids[ids[:] > -1] - 1
  print(ids_points8)

  node2cell_list = ComputeNode2CellList(nonconformal_faces, len_NCF, len(coordinates[:,0]))

  listQuad4Quad = []
  nfaces = len(nonconformal_faces)
  print("\nQuad4Quad: {} original non conformal faces ({} potential Quad4Quad)".format(nfaces, nfaces/5))
  t0 = time.time()
  print("Looking for non conformal faces in 3D mesh")

  for point8 in ids_points8:
    match_nonconformal_faces = node2cell_list[point8][1:]
    if len(match_nonconformal_faces)!=4:
        print("Something is off. {} non conformal faces match this hanging point at {}. 4 non conformal faces should match (Quad4Quad)".format(len(match_nonconformal_faces),ctr))

    list_nodes_B4B = nonconformal_faces[match_nonconformal_faces]
    for position in list_nodes_B4B:
      position_point8 = numpy.where(position==point8)[0][0]
      if position_point8 == 0:
          point5 = position[1]
          point2 = position[2]
          point6 = position[3]
      elif position_point8 == 1:
          point7 = position[0]
          point6 = position[2]
          point3 = position[3]
      elif position_point8 == 2:
          point0 = position[0]
          point4 = position[1]
          point7 = position[3]
      elif position_point8 == 3:
          point4 = position[0]
          point1 = position[1]
          point5 = position[2]

    listQuad4Quad.append([ point0, point1, point2, point3, point4, point5, point6, point7, point8])

  listQuad4Quad = numpy.array(listQuad4Quad)

  return listQuad4Quad,0

def isequal(a, b):
   return abs(a - b) < 1.e-10

def cmp(a, b):
  if isequal(a[2], b[2]):
    if isequal(a[1], b[1]):
      if isequal(a[0], b[0]):
        return 0
      elif a[0] > b[0]:
        return 1
      else:
        return -1
    elif a[1] > b[1]:
      return 1
    else:
      return -1
  elif a[2] > b[2]:
    return 1
  else:
    return -1

def ComputeNode2CellList(numpy_cell2node_not_raveled, NumElements, NumNodes, NumVerticesPerElement=4):

   node2cell_list  = [[i] for i in range(NumNodes)]
   for i in range(NumElements):
     for j in range(NumVerticesPerElement):
       node2cell_list[numpy_cell2node_not_raveled[i][j]].append(i)

   return node2cell_list

def myRoundNumpy(val, relTol, absTol):
  orderOfMagnitude = numpy.floor(numpy.log10(abs(val)))
  tol = numpy.maximum(relTol + orderOfMagnitude, absTol)
  return numpy.round(val * 10**(-tol)) / 10**(-tol)



def computeCellCenters_Quads(xNP, yNP, zNP, EC):
    n_vertex_per_elements = 4
    nb_elts = len(EC)/n_vertex_per_elements
    x = []; y = []; z= [];
    x = (xNP[EC[0::4]]+xNP[EC[1::4]]+xNP[EC[2::4]]+xNP[EC[3::4]])/4.0
    y = (yNP[EC[0::4]]+yNP[EC[1::4]]+yNP[EC[2::4]]+yNP[EC[3::4]])/4.0
    z = (zNP[EC[0::4]]+zNP[EC[1::4]]+zNP[EC[2::4]]+zNP[EC[3::4]])/4.0
    return [x,y,z]

def initializeCell2ProcOutsideClass(clac,t,newcell2Proc):
     nProcs = clac.NProcs()

     newcell2ProcGathered = ArrayOps.AllGather(newcell2Proc,clac)

     CORRECT_newcell2ProcGathered = FSIntArray(nProcs+1)
     CORRECT_newcell2ProcGathered.Fill(int(0))
     for i in range(nProcs):
       CORRECT_newcell2ProcGathered[i+1] = int(CORRECT_newcell2ProcGathered[i]+newcell2ProcGathered[i])
     return CORRECT_newcell2ProcGathered

#class Converter_FSDM_CGNS_MPI(Converter_FSDM_CGNS):

