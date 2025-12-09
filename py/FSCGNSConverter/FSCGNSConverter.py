# Import modules
import os
import sys
import time
from functools import cmp_to_key, wraps
import numpy

import Converter.PyTree as C
import Converter.Internal as Internal
import Converter.Mpi as Cmpi
import Generator.PyTree as G
import Transform.PyTree as T

from FSDataManager import (
    FSClac, FSMesh, FSError, FSFloatArray, FSIntArray, FSStringArray,
    FSDataName, FSDataSpecArray, FSDatasetInfo, FSMeshEnums,
    FSUnstructVolumeCellTypes, FSUnstructSurfaceCellTypes, FS_AT_CADGroupID,
    FSCellInfo
)

# Import optional packages
try:
    from FSDMPyUtils import ArrayOps
except ImportError as e:
    raise ImportError("ERROR: FSDMPyUtils not found") from e


__all__ = ['FSCGNSConverter', 'ENABLE_PROFILING', 'buildMeshOps']


# ---------------------------------------------------------------------------- #
# Global variables
# ---------------------------------------------------------------------------- #

# ENABLE_PROFILING: bool; Switch to toggle on or off profiling
ENABLE_PROFILING = False

# CGNS_CONTAINER_NAMES: list; Names of the CGNS containers
CGNS_CONTAINER_NAMES = [
    Internal.__FlowSolutionNodes__,
    Internal.__FlowSolutionCenters__
]


# ---------------------------------------------------------------------------- #
# Decorator
# ---------------------------------------------------------------------------- #

def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if ENABLE_PROFILING:
            t0 = time.perf_counter()
            res = func(*args, **kwargs)
            t1 = time.perf_counter()
            elapsed = max(Cmpi.allgather(t1 - t0))
            if Cmpi.master:
                print(f"  > {func.__name__}: executed in {elapsed:.3f} sec.")
            return res
        else:
            return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------- #
# Functions
# ---------------------------------------------------------------------------- #

@profile_time
def _fixNodesForFlowSolution(t):
    if Cmpi.master: print("Fixing Flow Solution.")
    n_FSC = Internal.getNodeFromName(t, "FlowSolution#Centers")
    if n_FSC is None: return

    dictio = {}
    for n in n_FSC[2][1:]:
        first, second = n[0].split(".")[:2]
        if first not in dictio:
            dictio[first] = []
        dictio[first].append(second)

    zone = Internal.getZones(t)
    for flowSolutionName, arrayNames in dictio.items():
        n_newFS = Internal.newFlowSolution(
            name=f"FlowSolution#{flowSolutionName}",
            gridLocation='CellCenter',
            parent=zone[0]
        )
        for arrayName in arrayNames:
            n = Internal.getNodeFromName(n_FSC, f"{flowSolutionName}.{arrayName}")
            Internal.newDataArray(arrayName, value=n[1], parent=n_newFS)
    Internal._rmNodesByName(t, "FlowSolution#Centers")
    return None

@profile_time
def recoverBCsC(a, BCs, BCNames, BCTypes, tol=1.e-11):
    """Recover given BCs on a tree.
    Usage: _recoverBCs(a, BCs, BCNames, BCTypes, tol)"""
    try: import Post.PyTree as P
    except: raise ImportError("_recoverBCs: requires Post module.")
    C._deleteZoneBC__(a)
    zones = Internal.getZones(a)
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
            n_elts = Internal.getElementNodes(b)
            size = 0
            for n_elt in n_elts:
                erange = Internal.getNodeFromName1(n_elt, 'ElementRange')[1]
                size += erange[1] - erange[0] + 1
            n = len(n_elts)
            if n == 1:
                ids = C.identifyElements(hook, b, tol)
            else:
                bb = C.breakConnectivity(b)
                ids = numpy.array([], dtype=Internal.E_NpyInt)
                for bc in bb:
                    ids = numpy.concatenate([ids, C.identifyElements(hook, bc, tol)])

            # Cree les BCs
            ids0 = ids  # keep ids for bcdata
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
                    ds = Internal.newBCDataSet(
                        name='BCDataSet', value='UserDefined',
                        gridLocation='FaceCenter', parent=bcz
                    )
                    d = Internal.newBCData('NeumannData', parent=ds)

                    for node in Internal.getChildren(fsc):
                        if Internal.isType(node, 'DataArray_t'):
                            val0 = Internal.getValue(node)
                            if isinstance(val0,numpy.ndarray):
                                val0 = numpy.reshape(val0, val0.size, order='F')
                            else:
                                val0 = numpy.reshape([val0], 1, order='F')
                            val1 = val0[ids0 > -1]
                            Internal._createUniqueChild(d, node[0], 'DataArray_t', value=val1)

        C.freeHook(hook)

    return list_BCs, list_BCNames, list_BCTypes

def create_Quad2Quad(coordinates, nonconformal_faces, ncFacesCentroids,
                     plane="xy", tol=1e-6):
    if plane == 'xy':
        planedir = [0,1]
        ndir = 2
    elif plane == 'xz':
        planedir = [0,2]
        ndir = 1
    else:
        print("ERROR: Plane in create_Quad2Quad must be 'xy' or 'xz' instead "
              "of {}.".format(plane))
        sys.exit(1)

    nNCFaces = len(nonconformal_faces)
    print(
        "Quad2Quad: {} original non conformal faces (incl. {} potential "
        "Quad2Quad)\nLooking for non conformal faces in 2D mesh, "
        "{} plane.".format(nNCFaces, nNCFaces//3, plane)
    )

    listQuad2Quad = []
    len_NCF = len(nonconformal_faces)

    node2cell_list = computeNode2CellList(nonconformal_faces, len_NCF, len(coordinates[:,0]))
    lengths = numpy.array([len(x) for x in node2cell_list])
    points45_init = numpy.where(lengths==3)[0]

    node2cell_list = numpy.array(node2cell_list,dtype=object)
    node2cell_list_shr = numpy.vstack(node2cell_list[points45_init])

    indexes = numpy.lexsort(numpy.vstack([node2cell_list_shr[:,1], node2cell_list_shr[:,2]]))

    node2cell_list_shr_sorted = node2cell_list_shr[indexes]

    remove_node = []
    for i in range(0, len(node2cell_list_shr_sorted[:,0]), 2):
        idx1, iface1, iface2 = node2cell_list_shr_sorted[i]
        idx2 = node2cell_list_shr_sorted[i+1][0]

        x1_ctr, y1_ctr, z1_ctr = ncFacesCentroids[iface1][:]
        x2_ctr, y2_ctr, z2_ctr = ncFacesCentroids[iface2][:]
        x1, y1, z1 = coordinates[idx1][:]
        x2, y2, z2 = coordinates[idx2][:]

        vector1 = numpy.array([
            (y1 - y1_ctr)*(z2 - z1) - (y2 - y1)*(z1 - z1_ctr),
            (x1 - x1_ctr)*(z2 - z1) - (z1 - z1_ctr)*(x2 - x1),
            (x1 - x1_ctr)*(y2 - y1) - (y1 - y1_ctr)*(x2 - x1)
        ])
        vector1norm = numpy.linalg.norm(vector1)
        vector1 = vector1/vector1norm
        vector2 = numpy.array([
            (y1 - y2_ctr)*(z2 - z1) - (y2 - y1)*(z1 - z2_ctr),
            (x1 - x2_ctr)*(z2 - z1) - (z1 - z2_ctr)*(x2 - x1),
            (x1 - x2_ctr)*(y2 - y1) - (y1 - y2_ctr)*(x2 - x1)
        ])
        vector2norm = numpy.linalg.norm(vector2)
        vector2 = vector2/vector2norm
        scalar_product = numpy.dot(vector1, vector2)
        if scalar_product > 0.0:
            remove_node.append(node2cell_list_shr_sorted[i][0])
            remove_node.append(node2cell_list_shr_sorted[i+1][0])

    for i in remove_node:
        points45_init = points45_init[points45_init!=i]
        node2cell_list_shr = node2cell_list_shr[node2cell_list_shr[:,0] != i]
        node2cell_list_shr_sorted = node2cell_list_shr_sorted[node2cell_list_shr_sorted[:,0] != i]

    if len(node2cell_list_shr_sorted)%2 != 0:
        print("ERROR: Something is off, some small faces are missing.")
        sys.exit(1)

    big_face = []
    hns = []
    for conn in range(0, len(node2cell_list_shr_sorted[:,0]), 2):
        nd1 = node2cell_list_shr_sorted[conn][0]
        nd2 = node2cell_list_shr_sorted[conn+1][0]

        el1 = node2cell_list_shr_sorted[conn][1]
        el2 = node2cell_list_shr_sorted[conn][2]

        big_face_concatenated = numpy.concatenate([nonconformal_faces[el1],nonconformal_faces[el2]])
        big_face_concatenated = big_face_concatenated[big_face_concatenated!=nd1]
        big_face_concatenated = big_face_concatenated[big_face_concatenated!=nd2]

        if nonconformal_faces[el1][1] == nd1:
            hns.append([nd2,nd1])
        else:
            hns.append([nd1,nd2])
        big_face.append(big_face_concatenated)

    for idx, nodes in enumerate(big_face):
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
        thisQuad2Quad = [point0, point1, point2, point3, point4, point5]
        # Append to list
        listQuad2Quad.append(thisQuad2Quad)
    np_Quad2Quad = numpy.array(listQuad2Quad)
    #if listQuad2Quad.shape[0] != nNCFaces/3:
    #    raise ValueError("Problem on non conformal faces: only %d out of %d have been matched." %(listQuad2Quad.shape[0],nNCFaces//3))
    return np_Quad2Quad

def create_Quad4Quad(coordinates, nonconformal_faces, ncFacesCentroids,tol=1e-6):
    nb_vertices = coordinates.shape[0]
    len_NCF = len(nonconformal_faces)

    z = Internal.newZone(name="Zone",zsize=[[nb_vertices,len_NCF]],ztype="Unstructured")
    gc = Internal.newGridCoordinates(parent=z)
    Internal.newDataArray('CoordinateX', value=coordinates[:,0], parent=gc)
    Internal.newDataArray('CoordinateY', value=coordinates[:,1], parent=gc)
    Internal.newDataArray('CoordinateZ', value=coordinates[:,2], parent=gc)

    zC = Internal.newZone(
        name="ZoneCenters",
        zsize=[[len_NCF,len_NCF]],
        ztype="Unstructured"
    )
    gcC = Internal.newGridCoordinates(parent=zC)
    Internal.newDataArray('CoordinateX', ncFacesCentroids[:,0], parent=gcC)
    Internal.newDataArray('CoordinateY', ncFacesCentroids[:,1], parent=gcC)
    Internal.newDataArray('CoordinateZ', ncFacesCentroids[:,2], parent=gcC)

    hook = C.createHook(z, 'nodes')
    ids = C.identifyNodes(hook, zC)
    ids_points8 = ids[ids[:] > -1] - 1

    node2cell_list = computeNode2CellList(nonconformal_faces, len_NCF, len(coordinates[:,0]))

    listQuad4Quad = []
    nfaces = len(nonconformal_faces)
    print(
        "Quad4Quad: {} original non conformal faces (incl. {} potential "
        "Quad4Quad). Looking for non conformal faces in 3D mesh.".format(
            nfaces, nfaces//5)
    )

    unique_ids_points8 = numpy.unique(ids_points8)
    for point8 in unique_ids_points8:
        match_nonconformal_faces = node2cell_list[point8][1:]
        if len(match_nonconformal_faces)!=4:
            print("ERROR: {} non conformal faces match this hanging "
                  "point. 4 non conformal faces should match (Quad4Quad).".format(
                      len(match_nonconformal_faces)))
            sys.exit(1)

        list_nodes_B4B = nonconformal_faces[match_nonconformal_faces]
        for position in list_nodes_B4B:
            position_point8 = numpy.where(position==point8)[0][0]
            if position_point8 == 0:
                point7 = position[1]
                point3 = position[2]
                point6 = position[3]
            elif position_point8 == 1:
                point5 = position[0]
                point6 = position[2]
                point2 = position[3]
            elif position_point8 == 2:
                point1 = position[0]
                point4 = position[1]
                point5 = position[3]
            elif position_point8 == 3:
                point4 = position[0]
                point0 = position[1]
                point7 = position[2]

        listQuad4Quad.append([point0, point1, point2, point3, point4, point5, point6, point7, point8])

    np_Quad4Quad = numpy.array(listQuad4Quad, dtype=Internal.E_NpyInt)
    return np_Quad4Quad

def _addBC2ZoneLoc(z, bndName, bndType, zbc, loc='FaceCenter', zdnrName=None):
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
    Internal.createUniqueChild(info, 'ElementRange', 'IndexRange_t',
                               value=numpy.array([[maxElt+1,maxElt+neb]]))
    return None

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

def computeNode2CellList(np_cell2NodeUnravelled, nelts, nnodes, nvpe=4):
    node2CellList  = [[i] for i in range(nnodes)]
    for i in range(nelts):
        for j in range(nvpe):
            node2CellList[np_cell2NodeUnravelled[i][j]].append(i)
    return node2CellList

def computeQuadCentroids(xNP, yNP, zNP, eltConn):
    quadIdc = eltConn.reshape(-1, 4)
    centroids = numpy.column_stack((
        xNP[quadIdc].mean(axis=1),
        yNP[quadIdc].mean(axis=1),
        zNP[quadIdc].mean(axis=1)
    ))
    return centroids

def initializeCell2ProcOutsideClass(clac, ncellsOfType=0):
    nprocs = Cmpi.size
    gath_cell2Proc = ArrayOps.AllGather(ncellsOfType, clac)
    fs_cell2Proc = FSIntArray(nprocs+1)
    fs_cell2Proc[0] = 0
    for i in range(nprocs):
        fs_cell2Proc[i+1] = fs_cell2Proc[i] + int(gath_cell2Proc[i])
    return fs_cell2Proc

def buildMeshOps(meshName, partitioningLibrary="FSZoltan",
                 preserveCellStacks=True, verbose=True):
    if partitioningLibrary == "FSZoltan":
        try:
            import FSZoltan
            partitioningCmd = "RepartitionMeshZOLTAN"
        except ImportError:
            partitioningCmd = "RepartitionMeshPARMETIS"
            print("WARNING: FSZoltan not found, ParMETIS used instead")
    else:
        partitioningCmd = "RepartitionMeshPARMETIS"

    meshFormat = meshName.split('.')[-1]
    if meshFormat == 'h5': meshImportCmd = "ImportMeshHDF5"
    elif meshFormat in ['grid', 'cdf']: meshImportCmd = "ImportMeshTAU"
    else:
        print(f"ERROR: Input mesh format, {meshFormat}, not supported. "
              "Must be h5, grid or cdf.")
        sys.exit(1)
    
    meshOps = [(meshImportCmd, {"MeshFilename": meshName})]
    if verbose: meshOps.append("PrintInfo")
    if Cmpi.size > 1:
        meshOps.extend([
            # Create a reasonable partitioning for ZOLTAN (prevent memory bottlenecks)
            "RepartitionMeshRCB",
            # Create local numbering
            "CreateLocalNumbering",
            # Main task
            (
                partitioningCmd,
                {
                    "PreserveCellStacks": preserveCellStacks,
                    "LineSectionsExtractionParameters": {
                        "ActiveNodesSelection": {
                            "CellTypes": (
                                "Prisms", "Hexahedra", "Tetrahedra",
                                "Pyramids", "Quadrilaterals", "Triangles",
                            )
                        },
                        "StartNodesSelection": {
                            "CellAttribute": "CADGroupID",
                        },
                    },
                    "GraphExtraction": {"GraphType": "CellBased"},
                    "Approach": "CoordinateGraphMultilevel",
                }
            ),
            # Create local numbering
            "CreateLocalNumbering"
        ])
        if verbose: meshOps.append("PrintInfo")
    return tuple(meshOps)


# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #

class FSCGNSConverter:

    # Map cell types from FS to CGNS and vice versa
    FS2CGNSCELLTYPES = {
        'Tri3': 'TRI',
        'Quad4': 'QUAD',
        'Tetra4': 'TETRA',
        'Pyra5': 'PYRA',
        'Prism6': 'PENTA',
        'Hexa8': 'HEXA'
    }
    CGNS2FSCELLTYPES = {v: k for k, v in FS2CGNSCELLTYPES.items()}

    # Map cell numbers from FS to CGNS and vice versa
    FS2CGNSCELLNOS = {
        3: 5,
        4: 7,
        5: 10,
        6: 12,
        7: 14,
        8: 17
    }
    CGNS2FSCELLNOS = {v: k for k, v in FS2CGNSCELLNOS.items()}

    def __init__(
        self,
        meshName=None,
        pyTree=None,
        clac=None,
        fsmesh=None,
        dimPb=2,
        flipYZAxes=False,
        conformal=True,
        IBMParameters=None,
        datasets='all',
        bcDict=None,
        coordsName="Coordinates",
        verbose=True,
        **kwargs
    ):
        # Sanitize and store input arguments
        if (clac is None) != (fsmesh is None):
            raise ValueError("Must provide both 'clac' and 'fsmesh', or "
                             "neither of them.")
        if ((meshName is None) == (fsmesh is None)) and (pyTree is None):
            raise ValueError("Must either provide 'meshName' or 'fsmesh', "
                             "but not both or neither of them.")
        if ((meshName is None) == (pyTree is None)) and (fsmesh is None):
            raise ValueError("Must either provide 'meshName' or 'pyTree', "
                             "but not both or neither of them.")
        if (fsmesh is not None) and (pyTree is not None):
            raise ValueError("Provide either 'fsmesh' or 'pyTree', but not "
                             "both of them.")
        if not (meshName is None or isinstance(meshName, str)):
            raise TypeError("'meshName' must be a string")

        self.meshName = meshName
        if (clac is None) != (fsmesh is None):
            raise ValueError("Must provide both 'clac' and 'fsmesh', or "
                             "neither of them.")
        self.clac = FSClac() if clac is None else clac
        self.fsmesh = fsmesh
        self.pyTree = pyTree
        self.dimPb = dimPb
        self.flipYZAxes = flipYZAxes
        self.conformal = conformal
        self.IBMParameters = {} if IBMParameters is None else IBMParameters.copy()
        self.IBM = len(self.IBMParameters) > 0
        if bcDict is None or not isinstance(bcDict, dict):
            self.bcDict = {}
        else:
            self.bcDict = {int(k): v for k, v in bcDict.items()}
            for marker in self.bcDict:
                if isinstance(self.bcDict[marker], tuple):
                    self.bcDict[marker] = list(self.bcDict[marker])
                elif isinstance(self.bcDict[marker], str):
                    self.bcDict[marker] = [None, self.bcDict[marker]]
                lenD = len(self.bcDict[marker])
                if lenD == 1: self.bcDict[marker].insert(0, None)
                elif lenD == 0 or lenD > 2:
                    raise ValueError(
                        "Invalid bcDict argument. Dictionary values can either "
                        "be a BCType (str) or a list containing "
                        "(BCName, BCType). When a BCName is provided, it "
                        "overrides the name present in the input file."
                    )
        self.coordsName = coordsName
        self.verbose = verbose

        if datasets is None:
            self.datasets = set()
        elif isinstance(datasets, (list, set)):
            self.datasets = set(datasets)
        elif isinstance(datasets, str):
            if datasets == 'all': self.datasets = datasets
            else: self.datasets = set([datasets])
        else:
            raise TypeError(
                "Invalid datasets argument. Must be None, 'all', a string "
                "(eg. a field name or a CGNS container name), or a list/set of "
                f"dataset names. Got: {datasets}"
            )

        # Determine which convert and export functions to use from the inputs
        isCGNSNode = (
            isinstance(self.pyTree, list) 
            and len(self.pyTree) == 4 
            and (Internal.isTopTree(self.pyTree) or self.pyTree[-1] == 'Zone_t')
        )
        isCGNSMeshName = (
            isinstance(self.meshName, str)
            and self.meshName.endswith(".cgns")
        )
        if isCGNSNode or isCGNSMeshName:
            self.__convert = self.convert2FSDM
            self.__export2Tecplot = self.exportFSMesh2Tecplot
        else:
            self.__convert = self.convert2CGNS
            self.__export2Tecplot = self.exportCGNS2Tecplot

        # Create other class attributes
        self.meshType = "Unstructured"
        self.nvertices = 0
        self.nvolumeCells = 0
        self.nsurfaceCells = 0
        self.fsSurfaceCellTypes = []
        self.fsVolumeCellTypes = []
        self.fsCellTypes = []
        self.fsCellTypesBCs = []
        self.np_coordinates = []
        self.cell2NodeDict = {}
        self.cell2ProcDict = {}
        self.cell2NodeVolumeList = []
        self.cell2NodeSurfaceList = []
        self.indicesPerBdr = []
        self.bcsNames = []
        self.fsMarkers = []
        self.bMarker2BCNameDict = {}
        self.bMarker2FacePLDict = {}

    def convert(self, **kwargs):
        """Main routine to convert a mesh from FSDM to CGNS or vice versa"""
        self.__convert(**kwargs)

    def convert2CGNS(self, forOverset=False, forFFDX=False, **kwargs):
        """Convert a mesh from FSDM to CGNS"""
        if forOverset:
            includeSurfaceData = False
            includeGhostCells = True
        else:
            includeSurfaceData = True
            includeGhostCells = False
        self.recoverFSMeshInfo(includeGhostCells=includeGhostCells)
        self.recoverFSCoordinates()
        self.initializeCGNSCoordinates()
        self.recoverFSConnectivity(
            includeGhostCells=includeGhostCells,
            includeSurfaceData=includeSurfaceData
        )
        self.createCGNSConnectivity(includeSurfaceData=includeSurfaceData)

        if forOverset: return

        self.recoverFSFlowSolution()
        if Cmpi.size > 1:
            Cmpi._setProc(self.pyTree, Cmpi.rank)
            zones = Internal.getZones(self.pyTree)
            for z in zones: z[0] += str(Cmpi.rank)

        if forFFDX:
            self.convert2NGon4FFD(**kwargs)
            self.mergeBCsByMarker()
        return

    def convert2FSDM(self):
        """Convert a mesh from CGNS to FSDM"""
        z_ncFaces = None
        IBMDatasets = None

        self.recoverCGNSMeshInfo()

        if self.nvertices > 0:
            if not self.conformal:
                self.prepareDatasetOfNonConformalFaces()
                z_ncFaces = self.createZoneOfNonConformalFaces()
            self.prepareCGNSConnectivities()
            self.recoverCGNSCoordinates()
            self.recoverCGNSConnectivity()

        self.initializeFSMesh(z_ncFaces)
        if self.nvertices > 0:
            if self.IBM: self.initializeIBMDatasets()
            IBMDatasets = self.recoverPointList2BoundaryMarkers()
        if Cmpi.size > 1:
            self.initializeFSBCs_MPI(IBMDatasets)
            self.parallelDeduplicateNodesFSMesh()
        else:
            self.initializeFSBCs(IBMDatasets)
        if Cmpi.size == 1: self.initializeFSFlowSolution()  # TODO

        self.checkFSMesh()
        return

    @profile_time
    def recoverFSMeshInfo(self, includeGhostCells=False):
        """Import and/or fetch FS mesh data and initialize the corresponding
        class attributes"""
        if Cmpi.master and self.verbose: print("Fetching FSDM mesh data.")
        if self.fsmesh is None:  # Import FS mesh
            self.fsmesh = FSMesh(self.clac)
            meshOps = buildMeshOps(self.meshName, preserveCellStacks=True,
                                   verbose=self.verbose)
            self.fsmesh.DoOps(meshOps) or FSError.PrintAndExit()

        # Get mesh info
        self.nsurfaceCells = 0
        self.nvolumeCells = 0
        self.fsVolumeCellTypes = []
        self.fsSurfaceCellTypes = []
        self.fsCellTypes = []

        # Get number of vertices
        self.nvertices = self.fsmesh.GetNOwnedCells(FSMeshEnums.CT_Node)

        # Compute number of surface cells
        for cellType in FSUnstructSurfaceCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsSurfaceCellTypes.append(cellType)
                if includeGhostCells:
                    self.nsurfaceCells += self.fsmesh.GetNCells(cellType)
                else:
                    self.nsurfaceCells += self.fsmesh.GetNOwnedCells(cellType)

        # Compute number of volume cells
        for cellType in FSUnstructVolumeCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsVolumeCellTypes.append(cellType)
                if includeGhostCells:
                    self.nvolumeCells += self.fsmesh.GetNCells(cellType)
                else:
                    self.nvolumeCells += self.fsmesh.GetNOwnedCells(cellType)

        self.fsCellTypes = self.fsVolumeCellTypes + self.fsSurfaceCellTypes
        return

    def setNumpyCoordinates(self, coords):
        if self.flipYZAxes:
            self.np_coordinates = numpy.empty_like(coords)
            self.np_coordinates[:,0] = coords[:,0]
            self.np_coordinates[:,1] = coords[:,2]
            self.np_coordinates[:,2] = -coords[:,1]
        else:
            self.np_coordinates = coords
        return

    @profile_time
    def recoverFSCoordinates(self):
        """
        Fetch FS mesh coordinates and initialize the corresponding numpy class
        attributes
        """
        if Cmpi.master and self.verbose: print("Fetching FS mesh coordinates.")
        fs_coordinates = self.fsmesh.GetUnstructDataset(self.coordsName).GetValues()
        np_coordinates = numpy.array(fs_coordinates.Buffer(), copy=True)
        self.setNumpyCoordinates(np_coordinates)
        return

    @profile_time
    def mergeUnstructSurfaceConnectivities(self):
        # Find CGNS surface element numbers corresponding to existing
        # FS surface cell types
        cgnsSurfCellNos = []
        for fsCellType in self.fsSurfaceCellTypes:
            cgnsSurfCellNos.append(FSCGNSConverter.FS2CGNSCELLNOS[fsCellType])

        z = Internal.getZones(self.pyTree)[0]
        n_elts = Internal.getNodesFromType1(z, "Elements_t")

        # Group surface element nodes by CGNS element number
        eltType2ElementNodesDict = {}
        for n_elt in n_elts:
            eltName = n_elt[0]
            eltNo = Internal.getValue(n_elt)[0]
            if eltNo in cgnsSurfCellNos and eltName != "NonConformalFaces":
                eltType2ElementNodesDict.setdefault(eltNo, [])
                eltType2ElementNodesDict[eltNo].append(n_elt)

        ntotFaces = 0
        for eltNo, n_eltOfType in eltType2ElementNodesDict.items():
            if len(n_eltOfType) <= 1: continue  # single connectivity at most
            # Concatenate CGNS surface mesh connectivities of this type
            eltConns = []
            for n_elt in n_eltOfType:
                eltName = n_elt[0]
                np_eltConn = Internal.getNodeFromName1(n_elt, 'ElementConnectivity')[1]
                eltConns.append(np_eltConn)
            eltConns = numpy.concatenate(eltConns)

            # Update element range and connectivity of the first element node of this type
            np_eltRange = Internal.getNodeFromName1(n_eltOfType[0], 'ElementRange')[1]
            ec = Internal.getNodeFromName1(n_eltOfType[0], 'ElementConnectivity')

            nvpe = Internal.eltNo2EltName(eltNo)[1]
            nfaces = eltConns.shape[0]
            np_eltRange[0] = self.nvolumeCells + ntotFaces + 1
            np_eltRange[1] = np_eltRange[0] + int(nfaces//nvpe)
            ec[1] = eltConns
            n_eltOfType[0][0] = Internal.eltNo2EltName(eltNo)[0]

            ntotFaces += nfaces

            # Delete all but the first element node of this type
            for n_elt in n_eltOfType[1:]:
                Internal._rmNode(z, n_elt)  # TODO Christophe, why not n_elts

            # TODO update this obscure variable!
            self.fsCellTypesBCs.append(eltNo)
        return

    def createBCZonePerSurfaceElementType(self):  # TODO rename TODO what if there are already present - no deletion
        z = Internal.getZones(self.pyTree)[0]
        n_zoneBCs = Internal.getNodesFromType1(z, "ZoneBC_t")
        if len(n_zoneBCs) == 0: return  # No BCs defined

        n_bcs = Internal.getNodesFromType1(n_zoneBCs, "BC_t")
        if len(self.bcsNames) != len(n_bcs):
            print(f"WARNING: Number of BCs in CGNS mesh, {len(n_bcs)}, does not "
                  f"match the number of BCs in bcsNames, {len(self.bcsNames)}.")
            #sys.exit(1)

        # Loop over all existing conformal BC nodes (IndexArray or IndexRange)
        # and offset vertex indices by the number of volume cells
        np_vertexPL = numpy.array([], dtype=Internal.E_NpyInt)
        ntotBCVertices = 0
        for i, n_bc in enumerate(n_bcs):
            bcName = Internal.getName(n_bc)
            if bcName == "NonConformalFaces": continue
            if not any(suffix in bcName for suffix in [".TRI", ".QUAD"]):
                n_bc[0] = self.bcsNames[i]

            offset = self.nvolumeCells + ntotBCVertices
            n_IA = Internal.getNodeFromType1(n_bc, "IndexArray_t")
            if n_IA is not None and n_IA[0] == "PointList":
                np_vertexPL = Internal.getNodeFromName1(n_bc, "PointList")[1][0]
                np_vertexPL += offset
                ntotBCVertices += np_vertexPL.shape[0]
            n_IR = Internal.getNodeFromType1(n_bc, "IndexRange_t")
            if n_IR is not None and n_IR[0] == "ElementRange":
                np_eltRange = Internal.getNodeFromName1(n_bc, "ElementRange")[1][0]
                nfaces = np_eltRange[1] - np_eltRange[0] + 1
                np_eltRange[0] = offset + 1
                np_eltRange[1] = offset + nfaces
                ntotBCVertices += nfaces
        return

    def prepareDatasetOfNonConformalFaces(self):
        bc_names = [bc[0] for bc in Internal.getNodesFromType(self.pyTree, 'BC_t')]
        if "QuadNQuad" in bc_names:
            rm = Internal.getNodeFromName(self.pyTree, 'QuadNQuad')
            old_name_hf = rm[0]
            Internal._renameNode(self.pyTree, old_name_hf, "NonConformalFaces")
            n_NCF = Internal.getNodeFromName(self.pyTree, "NonConformalFaces")
            if Internal.getNodeFromName(n_NCF, "PointList") != None:
                len_NCF = Internal.getNodeFromName(n_NCF, "PointList")[1][0].shape[0]
            elif Internal.getNodeFromName(n_NCF, "ElementRange") != None:
                eltRange = Internal.getNodeFromName(n_NCF, "ElementRange")[1]
                if len(eltRange) == 1:
                    ERmin, ERmax = eltRange[0][:]
                elif len(eltRange) == 2:
                    ERmin, ERmax = eltRange[0], eltRange[1]
                else:
                    print("ERROR: Problem with the element range of non  "
                          "conformal interfaces.")
                    sys.exit(1)
                len_NCF = ERmax - ERmin + 1
            self.nsurfaceCells -= len_NCF
        return

    @profile_time
    def createZoneOfNonConformalFaces(self):
        z_octreeFaces = None
        xCoord = Internal.getNodeFromName(self.pyTree, "CoordinateX")[1]
        yCoord = Internal.getNodeFromName(self.pyTree, "CoordinateY")[1]
        zCoord = Internal.getNodeFromName(self.pyTree, "CoordinateZ")[1]
        n_octreeFaces = Internal.getNodesFromName(self.pyTree, "NonConformalFaces")

        if n_octreeFaces != []:
            octreeFaces_EC_global = Internal.getNodeFromName(n_octreeFaces, "ElementConnectivity")[1]
            len_NCF = len(octreeFaces_EC_global)//4

            n_NCF = Internal.getNodesFromName(self.pyTree, 'NonConformalFaces')
            for i in n_NCF: Internal._rmNode(self.pyTree, i)

            idx = numpy.unique(octreeFaces_EC_global, return_index=True)[1]
            octreeFaces_idx_nodes = octreeFaces_EC_global[numpy.sort(idx)] - 1  # indices loc2glob
            len_nodes_NCF = len(octreeFaces_idx_nodes)
            if len_nodes_NCF > 0:
                xCoord_nodes_NCF = xCoord[octreeFaces_idx_nodes]
                yCoord_nodes_NCF = yCoord[octreeFaces_idx_nodes]
                zCoord_nodes_NCF = zCoord[octreeFaces_idx_nodes]

                global_ncFaces = numpy.reshape(octreeFaces_EC_global-1,(len_NCF,4))
                indices1 = numpy.linspace(0, len_nodes_NCF-1, len_nodes_NCF, dtype=int)
                glob2loc = numpy.zeros(self.nvertices, dtype=int)
                glob2loc[octreeFaces_idx_nodes] = indices1
                local_ncFaces = glob2loc[global_ncFaces]

                z_octreeFaces = Internal.newZone(
                    name="NonConformalFaces",
                    zsize=[[len(xCoord_nodes_NCF),len(octreeFaces_idx_nodes)]],
                    ztype="Unstructured"
                )
                gc = Internal.newGridCoordinates(parent=z_octreeFaces)
                Internal.newDataArray('CoordinateX', value=xCoord_nodes_NCF, parent=gc)
                Internal.newDataArray('CoordinateY', value=yCoord_nodes_NCF, parent=gc)
                Internal.newDataArray('CoordinateZ', value=zCoord_nodes_NCF, parent=gc)
                Internal.newElements(
                    name="NonconformalFaces",
                    etype=7,
                    econnectivity=numpy.ravel(local_ncFaces+1),
                    erange=[1, len_NCF],
                    eboundary=0,
                    parent=z_octreeFaces
                )
            else:
                z_octreeFaces = None #Internal.newZone(name = "NonConformalFaces",zsize=[[0,0]],ztype="Unstructured")
        else:
            z_octreeFaces = None #Internal.newZone(name = "NonConformalFaces",zsize=[[0,0]],ztype="Unstructured")
        return z_octreeFaces

    @profile_time
    def prepareCGNSConnectivities(self): # TODO split struct and unstruct
        """
        Structured meshes:

        Unstructured meshes:
            Merge connectivities such that there is at most one connectivity per element type 
        """
        eltTypeList = []
        if self.meshType == "Structured":
            C._rmBCOfType(self.pyTree, "BCMatch")
            C._rmBCOfType(self.pyTree, "BCDegeneratedLine")
        elif self.meshType == "Unstructured":
            n_elts = Internal.getNodesFromType(self.pyTree, "Elements_t")
            eltTypeList = [Internal.getValue(n_elt)[0] for n_elt in n_elts]
        foundMultipleEltNodeOfType = len(set(eltTypeList)) < len(eltTypeList)

        if Cmpi.size > 1 or self.meshType == "Structured" or foundMultipleEltNodeOfType:
            if Cmpi.master and self.verbose:
                print("Merging several QUAD-CGNS zones into one zone.")
            novol = len(self.fsVolumeCellTypes)
            zones = Internal.getZones(self.pyTree)
            z = zones[0]
            for noz in range(novol, len(zones)):
                z = T.join([z, zones[noz]])
            #z = T.join(zones) # TODO this deletes BCs unfortunately

            n_zoneBCs = Internal.getNodesFromType1(z, 'ZoneBC_t')
            n_bcs = Internal.getNodesFromType1(n_zoneBCs, 'BC_t')
            
            if self.meshType == "Unstructured":
                # Fill the list of root BC names
                for n_bc in n_bcs:
                    if not any(suffix in n_bc[0] for suffix in [".TRI", ".QUAD"]):
                        self.bcsNames.append(n_bc[0].replace('.', ''))
                    else:
                        self.bcsNames.append(n_bc[0].split(".")[0])
            else:  # "Structured"
                zbcs = []; bcTypes = []; bcs = []
                for n_bc in n_bcs:
                    if not any(suffix in n_bc[0] for suffix in [".TRI", ".QUAD"]):
                        self.bcsNames.append(n_bc[0].replace('.', ''))
                    else:
                        self.bcsNames.append(n_bc[0].split(".")[0])
                    bcType = Internal.getValue(n_bc)
                    if bcType not in bcTypes:
                        bcTypes.append(bcType)
                        bcs.append(n_bc)

                for bcType in bcTypes:
                    zbc = C.extractBCOfType(self.pyTree, bcType)
                    zbc = C.convertArray2Hexa(zbc)
                    zbc = T.join(zbc)
                    self.nsurfaceCells += Internal.getValue(zbc)[0][1] # TODO += careful ?
                    zbcs.append(zbc)

                z = C.convertArray2Hexa(z)
                if float(C.__version__) < 5.0: z = G.close(z)
                self.nvertices = int(Internal.getValue(z)[0][0])
                nBCs = len(bcTypes)
                for i in range(nBCs):
                    _addBC2ZoneLoc(z, bcTypes[i], bcTypes[i], zbcs[i])
                self.pyTree = C.newPyTree(["Unstructured", z])

            self.mergeUnstructSurfaceConnectivities()
            self.createBCZonePerSurfaceElementType()
        return None

    @profile_time
    def recoverCGNSMeshInfo(self):
        if Cmpi.master and self.verbose: print("Fetching CGNS mesh info.")

        # Read CGNS mesh
        if self.pyTree is None:
            if Cmpi.size == 1:
                self.pyTree = C.convertFile2PyTree(self.meshName)
            else:
                self.pyTree = Cmpi.convertFile2PyTree(self.meshName, proc=Cmpi.rank)
                bases = Internal.getBases(self.pyTree)
                if len(bases) != Cmpi.size:
                    import XCore.PyTree as XC
                    self.pyTree = XC.loadAndSplitElt(self.meshName)

        # Delete empty bases from the pyTree
        emptyBaseNames = []
        bases = Internal.getBases(self.pyTree)
        for base in bases:
            zones = Internal.getZones(base)
            if zones == []: emptyBaseNames.append(base[0])
        for name in emptyBaseNames:
            Internal._rmNodesByNameAndType(self.pyTree, name, "CGNSBase_t")

        # Get mesh info
        self.nvertices = 0
        self.nsurfaceCells = 0
        self.nvolumeCells = 0
        self.fsVolumeCellTypes = []
        self.fsSurfaceCellTypes = []
        self.fsCellTypes = []

        # Reorder volume and surface element types as in FSDM
        self.reorderCells()

        zones = Internal.getZones(self.pyTree)
        if zones != []:
            zone = zones[0]
            zoneDim = Internal.getZoneDim(zone)
            self.meshType = zoneDim[0]

            if self.meshType == "Unstructured":
                self.nvertices = zoneDim[1]
                self.nvolumeCells = zoneDim[2]

                n_elts = Internal.getNodesFromType(zone, "Elements_t")
                for n_elt in n_elts:
                    cgnsCellType = Internal.getValue(n_elt)[0]
                    fsCellType = FSCGNSConverter.CGNS2FSCELLNOS[cgnsCellType]
                    if (
                        fsCellType in FSUnstructVolumeCellTypes
                        and fsCellType not in self.fsVolumeCellTypes
                    ):
                        self.fsVolumeCellTypes.append(fsCellType)
                    elif fsCellType in FSUnstructSurfaceCellTypes:
                        eltRange = Internal.getNodeFromName(n_elt, "ElementRange")[1]
                        nfaces = eltRange[1] - eltRange[0] + 1
                        if nfaces == 0:
                            raise ValueError("ERROR: Empty surface connectivity.")
                        self.nsurfaceCells += nfaces
                        if fsCellType not in self.fsSurfaceCellTypes:
                            self.fsSurfaceCellTypes.append(fsCellType)

            elif self.meshType == "Structured":
                index_i, index_j, index_k = zoneDim[1], zoneDim[2], zoneDim[3]
                self.nvertices = index_i * index_j * index_k
                self.nvolumeCells = (index_i-1) * (index_j-1) * (index_k-1)
                if self.nvolumeCells > 0:
                    self.fsVolumeCellTypes.append(8)  # Hexa
                    self.fsSurfaceCellTypes.append(4)  # Quad

            self.fsCellTypes = self.fsVolumeCellTypes + self.fsSurfaceCellTypes

        #Cmpi.barrier()
        #gath_fsCellTypes = Cmpi.allgather(self.fsCellTypes)
        #gath_fsCellTypes = list({x for v in gath_fsCellTypes for x in v})
        #if zones == []:
        #    self.fsCellTypes = gath_fsCellTypes
        return

    @profile_time
    def recoverCGNSCoordinates(self):
        if Cmpi.master and self.verbose:
            print("Fetching CGNS PyTree coordinates.")
        foundCoords = [False for _ in range(3)]
        coordName2Pos = {"CoordinateX": 0, "CoordinateY": 1, "CoordinateZ": 2}
        np_coordinates = numpy.empty((self.nvertices, 3))

        z = Internal.getZones(self.pyTree)[0]
        n_gridCoords = Internal.getNodeFromType1(z, "GridCoordinates_t")
        n_coords = Internal.getNodesFromType1(n_gridCoords, "DataArray_t")

        for n_coord in n_coords:
            i = coordName2Pos.get(n_coord[0], None)
            if i is None: continue
            np_coordinates[:,i] = n_coord[1]
            foundCoords[i] = True

        if not all(foundCoords):
            print("ERROR: Coordinates missing in GridCoordinates_t, exiting")
            sys.exit(1)
        self.setNumpyCoordinates(np_coordinates)
        return

    def initializeCell2Proc(self, fsCellType, ncellsOfType=0):
        nprocs = Cmpi.size
        gath_cell2Proc = ArrayOps.AllGather(ncellsOfType, self.clac)
        fs_cell2Proc = FSIntArray(nprocs+1)
        fs_cell2Proc[0] = 0
        for i in range(nprocs):
            fs_cell2Proc[i+1] = fs_cell2Proc[i] + int(gath_cell2Proc[i])
        self.cell2ProcDict[fsCellType] = fs_cell2Proc
        return

    @profile_time
    def recoverCGNSConnectivity(self):
        self.cell2NodeDict = {}
        n_elts = Internal.getNodesFromType(self.pyTree, "Elements_t")
        for n_elt in n_elts:
            eltType = Internal.getValue(n_elt)[0]
            n_EC = Internal.getNodeFromName1(n_elt, "ElementConnectivity")
            if n_EC is None:
                print("ERROR: mesh connectivity not found in CGNS node of type Elements_t.")
                sys.exit(1)
            else:
                nvpe = Internal.eltNo2EltName(eltType)[1]
                nelts = n_EC[1].size // nvpe
                fsCellType = FSCGNSConverter.CGNS2FSCELLNOS[eltType]
                if fsCellType not in self.cell2NodeDict: self.cell2NodeDict[fsCellType] = []
                self.cell2NodeDict[fsCellType].append(n_EC[1].reshape(nelts, nvpe))
        for fsCellType in self.cell2NodeDict:
            self.cell2NodeDict[fsCellType] = numpy.concatenate(self.cell2NodeDict[fsCellType]) - 1
        return

    @profile_time
    def initializeFSConnectivity(self, cellType):
        hasCells = cellType in self.cell2NodeDict
        if hasCells:
            np_cell2node = self.cell2NodeDict[cellType]
            fs_cell2node = FSIntArray(*np_cell2node.shape)
        else:
            fs_cell2node = FSIntArray(0, FSCellInfo.NNodes(cellType))

        if Cmpi.size > 1:
            if hasCells:
                nnodesPrevious = self.cell2ProcDict[FSMeshEnums.CT_Node][Cmpi.rank]
                numpy.copyto(
                    numpy.array(fs_cell2node.Buffer(), copy=False),
                    np_cell2node + nnodesPrevious,
                    casting='same_kind'
                )
            self.fsmesh.InitUnstructCells(
                cellType,
                self.cell2ProcDict[cellType],
                fs_cell2node,
                True
            )
        else:
            if hasCells:
                numpy.copyto(
                    numpy.array(fs_cell2node.Buffer(), copy=False),
                    np_cell2node,
                    casting='same_kind'
                )
            self.fsmesh.InitUnstructCells(cellType, fs_cell2node, True)
        return

    @profile_time
    def initializeFSMesh(self, z_ncFaces=None):
        if self.fsmesh is None:
            self.fsmesh = FSMesh(self.clac)
        self.fsmesh.BeginInitialization()
        if Cmpi.size > 1:
            self.initializeCell2Proc(FSMeshEnums.CT_Node, self.nvertices)
            self.fsmesh.InitUnstructNodes(self.cell2ProcDict[FSMeshEnums.CT_Node])

            for fsCellType in FSCGNSConverter.FS2CGNSCELLNOS:
                ncellsOfType = 0
                if fsCellType in self.cell2NodeDict:
                    ncellsOfType = self.cell2NodeDict[fsCellType].shape[0]
                self.initializeCell2Proc(fsCellType, ncellsOfType)
                self.initializeFSConnectivity(fsCellType)
        else:
            self.fsmesh.InitUnstructNodes(self.nvertices)
            for fsCellType in self.fsCellTypes:
                self.initializeFSConnectivity(fsCellType)

        if not self.conformal:
            if Cmpi.size > 1:
                self.initializePseudoCell_QuadNQuad_MPI(z_ncFaces)
            else:
                self.initializePseudoCell_QuadNQuad(z_ncFaces)

        self.fsmesh.EndInitialization()
        self.initializeFSBCCoordinates(
            [FSDataName(self.coordsName)],
            FSMeshEnums.CT_Node
        )

        if self.nvertices > 0:
            fs_coordinates = self.fsmesh.GetUnstructDataset(
                FSDataName(self.coordsName)
            ).GetValues()
            numpy.copyto(
                numpy.array(fs_coordinates.Buffer(), copy=False),
                self.np_coordinates,
                casting='no'
            )
        return

    @profile_time
    def initializeIBMDatasets(self):
        # Flis wall distance initialization
        spatial_discretization = self.IBMParameters["spatial discretization"]["type"]
        if spatial_discretization == "FV":
            N_IP_per_element = 1
        else:
            try:
                import QuadratureDG as Q
            except ImportError:
                raise ImportError("QuadratureDG module not found.")
            degree = self.IBMParameters["spatial discretization"]["degree"]
            if spatial_discretization == "DG":
                integrationDegree = 2*degree + 1
                quadratureType = "GaussLegendre"
            elif spatial_discretization == "DGSEM":
                integrationDegree = 2*degree - 1
                quadratureType = "GaussLobatto"
            N_IP_per_element = Q.GetReferencePointsHexa(integrationDegree, quadratureType)[0]

        list_suffix_datasets = [""]
        list_suffix_datasets.extend(range(1, N_IP_per_element))
        for i in range(N_IP_per_element): # WHAT IS THIS?
            flis_node = Internal.getNodeFromName(
                self.pyTree,
                "FlisWallDistance" + str(list_suffix_datasets[i])
            )
            np_flisDistance = Internal.getNodeFromName(flis_node, "TurbulentDistance")[1]
            quantityName = "FlisWallDistance" + str(list_suffix_datasets[i])
            quantityNames = FSStringArray(1)
            quantityNames[0] = quantityName
            quantitySpecs = FSDataSpecArray(1)
            quantitySpecs[0].Length()

            fs_volumeCellTypes = FSIntArray(len(self.fsVolumeCellTypes))
            numpy.copyto(
                numpy.array(fs_volumeCellTypes.Buffer(), copy=False),
                self.fsVolumeCellTypes,
                casting='same_kind'
            )
            self.fsmesh.InitUnstructDataset(
                quantityName,
                FSDatasetInfo(quantityNames, quantitySpecs, fs_volumeCellTypes)
            )
            fs_var = self.fsmesh.GetUnstructDataset(quantityName).GetValues()
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                np_flisDistance[:,None],
                casting='no'
            )
        return

    @profile_time
    def recoverPointList2BoundaryMarkers(self):
        self.dict_bc_elts = {"TRI": [], "QUAD": []}
        n_zoneBC = Internal.getNodeFromType(self.pyTree, "ZoneBC_t")
        if n_zoneBC is None: return []
        pytree_bc_nodes = Internal.getNodesFromType(n_zoneBC, "BC_t")
        current_automatic_marker = 1

        if self.IBM:
            IBM_BC_coords_x = {}
            IBM_BC_coords_y = {}
            IBM_BC_coords_z = {}
            IBM_BC_names = []
            if self.IBMParameters["IBM type"]["type"] == "local":
                wall_bMarkers = self.IBMParameters["IBM type"]["wall boundary markers"]
                BC_wall_coords_x = []
                BC_wall_coords_y = []
                BC_wall_coords_z = []
                BC_wall_names = []

        # Loop on BCs
        for bc_node in pytree_bc_nodes:
            bc_name = bc_node[0]
            # Get list of BC points: based on name to make the difference between point list and point range ;
            if Internal.getNodeFromType(bc_node, "IndexArray_t") is not None:
                pointlist_node = Internal.getNodeFromType(bc_node, "IndexArray_t")
                point_list = numpy.ravel(pointlist_node[1]) - 1
                bMarker_node = Internal.getNodeFromName(bc_node, "BoundaryMarker")
                if bMarker_node is not None:
                    bc_bMarker = Internal.getValue(bMarker_node)
                else:
                    # Get boundary marker: generate one if it does not exist otherwise we expect it to be in a user defined node named "BoundaryMarker"
                    bc_bMarker = current_automatic_marker
                    current_automatic_marker += 1

            elif Internal.getNodeFromType(bc_node, "IndexRange_t") is not None:
                n_er = Internal.getNodeFromType(bc_node, "IndexRange_t")
                if len(n_er[1]) == 1:
                    point_list = numpy.arange(n_er[1][0][0]-1, n_er[1][0][1])
                elif len(n_er[1]) == 2:
                    point_list = numpy.arange(n_er[1][0]-1, n_er[1][1])
                bMarker_node = Internal.getNodeFromName(bc_node, "BoundaryMarker")
                if bMarker_node is not None:
                    print("WARNING: Boundary markers in the CGNS pyTree will "
                          "not be taken into account. Automatic boundary "
                          "markers are defined instead.")
                bc_bMarker = current_automatic_marker
                current_automatic_marker += 1

            # Fill boundary dicts
            self.bMarker2BCNameDict[bc_bMarker] = bc_name
            self.bMarker2FacePLDict[bc_bMarker] = point_list

            bcNameSplit = bc_name.split(".")
            if len(bcNameSplit) > 1:
                if bcNameSplit[1].startswith("TRI"):
                    self.dict_bc_elts["TRI"].append(bc_bMarker)
                elif bcNameSplit[1].startswith("QUAD"):
                    self.dict_bc_elts["QUAD"].append(bc_bMarker)

            bc_dataset_node = Internal.getNodeFromType(bc_node, "BCDataSet_t")
            if bc_dataset_node is not None:
                #raise ValueError("recoverPointList2BoundaryMarkers: empty BCDataset for IBM.")
                bc_data_nodes = Internal.getNodesFromType(bc_dataset_node, "BCData_t")
                for data_node in bc_data_nodes:
                    fs_bc_dataset_name = data_node[0]
                    if self.IBM and bc_name.startswith("IBMWall"):
                        if bc_name not in IBM_BC_coords_x:
                            IBM_BC_coords_x[bc_name] = []
                            IBM_BC_coords_y[bc_name] = []
                            IBM_BC_coords_z[bc_name] = []
                        IBM_BC_names.append(fs_bc_dataset_name)
                        if self.flipYZAxes:
                            IBM_BC_coords_x[bc_name].append(data_node[2][0][1])
                            IBM_BC_coords_y[bc_name].append(data_node[2][2][1])
                            IBM_BC_coords_z[bc_name].append(-data_node[2][1][1])
                        else:
                            IBM_BC_coords_x[bc_name].append(data_node[2][0][1])
                            IBM_BC_coords_y[bc_name].append(data_node[2][1][1])
                            IBM_BC_coords_z[bc_name].append(data_node[2][2][1])

                    elif self.IBM and self.IBMParameters["IBM type"]["type"] == "local" and bc_bMarker in wall_bMarkers:
                        BC_wall_names.append(fs_bc_dataset_name)
                        if self.flipYZAxes:
                            BC_wall_coords_x.append(data_node[2][0][1])
                            BC_wall_coords_y.append(data_node[2][2][1])
                            BC_wall_coords_z.append(-data_node[2][1][1])
                        else:
                            BC_wall_coords_x.append(data_node[2][0][1])
                            BC_wall_coords_y.append(data_node[2][1][1])
                            BC_wall_coords_z.append(data_node[2][2][1])

        if not self.IBM: return []
        elif self.IBMParameters["IBM type"]["type"] == "global":
            return [
                IBM_BC_names,
                IBM_BC_coords_x, IBM_BC_coords_y, IBM_BC_coords_z
            ]
        else:  # formulation 'locale'
            return [
                IBM_BC_names,
                IBM_BC_coords_x, IBM_BC_coords_y, IBM_BC_coords_z,
                BC_wall_names,
                BC_wall_coords_x, BC_wall_coords_y, BC_wall_coords_z
            ]

    @profile_time
    def initializeFSBCs(self, IBMDatasets=[]):
        #if not self.bcDict: return  # TODO VINCENT
        #We now have our point list for each marker so we can init the cell attribute in the fsmesh
        np_markerArray = numpy.zeros(self.nsurfaceCells, dtype=Internal.E_NpyInt)
        for marker, np_facePL in self.bMarker2FacePLDict.items():
            np_markerArray[np_facePL - self.nvolumeCells] = marker

        offset = 0
        # Loop on surface cell types in the mesh and slice the array above to get the data we need
        for cellType in self.fsSurfaceCellTypes:
            if self.dict_bc_elts["QUAD"] != [] and self.dict_bc_elts["TRI"] != []:
                cgnsCellNo = FSCGNSConverter.FS2CGNSCELLNOS[cellType]
                cgnsCellType = Internal.eltNo2EltName(cgnsCellNo)[0]
                temp = set(self.dict_bc_elts[cgnsCellType])
                res = [i for i, val in enumerate(np_markerArray) if val in temp]  # TODO REFACTOR
                np_marker_array_cell_type = np_markerArray[res]
            elif len(self.fsSurfaceCellTypes) == 1:
                np_marker_array_cell_type = np_markerArray
            else:
                ncells = self.fsmesh.GetNCells(cellType)
                res = numpy.arange(offset, offset + ncells)
                np_marker_array_cell_type = np_markerArray[res]
                offset += ncells

            fs_marker_array_cell_type = FSIntArray(np_marker_array_cell_type.shape[0])
            numpy.copyto(
                numpy.array(
                    fs_marker_array_cell_type.Buffer(),
                    copy=False
                ),
                np_marker_array_cell_type,
                casting='same_kind'
            )
            self.fsmesh.InitCellAttribute(FS_AT_CADGroupID, cellType, fs_marker_array_cell_type)

        # Then we attach our boundary marker to their name in the fsmesh
        if self.IBM:
            IBM_bMarkers = []
            IBM_names = []
        for marker in self.bMarker2BCNameDict:
            self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, self.bMarker2BCNameDict[marker])
            if self.IBM and self.bMarker2BCNameDict[marker].startswith("IBMWall"):
                IBM_bMarkers.append(marker)
                IBM_names.append(self.bMarker2BCNameDict[marker])

        if self.IBM:
            IBMDataset1 = []
            IBMDataset2 = []
            IBMDataset3 = []
            pointListIBC = []

            for IBM_bMarker, IBM_name in zip(IBM_bMarkers, IBM_names):
                IBMDataset1.append(IBMDatasets[1][IBM_name])
                IBMDataset2.append(IBMDatasets[2][IBM_name])
                IBMDataset3.append(IBMDatasets[3][IBM_name])
                pointListIBC.append(self.bMarker2FacePLDict[IBM_bMarker])
            IBMDataset1 = numpy.concatenate(IBMDataset1, axis=1)
            IBMDataset2 = numpy.concatenate(IBMDataset2, axis=1)
            IBMDataset3 = numpy.concatenate(IBMDataset3, axis=1)
            pointListIBC = numpy.concatenate(pointListIBC, dtype=Internal.E_NpyInt) - self.nvolumeCells

            self.createDatasetOfCoordinatesBC(
                self.fsmesh,
                IBMDataset1, IBMDataset2, IBMDataset3, IBMDatasets[0],
                self.nsurfaceCells,
                pointListIBC
            )

            if self.IBMParameters["IBM type"]["type"] == "local":
                wall_bMarkers = self.IBMParameters["IBM type"]["wall boundary markers"]
                self.createDatasetOfCoordinatesBC(
                    self.fsmesh,
                    IBMDatasets[5], IBMDatasets[6], IBMDatasets[7], IBMDatasets[4],
                    self.nsurfaceCells,
                    self.bMarker2FacePLDict[wall_bMarkers[0]] - self.nvolumeCells
                )
        return

    @profile_time
    def initializeFSBCs_MPI(self, IBMDatasets=[]):
        #if not self.bcDict: return  # TODO VINCENT
        np_markerArray = numpy.zeros(self.nsurfaceCells, dtype=int)
        values = list(self.bMarker2BCNameDict.values())
        gath_values = Cmpi.allgather(values)

        bc_names_all = sorted(set([item for row in gath_values for item in row]))
        bc_markers_all = numpy.arange(1, len(bc_names_all)+1).tolist()

        bMarker2BCName2 = {}
        bMarker2PL2 = {}
        for marker, name in zip(bc_markers_all, bc_names_all):
            if name in self.bMarker2BCNameDict.values():
                idx = list(self.bMarker2BCNameDict.keys())[list(self.bMarker2BCNameDict.values()).index(name)]
                bMarker2BCName2[marker] = self.bMarker2BCNameDict[idx]
                bMarker2PL2[marker] = self.bMarker2FacePLDict[idx]

        for marker in bMarker2BCName2:
            point_list = bMarker2PL2[marker] - self.nvolumeCells
            np_markerArray[point_list] = marker

        # Loop on surface cell types in the mesh and slice the array above to get the data we need
        for cellType in self.fsSurfaceCellTypes:
            if self.dict_bc_elts["QUAD"] != [] and self.dict_bc_elts["TRI"] != []:
                cgnsCellNo = FSCGNSConverter.FS2CGNSCELLNOS[cellType]
                cgnsCellType = Internal.eltNo2EltName(cgnsCellNo)[0]
                temp = set(self.dict_bc_elts[cgnsCellType])
                res = [i for i, val in enumerate(np_markerArray) if val in temp]
                np_marker_array_cell_type = np_markerArray[res]
            else:
                np_marker_array_cell_type = np_markerArray
            fs_marker_array_cell_type = FSIntArray(np_marker_array_cell_type.shape[0])
            numpy.copyto(
                numpy.array(fs_marker_array_cell_type.Buffer(), copy=False),
                np_marker_array_cell_type,
                casting='same_kind'
            )
            self.fsmesh.InitCellAttribute(
                FS_AT_CADGroupID, cellType,
                fs_marker_array_cell_type
            )

        # Then we attach our boundary marker to their name in the fsmesh
        # for marker in bMarker2BCName2.keys() :
        if self.IBM:
            IBM_bMarkers = []
            IBM_names = []
            fsdatanames = ["WallPointCoordinates", "DonorPointCoordinates"]
            fs_surfaceCellTypes = FSIntArray(len(self.fsSurfaceCellTypes))
            if self.fsSurfaceCellTypes:
                numpy.copyto(
                    numpy.array(fs_surfaceCellTypes.Buffer(), copy=False),
                    self.fsSurfaceCellTypes,
                    casting='same_kind'
                )
            self.initializeFSBCCoordinates(fsdatanames, fs_surfaceCellTypes)

        for marker, name in zip(bc_markers_all, bc_names_all) :
            self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, name)
            if self.IBM and marker in self.bMarker2BCNameDict and self.bMarker2BCNameDict[marker].startswith("IBMWall"):
                IBM_bMarkers.append(marker)
                IBM_names.append(self.bMarker2BCNameDict[marker])

        if self.IBM and IBM_names != []:
            IBMDataset1 = []
            IBMDataset2 = []
            IBMDataset3 = []
            pointListIBC = []

            for IBM_bMarker, IBM_name in zip(IBM_bMarkers, IBM_names):
                IBMDataset1.append(IBMDatasets[1][IBM_name])
                IBMDataset2.append(IBMDatasets[2][IBM_name])
                IBMDataset3.append(IBMDatasets[3][IBM_name])
                pointListIBC.append(self.bMarker2FacePLDict[IBM_bMarker])
            IBMDataset1 = numpy.concatenate(IBMDataset1, axis=1)
            IBMDataset2 = numpy.concatenate(IBMDataset2, axis=1)
            IBMDataset3 = numpy.concatenate(IBMDataset3, axis=1)
            pointListIBC = numpy.concatenate(pointListIBC, dtype=Internal.E_NpyInt) - self.nvolumeCells

            self.createDatasetOfCoordinatesBC(
                self.fsmesh,
                IBMDataset1, IBMDataset2, IBMDataset3, IBMDatasets[0],
                self.nsurfaceCells,
                pointListIBC
            )
        return

    def initializeFSBCCoordinates(self, bcNames, cellType):
        coordNames = FSStringArray(3)
        coordNames[0] = FSDataName.Coordinate().X()
        coordNames[1] = FSDataName.Coordinate().Y()
        coordNames[2] = FSDataName.Coordinate().Z()
        coordSpecs = FSDataSpecArray(3)
        coordSpecs[0].Length()
        coordSpecs[1].Length()
        coordSpecs[2].Length()

        for fsdataname in bcNames:
            self.fsmesh.InitUnstructDataset(
                fsdataname,
                FSDatasetInfo(coordNames, coordSpecs, cellType)
            )
        return

    def createDatasetOfCoordinatesBC(self, fsmesh, coords_x, coords_y, coords_z,
                                     BC_names, nb_cell_surf, point_list): # TODO VINCENT rewrite entirely for loop
        #point_list = numpy.asarray(point_list) TODO VINCENT
        if not Cmpi.size > 1:
            fs_surfaceCellTypes = FSIntArray(len(self.fsSurfaceCellTypes))
            numpy.copyto(
                numpy.array(fs_surfaceCellTypes.Buffer(), copy=False),
                self.fsSurfaceCellTypes,
                casting='same_kind'
            )
            self.initializeFSBCCoordinates(BC_names, fs_surfaceCellTypes)

        # TODO VINCENT
        # if len(coords_x) > 1:
        #     coords_x = numpy.concatenate(coords_x)
        #     coords_y = numpy.concatenate(coords_y)
        #     coords_z = numpy.concatenate(coords_z)
        # nVertices = coords_x.size
        # np_coordinates = numpy.column_stack((coords_x, coords_y, coords_z))

        for i in range(len(coords_x)):
            coord_x = coords_x[i]
            coord_y = coords_y[i]
            coord_z = coords_z[i]

            nVertices = coord_x.size
            np_coordinates = numpy.append(
                coord_x[:,None],
                coord_y[:,None],
                axis=1
            )
            np_coordinates = numpy.append(
                np_coordinates,
                coord_z[:,None],
                axis=1
            )
            fs_coordinates = FSFloatArray(nVertices, 3)
            numpy.copyto(
                numpy.array(fs_coordinates.Buffer(), copy=False),
                np_coordinates,
                casting='no'
            )

            dataset = numpy.zeros((nb_cell_surf, 3))
            #dataset[point_list] = np_coordinates # TODO VINCENT
            for j in range(nVertices):
                dataset[point_list[j]][0] = np_coordinates[j][0]
                dataset[point_list[j]][1] = np_coordinates[j][1]
                dataset[point_list[j]][2] = np_coordinates[j][2]
            fsdataname = FSDataName(BC_names[i])

            fs_var = self.fsmesh.GetUnstructDataset(fsdataname).GetValues()
            fs_var.Fill(0.0)
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                dataset,
                casting='safe'
            )
        return
    
    @profile_time
    def checkFSMesh(self):
        isMeshOK = self.fsmesh.Check()
        if all(Cmpi.allgather(isMeshOK)):
            if Cmpi.master and self.verbose: print("FSMesh successfully created.")
        else:
            if not isMeshOK:
                print(f"[{Cmpi.rank}] ERROR: failed checking the FSMesh.")
            sys.exit(1)

    @profile_time
    def initializeCGNSCoordinates(self):
        if Cmpi.master and self.verbose: print("Initialising CGNS PyTree coordinates.")
        self.pyTree = Internal.newCGNSTree()
        base = Internal.newCGNSBase('Base', 3, 3, parent=self.pyTree)
        self.nvertices = self.np_coordinates.shape[0]
        pyTree_zone = Internal.newZone(
            name='Zone1', # name='zone.{}'.format(Cmpi.rank), TODO
            zsize=[[self.nvertices, self.nvolumeCells, 0]],
            ztype=self.meshType,
            family=None,
            parent=base
        )

        # Create CGNS node for coordinates
        n_coordinates = Internal.newGridCoordinates(parent=pyTree_zone)
        Internal.newDataArray('CoordinateX', value=self.np_coordinates[:,0],
                              parent=n_coordinates)
        Internal.newDataArray('CoordinateY', value=self.np_coordinates[:,1],
                              parent=n_coordinates)
        Internal.newDataArray('CoordinateZ', value=self.np_coordinates[:,2],
                              parent=n_coordinates)
        return

    @profile_time
    def recoverFSConnectivity(self, includeGhostCells=False,
                              includeSurfaceData=True):
        """
        Fetch FS mesh connectivity data from an FS mesh and initialize the
        corresponding class attributes

        Args:
            includeGhostCells: bool; whether to consider ghost cells

            includeSurfaceData: bool; whether to consider surface data
        """
        if Cmpi.master and self.verbose: print("Fetching FS mesh connectivity data.")
        self.cell2NodeVolumeList = []
        for cellType in self.fsVolumeCellTypes:
            nownedCells = self.fsmesh.GetNOwnedCells(cellType)
            fs_cell2Node = self.fsmesh.GetCell2Node(cellType)
            np_cell2NodeUnravelled = 1 + numpy.array(
                fs_cell2Node.Buffer(),
                copy=True,
                dtype=Internal.E_NpyInt
            )
            if not includeGhostCells:
                np_cell2NodeUnravelled = np_cell2NodeUnravelled[:nownedCells]
            self.cell2NodeVolumeList.append(np_cell2NodeUnravelled.ravel())

        if includeSurfaceData:
            # Get marker list
            fs_bMarkerList = self.fsmesh.GetCellAttributeValuesWithNames("CADGroupID")
            try:
                np_bMarkerList = numpy.array(fs_bMarkerList.Buffer(), copy=True)
            except BufferError:
                np_bMarkerList = []
            uniqueMarkers = {}
            sharedMarkers = set()
            for cellType in self.fsSurfaceCellTypes:
                nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                fs_bMarkerCellType = self.fsmesh.GetCellAttribute("CADGroupID", cellType)
                np_bMarkerCellType = numpy.array(
                    fs_bMarkerCellType.Buffer(),
                    copy=True,
                    dtype=Internal.E_NpyInt
                )
                if not includeGhostCells:
                    np_bMarkerCellType = np_bMarkerCellType[:nownedCells]
                uniqueMarkers[cellType] = set(np_bMarkerCellType)
            if uniqueMarkers: sharedMarkers = set.intersection(*uniqueMarkers.values())

            bMarkerCellTypeDict = {}
            if sharedMarkers:
                if Cmpi.master and self.verbose:
                    print(
                        "Markers associated with different surface element "
                        "types: {}.".format(sharedMarkers)
                    )
                val = 0
                for cellType in self.fsSurfaceCellTypes:
                    nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                    fs_bMarkerCellType = self.fsmesh.GetCellAttribute("CADGroupID", cellType)
                    np_bMarkerCellType = numpy.array(fs_bMarkerCellType.Buffer(), copy=True)
                    if not includeGhostCells:
                        np_bMarkerCellType = np_bMarkerCellType[:nownedCells]
                    for sharedMarker in sharedMarkers:
                        np_bMarkerCellType = numpy.where(
                            np_bMarkerCellType == sharedMarker,
                            sharedMarker+val,
                            np_bMarkerCellType
                        )
                        if (sharedMarker+val) not in np_bMarkerList:
                            np_bMarkerList = numpy.append(np_bMarkerList, sharedMarker+val)
                        self.bcDict[sharedMarker+val] = self.bcDict[sharedMarker]
                    bMarkerCellTypeDict[cellType] = np_bMarkerCellType
                    val += 0.1
            else:
                for cellType in self.fsSurfaceCellTypes:
                    nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                    fs_bMarkerCellType = self.fsmesh.GetCellAttribute("CADGroupID", cellType)
                    bMarkerCellTypeDict[cellType] = numpy.array(
                        fs_bMarkerCellType.Buffer(),
                        copy=True
                    )[:nownedCells]  # no ghost cells

            for marker in np_bMarkerList:
                indices_vector = []
                offset = 0
                for cellType in self.fsSurfaceCellTypes:
                    nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                    fs_cell2Node = self.fsmesh.GetCell2Node(cellType)
                    np_cell2NodeUnravelled = (
                        1 + numpy.array(
                            fs_cell2Node.Buffer(),
                            copy=True,
                            dtype=Internal.E_NpyInt
                        )
                    )[:nownedCells]
                    np_bMarkerCellType = bMarkerCellTypeDict[cellType]

                    indices_vector = numpy.ravel(numpy.argwhere(np_bMarkerCellType == marker))
                    if len(indices_vector) > 0:
                        self.bcsNames.append(str(self.fsmesh.GetCellAttributeValueName("CADGroupID", int(marker))))
                        self.indicesPerBdr.append(indices_vector+offset)
                        self.cell2NodeSurfaceList.append(numpy.ravel(np_cell2NodeUnravelled[indices_vector]))
                        self.fsCellTypesBCs.append(cellType)
                        self.fsMarkers.append(marker)
                    offset = nownedCells
        return

    @profile_time
    def createCGNSConnectivity(self, includeSurfaceData=True):
        """Initialise CGNS mesh connectivity from FS mesh data"""
        if Cmpi.master and self.verbose:
            print("Initialising CGNS mesh connectivity from FS mesh info.")
        ntotCells = 1
        zone = Internal.getZones(self.pyTree)[0]

        for i, fsCellType in enumerate(self.fsVolumeCellTypes):
            nvpe = int(FSMeshEnums.CellTypeToString(fsCellType)[-1])
            nEltsPerConn = self.cell2NodeVolumeList[i].shape[0]//nvpe
            cgnsEltNo = FSCGNSConverter.FS2CGNSCELLNOS[fsCellType]
            cgnsCellType = Internal.eltNo2EltName(cgnsEltNo)[0]
            Internal.newElements(
                name="GridElements_" + cgnsCellType,
                etype=cgnsCellType,
                econnectivity=self.cell2NodeVolumeList[i],
                erange=[ntotCells, ntotCells + nEltsPerConn - 1],
                eboundary=0,
                parent=zone
            )
            ntotCells += nEltsPerConn

        if not includeSurfaceData: return

        ntotCells = self.nvolumeCells + 1
        for i, fsCellType in enumerate(self.fsCellTypesBCs):
            nvpe = int(FSMeshEnums.CellTypeToString(fsCellType)[-1])
            nFacesPerBC = len(self.cell2NodeSurfaceList[i])//nvpe
            cgnsEltNo = FSCGNSConverter.FS2CGNSCELLNOS[fsCellType]
            cgnsCellType = Internal.eltNo2EltName(cgnsEltNo)[0]
            # FS bcname in self.bcsNames can be overriden by self.bcDict
            bcname, bctype = self.bcDict[self.fsMarkers[i]]
            if bcname is None: bcname = self.bcsNames[i].split(".")[0]
            bcname = f"{bcname}.{cgnsCellType}_{int(self.fsMarkers[i])}"
            bcname = bcname[-32:]  # limitation of the CGNS format
            Internal.newElements(
                name=bcname,
                etype=cgnsCellType,
                erange=[ntotCells, ntotCells+nFacesPerBC-1],
                econnectivity=self.cell2NodeSurfaceList[i],
                eboundary=nFacesPerBC,
                parent=zone
            )

            C._addBC2Zone(
                zone, bcname, bctype,
                elementRange=[ntotCells, ntotCells + nFacesPerBC - 1]
            )
            zoneBC =  Internal.getNodeFromType(zone, "ZoneBC_t")
            lastbcname = C.getLastBCName(bcname)
            node_bc = Internal.getNodeFromName(zoneBC, lastbcname)
            node_bc[0] = bcname
            boundaryStateDataset = Internal.createNode(
                'BCDataSet','BCDataSet_t', parent=node_bc, value='Null'
            )
            boundaryState = Internal.createNode(
                "Boundary", 'BCData_t', parent=boundaryStateDataset
            )
            boundaryState[2].append([
                "BoundaryMarker",
                int(self.fsMarkers[i]), [], 'UserDefinedData_t'
            ])
            ntotCells += nFacesPerBC
        return

    @profile_time
    def recoverFSFlowSolution(self):
        """
        Fetch FSDM flow solution data and initialize that of the CGNS pyTree
        """
        datasets = set(str(d) for d in self.fsmesh.GetUnstructDatasetNames())
        if isinstance(self.datasets, set):
            datasets = datasets.intersection(self.datasets)
        if not datasets: return

        if Cmpi.master and self.verbose: print("Fetching FSDM flow solution data.")

        for dsName in datasets:
            if self.coordsName[:-1] in dsName: continue
            if Cmpi.master and self.verbose: print("  - Dataset: {}.".format(dsName))
            unstructDataset = self.fsmesh.GetUnstructDataset(dsName)
            fs_cellTypes = unstructDataset.GetCellTypes()
            flowSolutionNames = [str(name) for name in unstructDataset.GetNames()]
            fs_flowSolutionValues = unstructDataset.GetValues()

            # Compute the indices of the ghost cells such that they can be removed
            # from the flow solution dataset
            ntotCells = 0
            np_ghostCellsIndices = []
            for cellType in fs_cellTypes:
                nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                nCells = self.fsmesh.GetNCells(cellType)
                np_ghostCellsIndices.append(
                    numpy.arange(
                        ntotCells + nownedCells,
                        ntotCells + nCells,
                        dtype=Internal.E_NpyInt)
                )
                ntotCells += nCells
            np_ghostCellsIndices = numpy.concatenate(np_ghostCellsIndices)

            np_flowSolutionValues = numpy.array(fs_flowSolutionValues.Buffer(), copy=True)
            np_flowSolutionValues = numpy.delete(np_flowSolutionValues, np_ghostCellsIndices, axis=0)

            if fs_cellTypes[0] in self.fsVolumeCellTypes:
                zone = Internal.getZones(self.pyTree)[0]
                n_FS = Internal.newFlowSolution(
                    name='FlowSolution#Centers',
                    gridLocation='CellCenter',
                    parent=zone
                )
                for i, augStateName in enumerate(flowSolutionNames):
                    Internal.newDataArray(
                        f"{dsName}.{augStateName}",
                        value=np_flowSolutionValues[:,i],
                        parent=n_FS
                    )
            else:
                zoneBC = Internal.getNodeFromType(self.pyTree, "ZoneBC_t")
                if zoneBC is not None:
                    n_bcs = Internal.getNodesFromType(zoneBC, "BC_t")
                    for i, n_bc in enumerate(n_bcs):
                        bdrStateDataset = Internal.getNodeFromType(n_bc, 'BCDataSet_t')
                        bdrState = Internal.getNodeFromType(bdrStateDataset, 'BCData_t')
                        for j, flowSolutionName in enumerate(flowSolutionNames):
                            Internal.newDataArray(
                                flowSolutionName,
                                value=np_flowSolutionValues[:,j][self.indicesPerBdr[i]],
                                parent=bdrState
                            )
        return

    @profile_time
    def initializeFSFlowSolution(self):
        """
        Fetch CGNS flow solution data and initialize that in FS
        """
        def checkSetInsertion(s, item):
            return len(s) != (s.add(item) or len(s))
    
        def addFSUnstructDataset(datasetName, loc, varNames, np_vars):
            if Cmpi.master and self.verbose:
                print(f"  - Dataset: {datasetName}.")
                print("\n".join(f"    + Variable: {name} @ {loc}." for name in varNames))
            nvars = len(varNames)
            fsVarNames = FSStringArray(nvars)
            for i, name in enumerate(varNames): fsVarNames[i] = FSDataName(name)
            fsVarSpecs = FSDataSpecArray(nvars)
            if loc == 'CellCenter':
                fs_loc = FSIntArray(len(self.fsVolumeCellTypes))
                numpy.copyto(
                    numpy.array(fs_loc.Buffer(), copy=False),
                    self.fsVolumeCellTypes,
                    casting='same_kind'
                )
            elif loc == 'FaceCenter':
                fs_loc = FSIntArray(len(self.fsSurfaceCellTypes))
                numpy.copyto(
                    numpy.array(fs_loc.Buffer(), copy=False),
                    self.fsSurfaceCellTypes,
                    casting='same_kind'
                )
            else:
                fs_loc = FSIntArray(1)
                fs_loc[0] = FSMeshEnums.CT_Node

            self.fsmesh.InitUnstructDataset(
               datasetName,
               FSDatasetInfo(fsVarNames, fsVarSpecs, fs_loc)
            )
            try:
                fs_vars = self.fsmesh.GetUnstructDataset(datasetName).GetValues()
                numpy.copyto(
                    numpy.array(fs_vars.Buffer(), copy=False),
                    np_vars,
                    casting='no'
                )
            except BufferError:
                pass
            
        if not self.datasets: return
        elif self.datasets == 'all': self.datasets = set(CGNS_CONTAINER_NAMES)
        if Cmpi.master and self.verbose: print("Fetching CGNS flow solution data.")
        
        z = Internal.getZones(self.pyTree)[0]
        n_flowSolns = Internal.getNodesFromType1(z, 'FlowSolution_t')
        if n_flowSolns == []: return
        
        foundVarNames = set()
        for datasetName in self.datasets:
            if datasetName in CGNS_CONTAINER_NAMES:
                n = Internal.getNodeFromName1(z, datasetName)
                if n is None: continue
                n_loc = Internal.getNodeFromType1(n, 'GridLocation_t')
                if n_loc is None: loc = 'Node'
                else: loc = Internal.getValue(n_loc)
                n_vars = Internal.getNodesFromType1(n, 'DataArray_t')
                varNames = []
                np_vars = []
                for n_var in n_vars:
                    varName = Internal.getName(n_var)
                    if self.coordsName[:-1] in varName: continue
                    if not checkSetInsertion(foundVarNames, f"{loc}:{varName}"): continue
                    varNames.append(varName)
                    np_vars.append(Internal.getValue(n_var))
                if not np_vars: continue
                np_vars = numpy.stack(np_vars, axis=-1)
                addFSUnstructDataset(datasetName, loc, varNames, np_vars)
            else:
                if ':' in datasetName:
                    varLoc, varName = datasetName.split(':')[:2]
                    if varLoc == 'nodes': varLoc = 'Node'
                    elif varLoc == 'centers': varLoc = 'CellCenter'
                else:
                    varLoc, varName = None, datasetName
                varNames = []
                np_vars = []
                n_flowSolns = Internal.getNodesFromType1(z, 'FlowSolution_t')
                for n_flowSoln in n_flowSolns:
                    n_loc = Internal.getNodeFromType1(n_flowSoln, 'GridLocation_t')
                    if n_loc is None: loc = 'Node'
                    else: loc = Internal.getValue(n_loc)
                    if varLoc is not None and varLoc != loc: continue
                    n_var = Internal.getNodeFromName1(n_flowSoln, varName)
                    if n_var is None: continue
                    if self.coordsName[:-1] in varName: continue
                    if not checkSetInsertion(foundVarNames, f"{loc}:{varName}"): continue
                    np_var = Internal.getValue(n_var)[:,None]
                    addFSUnstructDataset(f"{loc}:{varName}", loc, [varName], np_var)
        return

    @profile_time
    def initializePseudoCell_QuadNQuad(self, z_ncFaces):
        local_ncFaces = Internal.getNodeFromName(z_ncFaces, "ElementConnectivity")[1] - 1
        len_NCF = len(local_ncFaces)//4
        n_ncFacesX = Internal.getNodeFromName(z_ncFaces, "CoordinateX")[1]
        n_ncFacesY = Internal.getNodeFromName(z_ncFaces, "CoordinateY")[1]
        n_ncFacesZ = Internal.getNodeFromName(z_ncFaces, "CoordinateZ")[1]

        n_ncFaces = numpy.column_stack((n_ncFacesX, n_ncFacesY, n_ncFacesZ))
        ncFacesCentroids = computeQuadCentroids(
            n_ncFacesX,
            n_ncFacesY,
            n_ncFacesZ,
            local_ncFaces
        )

        local_ncFaces = numpy.reshape(local_ncFaces, (len_NCF,4))
        tic = time.perf_counter()
        if self.dimPb == 2: create_fct =  create_Quad2Quad
        else: create_fct = create_Quad4Quad
        listQuadNQuad_local = create_fct(
            n_ncFaces,
            local_ncFaces,
            ncFacesCentroids
        )  # plane, tol

        toc = time.perf_counter()
        print("[{rank}] DEBUG: Time for hanging nodes search: {toc-tic:.3f}.")
        Internal._rmNodesFromType(self.pyTree, "Elements_t")
        hook = C.createHook(self.pyTree, 'nodes')
        ids = C.identifyNodes(hook, z_ncFaces)
        ids = ids[ids != -1] - 1
        listQuadNQuad = ids[listQuadNQuad_local]

        fs_cell2node = FSIntArray(*listQuadNQuad.shape)
        numpy.copyto(
            numpy.array(fs_cell2node.Buffer(), copy=False),
            listQuadNQuad,
            casting='same_kind'
        )
        if self.dimPb == 2:
            self.fsmesh.InitUnstructCells(
                FSMeshEnums.PCT_Quad2Quad,
                fs_cell2node,
                False
            )
        else:
            self.fsmesh.InitUnstructCells(
                FSMeshEnums.PCT_Quad4Quad,
                fs_cell2node,
                False
            )
        return

    def initializePseudoCell_QuadNQuad_MPI(self, z_ncFaces):
        if Cmpi.master and self.verbose: print("Creating QuadNQuad pseudo connectivity.")
        rank = self.clac.ProcID()
        if z_ncFaces is not None:
            z_ncFaces[0] = z_ncFaces[0] + str(Cmpi.rank)
            n_ncFacesX = Internal.getNodeFromName(z_ncFaces, "CoordinateX")[1]
            n_ncFacesY = Internal.getNodeFromName(z_ncFaces, "CoordinateY")[1]
            n_ncFacesZ = Internal.getNodeFromName(z_ncFaces, "CoordinateZ")[1]
        else:
            n_ncFacesX = numpy.empty(0)
            n_ncFacesY = numpy.empty(0)
            n_ncFacesZ = numpy.empty(0)

        allgathered_x = Cmpi.gather(n_ncFacesX, 0)
        allgathered_y = Cmpi.gather(n_ncFacesY, 0)
        allgathered_z = Cmpi.gather(n_ncFacesZ, 0)

        len_NCF = len(n_ncFacesX)

        del n_ncFacesX, n_ncFacesY, n_ncFacesZ

        if self.dimPb == 2: fsCellType = FSMeshEnums.PCT_Quad2Quad
        else: fsCellType = FSMeshEnums.PCT_Quad4Quad

        self.initializeCell2Proc(fsCellType, len_NCF)
        if z_ncFaces is not None:
            local_ncFaces = Internal.getNodeFromName(
                z_ncFaces,
                "ElementConnectivity"
            )[1] - 1 + self.cell2ProcDict[fsCellType][rank]
        else:
            local_ncFaces = numpy.empty(0, dtype=Internal.E_NpyInt)

        allgathered_local_ncFaces = Cmpi.gather(local_ncFaces, 0)
        del local_ncFaces

        listQuadNQuad_local = []

        if Cmpi.master:
            allgathered_x = numpy.concatenate(allgathered_x)
            allgathered_y = numpy.concatenate(allgathered_y)
            allgathered_z = numpy.concatenate(allgathered_z)
            allgathered_local_ncFaces = numpy.concatenate(allgathered_local_ncFaces)
            allgathered_nodes = numpy.hstack([
                allgathered_x.reshape(len(allgathered_x),1),
                allgathered_y.reshape(len(allgathered_x),1),
                allgathered_z.reshape(len(allgathered_x),1)
            ])

            cmpIdx = lambda a, b : cmp(allgathered_nodes[a], allgathered_nodes[b])
            idx_sorted = sorted(range(len(allgathered_nodes)), key=cmp_to_key(cmpIdx))

            nnodes_old = len(allgathered_nodes)

            np_sortedCoords = allgathered_nodes[idx_sorted]

            uniqueCoords = numpy.empty((nnodes_old,3))
            dedupMap = numpy.empty((nnodes_old))
            dedup2dup = numpy.empty((nnodes_old), dtype=Internal.E_NpyInt)
            previous = None
            j = -1

            for i in range(nnodes_old):
                if i == 0 or (abs(previous-np_sortedCoords[i])>(10**(-10))).any():
                    j=j+1
                    uniqueCoords[j] = np_sortedCoords[i]
                    previous = np_sortedCoords[i]
                    dedupMap[idx_sorted[i]] = j
                    dedup2dup[j] = idx_sorted[i]
                else:
                    dedupMap[idx_sorted[i]] = j

            for i in range(len(allgathered_local_ncFaces)):
                allgathered_local_ncFaces[i] = dedupMap[allgathered_local_ncFaces[i]]

            uniqueCoords.resize((j+1,3))
            allgathered_nodes = uniqueCoords
            del uniqueCoords

            len_NCF = len(allgathered_local_ncFaces)//4
            ncFacesCentroids = computeQuadCentroids(
                allgathered_nodes[:,0],
                allgathered_nodes[:,1],
                allgathered_nodes[:,2],
                allgathered_local_ncFaces
            )

            local_ncFaces = numpy.reshape(allgathered_local_ncFaces,(len_NCF,4))

            del allgathered_local_ncFaces

            print("Rank {}: local_ncFaces {}.".format(Cmpi.rank, local_ncFaces.shape[0]))
            tic = time.perf_counter()

            if self.dimPb == 2: create_fct =  create_Quad2Quad
            else: create_fct = create_Quad4Quad
            listQuadNQuad_local = create_fct(
                allgathered_nodes,
                local_ncFaces,
                ncFacesCentroids
            )
            toc = time.perf_counter()
            print("Rank {}: time for hanging nodes search: {:.3f}.".format(Cmpi.rank, toc-tic))
            print("Rank {}: size listQuadNQuad {}.".format(Cmpi.rank, listQuadNQuad_local.shape[0]))
            listQuadNQuad_local = dedup2dup[listQuadNQuad_local]

        self.initializeCell2Proc(fsCellType, len(listQuadNQuad_local))
        Internal._rmNodesFromType(self.pyTree, "Elements_t")
        if z_ncFaces is not None:
            hook = C.createHook(self.pyTree, 'nodes')
            ids = C.identifyNodes(hook, z_ncFaces)
        else:
            ids = numpy.empty(0, dtype=Internal.E_NpyInt)
        ids = ids[ids>-1]-1 + self.cell2ProcDict[1][rank]
        gath_ids = numpy.concatenate(Cmpi.allgather(ids))

        if Cmpi.master:
            listQuadNQuad = gath_ids[listQuadNQuad_local]
            fs_cell2node = FSIntArray(
                listQuadNQuad.shape[0], listQuadNQuad.shape[1]
            )
            numpy.copyto(
                numpy.array(fs_cell2node.Buffer(), copy=False),
                listQuadNQuad,
                casting='same_kind'
            )
        else:
            fs_cell2node = FSIntArray(0, FSCellInfo.NNodes(fsCellType))
        self.fsmesh.InitUnstructCells(
            fsCellType, self.cell2ProcDict[fsCellType],
            fs_cell2node,
            False
        )
        return None

    @profile_time
    def parallelDeduplicateNodesFSMesh(self):
        tol = 1e-10
        decimals = int(-numpy.log10(tol))
        if self.dimPb == 2: quadNQuad = 15
        else: quadNQuad = 16

        fs_coords = self.fsmesh.GetUnstructDataset("Coordinates").GetValues()
        np_coords = numpy.array(fs_coords.Buffer(), copy=True)
        np_coords = ArrayOps.Gather(np_coords, self.clac)

        uniqueCoords = numpy.empty((0,3))
        dedupMap = numpy.empty((0), dtype=Internal.E_NpyInt)

        tic = time.perf_counter()
        rank = self.clac.GetProcID()
        if rank == 0:
            # Use lexicographical order to sort by z, then y, then x
            lexOrder = numpy.lexsort(numpy.around(np_coords, decimals=decimals).T)
            np_sortedCoords = np_coords[lexOrder]
            nvertices = len(np_sortedCoords)
            toc = time.perf_counter()
            elapsed = toc-tic
            print(f"DEBUG: Time for sorting coords = {elapsed:.3f}.")

            # Find unique (sorted) coordinates and their indices
            uniqueMask = numpy.ones(nvertices, dtype=bool)
            uniqueMask[1:] = numpy.any(numpy.abs(numpy.diff(np_sortedCoords, axis=0)) > tol, axis=1)
            uniqueIndices = numpy.where(uniqueMask)[0]
            uniqueCoords = np_sortedCoords[uniqueMask]

            # Compute the gap (ie, number of duplicates) for each unique vertex
            duplicatesCount = numpy.diff(numpy.append(uniqueIndices, nvertices))

            # Map each duplicate to its corresponding unique vertex index
            dedupMap = numpy.full(nvertices, -1, dtype=Internal.E_NpyInt)
            dedupMap[lexOrder] = numpy.repeat(
                numpy.arange(len(uniqueCoords)),
                duplicatesCount
            )
            toc = time.perf_counter()
            elapsed = toc-tic
            print(f"DEBUG: Time for dedup map build = {elapsed:.3f}.")

        if rank == 0:
            print("Removing duplicated vertices from mesh connectivity.")
        cell2Proc_nodes = initializeCell2ProcOutsideClass(self.clac, len(uniqueCoords))
        dedupMap = ArrayOps.Broadcast(dedupMap, self.clac)
        fsCellTypesNC = self.fsCellTypes
        if not self.conformal: fsCellTypesNC.append(quadNQuad)
        cell2NodeDict = {}
        for cellType in fsCellTypesNC:
            fs_cell2Node = self.fsmesh.GetCell2Node(cellType)
            np_cell2Node = numpy.array(fs_cell2Node.Buffer(), copy=True)
            cell2NodeDict[cellType] = dedupMap[np_cell2Node]
        gath_fsCellTypesNC = Cmpi.allgather(fsCellTypesNC)
        fsCellTypesNC = set(ct for cellTypes in gath_fsCellTypesNC for ct in cellTypes)

        fs_bMarkerList = self.fsmesh.GetCellAttributeValuesWithNames("CADGroupID")
        names = []
        fsCellTypeMarkersDict = {}
        npCellTypeMarkersDict = {}
        try:
            np_bMarkerList = numpy.array(fs_bMarkerList.Buffer(), copy=True)
            for cellType in self.fsSurfaceCellTypes:
                fs_markers_array_cell_type = self.fsmesh.GetCellAttribute("CADGroupID", cellType)
                np_markers_array_cell_type = numpy.array(fs_markers_array_cell_type.Buffer(), copy=True)
                npCellTypeMarkersDict[cellType] = np_markers_array_cell_type
                fs_markers_array_cell_type = FSIntArray(len(np_markers_array_cell_type))
                for i in range(len(np_markers_array_cell_type)):
                    fs_markers_array_cell_type[i] = int(np_markers_array_cell_type[i])
                fsCellTypeMarkersDict[cellType] = fs_markers_array_cell_type
                #np_markers_celltype_dict_gath = Cmpi.gather(npCellTypeMarkersDict)

            fs_bMarkerList = FSIntArray(len(np_bMarkerList))
            for i in range(len(np_bMarkerList)):
                fs_bMarkerList[i] = int(np_bMarkerList[i])

            for marker in fs_bMarkerList:
                names.append(self.fsmesh.GetCellAttributeValueName("CADGroupID", marker))
        except BufferError:
            pass

        if self.IBM:
            if self.fsVolumeCellTypes:
                flis_distance = self.fsmesh.GetUnstructDataset("FlisWallDistance").GetValues()
                np_flisDistance = numpy.array(flis_distance.Buffer(), copy=True)
            else:
                np_flisDistance = numpy.empty(0)

            fsDataNames = ["WallPointCoordinates", "DonorPointCoordinates"]
            datasets = []
            for fsdataname in fsDataNames:
                if self.fsSurfaceCellTypes:
                    dataset = numpy.array(
                        self.fsmesh.GetUnstructDataset(fsdataname).GetValues().Buffer(),
                        copy=True
                    )
                else:
                    dataset = numpy.empty(0)
                datasets.append(dataset)

        self.fsmesh.Reset()
        self.fsmesh.BeginInitialization()
        self.fsmesh.InitUnstructNodes(cell2Proc_nodes)

        for cellType in fsCellTypesNC:
            ncellsOfType = 0
            if cellType in cell2NodeDict:
                np_cell2node = cell2NodeDict[cellType]
                fs_cell2node = FSIntArray(*np_cell2node.shape)
                ncellsOfType = np_cell2node.shape[0]
                cell2Proc = initializeCell2ProcOutsideClass(
                    self.clac, ncellsOfType
                )
                if ncellsOfType > 0:
                    numpy.copyto(
                        numpy.array(fs_cell2node.Buffer(), copy=False),
                        np_cell2node,
                        casting='same_kind'
                    )
            else:
                cell2Proc = initializeCell2ProcOutsideClass(self.clac, 0)
                fs_cell2node = FSIntArray(0, FSCellInfo.NNodes(cellType))
            self.fsmesh.InitUnstructCells(cellType, cell2Proc, fs_cell2node, True)

        self.fsmesh.EndInitialization()
        self.initializeFSBCCoordinates(
            [FSDataName(self.coordsName)],
            FSMeshEnums.CT_Node
        )
        if uniqueCoords.shape[0] > 0:
            fs_coordinates = self.fsmesh.GetUnstructDataset(
                FSDataName(self.coordsName)
            ).GetValues()
            numpy.copyto(
                numpy.array(fs_coordinates.Buffer(), copy=False),
                uniqueCoords,
                casting='no'
            )

        for name, marker in zip(names, fs_bMarkerList):
            self.fsmesh.SetCellAttributeValueName(FS_AT_CADGroupID, marker, name)
        for cellType in self.fsSurfaceCellTypes:
            self.fsmesh.InitCellAttribute(
                FS_AT_CADGroupID,
                cellType,
                fsCellTypeMarkersDict[cellType]
            )

        if self.IBM:
            quantityName = "FlisWallDistance"
            quantityNames = FSStringArray(1)
            quantityNames[0] = quantityName
            quantitySpecs = FSDataSpecArray(1)
            quantitySpecs[0].Length()
            fs_volumeCellTypes = FSIntArray(len(self.fsVolumeCellTypes))
            if self.fsVolumeCellTypes:
                numpy.copyto(
                    numpy.array(fs_volumeCellTypes.Buffer(), copy=False),
                    self.fsVolumeCellTypes,
                    casting='same_kind'
                )

            self.fsmesh.InitUnstructDataset(
                quantityName,
                FSDatasetInfo(quantityNames, quantitySpecs, fs_volumeCellTypes)
            )
            if self.fsVolumeCellTypes:
                fs_var = self.fsmesh.GetUnstructDataset(quantityName).GetValues()
                numpy.copyto(
                    numpy.array(fs_var.Buffer(), copy=False),
                    np_flisDistance,
                    casting='no'
                )
            fs_surfaceCellTypes = FSIntArray(len(self.fsSurfaceCellTypes))
            if self.fsSurfaceCellTypes:
                numpy.copyto(
                    numpy.array(fs_surfaceCellTypes.Buffer(), copy=False),
                    self.fsSurfaceCellTypes,
                    casting='same_kind'
                )
            self.initializeFSBCCoordinates(fsDataNames, fs_surfaceCellTypes)
            if self.fsSurfaceCellTypes:
                for fsdataname, dataset in zip(fsDataNames, datasets):
                    fs_var = self.fsmesh.GetUnstructDataset(fsdataname).GetValues()
                    numpy.copyto(
                        numpy.array(fs_var.Buffer(), copy=False),
                        dataset,
                        casting='no'
                    )

        toc = time.perf_counter()
        elapsed = max(Cmpi.allgather(toc-tic))
        if rank == 0 and self.verbose:
            print(f"DEBUG: Time for parallelDeduplicateNodesFSMesh = {elapsed:.3f}.")
        return

    def releaseResources(self):
        """Delete class attributes that are no longer needed"""
        del (
            self.fsmesh, self.fsMarkers,
            self.np_coordinates,
            self.cell2NodeDict, self.cell2NodeVolumeList,
            self.cell2NodeSurfaceList,
            self.indicesPerBdr,
            self.bMarker2BCNameDict, self.bMarker2FacePLDict
        )

    @profile_time
    def convert2NGon4FFD(self, reorient=True, tol=1e-6, **kwargs):
        """
        Convert a CGNS ME mesh to NGon for use in FFD

        Args:
            reorient; bool: Reorient surface normals. Default is True

            tol; float: tolerance. Default is 1e-6
        """
        if self.pyTree is None:
            filename = self.meshName.split(".")[0]
            if Cmpi.size == 1:
                self.pyTree = C.convertFile2PyTree(filename + ".cgns")
            else:
                self.pyTree = Cmpi.convertFile2PyTree(filename + ".cgns",
                                                      proc=Cmpi.rank)
        zones = Internal.getZones(self.pyTree)
        if len(zones) == 0:
            if self.meshName is not None:
                filename = self.meshName.split(".")[0]
                if Cmpi.size == 1:
                    self.pyTree = C.convertFile2PyTree(filename + ".cgns")
                else:
                    self.pyTree = Cmpi.convertFile2PyTree(filename + ".cgns",
                                                          proc=Cmpi.rank)
            else:
                if Cmpi.master:
                    print("ERROR: FSCGNSConverter.convert(forFFDX=True, "
                          "**kwargs) must be called instead of "
                          "FSCGNSConverter.convert2NGon4FFD")
                sys.exit(1)
        else:
            self.releaseResources()

        if float(C.__version__) < 5.0:
            # Break zones such that there is one type of volume element per zone
            if Cmpi.master and self.verbose:
                print("Breaking ME connectivity: 1 type of volume element per zone.")
            t3 = C.breakConnectivity(self.pyTree)

            # Limitation fixed in Cassiopee 5.0
            zones = Internal.getZones(t3)
            for zone in zones:
                n_elts = Internal.getNodesFromType(zone, "Elements_t")
                n_bcs = Internal.getNodesFromType(zone, "BC_t")
                for n_bc, (elt, bc) in enumerate(zip(n_elts[1:], n_bcs)):
                    ER_el = Internal.getNodeFromName(elt, "ElementRange")
                    ER_bc = Internal.getNodeFromName(bc, "ElementRange")
                    if (ER_el[1] != ER_bc[1][0]).all():
                        ER_bc[1][0] = ER_el[1]
            C._deleteEmptyZones(t3)

            # Convert ME to NGon
            if Cmpi.master and self.verbose: print("Converting ME to NGon.")
            self.pyTree = C.convertArray2NGon(t3, recoverBC=False)
            C._deleteFlowSolutions__(t3)

            # Save BCs in the correct format for later reassignment
            if Cmpi.master and self.verbose:
                print("Saving BCs temporarily for later reassignment.")
            BCs, BCNames, BCTypes = C.getBCs(t3)
            true_len_BCs = len(BCs)//len(Internal.getZones(t3))
            BCs = BCs[:true_len_BCs]
            BCNames = BCNames[:true_len_BCs]
            BCTypes = BCTypes[:true_len_BCs]
            for BC in BCs:
                n_elts = Internal.getNodesFromType(BC, "Elements_t")
                if len(n_elts) > 1 and n_elts[0][0].startswith("GridElements"):
                    Internal._rmNodesByName(BC, n_elts[0][0])
            del t3

            # Multizone NGon - delete useless nodes
            zones = Internal.getZones(self.pyTree)
            for z in zones:
                n_elts = Internal.getNodesFromType(z, "Elements_t")
                for n_elt in n_elts:
                    if n_elt[0] not in ["NGonElements", "NFaceElements"]:
                        Internal._rmNodesByName(n_elt, n_elt[0])

            # Merge all zones into a single zone and change its name
            if Cmpi.master and self.verbose: print("Merging all zones.")
            self.pyTree = T.merge(self.pyTree)
            z = Internal.getZones(self.pyTree)[0]
            z[0] = "zone." + str(Cmpi.rank)

            # Reassign BCs
            if Cmpi.master and self.verbose: print("Reassigning BCs.")
            if Cmpi.size > 1:
                list_BCs, list_BCNames, list_BCTypes = recoverBCsC(
                    self.pyTree, BCs, BCNames, BCTypes
                )
                list_BCs = Cmpi.allgather(list_BCs)
                list_BCNames = Cmpi.allgather(list_BCNames)
                list_BCTypes = Cmpi.allgather(list_BCTypes)
                removeBC = False
            else:
                list_BCs = [BCs]
                list_BCNames = [BCNames]
                list_BCTypes = [BCTypes]
                removeBC = True

            for (BCs_h, BCNames_h, BCTypes_h) in zip(list_BCs, list_BCNames, list_BCTypes):
                C._recoverBCs(
                    self.pyTree,
                    (BCs_h, BCNames_h, BCTypes_h),
                    tol=tol,
                    removeBC=removeBC
                )

            nassignedBCs = 0
            n_bcs = Internal.getNodesFromType(self.pyTree, "BC_t")
            for n_bc in n_bcs:
                ptList = Internal.getNodeFromName(n_bc, "PointList")[1][0]
                nassignedBCs += len(ptList)
            ntotAssignedBCs = sum(Cmpi.allgather(nassignedBCs))
            ntotSurfaceCells = sum(Cmpi.allgather(self.nsurfaceCells))

            if ntotAssignedBCs != ntotSurfaceCells:
                print("ERROR: BCs missing! {}/{} surface elements have no BC "
                      "assigned.".format(ntotSurfaceCells - ntotAssignedBCs,
                                         ntotSurfaceCells))
                sys.exit(1)
            else:
                if Cmpi.master and self.verbose:
                    print("All {} surface elements successfully mapped to "
                          "a BC.".format(ntotSurfaceCells))

            self.pyTree = C.newPyTree(['Base', self.pyTree])

        else:
            # Convert ME to NGon
            if Cmpi.master and self.verbose: print("Converting ME to NGon.")
            C._deleteEmptyZones(self.pyTree)
            C._convertArray2NGon(self.pyTree, recoverBC=True, api=3) # TODO attempt - fix it otherwise
            # Change zone name
            z = Internal.getZones(self.pyTree)[0]
            z[0] = "zone." + str(Cmpi.rank)
            n_bcs = C.getBCs(self.pyTree)[0]
            Internal._rmNodesByNameAndType(self.pyTree, "QUAD_bdy*", "Elements_t")
            Internal._rmNodesByNameAndType(self.pyTree, "TRI_bdy*", "Elements_t")
            # Rename BC nodes
            n_zbcs = Internal.getNodesFromType(self.pyTree, "BC_t")
            for n in n_zbcs:
                n[0] = n[0][:-2].replace('.QUAD', '').replace('.TRI', '')
            # Delete Element boundary nodes
            n_bElts = Internal.getElementBoundaryNodes(z)
            for n in n_bElts: Internal._rmNode(self.pyTree, n)

        if reorient:
            import Intersector.PyTree as XOR
            if Cmpi.master and self.verbose: print("Reorienting mesh for use in FFD.")
            XOR._reorient(self.pyTree)

        if self.datasets == 'all' or len(self.datasets) > 0:
            _fixNodesForFlowSolution(self.pyTree)
        return

    @profile_time
    def mergeBCsByMarker(self, tol=1e-11):
        if Cmpi.master and self.verbose:
            print("Merging BCs: one CGNS BC per boundary marker.")
        familyNames = []
        familyTypes = []
        fsMarkers = []

        z = Internal.getZones(self.pyTree)[0]
        n_zoneBCs = Internal.getNodesFromType1(z, 'ZoneBC_t')
        n_bcs = Internal.getNodesFromType1(n_zoneBCs, 'BC_t')

        for n_bc in n_bcs:
            bcName = Internal.getName(n_bc)
            bcType = Internal.getValue(n_bc)
            familyName, marker = bcName.split('.')[:2]
            marker = marker.split('_')[1]
            familyNameMarker = f"{familyName}_{marker}"
            if marker not in fsMarkers:
                fsMarkers.append(marker)
                familyNames.append(familyNameMarker)
                familyTypes.append(bcType)

            Internal.createChild(
                n_bc, 'FamilyName', 'FamilyName_t',
                value=familyNameMarker, pos=0
            )
            Internal.setValue(n_bc, 'FamilySpecified')

        zbcs = []
        FS = Internal.getNodesFromType(self.pyTree, "FlowSolution_t")
        Internal._rmNodesByType(self.pyTree, "FlowSolution_t")
        for familyName in familyNames:
            zbc = C.extractBCOfType(self.pyTree, "FamilySpecified:" + familyName)
            zbc = T.join(zbc)
            zbcs.append(zbc)

        C._recoverBCs(self.pyTree, (zbcs, familyNames, familyTypes), tol=tol, removeBC=True)
        zone = Internal.getZones(self.pyTree)
        for FS_node in FS:
            Internal._addChild(zone[0], FS_node, pos=-1) # at the end
        return

    def merge_BCs_FamilySpecified(self):
        if Cmpi.master and self.verbose:
            print("Specifying FamilySpecified BCs: one FamilyName per boundary marker.")
        BCs = []

        base = Internal.getNodeFromType1(self.pyTree, 'CGNSBase_t')
        z = Internal.getZones(self.pyTree)[0]
        n_zoneBCs = Internal.getNodesFromType1(z, 'ZoneBC_t')
        n_bcs = Internal.getNodesFromType1(n_zoneBCs, 'BC_t')

        for n_bc in n_bcs:
            bcName = Internal.getName(n_bc)
            bcType = Internal.getValue(n_bc)
            familyName = bcName.split('.')[0]
            BCs.append((bcName, bcType, familyName))
            Internal.createChild(n_bc, 'FamilyName', 'FamilyName_t', value=familyName, pos=0)
            Internal.setValue(n_bc, 'FamilySpecified')

        for bc in BCs:
            if Internal.getNodesFromNameAndType(self.pyTree, bc[2], 'Family_t') == []:
                n_family = Internal.createNode(bc[2], 'Family_t', parent=base)
                Internal.createChild(n_family, 'FamilyBC', 'FamilyBC_t', value=bc[1], pos=0)
        return None

    @profile_time
    def reorderCells(self):
        """Reorder CGNS volume and surface element types as in FSDM"""
        # Loop over volume and surface element types and store their range
        cellType2RangeDict = {}
        n_elts = Internal.getNodesFromType(self.pyTree, "Elements_t")
        for n_elt in n_elts:
            cellType = Internal.getValue(n_elt)[0]
            eltRange = Internal.getNodeFromName(n_elt, "ElementRange")[1]
            cellType2RangeDict[cellType] = eltRange

        # Sort element types by minimum range
        cellType2RangeDict = dict(
            sorted(cellType2RangeDict.items(), key=lambda item: item[1][0])
        )

        # Detect if the minimum element range of surface element types is less
        # than that of volume element types, in which case CGNS element types
        # must be rearranged: surface elements come last in FSDM
        areCellTypesOrdered = True
        cellTypeList = list(cellType2RangeDict.keys())
        for i, cellType in enumerate(cellTypeList):
            if cellType in [5, 7]:
                if any(ct not in [5, 7] for ct in cellTypeList[i+1:]):
                    areCellTypesOrdered = False
                    break
        if areCellTypesOrdered: return

        if Cmpi.master and self.verbose:
            print("Reording CGNS element types as in FSDM.")

        # Loop over all cell types to determine the new element ranges
        ntotCells = 0
        newCellType2RangeDict = {}
        cellTypeList = (
            [ct for ct in cellTypeList if ct not in [5, 7]] +
            [ct for ct in cellTypeList if ct in [5, 7]]
        )

        for cellType in cellTypeList:
            eltRange = cellType2RangeDict[cellType]
            nCells = eltRange[1] - eltRange[0] + 1
            newCellType2RangeDict[cellType] = [ntotCells + 1, ntotCells + nCells]
            ntotCells += nCells

        # Loop over volume and surface element types to assign their new range
        for n_elt in n_elts:
            cellType = Internal.getValue(n_elt)[0]
            eltRange = Internal.getNodeFromName(n_elt, "ElementRange")
            eltRange[1] = newCellType2RangeDict[cellType]
            print("cellType:", cellType, ", eltRange[1]:", eltRange[1])

        # Loop over all CGNS BC nodes and offset vertex point lists
        # using the difference between new and old TODO are they vertex point list?
        n_bcs = Internal.getNodesFromType(self.pyTree, "BC_t")
        origRangeTri = cellType2RangeDict.get(5)
        newRangeTri = newCellType2RangeDict.get(5)
        origRangeQuad = cellType2RangeDict.get(7)
        newRangeQuad = newCellType2RangeDict.get(7)
        for n_bc in n_bcs:
            ptList = Internal.getNodeFromName(n_bc, "PointList")
            if ptList is None: continue
            np_vertexPL = ptList[1].copy()

            # Mark which faces are tris and which are quads
            if newRangeTri is not None and newRangeQuad is not None:
                isQuad = numpy.logical_and(
                    np_vertexPL >= origRangeQuad[0],
                    np_vertexPL <= origRangeQuad[1]
                )
                np_vertexPL[isQuad] += newRangeQuad[0] - origRangeQuad[0]
                np_vertexPL[~isQuad] += newRangeTri[0] - origRangeTri[0]
            elif newRangeQuad is not None:
                np_vertexPL[:] += newRangeQuad[0] - origRangeQuad[0]
            else:
                np_vertexPL[:] += newRangeTri[0] - origRangeTri[0]

            ptList[1][:] = np_vertexPL
        return

    def export(self, filename, **kwargs):
        """Export to file based on the filename extension"""
        validExtensions = ['cgns', 'h5', 'plt']
        ext = filename.split('.')[-1]
        if ext not in ['cgns', 'h5', 'plt']:
            print("ERROR in FSCGNSConverter.FSCGNSConverter.export: Input "
                  "filename does not have a valid extension. It can "
                  "either be {}.".format(', '.join(e for e in validExtensions)))
            sys.exit(1)
        elif ext == 'cgns':
            self.exportCGNS(filename=filename, **kwargs)
        elif ext == 'h5':
            self.exportFSMesh(filename=filename, **kwargs)
        else:
            self.__export2Tecplot(filename=filename)
        return

    def exportFSMesh(self, filename="", verbose=False):
        """Export FSMesh to file"""
        if Cmpi.master and self.verbose: print("Export FS mesh.")
        if filename:
            if filename.endswith(".h5"): filename = filename[:-3]
        elif isinstance(self.meshName, str):
            filename = self.meshName.split('.')[0]  # Mesh saved in the same dir. as the input mesh
        else:
            filename = "t"
        self.fsmesh.ExportMeshHDF5(Filename=filename + ".h5") or FSError.PrintAndExit()
        if verbose: self.fsmesh.PrintInfo()
        return

    def exportFSMesh2Tecplot(self, filename=""):
        """Export FS mesh to tecplot format"""
        if Cmpi.master and self.verbose: print("Export FS mesh to tecplot format.")
        ext = "plt"
        if filename:
            if filename.endswith(".pt"): ext = "pt"
            if any(filename.endswith(i) for i in [".h5", ".plt", ".pt"]):
                filename = filename.rsplit('.', 1)[0]
        elif isinstance(self.meshName, str):
            filename = self.meshName.rsplit('.', 1)[0]  # Output saved in the same dir. as the input mesh
        else:
            filename = "t"

        volumeCellTypes = tuple(
            FSMeshEnums.CellTypeToString(vct)
            for vct in self.fsVolumeCellTypes
        )
        self.fsmesh.ExportMeshTECPLOT(
            Filename=f"{filename}_vol.{ext}",
            PrefixDatasetName=True,
            ExportCellTypes=volumeCellTypes
        ) or FSError.PrintAndExit()

        surfaceCellTypes = tuple(
            FSMeshEnums.CellTypeToString(sct)
            for sct in self.fsSurfaceCellTypes
        )
        self.fsmesh.ExportMeshTECPLOT(
            Filename=f"{filename}_surf.{ext}",
            PrefixDatasetName=True,
            ZonePerCellAttributeValue=True,
            CellAttribute=FS_AT_CADGroupID,
            UseCellAttributeValueName=True,
            ExportCellTypes=surfaceCellTypes
        ) or FSError.PrintAndExit()
        return

    def exportCGNS(self, filename="", verbose=False):
        """Export CGNS to file"""
        if Cmpi.master and self.verbose: print("Export CGNS mesh.")
        if filename:
            if filename.endswith(".cgns"): filename = filename[:-5]
        elif isinstance(self.meshName, str):
            filename = self.meshName.rsplit('.', 1)[0]  # Mesh saved in the same dir. as the input mesh
        else:
            filename = "t"
        Cmpi.convertPyTree2File(self.pyTree, filename + ".cgns")
        if verbose: Internal.printTree(self.pyTree)
        return

    def exportCGNS2Tecplot(self, filename=""):
        """Export CGNS to tecplot format"""
        if Cmpi.master and self.verbose: print("Export CGNS mesh to tecplot format.")
        if filename:
            if filename.endswith(".cgns"): filename = filename[:-5]
            elif filename.endswith(".plt"): filename = filename[:-4]
        elif isinstance(self.meshName, str):
            filename = self.meshName.split('.')[0]  # Output saved in the same dir. as the input mesh
        else:
            filename = "t"
        if Cmpi.size == 1: C.convertPyTree2File(self.pyTree, f"{filename}.plt")
        else: C.convertPyTree2File(self.pyTree, f"{filename}_{Cmpi.rank:03d}.plt")
        return
