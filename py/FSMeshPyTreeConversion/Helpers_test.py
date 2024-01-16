from FSDataManager import FSMesh, FSClac, FSError, FSMeshEnums, FSUnstructSurfaceCellTypes,FSMeshSelectionOpAttribute
import Converter.PyTree as C
import Converter.Internal as Internal
import Generator.PyTree as G
import Transform.PyTree as T
import Post.PyTree as P
import numpy, os, math



def computeSurfaceTri(cn,xt,yt,zt):
      nelts = len(cn)
      surface = numpy.empty(nelts)
      for i in range(nelts):
         ind1 = cn[i][0]
         ind2 = cn[i][1]
         ind3 = cn[i][2]
         l1x = xt[ind1]-xt[ind2]
         l1y = yt[ind1]-yt[ind2]
         l1z = zt[ind1]-zt[ind2]
         l2x = xt[ind1]-xt[ind3]
         l2y = yt[ind1]-yt[ind3]
         l2z = zt[ind1]-zt[ind3]
         surfx = l1y*l2z-l1z*l2y
         surfy = l1z*l2x-l1x*l2z
         surfz = l1x*l2y-l1y*l2x
         surf = math.sqrt(surfx*surfx+surfy*surfy+surfz*surfz)
         surface[i] = 0.5 * surf
      return surface

def computeSurfaceQuad(cn,xt,yt,zt):
      nelts = len(cn)
      surface = numpy.empty(nelts)

      for i in range(nelts):
         ind1 = cn[i][0]
         ind2 = cn[i][1]
         ind3 = cn[i][2]
         ind4 = cn[i][3]

         l1x = xt[ind2] - xt[ind1]
         l1y = yt[ind2] - yt[ind1]
         l1z = zt[ind2] - zt[ind1]

         l2x = xt[ind3] - xt[ind1]
         l2y = yt[ind3] - yt[ind1]
         l2z = zt[ind3] - zt[ind1]

         surf1x = l1y*l2z-l1z*l2y
         surf1y = l1z*l2x-l1x*l2z
         surf1z = l1x*l2y-l1y*l2x

         l1x = xt[ind3] - xt[ind1]
         l1y = yt[ind3] - yt[ind1]
         l1z = zt[ind3] - zt[ind1]

         l2x = xt[ind4] - xt[ind1]
         l2y = yt[ind4] - yt[ind1]
         l2z = zt[ind4] - zt[ind1]

         surf2x = l1y*l2z-l1z*l2y
         surf2y = l1z*l2x-l1x*l2z
         surf2z = l1x*l2y-l1y*l2x

         surfx = surf1x + surf2x
         surfy = surf1y + surf2y
         surfz = surf1z + surf2z
         surface[i] = 0.5 * math.sqrt(surfx*surfx+surfy*surfy+surfz*surfz)
      return surface


def computeWallSurfaceCODA(mesh_name,dict_BCs,alpha=0.0):
    # import our stuff from CODA
    from CODA import DiscretizationFactory, TimeIntegrationFactory, DataExportSelection
    from CODA import BoundaryIntegralValuesContainer, PrintSumOfValuesForSelectedMarkers
    from CODA.Utilities import ComputeSumOfValuesForSelectedMarkers

    wall_markers = []
    other_markers = []
    for key in dict_BCs.keys():
      if dict_BCs[key].startswith("BCWall"):
        wall_markers.append(key)
      else:
        other_markers.append(key)
    clac = FSClac()
    fsmesh = FSMesh(clac)
    fsmesh.ImportMeshHDF5(Filename=mesh_name)

    fsmesh.RemoveDataset("AugStateGradient")
    fsmesh.RemoveDataset("AugState")
    fsmesh.RemoveDataset("State")
    fsmesh.RemoveDataset("FlisWallDistance")
    fsmesh.RemoveDataset("Residual")

    #fsmesh.RepartitionMeshRCB() or FSError.PrintAndExit()
    fsmesh.HasLocalNumbering() or fsmesh.CreateLocalNumbering()

    fsmesh.PrintInfo()

    mach = 0.1
    discSelectionParaDict = {
        "spatial scheme": "FV",
        "convection scheme": "Roe upwinding",
        "PDE": "Euler",
        "order": 2,
    }

    discParaDict = {
        "testing": {
            "willfully ignore excessive load imbalance among domains w.r.t. the number of faces": True,
        },
        "reference state": {
            "flow speed specification": {
                "type": "Mach number based",
                "Mach": mach,
            },
        },
        "boundary treatments": [
            {
                # "treatment type" defined later depending on the PDE
                "treatment type": "BCWallInviscid",
                "boundary markers": wall_markers,
            },
            {
                "treatment type": "BCFarfield",
                "boundary markers": other_markers,
            },
        ]
        }
    disc = DiscretizationFactory.GetSingleton().Create(discSelectionParaDict, clac, fsmesh, discParaDict)

    state = disc.CreateZeroFieldVector()
    disc.InitializeFieldVector({"type": "free stream", "state": {
        "flow speed specification": {
            "type": "Mach number based",
            "Mach": mach,
        }
    }}, state)
    integralValues = BoundaryIntegralValuesContainer()
    disc.ComputeBoundaryIntegralValues(state, integralValues)
    dictWithSummedIntegralValues = ComputeSumOfValuesForSelectedMarkers(integralValues, wall_markers, ["Area"])
    surface = dictWithSummedIntegralValues["Area"]

    bdr_flag = False; normals_flag = False; bdr_viscous_flag = False
    clp=0; cdp=0; clf=0; cdf=0
    for datasetName in fsmesh.GetUnstructDatasetNames():
      if datasetName.StartsWith("Boundary"):
        import copy
        boundary_values_datasetname = copy.deepcopy(datasetName)
        boundary_values = fsmesh.GetUnstructDataset(datasetName).GetValues()
        boundary_names = fsmesh.GetUnstructDataset(datasetName).GetNames()
        boundary_values_numpy = numpy.array(boundary_values.Buffer(), copy=True)

        for index,boundary_name in enumerate(boundary_names):
          if boundary_name == "CoefPressure":
            bdr_flag = True
            index_cp = index
          if boundary_name == "CoefSkinFrictionX":
            bdr_viscous_flag = True
            index_cf = index
      if datasetName.StartsWith("Surface"):
        normals_flag = True
        normals_datasetname = copy.deepcopy(datasetName)

    if bdr_flag == True:
      if normals_flag == False:
        normals_datasetname = "SurfaceNormals"
        disc.GetMeshInterface().ExportDataToFSMesh(fsmesh, DataExportSelection.AreaWeightedSurfaceNormal, normals_datasetname) or FSError.PrintAndExit()

      cosa = math.cos(math.radians(alpha))
      sina = math.sin(math.radians(alpha))

      for wall_marker in wall_markers:
        print("wall_marker",wall_marker)
        selection = FSMeshSelectionOpAttribute(FSMeshEnums.CT_Undefined, "CADGroupID", wall_marker)
        wall = selection.Apply(fsmesh)
        #wall.ExportMeshTECPLOT(Filename="wall%d.plt" %wall_marker,PrefixDatasetName=True)
        boundary_values = wall.GetUnstructDataset(boundary_values_datasetname).GetValues()
        boundary_values_numpy = numpy.array(boundary_values.Buffer(), copy=True)
        surface_normals = wall.GetUnstructDataset(normals_datasetname).GetValues()
        surface_normals_numpy = numpy.array(surface_normals.Buffer(), copy=True)
        cp = boundary_values_numpy[:,index_cp]
        nx = surface_normals_numpy[:,0]
        ny = surface_normals_numpy[:,1]
        nz = surface_normals_numpy[:,2]

        node_coordinates = wall.GetUnstructDataset("Coordinates").GetValues()
        node_coordinates_numpy = numpy.array(node_coordinates.Buffer(), copy=True)
        xt = numpy.ravel(node_coordinates_numpy[:,0])
        yt = numpy.ravel(node_coordinates_numpy[:,1])
        zt = numpy.ravel(node_coordinates_numpy[:,2])

        area = numpy.empty(0)
        for t in FSUnstructSurfaceCellTypes:
          if wall.HasCellType(t):
            cell2Node = wall.GetCell2Node(t)
            cn = numpy.array(cell2Node.Buffer(), copy=True)
            if t == 4: surface_2D = computeSurfaceQuad(cn,xt,yt,zt)
            elif t == 3: surface_2D = computeSurfaceTri(cn,xt,yt,zt)
            area = numpy.concatenate([area,surface_2D])

        if bdr_viscous_flag == False:
          cfx = numpy.zeros(cp.size)
          cfy = numpy.zeros(cp.size)
          cfz = numpy.zeros(cp.size)
        else:
          print("index_cf",index_cf)
          cfx = boundary_values_numpy[:,index_cf]
          cfy = boundary_values_numpy[:,(index_cf+1)]
          cfz = boundary_values_numpy[:,(index_cf+2)]

        clp_i = cp*(-sina*nx+cosa*nz)
        cdp_i = cp*(+cosa*nx+sina*nz)
        clf_i = -sina*cfx+cosa*cfz
        cdf_i = +cosa*cfx+sina*cfz
        clp += numpy.sum(clp_i)
        cdp += numpy.sum(cdp_i)
        clf += numpy.sum(clf_i*area)
        cdf += numpy.sum(cdf_i*area)

      print("CLp = {:.5f}".format(clp))
      print("CLf = {:.5f}".format(clf))
      print("CL  = {:.5f}".format(clp+clf))

      print("CDp = {:.4f} x 10-4".format(1.e4*cdp))
      print("CDf = {:.4f} x 10-4".format(1.e4*cdf))
      print("CD  = {:.4f} x 10-4".format(1.e4*(cdp+cdf)))

    res = ['surf,clp,cdp,clf,cdf', numpy.array([[surface], [clp], [cdp], [clf], [cdf]]),1,1,1]
    return res

def computeWallSurfaceCassiopee(mesh_name,alpha=0.0):

    t = C.convertFile2PyTree(mesh_name,"bin_pickle")
    wall = C.extractBCOfType(t,"BCWall")
    G._getVolumeMap(wall)
    G._getNormalMap(wall)
    volumes = Internal.getNodesFromName(wall,"vol")
    cp = Internal.getNodesFromName(wall,"CoefPressure")
    surface = 0; clp=0; cdp=0; clf=0; cdf=0
    if cp != []:
      fluxcoef = 1.0
      cosa = math.cos(math.radians(alpha))
      sina = math.sin(math.radians(alpha))
      C._initVars(wall, '{centers:nx} = {sx}/sqrt({sx}*{sx}+{sy}*{sy}+{sz}*{sz})')
      C._initVars(wall, '{centers:ny} = {sy}/sqrt({sx}*{sx}+{sy}*{sy}+{sz}*{sz})')
      C._initVars(wall, '{centers:nz} = {sz}/sqrt({sx}*{sx}+{sy}*{sy}+{sz}*{sz})')
      C._initVars(wall, "{{centers:clp}}={{centers:CoefPressure}}*(-{sin:20.16g}*{{centers:nx}}+{cos:20.16g}*{{centers:nz}})".format(cos=cosa,sin=sina))
      C._initVars(wall, "{{centers:cdp}}={{centers:CoefPressure}}*(+{cos:20.16g}*{{centers:nx}}+{sin:20.16g}*{{centers:nz}})".format(cos=cosa,sin=sina))
      clp = P.integ(wall, 'centers:clp')[0]
      cdp = P.integ(wall, 'centers:cdp')[0]

      for xyz in ['X','Y','Z']:
          if C.isNamePresent(wall, 'centers:CoefSkinFriction'+xyz) != +1:
              C._initVars(wall, 'centers:CoefSkinFriction'+xyz, 0.)
              print('  warning: centers:CoefSkinFriction{} not found on skin and initialized using zeroes'.format(xyz))

      C._initVars(wall, "{{centers:clf}}=(-{sin:20.16g}*{{centers:CoefSkinFrictionX}}+{cos:20.16g}*{{centers:CoefSkinFrictionZ}})".format(cos=cosa,sin=sina))
      C._initVars(wall, "{{centers:cdf}}=(+{cos:20.16g}*{{centers:CoefSkinFrictionX}}+{sin:20.16g}*{{centers:CoefSkinFrictionZ}})".format(cos=cosa,sin=sina))
      clf = Internal.getNodeFromName(wall,"clf")[1]
      cdf = Internal.getNodeFromName(wall,"cdf")[1]

      print("clf",numpy.sum(clf))
      print("cdf",numpy.sum(cdf))


      clf = P.integ(wall, 'centers:clf')[0]
      cdf = P.integ(wall, 'centers:cdf')[0]
      print("CLp = {:.5f}".format(clp))
      print("CLf = {:.5f}".format(clf))
      print("CL  = {:.5f}".format(clp+clf))

      print("CDp = {:.4f} x 10-4".format(1.e4*cdp))
      print("CDf = {:.4f} x 10-4".format(1.e4*cdf))
      print("CD  = {:.4f} x 10-4".format(1.e4*(cdp+cdf)))

      #res = 'cdf,cdp,clf,clp'
    wall = T.breakElements(wall)
    G._getVolumeMap(wall)
    volumes = Internal.getNodesFromName(wall,"vol")

    for i in range(len(volumes)):
      surface += numpy.sum(volumes[i][1])
    res = ['surf,clp,cdp,clf,cdf', numpy.array([[surface], [clp], [cdp], [clf], [cdf]]),1,1,1]
    return res
