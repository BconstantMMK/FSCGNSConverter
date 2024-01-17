import os
import Converter.PyTree as C
import Converter.Mpi as Cmpi
import KCore.test as test
from FSDM_CGNS import Converter_FSDM_CGNS
import Helpers_test

dictionary_BCs = {
    "naca0012_hexa.h5": {1 : 'BCSymmetryPlane', 2 : 'BCSymmetryPlane', 3 : 'BCOutflow', 4 : 'BCOutflow', 5 : 'BCWallViscous', 6 : 'BCFarfield'}, #1
    "naca0012_prism.h5": {1 : 'BCSymmetryPlane', 2 : 'BCSymmetryPlane', 3 : 'BCFarfield', 4 : 'BCWallViscous', 5 : 'BCWallViscous'}, #2
    "rae_hexa_prism.h5": {1:"BCSymmetryPlane", 2:"BCSymmetryPlane", 3:"BCWallViscous", 5:"BCFarfield"}, #6
    "M6_pyra_tetra_hexa.h5": {1 : 'BCWallViscous', 2 : 'BCWallViscous', 3 : 'BCWallViscous', 4 : 'BCSymmetryPlane', 5 : 'BCFarfield'}, #10
}

directory_new_results =  "./Data_new_test"
if Cmpi.rank==0:
  os.system("mkdir -p "+ directory_new_results)

meshes = dictionary_BCs.keys()
bcs = dictionary_BCs.values()

for i,(mesh,bc) in enumerate(zip(meshes,bcs)):
    print("Test n. %d, mesh %s.\n BCs=%s" %(i+1, mesh,bc))
    C_FC = Converter_FSDM_CGNS("./InitialMeshes_wSolution/"+mesh,keepFlowSolution=True,inmemory=True,dict_BCs=bc)
    C_FC.convertFSDM2CGNS()
    C_FC.convertMonozoneME2Ngon4FFD(reorient=True,mergeOnProc0=False)
    Cmpi.convertPyTree2File(C_FC.pytree,directory_new_results+"/"+mesh.split(".")[0]+".ref"+str(i+1),"bin_cgns")
    if Cmpi.rank==0:
      t = C.convertFile2PyTree(directory_new_results+"/"+mesh.split(".")[0]+".ref"+str(i+1),"bin_cgns")
      r = Helpers_test.test(t,i+1)
      if r==False:
          raise ValueError("Test n. %d, mesh %s has failed." %(i+1, mesh))
      else:
          print("Test n. %d, mesh %s has succeeded." %(i+1, mesh))
      print("\n\n")
    Cmpi.barrier()

if Cmpi.rank==0 and r == True:
    os.system("rm -r "+directory_new_results)
