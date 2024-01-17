import Converter.PyTree as C
import KCore.test as test
import Helpers_test

dictionary_BCs = {
    "naca0012_hexa.h5": {1 : 'BCSymmetryPlane', 2 : 'BCSymmetryPlane', 3 : 'BCOutflow', 4 : 'BCOutflow', 5 : 'BCWallViscous', 6 : 'BCFarfield'}, #1
    "naca0012_prism.h5": {1 : 'BCSymmetryPlane', 2 : 'BCSymmetryPlane', 3 : 'BCFarfield', 4 : 'BCWallViscous', 5 : 'BCWallViscous'}, #2
    "rae_hexa_prism.h5": {1:"BCSymmetryPlane", 2:"BCSymmetryPlane", 3:"BCWallViscous", 5:"BCFarfield"}, #6
    "M6_pyra_tetra_hexa.h5": {1 : 'BCWallViscous', 2 : 'BCWallViscous', 3 : 'BCWallViscous', 4 : 'BCSymmetryPlane', 5 : 'BCFarfield'}, #10
}
alpha = [1.25,1.25,2.79,3.06]

meshes = dictionary_BCs.keys()
bcs = dictionary_BCs.values()

for i,(mesh,bc) in enumerate(zip(meshes,bcs)):
    res_CODA = Helpers_test.computeWallSurfaceCODA("./InitialMeshes_wSolution/"+mesh,bc,alpha=alpha[i])
    surface_CODA = res_CODA[1][0][0]
    cd_CODA = res_CODA[1][2][0] + res_CODA[1][4][0]
    print("Total surface of skin from CODA = ",surface_CODA)
    print("Cd on skin from CODA = ",cd_CODA)
    #res_Cassiopee = Helpers_test.computeWallSurfaceCassiopee("Data_new_test/"+mesh.split(".")[0]+".ref"+str(i+1),alpha=alpha[i])
    res_Cassiopee = Helpers_test.computeWallSurfaceCassiopee("Data/launchConversion.ref"+str(i+1),alpha=alpha[i])
    surface_Cassiopee = res_Cassiopee[1][0][0]
    cd_Cassiopee = res_Cassiopee[1][2][0] + res_Cassiopee[1][4][0]
    print("Total surface of skin from Cassiopee = ",surface_Cassiopee)
    print("Cd on skin from Cassiopee = ",cd_Cassiopee)
    if abs(cd_Cassiopee - cd_CODA) > 1e-10 or abs(surface_Cassiopee - surface_CODA) > 1e-10:
        raise ValueError("Test n. %d, mesh %s has failed. Total wall surface and/or cd computed with .h5 and .cgns differ." %(i+1, mesh))
    # test array with reference arrays in repository Data
    r = test.testA(res_Cassiopee, number=i+1)
    if r == False:
        raise ValueError("Test n. %d, mesh %s has failed. Results differ from reference." %(i+1, mesh))
    else:
        print("Test n. %d, mesh %s has succeeded." %(i+1, mesh))

    print("\n\n")


