from FSDataManager import FSClac
from FSDataManager import FSLog
from FSDataManager import FSError
from FSDataManager import FSMesh
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter


bcDict = {
    1: "BCWallViscous",
    2: "BCWallViscous",
    3: "BCWallViscous",
    4: "BCSymmetryPlane",
    5: "BCFarfield"
}


def main():
    clac = FSClac()
    fsmeshOrig = FSMesh(clac)

    if not fsmeshOrig.ImportMeshTAU(Filename="./tau.grid"):
        FSError.PrintAndExit()
    if not fsmeshOrig.RepartitionMeshRCB():
        FSError.PrintAndExit()

    # Unlike what's shown in this example, FlowSolutions must be kept in FFD
    FSLog(clac, 0, "------ Conversion FSDM -> CGNS NGon -------")
    convObj = FSCGNSConverter(meshName="mesh.grid", bcDict=bcDict, datasets=[])
    convObj.convert(forFFDX=True)
    convObj.export(filename="mesh_ngon.cgns")
    FSLog(clac, 0, "------ Conversion done -------")


if __name__ == "__main__":
   main()

