# FSCGNSConverter

FSCGNSConverter is a mesh conversion FS module that
facilitates data exchange between the CODA ecosystem and ONERA's CGNS pre- and
post-processing tools Cassiopee and maia.
It is a newer iteration of FSMeshPyTreeConversion offering an improved API and
increased performance.

<br>

## Table of Contents
1. [Repository structure](#1-repository-structure)  
2. [Cloning a module](#2-cloning-a-module)  
2.1 [Main branch](#21-main-branch)  
2.2 [Dev branch](#22-dev-branch)  
2.3 [User dev branches](#23-user-dev-branches)  
2.4 [PYTHONPATH variable](#24-pythonpath-variable)  
3. [Pulling updates](#3-pulling-updates)  
3.1 [Resolving conflicts](#31-resolving-conflicts)
4. [Contributing to a module](#4-contributing-to-a-module)  
5. [Merging your contributions](#5-merging-your-contributions)  
6. [Managing remotes](#6-managing-remotes)

<br>

## 1. API

The API of the FSCGNSConverter class is given below and includes  
  - constructor arguments: a list of mandatory and optional parameters required to create an instance of the class;  
  - public methods: the functions that can be called on the instance.

### 1.1 Instantiation

The names of all the input arguments for the `FSCGNSConverter` class are listed
hereafter:

```py
convObj = FSCGNSConverter(
    meshName=None,
    pyTree=None,
    clac=None,
    fsmesh=None,
    dimPb=2,  # soon deprecated
    flipYZAxes=False,
    conformal=True,
    IBMParameters=None,
    datasets='all',
    bcDict=None,
    coordsName="Coordinates",
    **kwargs
)
```

| Input argument | Name in FSMeshPyTreeConversion | Type | Default | Description |
|--------------|------------|--------------|--------------|-------------|
| meshName | mesh_name | str | None | Mesh filename |
| pyTree | pytree | CGNS PyTree / base / zone | None | Input CGNS mesh |
| clac | | object | None, ie, `FSClac` | FSDataManager's `clac` object |
| fsmesh | | object | None | FSDataManager's `fsmesh` object |
| flipYZAxes | invertPlanesYZ | bool | False | Whether to flip the y- and z-axes |
| conformal | | bool | True | Whether the mesh presents hanging nodes |
| IBMParameters | IBM_parameters | dict | None | Dictionary of IBM parameters |
| datasets | whichDatasets | str, list or set | 'all' | Names of the datasets to consider during conversion |
| bcDict | dict_BCs | dict | None | Dictionary mapping FS BC indices to CGNS BC families |
| coordsName | coords_name  | str | "Coordinates" | Name of the Coordinates field |
| verbose | x | bool | True | Whether to display the step-by-step script progression |

NB: Input arguments `inmemory`, `keepFlowSolution` and `IBM` were deleted.

### 1.2 FSCGNSConverter class methods

#### 1.2.1 Conversion

There is a single function to convert the mesh from one format to the other:

```py
convObj.convert(**kwargs)
```

The resulting format was inferred during the class instantiation.  
When converting to CGNS, the keyworded arguments are:
 - `forOverset`: `bool`, Whether convert for overset cases (default: `False`)
 - `forFFDX`: `bool`, Whether convert for use in FFD / FFX (Multiple-Elements -> monozone NGon, normals pointing inwards) (default: `False`)

#### 1.2.2 Export

There is a single function to convert the mesh from one format to the other:

```py
convObj.export(filename=outfile, **kwargs)
```

The output format is inferred from the filename's extension.  
When exporting the CGNS PyTree or the FS mesh, the keyworded arguments are:
 - `verbose`: `bool`, print CGNS tree / FS mesh info (default: `True`)

### 1.3 FSCGNSConverter class attributes

After calling the `convert` function, the CGNS pyTree or the FS mesh can be obtained with:

```py
t = convObj.pyTree
fsmesh = convObj.fsmesh
```

### 1.4 FSCGNSConverter class attributes

After calling one of the `convert` functions, the CGNS pyTree or the FS mesh can be obtained with:
```py
t = convObj.pyTree
fsmesh = convObj.fsmesh
```

<br>

## 2. Examples

CGNS files comprising both the mesh and solution fields can be converted to
h5 format, and vice-versa.

### 2.1 Conversion from h5 to cgns

In this example, an h5 file is converted to cgns format.

```py
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

meshFilename = "mesh.h5"
caseConfig = {
    # list of optional arguments
}

convObj = FSCGNSConverter(meshName=meshFilename, **caseConfig)
convObj.convert()

# Get the CGNS tree or export the CGNS mesh
t = convObj.pyTree
convObj.exportCGNSMesh(filename="t.cgns", verbose=True)
```

### 2.2 Conversion from cgns to h5

#### 2.2.1 Using a filename 

In this example, an cgns file is converted to h5 format.

```py
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

meshFilename = "mesh.cgns"
caseConfig = {
    # list of optional arguments
}

convObj = FSCGNSConverter(meshName=meshFilename, **caseConfig)
convObj.convert()

# Get the FS mesh or export the FS mesh
fsmesh = convObj.fsmesh
convObj.exportFSMesh(filename="mesh.h5", verbose=True)
```

#### 2.2.2 From a CGNS tree 

In this example, an cgns tree is converted to h5 format and saved to tecplot format.

```py
import Converter.PyTree as C
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

meshFilename = "mesh.cgns"
t = C.convertFile2PyTree(meshFilename)
caseConfig = {
    # list of optional arguments
}

convObj = FSCGNSConverter(meshName=t, **caseConfig)
convObj.convert2FSDM()

# Save to tecplot format
convObj.exportFSMesh2Tecplot(filename="mesh.plt")
```
