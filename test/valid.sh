source /stck/rhea/dist/spiro_v2023_10/source.sh --env coda_users --compiler gcc@10 --mpi openmpi

export FSMESHPYTREECONVERSION_DIR=/stck/fbasile/Private/Sources/FSMeshPyTreeConversion/
export PYTHONPATH=$FSMESHPYTREECONVERSION_DIR/py/FSMeshPyTreeConversion/:$PYTHONPATH

set -e
#./run_MPI-OPENMP.sh -n 1 -t 1 -cmd python3 launchConversion.py
#./run_MPI-OPENMP.sh -n 1 -t 1 -cmd python3 launchCheckSurfaceAndBoundaryCoefficients.py

./run_MPI-OPENMP.sh -n 24 -t 1 -cmd python3 launchConversionParallel24.py
./run_MPI-OPENMP.sh -n 1 -t 1 -cmd python3 launchCheckSurfaceAndBoundaryCoefficientsParallel24.py
