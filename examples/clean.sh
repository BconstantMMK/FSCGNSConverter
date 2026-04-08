#! /bin/sh

# Clean
rm -rf __pycache__/
rm -f *.plt *.cgns *.txt *.dat *.h5
find . -type f -name "*.grid" -not -name "tau.grid" -delete
