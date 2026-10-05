#!/bin/bash
# Run inside a Docker Hub image by .dockerhub-test-template: a quick parallel
# solve and a parallel Python run. The files listed in test_paths are copied to
# /tmp by the CI job; copy them to the home directory so they are writable.
set -e
WORK=$HOME/ci-test
mkdir -p $WORK
cd /tmp
cp -r --parents solvers/IncNavierStokesSolver/Tests library/Demos/Python/MultiRegions library/Demos/MultiRegions/Tests $WORK
cd $WORK

mpirun -n 3 IncNavierStokesSolver solvers/IncNavierStokesSolver/Tests/ChanFlow_m3_par.xml
test_output=`IncNavierStokesSolver solvers/IncNavierStokesSolver/Tests/ChanFlow_m3_par.xml | grep "L 2 error (variable u)" | awk '{print ($7 < 1e-7)}'`
if [ "$test_output" -eq 0 ]; then echo "Tolerance test failed on parallel IncNavierStokesSolver run"; exit 1; fi
cd library/Demos/Python/MultiRegions
mpirun -n 2 python3 Helmholtz2D.py ../../MultiRegions/Tests/Helmholtz2D_P7.xml
test_output=`mpirun -n 2 python3 Helmholtz2D.py ../../MultiRegions/Tests/Helmholtz2D_P7.xml | grep "L 2 error (variable nek)" | awk '{print ($7 < 1e-4)}'`
if [ "$test_output" -eq 0 ]; then echo "Tolerance test failed on parallel Python run"; exit 1; fi
