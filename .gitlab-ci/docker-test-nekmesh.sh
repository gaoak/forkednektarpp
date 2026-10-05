#!/bin/bash
# Run inside the nekmesh image by .dockerhub-test-template: check that a simple
# mesh generation works.
set -e
WORK=$HOME/ci-test
mkdir -p $WORK
cd /tmp
cp -r --parents utilities/NekMesh/Tests/MeshGen/STEP $WORK
cd $WORK/utilities/NekMesh/Tests/MeshGen/STEP
NekMesh -v -m jac:list 3d_bl_cyl.mcf 3d_bl_cyl-out.xml
