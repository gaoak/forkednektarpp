#!/bin/bash
# Run inside the nektar-dev image by .dockerhub-test-template: compile and run
# the standalone executable template against the installed Nektar++.
set -e
WORK=$HOME/ci-test
mkdir -p $WORK
cd /tmp
cp -r --parents templates/executable $WORK
cd $WORK/templates/executable
./test.sh /usr/local/lib64/nektar++/cmake 3 /usr/bin/cmake
