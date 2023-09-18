# How to install redesign-prototye
## Build Nektar++ e.g. on current master
- See README.md under [Nektar++ repository](https://gitlab.nektar.info/nektar/nektar)
- For development, the `Debug` option is recommended
- Note that the SIMD options have to be turned off, if you want to use CUDA support (Should be by default).

## Build Nektar++ redesign-prototype
In main directory of the redesign, create a `build` directory `cd` into it, configure and build via:
```bash
mkdir build
cd build
cmake ../ -DNektar++_DIR=~/code/nektarMaster/build/dist/lib64/nektar++/cmake -DCMAKE_BUILD_TYPE=Debug
make 
```

If boost is not available on system, build boost (e.g. v1.71.1)
- One option is via the Ubuntu package manager `sudo apt install libboost-all-dev`
- Another option is to download boost and install it manually

After installing boost, run again:
```bash
cmake ../ -DNektar++_DIR=~/code/nektarMaster/build/dist/lib64/nektar++/cmake -DCMAKE_BUILD_TYPE=Debug
make
```

When the installation was successful, check it via tests:
```bash
ctest --output-on-failure
```

## Build Nektar++ redesign-prototype with CUDA
To build the redesign project with CUDA:
- The CUDA option must be turned on
- The SIMD option must be turned off

This can be obtained by using the following command
```bash
cmake ../ -DNektar++_DIR=~/code/nektarMaster/build/dist/lib64/nektar++/cmake -DCMAKE_BUILD_TYPE=Debug -DNEKTAR_USE_CUDA=ON -DNEKTAR_USE_SIMD=OFF 
make
```
