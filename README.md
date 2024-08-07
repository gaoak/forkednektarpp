Nektar++
========
Nektar++ is an open-source software framework designed to support the
development of high-performance scalable solvers for partial differential
equations (PDEs) using the spectral/hp element method.

This package consists of a set of libraries (the framework) and a number of
pre-written PDE solvers for a selection of application domains.

The software and User Guide is available for download from
<http://www.nektar.info/>.


User Guide
----------
Detailed information on compiling, installing and using the software is
available in the User Guide. This document is available as a pre-compiled PDF
from the downloads section of the project website.


Tutorials
---------
A number of tutorials are available, designed to walk the user through the
basics of spectral/hp element methods, through the use of individual solvers and
performing specific types of calculations.

The tutorials are available from <http://doc.nektar.info/tutorials/latest>.


Pre-requisites
--------------
Nektar++ requires the following software to be installed on the users system:

- CMake
- BLAS/LAPACK

Additional software is also required. This can either be installed system-wide
or it can be downloaded and compiled automatically during the build process.

For more detailed information, please see the User Guide.


Compilation
-----------
On most UNIX-based systems a default compilation can be performed using the
following commands from the top-level of the source tree:

    mkdir build
    cd build
    cmake ..
    make

To alter the build configuration (for example, to enable parallel execution
support) we recommend using the `ccmake` command instead of `cmake`. 

For more detailed operating-system specific instructions, please see the
User Guide.


Redesign
-----------
A minimalist compilation command example is shown below for each available backend:

### Serial
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON 

### AVX2 
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SIMD_AVX2=ON 

### AVX512 
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SIMD_AVX512=ON 

### CUDA
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_CUDA=ON \
             -DCMAKE_CUDA_FLAGS="--extended-lambda --expt-relaxed-constexpr" \
             -DCMAKE_CUDA_ARCHITECTURES=86 

Note:
- For A40, please use `-DCMAKE_CUDA_ARCHITECTURES=86`
- For A100, please use `-DCMAKE_CUDA_ARCHITECTURES=80` 

### SYCL (CUDA)
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SYCL=ON \
             -DCMAKE_CXX_COMPILER="/path-to-sycl-llvm/bin/clang-19" \
             -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda" 

Note:
- If desired, one can build the SYCL LLVM locally using
```
    module load cuda
    python3 -m pip install ninja
    git clone https://github.com/intel/llvm -b sycl
    python3 llvm/buildbot/configure.py --cuda
    python3 llvm/buildbot/compile.py  -j32
```
- If the SYCL LLVM is build locally, the `bin` and `lib` must be added to the PATHs environment variable
```
    PATH="$PATH:/path-to-llvm/build/bin/"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path-to-llvm/build/lib/"
```

### Kokkos (Serial)
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_KOKKOS=ON \
             -DKokkos_DIR=/path/kokkos-serial-install/lib/cmake/Kokkos/

Note:
- If desired, one can build Kokkos locally using
```
    git clone https://github.com/kokkos/kokkos.git
    cd kokkos
    mkdir build && cd build 
    cmake .. -DCMAKE_CXX_COMPILER=g++ \
             -DCMAKE_CXX_FLAGS=-fPIC \
             -DCMAKE_INSTALL_PREFIX=~/path/kokkos-serial-install
    make install
```

### Kokkos (CUDA)
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_KOKKOS=ON \
             -DKokkos_DIR=/path/kokkos-cuda-install/lib/cmake/Kokkos/

Note:
- If desired, one can build Kokkos locally using
```
    git clone https://github.com/kokkos/kokkos.git
    cd kokkos
    mkdir build && cd build 
    cmake .. -DCMAKE_CXX_COMPILER=g++ \
             -DCMAKE_CXX_FLAGS=-fPIC \
             -DCMAKE_INSTALL_PREFIX=/path/kokkos-cuda-install \
             -DKokkos_ARCH_AMPERE86=ON \
             -DKokkos_ENABLE_CUDA=ON \
             -DKokkos_ENABLE_CUDA_LAMBDA=ON \
             -DKokkos_ENABLE_CUDA_CONSTEXPR=ON
    make install
```
- For A40, please use `-DKokkos_ARCH_AMPERE86=ON`
- For A100, please use `-DKokkos_ARCH_AMPERE80=ON` 

Installation
------------
The default installation location is in a `dist` subdirectory of the `build`
directory. This can be changed by setting the `CMAKE_INSTALL_PREFIX` option
using `ccmake`. To install the compiled libraries, solvers and header files, on
UNIX-based systems run:

    make install
