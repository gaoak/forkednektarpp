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
             -DNEKTAR_DEVICE_ARCH=sm_xx \ 

Note:
- For H100, please use `sm_xx=sm_90` 
- For A40, please use `sm_xx=sm_86`
- For A100, please use `sm_xx=sm_80` 
- For V100, please use `sm_xx=sm_70` 

### HIP
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_HIP=ON \
             -DCMAKE_CXX_COMPILER=hipcc \
             -DNEKTAR_DEVICE_ARCH=gfxzzz \

Note:
- For MI100, please use `gfxzzz=gfx908` 
- For MI210, please use `gfxzzz=gfx90a` 
- For MI250, please use `gfxzzz=gfx90a` 
- For MI300, please use `gfxzzz=gfx942` 
- For MI325, please use `gfxzzz=gfx942` 

### Intel SYCL (Default)
    export CL_CONFIG_CPU_FORCE_LOCAL_MEM_SIZE=64K

    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SYCL=Default \
             -DCMAKE_CXX_COMPILER="/path-to-intel-compiler/bin/icpx" 

### Intel SYCL (CUDA)
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SYCL=CUDA \
             -DNEKTAR_DEVICE_ARCH=sm_xx \ 
             -DCMAKE_CXX_COMPILER="/path-to-intel-compiler/bin/icpx" 

Note:
- For H100, please use `sm_xx=sm_90` 
- For A40, please use `sm_xx=sm_86`
- For A100, please use `sm_xx=sm_80` 
- For V100, please use `sm_xx=sm_70` 

### AdaptiveCpp SYCL (Default)
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SYCL=Default \
             -DCMAKE_CXX_COMPILER="/path-to-adaptivecpp-compiler/bin/acpp" 

### AdaptiveCpp SYCL (CUDA)
    cmake .. -DNEKTAR_BUILD_REDESIGN=ON \
             -DNEKTAR_ENABLE_SYCL=CUDA \
             -DCMAKE_CXX_COMPILER="/path-to-adaptivecpp-compiler/bin/acpp" 
 

Installation
------------
The default installation location is in a `dist` subdirectory of the `build`
directory. This can be changed by setting the `CMAKE_INSTALL_PREFIX` option
using `ccmake`. To install the compiled libraries, solvers and header files, on
UNIX-based systems run:

    make install
