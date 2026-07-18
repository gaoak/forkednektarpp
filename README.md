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


Device Support
-----------
A minimalist compilation command example is shown below for each available backend:

### Serial
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON 
```

### AVX2 
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_SIMD=AVX2 
```

### AVX512 
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_SIMD=AVX512 
```

### SVE 
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_SIMD=SVE \
         -DNEKTAR_SVE_BITS=xxx
```

Notes:
 1. xxx can be 128, 256, 512, 1024, or 2048 depending of the architecture

### SVE2 
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_SIMD=SVE2 \
         -DNEKTAR_SVE_BITS=xxx
```

Notes:
 1. xxx can be 128, 256, 512, 1024, or 2048 depending of the architecture

### CUDA
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=CUDA \
         -DNEKTAR_DEVICE_ARCH=sm_xx \ 
```

For device architecture NEKTAR_DEVICE_ARCH, please use:

```:
V100: `sm_xx=sm_70`
A100: `sm_xx=sm_80` 
A40: `sm_xx=sm_86`
H100: `sm_xx=sm_90` 
B100: `sm_xx=sm_100` 
```

The `sm_xx` value can also be queried using the following command:
`nvidia-smi --query-gpu=compute_cap --format=csv`

### HIP
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=HIP \
         -DCMAKE_CXX_COMPILER=hipcc \
         -DNEKTAR_DEVICE_ARCH=gfxzzz \
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - MI210: `gfxzzz=gfx908` 
 - MI250: `gfxzzz=gfx90a` 
 - MI300: `gfxzzz=gfx942` 
 - MI325: `gfxzzz=gfx942` 
 - MI350: `gfxzzz=gfx950` 
 - MI355: `gfxzzz=gfx950` 
``` 

The `gfxzzz` value can also be queried using the following command:
`rocm-smi --showproductname | grep gfx`

### AdaptiveCpp SYCL (CUDA)
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DCMAKE_CXX_COMPILER="/path-to-adaptivecpp-compiler/bin/acpp" 
 ```

Note: 
1. Just-in-time compilation does not work properly on `sm_90` architecture. To 
  enable ahead-of-time compilation, the architecture must be specified explictly
  as fellow:

```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DNEKTAR_DEVICE_ARCH=sm_90 \
         -DCMAKE_CXX_COMPILER="/path-to-adaptivecpp-compiler/bin/acpp" 
 ```

2. Due to massive library size, linking problems may occur when using the AdaptiveCpp compiler. Using the lld linker instead of the GNU linker (bfd) can fix the problem:

```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DCMAKE_CXX_COMPILER="/path-to-adaptivecpp-compiler/bin/acpp" \
         -DCMAKE_LINKER=lld \
         -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
```     

3. Possible fix for a `Could NOT find MPI_CXX (missing: MPI_CXX_WORKS)` error is provied below:

```
export OMPI_CC=/path-to-llvm/bin/clang
export OMPI_CXX=/path-to-adaptivecpp-compiler/bin/acpp

cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DNEKTAR_USE_MPI=ON \
         -DCMAKE_C_COMPILER=mpicc \
         -DCMAKE_CXX_COMPILER=mpicxx 
```

4.  See [guideline](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) for instructions to compile AdpativeCpp. An example of configuration is provided below:

```
cmake .. -DWITH_CUDA_BACKEND=ON \
         -DWITH_OPENCL_BACKEND=OFF \
         -DWITH_ROCM_BACKEND=OFF \
         -DWITH_SSCP_COMPILER=ON \
         -DCMAKE_C_COMPILER=clang \
         -DWITH_LEVEL_ZERO_BACKEND=OFF \
         -DCMAKE_CXX_EXTENSIONS=OFF \
         -DCMAKE_INSTALL_PREFIX=./
```


### AdaptiveCpp SYCL (HIP)
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-HIP \
         -DNEKTAR_DEVICE_ARCH=gfx90a \
         -DROCM_PATH="path-to-rocm" \
         -DCMAKE_CXX_COMPILER="/path-to-adaptivecpp-compiler/bin/acpp" 
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - MI210: `gfxzzz=gfx908` 
 - MI250: `gfxzzz=gfx90a` 
 - MI300: `gfxzzz=gfx942` 
 - MI325: `gfxzzz=gfx942` 
 - MI350: `gfxzzz=gfx950` 
 - MI355: `gfxzzz=gfx950` 
```

The `gfxzzz` value can also be queried using the following command:
`rocm-smi --showproductname | grep gfx`

2. See [guideline](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) for instructions to compile AdpativeCpp. An example of configuration is provided below:

```
cmake .. -DWITH_CUDA_BACKEND=OFF \
         -DWITH_OPENCL_BACKEND=OFF \
         -DWITH_ROCM_BACKEND=ON \
         -DWITH_SSCP_COMPILER=ON \
         -DCMAKE_C_COMPILER=clang \
         -DWITH_LEVEL_ZERO_BACKEND=OFF \
         -DCMAKE_CXX_EXTENSIONS=OFF \
         -DCMAKE_INSTALL_PREFIX=./
```


Installation
------------
The default installation location is in a `dist` subdirectory of the `build`
directory. This can be changed by setting the `CMAKE_INSTALL_PREFIX` option
using `ccmake`. To install the compiled libraries, solvers and header files, on
UNIX-based systems run:

    make install
