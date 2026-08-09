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

### AVX2/AVX512
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_SIMD=AVX2 #AVX512
```


### SVE/SVE2
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_SVE_BITS=xxx \
         -DNEKTAR_ENABLE_SIMD=SVE  #SVE2 
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
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp" 
```

Note: 
1. Just-in-time compilation does not work properly on `sm_90` architecture. To 
  enable ahead-of-time compilation, the architecture must be specified explictly
  as fellow:

```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DNEKTAR_DEVICE_ARCH=sm_90 \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp" 
```

2. Due to massive library size, linking problems may occur when using the AdaptiveCpp compiler. Using the lld linker instead of the GNU linker (bfd) can fix the problem:

```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp" \
         -DCMAKE_LINKER=lld \
         -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
```     

3. Possible fix for a `Could NOT find MPI_CXX (missing: MPI_CXX_WORKS)` error is provied below:

```
export OMPI_CC=clang
export OMPI_CXX=$ACPP_HOME/AdaptiveCpp/build/bin/acpp

cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DNEKTAR_USE_MPI=ON \
         -DCMAKE_C_COMPILER=mpicc \
         -DCMAKE_CXX_COMPILER=mpicxx 
```

4. See [guideline](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) for instructions to compile AdpativeCpp with CUDA backend. Nektar++ has been tested with AdaptiveCpp v25.10.0. An example of configuration command is provided below:

```
cmake $ACPP_HOME/AdaptiveCpp/ -B $ACPP_HOME/AdaptiveCpp/build/ \
         -DWITH_CUDA_BACKEND=ON \
         -DWITH_OPENCL_BACKEND=OFF \
         -DWITH_ROCM_BACKEND=OFF \
         -DWITH_SSCP_COMPILER=ON \
         -DCMAKE_C_COMPILER=clang \
         -DWITH_LEVEL_ZERO_BACKEND=OFF \
         -DCMAKE_CXX_EXTENSIONS=OFF \
         -DCMAKE_INSTALL_PREFIX=$ACPP_HOME/AdaptiveCpp/build/

make install -C $ACPP_HOME/AdaptiveCpp/build/ -j 8

export LD_LIBRARY_PATH=$ACPP_HOME/AdaptiveCpp/build/lib/:${LD_LIBRARY_PATH}
```


### AdaptiveCpp SYCL (HIP)
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-HIP \
         -DNEKTAR_DEVICE_ARCH=gfx90a \
         -DROCM_PATH="path-to-rocm" \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp" 
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

2. See [guideline](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) for instructions to compile AdpativeCpp with HIP/ROCm backend. Nektar++ has been tested with AdaptiveCpp v25.10.0. An example of configuration command is provided below:

```
cmake $ACPP_HOME/AdaptiveCpp/ -B $ACPP_HOME/AdaptiveCpp/build/ \
         -DWITH_CUDA_BACKEND=OFF \
         -DWITH_OPENCL_BACKEND=OFF \
         -DWITH_ROCM_BACKEND=ON \
         -DWITH_SSCP_COMPILER=ON \
         -DCMAKE_C_COMPILER=clang \
         -DWITH_LEVEL_ZERO_BACKEND=OFF \
         -DCMAKE_CXX_EXTENSIONS=OFF \
         -DCMAKE_INSTALL_PREFIX=$ACPP_HOME/AdaptiveCpp/build/

make install -C $ACPP_HOME/AdaptiveCpp/build/ -j 8

export LD_LIBRARY_PATH=$ACPP_HOME/AdaptiveCpp/build/lib/:${LD_LIBRARY_PATH}
```

### Intel LLVM SYCL (CUDA) - EXPERIMENTAL
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DCMAKE_C_COMPILER="$DPCPP_HOME/llvm/build/bin/clang" \ 
         -DCMAKE_CXX_COMPILER="$DPCPP_HOME/llvm/build/bin/clang++" 
```

Note:
1. See [guideline](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) for instructions to compile Intel LLVM with CUDA backend. Nektar++ has been tested with Intel LLVM v6.3.0 on Ubuntu 22.04 and LLVM v7.0.0 on Ubuntun 24.04. An example of configuration command is provided below:

```
python3 $DPCPP_HOME/llvm/buildbot/configure.py --cuda --obj-dir=$DPCPP_HOME/llvm/build 
python3 $DPCPP_HOME/llvm/buildbot/compile.py --obj-dir=$DPCPP_HOME/llvm/build

export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib/:${LD_LIBRARY_PATH}

```

For GH200, please use:

```
  python3 $DPCPP_HOME/llvm/buildbot/configure.py --host-target "AArch64;ARM;X86" --cuda --obj-dir=$DPCPP_HOME/llvm/build
  python3 $DPCPP_HOME/llvm/buildbot/compile.py --obj-dir=$DPCPP_HOME/llvm/build

  export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib/:${LD_LIBRARY_PATH}
```

### Intel LLVM SYCL (HIP) - EXPERIMENTAL
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-HIP \
         -DNEKTAR_DEVICE_ARCH=gfxzzz \
         -DCMAKE_C_COMPILER="$DPCPP_HOME/llvm/build/bin/clang" \ 
         -DCMAKE_CXX_COMPILER="$DPCPP_HOME/llvm/build/bin/clang++" 
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

2. See [guideline](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) for instructions to compile Intel LLVM with HIP/ROCm backend. Nektar++ has been tested with Intel LLVM v6.3.0 on Debian Trixie. An example of configuration command is provided below:

```
python3 $DPCPP_HOME/llvm/buildbot/configure.py --hip --obj-dir=$DPCPP_HOME/llvm/build --llvm-external-projects compiler-rt
python3 $DPCPP_HOME/llvm/buildbot/compile.py --obj-dir=$DPCPP_HOME/llvm/build

export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib/:${LD_LIBRARY_PATH}
```


Installation
------------
The default installation location is in a `dist` subdirectory of the `build`
directory. This can be changed by setting the `CMAKE_INSTALL_PREFIX` option
using `ccmake`. To install the compiled libraries, solvers and header files, on
UNIX-based systems run:

    make install
