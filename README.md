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


### NEON
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_SIMD=NEON
```

Notes:
 1. AArch64 (64-bit Arm) only. NEON gives 128-bit vectors, so 2 lanes in
    double precision and 4 lanes in single precision.
 2. No extra compiler flags are needed, as NEON is part of the AArch64
    baseline.
 3. This is the option to use on Apple silicon (M1-M4), which implements
    NEON but not SVE.


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
         -DNEKTAR_DEVICE_ARCH=sm_XX
```

For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - V100: `sm_XX=sm_70`
 - A100: `sm_XX=sm_80`
 - A40:  `sm_XX=sm_86`
 - H100: `sm_XX=sm_90`
 - B100: `sm_XX=sm_100`
```

The `sm_XX` value can also be queried using the following command:
`nvidia-smi --query-gpu=compute_cap --format=csv`

### HIP
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=HIP \
         -DCMAKE_CXX_COMPILER=hipcc \
         -DNEKTAR_DEVICE_ARCH=gfxXXX
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - MI210: `gfxXXX=gfx90a`
 - MI250: `gfxXXX=gfx90a`
 - MI300: `gfxXXX=gfx942`
 - MI325: `gfxXXX=gfx942`
 - MI350: `gfxXXX=gfx950`
 - MI355: `gfxXXX=gfx950`
```

The `gfxXXX` value can also be queried using the following command:
`rocm-smi --showproductname | grep gfx`

### AdaptiveCpp SYCL (CUDA)
For just-in-time compilation, please use:
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp"
```

For ahead-of-time compilation, please use:
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-CUDA \
         -DNEKTAR_DEVICE_ARCH=sm_XX \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp"
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - V100: `sm_XX=sm_70`
 - A100: `sm_XX=sm_80`
 - A40:  `sm_XX=sm_86`
 - H100: `sm_XX=sm_90`
 - B100: `sm_XX=sm_100`
```

The `sm_XX` value can also be queried using the following command:
`nvidia-smi --query-gpu=compute_cap --format=csv`

`NEKTAR_DEVICE_ARCH` is optional and triggers ahead-of-time compilation
(`--acpp-targets=cuda:sm_XX`), so the resulting binary only runs on the
architecture it was built for. If it is omitted, the kernels are compiled
just-in-time (`--acpp-targets=generic`) instead and the build runs on any
supported GPU, at the cost of a one-off compilation delay when the solver
starts. Just-in-time compilation does not work properly on the `sm_90`
architecture, so `NEKTAR_DEVICE_ARCH` must be specified for H100.

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
For just-in-time compilation, please use:
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-HIP \
         -DROCM_PATH="path-to-rocm" \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp"
```

For ahead-of-time compilation, please use:
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-HIP \
         -DNEKTAR_DEVICE_ARCH=gfxXXX \
         -DROCM_PATH="path-to-rocm" \
         -DCMAKE_CXX_COMPILER="$ACPP_HOME/AdaptiveCpp/build/bin/acpp"
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - MI210: `gfxXXX=gfx90a`
 - MI250: `gfxXXX=gfx90a`
 - MI300: `gfxXXX=gfx942`
 - MI325: `gfxXXX=gfx942`
 - MI350: `gfxXXX=gfx950`
 - MI355: `gfxXXX=gfx950`
```

The `gfxXXX` value can also be queried using the following command:
`rocm-smi --showproductname | grep gfx`

`NEKTAR_DEVICE_ARCH` is optional and triggers ahead-of-time compilation
(`--acpp-targets=hip:gfxXXX`), so the resulting binary only runs on the
architecture it was built for. If it is omitted, the kernels are compiled
just-in-time (`--acpp-targets=generic`) instead and the build runs on any
supported GPU, at the cost of a one-off compilation delay when the solver
starts.

2. AdaptiveCpp must be built against an LLVM with the same major version as
the LLVM of the ROCm installation (ROCm 7.0.x uses LLVM 20). The LLVM shipped
with ROCm cannot be used, as it does not provide `libLLVM.so`. An example of
LLVM build is provided below:

```
git clone --depth 1 -b llvmorg-20.1.8 https://github.com/llvm/llvm-project.git $LLVM_HOME

cmake -S $LLVM_HOME/llvm -B $LLVM_HOME/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=$LLVM_HOME/install \
         -DLLVM_ENABLE_PROJECTS="clang;lld" \
         -DLLVM_ENABLE_RUNTIMES="compiler-rt;openmp" \
         -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
         -DLLVM_BUILD_LLVM_DYLIB=ON \
         -DLLVM_LINK_LLVM_DYLIB=ON \
         -DLLVM_INCLUDE_TESTS=OFF \
         -DLLVM_INCLUDE_BENCHMARKS=OFF \
         -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
         -DCOMPILER_RT_BUILD_XRAY=OFF \
         -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
         -DCOMPILER_RT_BUILD_MEMPROF=OFF \
         -DCOMPILER_RT_BUILD_PROFILE=OFF \
         -DCOMPILER_RT_BUILD_ORC=OFF \
         -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
         -DCOMPILER_RT_BUILD_GWP_ASAN=OFF

cmake --build $LLVM_HOME/build --target install -j 8

export LD_LIBRARY_PATH=$LLVM_HOME/install/lib/:$LLVM_HOME/install/lib/x86_64-unknown-linux-gnu/:${LD_LIBRARY_PATH}
```

Only the compiler-rt builtins are needed, which ROCm links into HIP
libraries. The sanitizers are disabled as they fail to build against recent
kernel headers.

3. Possible fix for a `Could NOT find MPI_CXX (missing: MPI_CXX_WORKS)` error is provied below:

```
export OMPI_CC=$LLVM_HOME/install/bin/clang
export OMPI_CXX=$ACPP_HOME/AdaptiveCpp/build/bin/acpp

cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-HIP \
         -DNEKTAR_USE_MPI=ON \
         -DCMAKE_C_COMPILER=mpicc \
         -DCMAKE_CXX_COMPILER=mpicxx
```

4. See [guideline](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) for instructions to compile AdpativeCpp with HIP/ROCm backend. Nektar++ has been tested with AdaptiveCpp v25.10.0. An example of configuration command is provided below:

```
cmake $ACPP_HOME/AdaptiveCpp/ -B $ACPP_HOME/AdaptiveCpp/build/ \
         -DWITH_CUDA_BACKEND=OFF \
         -DWITH_OPENCL_BACKEND=OFF \
         -DWITH_ROCM_BACKEND=ON \
         -DWITH_SSCP_COMPILER=ON \
         -DWITH_LEVEL_ZERO_BACKEND=OFF \
         -DCMAKE_C_COMPILER=$LLVM_HOME/install/bin/clang \
         -DCMAKE_CXX_COMPILER=$LLVM_HOME/install/bin/clang++ \
         -DLLVM_DIR=$LLVM_HOME/install/lib/cmake/llvm \
         -DCLANG_EXECUTABLE_PATH=$LLVM_HOME/install/bin/clang++ \
         -DROCM_PATH="path-to-rocm" \
         -DCMAKE_CXX_EXTENSIONS=OFF \
         -DCMAKE_INSTALL_PREFIX=$ACPP_HOME/AdaptiveCpp/build/

make install -C $ACPP_HOME/AdaptiveCpp/build/ -j 8

export LD_LIBRARY_PATH=$ACPP_HOME/AdaptiveCpp/build/lib/:${LD_LIBRARY_PATH}
```

### Intel GPU - EXPERIMENTAL
```
cmake .. -DNEKTAR_ENABLE_DEVICE_SUPPORT=ON \
         -DNEKTAR_ENABLE_DEVICE=SYCL-Intel \
         -DNEKTAR_DEVICE_ARCH=arch \
         -DCMAKE_C_COMPILER=icx \
         -DCMAKE_CXX_COMPILER=icpx
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use the following
values:

```
 - Data Center GPU Max 1100/1350/1550 (Ponte Vecchio): `arch=pvc`
 - Data Center GPU Max with VG (Ponte Vecchio VG):     `arch=pvc_vg`
 - Data Center GPU Flex 170 (Arctic Sound, ACM-G10):   `arch=acm_g10`
 - Data Center GPU Flex 140 (Arctic Sound, ACM-G11):   `arch=acm_g11`
```

The names known to your installation can be listed with:

```
ocloc compile --help
```

and a single name checked with `ocloc ids arch`, for example `ocloc ids pvc`.
The architecture of the GPUs present on a machine is reported by
`sycl-ls --verbose`.

2. `NEKTAR_DEVICE_ARCH` triggers ahead-of-time compilation, so the resulting
binary only runs on the architecture it was built for. If it is omitted, the
kernels are compiled just-in-time (`-fsycl-targets=spir64`) instead and the
build runs on any Intel GPU, at the cost of a one-off compilation delay when
the solver starts.


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
         -DNEKTAR_DEVICE_ARCH=gfxXXX \
         -DCMAKE_C_COMPILER="$DPCPP_HOME/llvm/build/bin/clang" \
         -DCMAKE_CXX_COMPILER="$DPCPP_HOME/llvm/build/bin/clang++"
```

Note:
1. For device architecture `NEKTAR_DEVICE_ARCH`, please use:

```
 - MI210: `gfxXXX=gfx90a`
 - MI250: `gfxXXX=gfx90a`
 - MI300: `gfxXXX=gfx942`
 - MI325: `gfxXXX=gfx942`
 - MI350: `gfxXXX=gfx950`
 - MI355: `gfxXXX=gfx950`
```

The `gfxXXX` value can also be queried using the following command:
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
