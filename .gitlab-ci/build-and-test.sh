#!/bin/bash -x

declare -a CMAKEARGS

if [[ $OS_VERSION != "macos" ]]; then
    # Make environment modules cmd available
    . /etc/profile.d/modules.sh

    # Display ccache usage and set limit
    ccache -s && ccache -M 8G
fi

echo "Running build with:"
echo "  - BUILD_CC                : $BUILD_CC"
echo "  - BUILD_CXX               : $BUILD_CXX"
echo "  - BUILD_FC                : $BUILD_FC"
echo "  - BUILD_TYPE              : $BUILD_TYPE"
echo "  - BUILD_SIMD              : $BUILD_SIMD"
echo "  - BUILD_SINGLE_PRECISION  : $BUILD_SINGLE_PRECISION"
echo "  - DISABLE_CWIPI           : $DISABLE_CWIPI"
echo "  - DISABLE_MCA             : $DISABLE_MCA"
echo "  - ENABLE_ALIGN_MEM        : $ENABLE_ALIGN_MEM"
echo "  - EXPORT_COMPILE_COMMANDS : $EXPORT_COMPILE_COMMANDS"
echo "  - NUM_CPUS                : $NUM_CPUS"
echo "  - OS_VERSION              : $OS_VERSION"
echo "  - PYTHON_EXECUTABLE       : $PYTHON_EXECUTABLE"
echo "  - USE_NINJA               : $USE_NINJA"

# Use ninja for builds: defaults to true
if [[ $USE_NINJA == "true" ]]; then
    CMAKEARGS=(.. "-G" "Ninja")
    MAKE_EXEC=ninja
else
    CMAKEARGS=(.. "-G" "Unix Makefiles")
    MAKE_EXEC=make
fi

if [[ $BUILD_TYPE == "default" ]]; then
    CMAKEARGS+=(
               "-DCMAKE_BUILD_TYPE=Release"
               "-DNEKTAR_BUILD_REDESIGN:BOOL=ON"
               "-DNEKTAR_TEST_ALL=ON"
               "-DNEKTAR_ERROR_ON_WARNINGS=OFF"
               )
elif [[ $BUILD_TYPE == "full" ]]; then
    CMAKEARGS+=(
               "-DCMAKE_BUILD_TYPE:STRING=Debug"
               "-DNEKTAR_FULL_DEBUG:BOOL=ON"
               "-DNEKTAR_TEST_ALL:BOOL=ON"
               "-DNEKTAR_USE_ARPACK:BOOL=ON"
               "-DNEKTAR_USE_FFTW:BOOL=ON"
               "-DNEKTAR_USE_MPI:BOOL=ON"
               "-DNEKTAR_USE_SCOTCH:BOOL=ON"
               "-DNEKTAR_USE_PETSC:BOOL=ON"
               "-DNEKTAR_USE_HDF5:BOOL=ON"
               "-DNEKTAR_USE_METIS:BOOL=ON"
               "-DNEKTAR_USE_MESHGEN:BOOL=ON"
               "-DNEKTAR_USE_CCM:BOOL=ON"
               "-DNEKTAR_USE_CGNS:BOOL=ON"
               "-DNEKTAR_CCMIO_URL=https://www.nektar.info/ccmio/libccmio-2.6.1.tar.gz"
               "-DNEKTAR_USE_VTK:BOOL=ON"
               "-DNEKTAR_BUILD_REDESIGN:BOOL=ON"
	       "-DNEKTAR_USE_LST:BOOL=ON"
               "-DNEKTAR_BUILD_PYTHON:BOOL=ON"
               "-DNEKTAR_TEST_USE_HOSTFILE=ON"
               "-DNEKTAR_UTILITY_EXTRAS=ON"
               "-DNEKTAR_ERROR_ON_WARNINGS=OFF"
               )

    if [[ $DISABLE_CWIPI != "true" ]]; then
        CMAKEARGS+=("-DNEKTAR_USE_CWIPI:BOOL=ON")
    fi
    if [[ $BUILD_SIMD == "avx2" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_SIMD:STRING=AVX2")
    elif [[ $BUILD_SIMD == "avx512" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_SIMD:STRING=AVX512")
    fi
    if [[ $BUILD_DEVICE == "DEVICEONHOST" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_DEVICE:STRING=DEVICEONHOST")
    elif [[ $BUILD_DEVICE == "CUDA" ]]; then
        # Load CUDA on Linux
        [[ $OS_VERSION != "macos" ]] && module load cuda/13.0.2

        # Enable CUDA in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_DEVICE:STRING=CUDA")
        CMAKEARGS+=("-DNEKTAR_DEVICE_ARCH=sm_86")
    elif [[ $BUILD_DEVICE == "SYCL-CPU" ]]; then
        if [[ $BUILD_CXX == "acpp" ]]; then
            # Load AdaptiveCpp compiler module for SYCL support on Linux
            [[ $OS_VERSION != "macos" ]] && module load adaptive-cpp

            # Change environment variable
            export OMP_NUM_THREADS=1
        elif [[ $BUILD_CXX == "icpx" ]]; then
            # Load Intel compiler module for SYCL support on Linux
            [[ $OS_VERSION != "macos" ]] && module load intel/compiler intel/mkl

            # Change SYCL environment variable
            export CL_CONFIG_CPU_FORCE_LOCAL_MEM_SIZE=64K
        fi

        # Enable SYCL in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_DEVICE:STRING=SYCL-CPU")
    elif [[ $BUILD_DEVICE == "SYCL-CUDA" ]]; then
        if [[ $BUILD_CXX == "acpp" ]]; then
            # This hack is necessary to compile oneMath when using Apptainer
            export PATH=$PATH:/apps/cuda/12.6.2/lib64/stubs/
            # Load LLVM support on Linux
            [[ $OS_VERSION != "macos" ]] && module load adaptive-cpp
        elif [[ $BUILD_CXX == "icpx" ]]; then
            # Load CUDA and Intel compiler module for SYCL support on Linux
            [[ $OS_VERSION != "macos" ]] && module load cuda/12.6.2 llvm/intel-6.2.0
        fi

        # Enable SYCL in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_DEVICE:STRING=SYCL-CUDA")
    fi
    if [[ $BUILD_SINGLE_PRECISION == "on" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_SINGLE_PRECISION:BOOL=ON")
        CMAKEARGS+=("-DNEKTAR_ENABLE_DOUBLE_PRECISION:BOOL=OFF")
    fi
    if [[ $ENABLE_ALIGN_MEM == "true" ]]; then
        CMAKEARGS+=("-DNEKTAR_USE_MEMORY_POOLS:BOOL=OFF")
        CMAKEARGS+=("-DNEKTAR_USE_ALIGNED_MEM:BOOL=ON")
    fi
elif [[ $BUILD_TYPE == "performance" ]]; then
    CMAKEARGS+=(
               "-DCMAKE_BUILD_TYPE=Release"
               "-DNEKTAR_BUILD_TESTS=OFF"
               "-DNEKTAR_BUILD_UNIT_TESTS=OFF"
               "-DNEKTAR_BUILD_PERFORMANCE_TESTS=ON"
               "-DNEKTAR_ERROR_ON_WARNINGS=OFF"
               )
fi

if [[ $DO_COVERAGE != "" ]]; then
    CMAKEARGS+=("-DCMAKE_CXX_FLAGS=-fprofile-arcs -ftest-coverage")
    pip3 install --user fastcov lxml
fi

if [[ $BUILD_TYPE != "performance" ]]; then
    TEST_JOBS="$NUM_CPUS"
else
    TEST_JOBS="1"
fi

if [[ $EXPORT_COMPILE_COMMANDS != "" ]]; then
    CMAKEARGS+=("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
fi

# Custom compiler
if [[ $BUILD_CC != "" ]]; then
    CMAKEARGS+=("-DCMAKE_C_COMPILER=${BUILD_CC}")
fi
if [[ $BUILD_CXX != "" ]]; then
    CMAKEARGS+=("-DCMAKE_CXX_COMPILER=${BUILD_CXX}")
fi
if [[ $BUILD_FC != "" ]]; then
    CMAKEARGS+=("-DCMAKE_Fortran_COMPILER=${BUILD_FC}")
fi

# Custom Python executable
if [[ $PYTHON_EXECUTABLE != "" ]]; then
    CMAKEARGS+=("-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}")
fi

rm -rf build && mkdir -p build && (cd build && cmake "${CMAKEARGS[@]}" ..)

if [[ $DISABLE_MCA != "" ]]; then
    export OMPI_MCA_btl_base_warn_component_unused=0
fi

if [[ $EXPORT_COMPILE_COMMANDS != "" ]]; then
    # If we are just exporting compile commands for clang-tidy, just build any
    # third-party dependencies that we need.
    $MAKE_EXEC -C build -j $NUM_CPUS thirdparty 2>&1
    exit_code=$?
else
    # Otherwise build and test the code.
    $MAKE_EXEC -C build -j $NUM_CPUS all 2>&1 && $MAKE_EXEC -C build -j $NUM_CPUS install && \
        (cd build && ctest -j $TEST_JOBS --output-on-failure)
    exit_code=$?

    # Build coverage
    if [[ $DO_COVERAGE != "" && $exit_code -eq 0 ]]; then
        set -e
        $HOME/.local/bin/fastcov --exclude '/usr' --lcov -o coverage.info
        lcov --summary coverage.info
        python3 cmake/python/lcov_cobertura.py coverage.info
        mkdir coverage
        python3 cmake/python/split_cobertura.py coverage.xml coverage
        exit 0;
    fi
fi

if [[ $exit_code -ne 0 ]]; then
    [[ $OS_VERSION != "macos" ]] && rm -rf build/dist
    exit $exit_code
fi
