#!/bin/bash -x

declare -a CMAKEARGS

if [[ $OS_VERSION != "macos" ]]; then
    # Make environment modules cmd available
    . /etc/profile.d/modules.sh

    # Display ccache usage and set limit
    ccache -s && ccache -M 5G
fi

echo "Running build with:"
echo "  - BUILD_CC                : $BUILD_CC"
echo "  - BUILD_CXX               : $BUILD_CXX"
echo "  - BUILD_FC                : $BUILD_FC"
echo "  - BUILD_TYPE              : $BUILD_TYPE"
echo "  - BUILD_SIMD              : $BUILD_SIMD"
echo "  - DISABLE_MCA             : $DISABLE_MCA"
echo "  - EXPORT_COMPILE_COMMANDS : $EXPORT_COMPILE_COMMANDS"
echo "  - NUM_CPUS                : $NUM_CPUS"
echo "  - OS_VERSION              : $OS_VERSION"
echo "  - PYTHON_EXECUTABLE       : $PYTHON_EXECUTABLE"

if [[ $BUILD_TYPE == "default" ]]; then
    CMAKEARGS=(..
               "-DCMAKE_BUILD_TYPE=Release"
               "-DNEKTAR_BUILD_REDESIGN:BOOL=ON"
               "-DNEKTAR_TEST_ALL=ON"
               "-DNEKTAR_ERROR_ON_WARNINGS=OFF"
              )
elif [[ $BUILD_TYPE == "full" ]]; then
    CMAKEARGS=(..
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
               "-DNEKTAR_CCMIO_URL=https://www.nektar.info/ccmio/libccmio-2.6.1.tar.gz"
               "-DNEKTAR_USE_CWIPI:BOOL=ON"
               "-DNEKTAR_USE_VTK:BOOL=ON"
               "-DNEKTAR_BUILD_REDESIGN:BOOL=ON"
               "-DNEKTAR_BUILD_PYTHON:BOOL=ON"
               "-DNEKTAR_TEST_USE_HOSTFILE=ON"
               "-DNEKTAR_UTILITY_EXTRAS=ON"
               "-DNEKTAR_ERROR_ON_WARNINGS=OFF"
              )

    if [[ $BUILD_SIMD == "avx2" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_SIMD_AVX2:BOOL=ON")
    elif [[ $BUILD_SIMD == "avx512" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_SIMD_AVX512:BOOL=ON")
    fi
    if [[ $BUILD_CUDA == "on" ]]; then
        # Load CUDA on Linux
        [[ $OS_VERSION != "macos" ]] && module load cuda/12.2.2

        # Enable CUDA in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_CUDA:BOOL=ON")
        CMAKEARGS+=("-DCMAKE_CUDA_ARCHITECTURES=86")
    fi
    if [[ $BUILD_KOKKOS == "Serial" ]]; then
        CMAKEARGS+=("-DNEKTAR_ENABLE_KOKKOS:STRING=Serial")
    elif [[ $BUILD_KOKKOS == "CUDA" ]]; then
        # Load Boost and CUDA on Linux
        [[ $OS_VERSION != "macos" ]] && module load boost cuda/12.2.2

        # Enable CUDA in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_KOKKOS:STRING=CUDA")
    fi
    if [[ $BUILD_SYCL == "Default" ]]; then
        # Load Intel compiler module for SYCL support on Linux
        [[ $OS_VERSION != "macos" ]] && module load intel/compiler intel/mkl

        # Enable SYCL in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_SYCL:STRING=Default")
    elif [[ $BUILD_SYCL == "CUDA" ]]; then
        # Load CUDA and Intel compiler module for SYCL support on Linux
        [[ $OS_VERSION != "macos" ]] && module load cuda/12.2.2 intel/compiler intel/mkl

        # Enable SYCL in CMake configuration
        CMAKEARGS+=("-DNEKTAR_ENABLE_SYCL:STRING=CUDA")
    fi
elif [[ $BUILD_TYPE == "performance" ]]; then
    CMAKEARGS=(..
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
    CMAKEARGS+=("-DPython_EXECUTABLE=${PYTHON_EXECUTABLE}")
fi

rm -rf build && mkdir -p build && (cd build && cmake -G 'Unix Makefiles' "${CMAKEARGS[@]}" ..)

if [[ $DISABLE_MCA != "" ]]; then
    export OMPI_MCA_btl_base_warn_component_unused=0
fi

if [[ $EXPORT_COMPILE_COMMANDS != "" ]]; then
    # If we are just exporting compile commands for clang-tidy, just build any
    # third-party dependencies that we need.
    make -C build -j $NUM_CPUS thirdparty 2>&1
    exit_code=$?
else
    # Otherwise build and test the code.
    if [[ $BUILD_KOKKOS == "CUDA" ]]; then
        # Do not run test with Kokkos-CUDA
        make -C build -j $NUM_CPUS all 2>&1 && make -C build -j $NUM_CPUS install
    else
        make -C build -j $NUM_CPUS all 2>&1 && make -C build -j $NUM_CPUS install && \
            (cd build && ctest -j $TEST_JOBS --output-on-failure)
    fi
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
