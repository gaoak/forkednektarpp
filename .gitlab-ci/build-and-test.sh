#!/bin/bash -x

[[ $OS_VERSION != "macos" ]] && ccache -s && ccache -M 5G

if [[ $BUILD_TYPE == "default" ]]; then
    BUILD_OPTS="-DCMAKE_BUILD_TYPE=Release \
        -DNEKTAR_BUILD_REDESIGN:BOOL=ON \
        -DNEKTAR_TEST_ALL=ON \
        -DNEKTAR_ERROR_ON_WARNINGS=OFF"
elif [[ $BUILD_TYPE == "full" ]]; then
    BUILD_OPTS="-DCMAKE_BUILD_TYPE:STRING=Debug \
        -DNEKTAR_FULL_DEBUG:BOOL=ON \
        -DNEKTAR_TEST_ALL:BOOL=ON \
        -DNEKTAR_USE_ARPACK:BOOL=ON \
        -DNEKTAR_USE_FFTW:BOOL=ON \
        -DNEKTAR_USE_MPI:BOOL=ON \
        -DNEKTAR_USE_SCOTCH:BOOL=ON \
        -DNEKTAR_USE_PETSC:BOOL=ON \
        -DNEKTAR_USE_HDF5:BOOL=ON \
        -DNEKTAR_USE_METIS:BOOL=ON \
        -DNEKTAR_USE_MESHGEN:BOOL=ON \
        -DNEKTAR_USE_CCM:BOOL=ON \
        -DNEKTAR_CCMIO_URL=https://www.nektar.info/ccmio/libccmio-2.6.1.tar.gz \
        -DNEKTAR_USE_CWIPI:BOOL=ON \
        -DNEKTAR_USE_VTK:BOOL=ON \
        -DNEKTAR_BUILD_REDESIGN:BOOL=ON \
        -DNEKTAR_BUILD_PYTHON:BOOL=ON \
        -DNEKTAR_TEST_USE_HOSTFILE=ON \
        -DNEKTAR_UTILITY_EXTRAS=ON \
        -DNEKTAR_ERROR_ON_WARNINGS=OFF"
    if [[ $BUILD_SIMD == "avx2" ]]; then
        BUILD_OPTS="$BUILD_OPTS -DNEKTAR_ENABLE_SIMD_AVX2:BOOL=ON"
    elif [[ $BUILD_SIMD == "avx512" ]]; then
        BUILD_OPTS="$BUILD_OPTS -DNEKTAR_ENABLE_SIMD_AVX512:BOOL=ON"
    fi
    if [[ $BUILD_KOKKOS == "on" ]]; then
        BUILD_OPTS="$BUILD_OPTS -DNEKTAR_ENABLE_KOKKOS:STRING=Serial" 
    fi
elif [[ $BUILD_TYPE == "performance" ]]; then
    BUILD_OPTS="-DCMAKE_BUILD_TYPE=Release \
        -DNEKTAR_BUILD_TESTS=OFF \
        -DNEKTAR_BUILD_UNIT_TESTS=OFF \
        -DNEKTAR_BUILD_PERFORMANCE_TESTS=ON \
        -DNEKTAR_ERROR_ON_WARNINGS=OFF"
fi

if [[ $BUILD_TYPE != "performance" ]]; then
    TEST_JOBS="$NUM_CPUS"
else 
    TEST_JOBS="1"
fi

if [[ $EXPORT_COMPILE_COMMANDS != "" ]]; then
    BUILD_OPTS="$BUILD_OPTS -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
fi

# Custom compiler
if [[ $BUILD_CC != "" ]]; then
   BUILD_OPTS="$BUILD_OPTS -DCMAKE_C_COMPILER=${BUILD_CC}"
fi
if [[ $BUILD_CXX != "" ]]; then
   BUILD_OPTS="$BUILD_OPTS -DCMAKE_CXX_COMPILER=${BUILD_CXX}"
fi
if [[ $BUILD_FC != "" ]]; then
   BUILD_OPTS="$BUILD_OPTS -DCMAKE_Fortran_COMPILER=${BUILD_FC}"
fi

rm -rf build && mkdir -p build && (cd build && cmake -G 'Unix Makefiles' $BUILD_OPTS ..)

if [[ $EXPORT_COMPILE_COMMANDS != "" ]]; then
    # If we are just exporting compile commands for clang-tidy, just build any
    # third-party dependencies that we need.
    make -C build -j $NUM_CPUS thirdparty 2>&1
else
    # Otherwise build and test the code.
    make -C build -j $NUM_CPUS all 2>&1 && make -C build -j $NUM_CPUS install && \
        (cd build && ctest -j $TEST_JOBS --output-on-failure)
fi

exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    [[ $OS_VERSION != "macos" ]] && rm -rf build/dist
    exit $exit_code
fi
