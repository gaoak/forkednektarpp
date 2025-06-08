#
# NektarIntelSYCL.cmake
#
# Sets up cmake variables needed for using SYCL in Nektar++
#

ADD_DEFINITIONS(-DNEKTAR_ENABLE_SYCL)

IF (USE_SYCL_BUILTIN_REDUCER)
    ADD_DEFINITIONS(-DUSE_SYCL_BUILTIN_REDUCER)
ENDIF()

IF (NEKTAR_ENABLE_SYCL STREQUAL "Default")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=x86_64 -Wno-enum-constexpr-conversion")
    ADD_DEFINITIONS(-DSYCL_ENABLE_CPU)
    SET(BOOST_MIN_VERSION "1.76.0")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
    ADD_DEFINITIONS(-DSYCL_ENABLE_CUDA)
    IF (NEKTAR_DEVICE_ARCH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvidia_gpu_${NEKTAR_DEVICE_ARCH} -Wno-enum-constexpr-conversion ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-enum-constexpr-conversion ")
    ENDIF()
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(BOOST_MIN_VERSION "1.82.0")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "HIP")
    ADD_DEFINITIONS(-DSYCL_ENABLE_HIP)
    IF (NEKTAR_DEVICE_ARCH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amd_gpu_${NEKTAR_DEVICE_ARCH} -Wno-enum-constexpr-conversion ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Wno-enum-constexpr-conversion ")
    ENDIF()
    SET(BOOST_MIN_VERSION "1.82.0")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "Intel")
    ADD_DEFINITIONS(-DSYCL_ENABLE_INTEL)
    IF (NEKTAR_DEVICE_ARCH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_${NEKTAR_DEVICE_ARCH} -Wno-enum-constexpr-conversion ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen -Wno-enum-constexpr-conversion ")
    ENDIF()
ENDIF()

FIND_PACKAGE(IntelSYCL REQUIRED)

IF (NEKTAR_ENABLE_SYCL STREQUAL "Default" OR NEKTAR_ENABLE_SYCL STREQUAL "Intel")
    FIND_PACKAGE(MKL CONFIG REQUIRED)
    SET(NEKTAR_SYCL_DEPENDS MKL::MKL_SYCL)
ELSE()
    # Check if oneMath library is already built
    SET(oneMath_DIR ${TPBUILD}/oneMath)
    FIND_PACKAGE(oneMath)
    UNSET(oneMath_DIR CACHE)

    IF (NOT oneMath_FOUND)
        EXECUTE_PROCESS(COMMAND rm -rf ${TPSRC}/oneMath)
        EXECUTE_PROCESS(COMMAND rm -rf ${TPBUILD}/oneMath)
        EXECUTE_PROCESS(COMMAND git clone -b develop https://github.com/uxlfoundation/oneMath.git ${TPSRC}/oneMath)
        IF (NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
            EXECUTE_PROCESS(COMMAND 
                        cmake ${TPSRC}/oneMath
                        -B ${TPBUILD}/oneMath
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DENABLE_GENERIC_BLAS_BACKEND=OFF
                        -DENABLE_ARMPL_BACKEND=OFF
                        -DENABLE_NETLIB_BACKEND=OFF
                        -DENABLE_MKLCPU_BACKEND=OFF
                        -DENABLE_MKLGPU_BACKEND=OFF
                        -DENABLE_CUBLAS_BACKEND=ON
                        -DENABLE_CUFFT_BACKEND=ON
                        -DENABLE_CURAND_BACKEND=ON
                        -DENABLE_CUSOLVER_BACKEND=ON
                        -DENABLE_CUSPARSE_BACKEND=ON
                        -DENABLE_ROCBLAS_BACKEND=OFF
                        -DENABLE_ROCFFT_BACKEND=OFF
                        -DENABLE_ROCRAND_BACKEND=OFF
                        -DENABLE_ROCSOLVER_BACKEND=OFF
                        -DENABLE_ROCSPARSE_BACKEND=OFF
                        -DBUILD_FUNCTIONAL_TESTS=OFF
                        -DBUILD_EXAMPLES=OFF
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST})
            EXECUTE_PROCESS(COMMAND make install -j8 -C ${TPBUILD}/oneMath)
        ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "HIP")
            EXECUTE_PROCESS(COMMAND 
                        cmake ${TPSRC}/oneMath
                        -B ${TPBUILD}/oneMath
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DHIP_TARGETS=${NEKTAR_DEVICE_ARCH}
                        -DENABLE_GENERIC_BLAS_BACKEND=OFF
                        -DENABLE_ARMPL_BACKEND=OFF
                        -DENABLE_NETLIB_BACKEND=OFF
                        -DENABLE_MKLCPU_BACKEND=OFF
                        -DENABLE_MKLGPU_BACKEND=OFF
                        -DENABLE_CUBLAS_BACKEND=OFF
                        -DENABLE_CUFFT_BACKEND=OFF 
                        -DENABLE_CURAND_BACKEND=OFF
                        -DENABLE_CUSOLVER_BACKEND=OFF
                        -DENABLE_CUSPARSE_BACKEND=OFF
                        -DENABLE_ROCBLAS_BACKEND=ON
                        -DENABLE_ROCFFT_BACKEND=ON
                        -DENABLE_ROCRAND_BACKEND=ON
                        -DENABLE_ROCSOLVER_BACKEND=ON
                        -DENABLE_ROCSPARSE_BACKEND=ON
                        -DBUILD_FUNCTIONAL_TESTS=OFF
                        -DBUILD_EXAMPLES=OFF
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST})
            EXECUTE_PROCESS(COMMAND make install -j8 -C ${TPBUILD}/oneMath)
        ENDIF()
    ENDIF()

    SET(oneMath_DIR ${TPBUILD}/oneMath)
    # --------- This hack is necessary until oneMath fix this problem --------
    EXECUTE_PROCESS(COMMAND cp ${TPSRC}/oneMath/cmake/FindCompiler.cmake ${TPBUILD}/oneMath/FindCompiler.cmake)
    # ------------------------------END---------------------------------------
    FIND_PACKAGE(oneMath REQUIRED)
    UNSET(oneMath_DIR CACHE)
    SET(NEKTAR_SYCL_DEPENDS ONEMATH::onemath)
ENDIF()
