#
# NektarOneMath.cmake
#
# Sets up cmake variables needed for using OneMath in Nektar++
#

IF (${AdaptiveCpp_FOUND})
    SET(ONEMATH_SYCL_IMPLEMENTATION "adaptivecpp")
ELSE()
    SET(ONEMATH_SYCL_IMPLEMENTATION "dpc++")
ENDIF()

IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CPU")
    IF (ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++")
        FIND_PACKAGE(MKL CONFIG REQUIRED)
        SET(NEKTAR_SYCL_DEPENDS MKL::MKL_SYCL)
    ENDIF()
ELSE()
    # Check if oneMath library is already built
    SET(oneMath_DIR ${TPBUILD}/oneMath)
    FIND_PACKAGE(oneMath)
    UNSET(oneMath_DIR CACHE)

    IF (NOT oneMath_FOUND)
        EXECUTE_PROCESS(COMMAND rm -rf ${TPSRC}/oneMath)
        EXECUTE_PROCESS(COMMAND rm -rf ${TPBUILD}/oneMath)
        EXECUTE_PROCESS(COMMAND git clone -b develop https://github.com/uxlfoundation/oneMath.git ${TPSRC}/oneMath)
        IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA")
            EXECUTE_PROCESS(COMMAND 
                        cmake ${TPSRC}/oneMath
                        -B ${TPBUILD}/oneMath
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DAdaptiveCpp_DIR=${AdaptiveCpp_DIR}
                        -DONEMATH_SYCL_IMPLEMENTATION=${ONEMATH_SYCL_IMPLEMENTATION}
                        -DENABLE_GENERIC_BLAS_BACKEND=OFF
                        -DENABLE_ARMPL_BACKEND=OFF
                        -DENABLE_NETLIB_BACKEND=OFF
                        -DENABLE_MKLCPU_BACKEND=OFF
                        -DENABLE_MKLGPU_BACKEND=OFF
                        -DENABLE_CUBLAS_BACKEND=ON
                        -DENABLE_CUFFT_BACKEND=ON
                        -DENABLE_CURAND_BACKEND=ON
                        -DENABLE_CUSOLVER_BACKEND=OFF
                        -DENABLE_CUSPARSE_BACKEND=OFF
                        -DENABLE_ROCBLAS_BACKEND=OFF
                        -DENABLE_ROCFFT_BACKEND=OFF
                        -DENABLE_ROCRAND_BACKEND=OFF
                        -DENABLE_ROCSOLVER_BACKEND=OFF
                        -DENABLE_ROCSPARSE_BACKEND=OFF
                        -DBUILD_FUNCTIONAL_TESTS=OFF
                        -DBUILD_EXAMPLES=OFF
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST})
            EXECUTE_PROCESS(COMMAND make install -j8 -C ${TPBUILD}/oneMath)
        ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
            EXECUTE_PROCESS(COMMAND 
                        cmake ${TPSRC}/oneMath
                        -B ${TPBUILD}/oneMath
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DHIP_TARGETS=${NEKTAR_DEVICE_ARCH}
                        -DCMAKE_PREFIX_PATH=${ROCM_PATH}
                        -DONEMATH_SYCL_IMPLEMENTATION=${ONEMATH_SYCL_IMPLEMENTATION}
                        -DAdaptiveCpp_DIR=${AdaptiveCpp_DIR}
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
                        -DENABLE_ROCRAND_BACKEND=OFF
                        -DENABLE_ROCSOLVER_BACKEND=OFF
                        -DENABLE_ROCSPARSE_BACKEND=OFF
                        -DBUILD_FUNCTIONAL_TESTS=OFF
                        -DBUILD_EXAMPLES=OFF
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST})
            EXECUTE_PROCESS(COMMAND make install -j8 -C ${TPBUILD}/oneMath)
        ENDIF()
    ENDIF()

    SET(oneMath_DIR ${TPBUILD}/oneMath)
    # --------- This hack is necessary until oneMath fix this problem --------
    EXECUTE_PROCESS(COMMAND cp ${TPSRC}/oneMath/cmake/FindCompiler.cmake ${TPBUILD}/oneMath/FindCompiler.cmake)
    IF (NOT ${AdaptiveCpp_FOUND} AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        EXECUTE_PROCESS(COMMAND sed -i "/check_cxx_compiler_flag(\"-fsycl\" is_dpcpp)/a set(is_dpcpp ON)" ${TPBUILD}/oneMath/FindCompiler.cmake)
    ENDIF()
    # ------------------------------END---------------------------------------
    FIND_PACKAGE(oneMath REQUIRED)
    UNSET(oneMath_DIR CACHE)
    SET(NEKTAR_SYCL_DEPENDS ONEMATH::onemath)
ENDIF()
