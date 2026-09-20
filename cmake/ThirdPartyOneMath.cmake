########################################################################
#
# ThirdParty configuration for Nektar++
#
# OneMath library
#
########################################################################

# Intel targets, CPU and GPU alike, are served by MKL's own SYCL interfaces
# rather than by oneMath: MKL covers both device kinds through the same
# oneapi/mkl.hpp headers. That interface is DPC++ only, so under AdaptiveCpp
# these configurations are left without a backend and the sources take their
# fallback path.
IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CPU" OR
    NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-Intel")
    IF (${AdaptiveCpp_FOUND})
       # Do nothing.
    ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        FIND_PACKAGE(MKL CONFIG REQUIRED)
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_ONEMKL)
        SET(ONEMATH_LIBRARY MKL::MKL_SYCL)
    ENDIF()
ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
    SET(NEKTAR_USE_ONEMATH ON)
    IF (${AdaptiveCpp_FOUND})
        SET(ONEMATH_SYCL_IMPLEMENTATION "adaptivecpp")
    ELSE()
        SET(ONEMATH_SYCL_IMPLEMENTATION "dpc++")
    ENDIF()

    FIND_PATH(ONEMATH_INCLUDE_DIR oneapi/math.hpp)
    FIND_LIBRARY(ONEMATH_LIBRARY NAMES "onemath")

    # If we have the library already then don't build it.
    IF (ONEMATH_LIBRARY AND ONEMATH_INCLUDE_DIR)
        SET(BUILD_ONEMATH OFF)
    ELSE()
        SET(BUILD_ONEMATH ON)
    ENDIF ()

    OPTION(THIRDPARTY_BUILD_ONEMATH
        "Build OneMath library from ThirdParty" ${BUILD_ONEMATH})

    # Selects the oneMath API over oneMKL in the sources, so it has to hold
    # whether the library is built below or was found on the system.
    ADD_DEFINITIONS(-DNEKTAR_ENABLE_ONEMATH)

    IF(THIRDPARTY_BUILD_ONEMATH)
        INCLUDE(ExternalProject)

        THIRDPARTY_LIBRARY(ONEMATH_LIBRARY SHARED onemath DESCRIPTION "OneMath library")
        IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA")
            FIND_PACKAGE(CUDAToolkit REQUIRED)
            IF("${OPENCL_INCLUDE_DIR}" STREQUAL "")
                SET(OPENCL_INCLUDE_DIR "/usr/include")
            ENDIF()
            EXTERNALPROJECT_ADD(
                    onemath-v0.9
                    PREFIX ${TPSRC}
                    GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
                    GIT_TAG v0.9
                    STAMP_DIR ${TPBUILD}/stamp
                    DOWNLOAD_DIR ${TPSRC}
                    SOURCE_DIR ${TPSRC}/onemath-v0.9
                    BINARY_DIR ${TPBUILD}/onemath-v0.9
                    TMP_DIR ${TPBUILD}/onemath-v0.9-tmp
                    INSTALL_DIR ${TPDIST}
                    BUILD_BYPRODUCTS ${ONEMATH_LIBRARY}
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR>
                        ${NEKTAR_EXTERNAL_PROJECT_CMAKE_GENERATOR_ARGS}
                        -B <BINARY_DIR>
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCUDA_CUDA_LIBRARY=${CUDAToolkit_LIBRARY_DIR}/stubs/libcuda.so
                        -DOPENCL_INCLUDE_DIR=${OPENCL_INCLUDE_DIR}
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
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                        -Wno-dev
                    UPDATE_COMMAND ""
                    )
        ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
            EXTERNALPROJECT_ADD(
                    onemath-v0.9
                    PREFIX ${TPSRC}
                    GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
                    GIT_TAG v0.9
                    STAMP_DIR ${TPBUILD}/stamp
                    DOWNLOAD_DIR ${TPSRC}
                    SOURCE_DIR ${TPSRC}/onemath-v0.9
                    BINARY_DIR ${TPBUILD}/onemath-v0.9
                    TMP_DIR ${TPBUILD}/onemath-v0.9-tmp
                    INSTALL_DIR ${TPDIST}
                    BUILD_BYPRODUCTS ${ONEMATH_LIBRARY}
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR>
                        ${NEKTAR_EXTERNAL_PROJECT_CMAKE_GENERATOR_ARGS}
                        -B <BINARY_DIR>
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
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                        -Wno-dev
                    UPDATE_COMMAND ""
                    )
        ENDIF()

        SET(ONEMATH_INCLUDE_DIR ${TPDIST}/include CACHE FILEPATH "onemath include" FORCE)
        MESSAGE(STATUS "Build onemath: ${ONEMATH_LIBRARY}")
        SET(ONEMATH_CONFIG_INCLUDE_DIR ${TPINC})
    ELSE()
        ADD_CUSTOM_TARGET(onemath-v0.9 ALL)
        MESSAGE(STATUS "Found onemath: ${ONEMATH_LIBRARY}")
        SET(ONEMATH_CONFIG_INCLUDE_DIR ${ONEMATH_INCLUDE_DIR})
    ENDIF()

    # Both branches leave ONEMATH_INCLUDE_DIR as the directory holding
    # oneapi/math.hpp, which is how the sources include it.
    INCLUDE_DIRECTORIES(${ONEMATH_INCLUDE_DIR})

    ADD_DEPENDENCIES(thirdparty onemath-v0.9)

    MARK_AS_ADVANCED(ONEMATH_LIBRARY)
    MARK_AS_ADVANCED(ONEMATH_INCLUDE_DIR)
    MARK_AS_ADVANCED(ONEMATH_CONFIG_INCLUDE_DIR)
ENDIF()
