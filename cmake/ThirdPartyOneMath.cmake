########################################################################
#
# ThirdParty configuration for Nektar++
#
# OneMath library
#
########################################################################

IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CPU")
    IF (ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++")
        FIND_PACKAGE(MKL CONFIG REQUIRED)
        SET(NEKTAR_SYCL_DEPENDS MKL::MKL_SYCL)
    ENDIF()
ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
    SET(NEKTAR_USE_ONEMATH ON)
    IF (${AdaptiveCpp_FOUND})
        SET(ONEMATH_SYCL_IMPLEMENTATION "adaptivecpp")
    ELSE()
        SET(ONEMATH_SYCL_IMPLEMENTATION "dpc++")
    ENDIF()

    FIND_PATH(ONEMATH_INCLUDE_DIR math.hpp)
    FIND_LIBRARY(ONEMATH_LIBRARY NAMES "onemath")

    # If we have our library then don't build libmagma.
    IF (ONEMATH_LIBRARY AND ONEMATH_INCLUDE_DIR)
        SET(BUILD_ONEMATH OFF)
    ELSE()
        SET(BUILD_ONEMATH ON)
    ENDIF ()

    OPTION(THIRDPARTY_BUILD_ONEMATH
        "Build OneMath library from ThirdParty" ${BUILD_ONEMATH})

    IF(THIRDPARTY_BUILD_ONEMATH)
        INCLUDE(ExternalProject)

        THIRDPARTY_LIBRARY(ONEMATH_LIBRARY SHARED onemath DESCRIPTION "OneMath library")
        IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA")
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
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR> -G ${CMAKE_GENERATOR}
                        -B <BINARY_DIR>
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
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR> -G ${CMAKE_GENERATOR}
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

        INCLUDE_DIRECTORIES(${ONEMATH_INCLUDE_DIR})
    ELSE()
        ADD_CUSTOM_TARGET(onemath-v0.9 ALL)
        MESSAGE(STATUS "Found onemath: ${ONEMATH_LIBRARY}")
        SET(ONEMATH_CONFIG_INCLUDE_DIR ${ONEMATH_INCLUDE_DIR})
    ENDIF()

    ADD_DEPENDENCIES(thirdparty onemath-v0.9)

    MARK_AS_ADVANCED(ONEMATH_LIBRARY)
    MARK_AS_ADVANCED(ONEMATH_INCLUDE_DIR)
    MARK_AS_ADVANCED(ONEMATH_CONFIG_INCLUDE_DIR)
ENDIF()
