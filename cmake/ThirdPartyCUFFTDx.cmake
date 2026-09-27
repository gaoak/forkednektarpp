########################################################################
#
# ThirdParty configuration for Nektar++
#
# cuFFTDx
#
# cuFFTDx is the device-side FFT library from NVIDIA's MathDx package, used
# by the homogeneous-direction device FFT in LibUtilities. It is header-only
# and distributed as a binary package rather than as sources, so the
# ThirdParty build below only unpacks the headers: there is nothing to
# configure, compile or install. CUFFTDX_INCLUDE_DIR and
# CUFFTDX_CUTLASS_INCLUDE_DIR are therefore left pointing inside the
# unpacked tree rather than at ${TPDIST}, and are added to the one target
# that needs them rather than globally - only DerivZDeviceFFT.cu includes
# these headers, and they are not installed with Nektar++.
#
# This module must be included AFTER NektarCUDA: the target architecture
# below is derived from CMAKE_CUDA_ARCHITECTURES.
#
########################################################################

IF (NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
    OPTION(NEKTAR_USE_CUFFTDX "Use cuFFTDx" OFF)
    MARK_AS_ADVANCED(NEKTAR_USE_CUFFTDX)
ENDIF()

IF (NEKTAR_USE_CUFFTDX)
    IF (NOT NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        MESSAGE(FATAL_ERROR "cuFFTDx is only available for the CUDA device backend")
    ENDIF()

    # First search for system or user-supplied MathDx installs.
    FIND_PATH(CUFFTDX_INCLUDE_DIR cufftdx/cufftdx.hpp
              HINTS "$ENV{CUFFTDX_HOME}/include"
                    /opt/nvidia/mathdx/24.01/include
                    /opt/nvidia/mathdx/24.04/include
                    /opt/nvidia/mathdx/25.12/include)

    # If we have the headers already then don't unpack our own copy.
    IF (CUFFTDX_INCLUDE_DIR)
        SET(BUILD_CUFFTDX OFF)
    ELSE()
        SET(BUILD_CUFFTDX ON)
    ENDIF()

    OPTION(THIRDPARTY_BUILD_CUFFTDX
        "Build cuFFTDx headers from ThirdParty." ${BUILD_CUFFTDX})

    IF (THIRDPARTY_BUILD_CUFFTDX)
        INCLUDE(ExternalProject)

        # The package unpacks to nvidia/mathdx/<version> below SOURCE_DIR,
        # with the CUTLASS copy cuFFTDx needs as a sibling of its include
        # directory. Note that the archive is the CUDA 13 redistributable.
        SET(CUFFTDX_MATHDX_DIR ${TPSRC}/cufftdx-25.12/nvidia/mathdx/25.12)

        EXTERNALPROJECT_ADD(
            cufftdx-25.12
            PREFIX ${TPSRC}
            URL ${TPURL}/nvidia-mathdx-25.12.1-cuda13.tar.gz
            URL_MD5 e207c5fbffbf22156de629964044ffdc
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            SOURCE_DIR ${TPSRC}/cufftdx-25.12
            BINARY_DIR ${TPBUILD}/cufftdx-25.12
            TMP_DIR ${TPBUILD}/cufftdx-25.12-tmp
            INSTALL_DIR ${TPDIST}
            BUILD_BYPRODUCTS ${CUFFTDX_MATHDX_DIR}/include/cufftdx/cufftdx.hpp
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
        )

        SET(CUFFTDX_INCLUDE_DIR ${CUFFTDX_MATHDX_DIR}/include CACHE FILEPATH
            "cuFFTDx include" FORCE)
        SET(CUFFTDX_CUTLASS_INCLUDE_DIR
            ${CUFFTDX_MATHDX_DIR}/external/cutlass/include CACHE FILEPATH
            "CUTLASS include shipped with cuFFTDx" FORCE)
        MESSAGE(STATUS "Build cuFFTDx: ${CUFFTDX_INCLUDE_DIR}")
    ELSE()
        ADD_CUSTOM_TARGET(cufftdx-25.12 ALL)

        # An unpacked MathDx install keeps CUTLASS beside the includes, as
        # the ThirdParty copy above does.
        GET_FILENAME_COMPONENT(CUFFTDX_MATHDX_DIR ${CUFFTDX_INCLUDE_DIR} DIRECTORY)
        SET(CUFFTDX_CUTLASS_INCLUDE_DIR
            ${CUFFTDX_MATHDX_DIR}/external/cutlass/include CACHE FILEPATH
            "CUTLASS include shipped with cuFFTDx" FORCE)
        MESSAGE(STATUS "Found cuFFTDx: ${CUFFTDX_INCLUDE_DIR}")
    ENDIF (THIRDPARTY_BUILD_CUFFTDX)

    ADD_DEPENDENCIES(thirdparty cufftdx-25.12)

    # cuFFTDx compiles for one architecture, given as a three-digit SM
    # number. Take it from the CUDA architectures Nektar is being built for,
    # falling back on the architecture of the GPU in this machine when those
    # are 'native'. Override by passing -DCUFFTDX_TARGET_SM=<value> on the
    # cmake command line.
    IF (NOT CUFFTDX_TARGET_SM)
        LIST(GET CMAKE_CUDA_ARCHITECTURES 0 CUFFTDX_ARCH_RAW)
        IF (CUFFTDX_ARCH_RAW MATCHES "^[0-9]+$")
            SET(CUFFTDX_ARCH_2DIGIT "${CUFFTDX_ARCH_RAW}")
        ELSE()
            EXECUTE_PROCESS(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE CUFFTDX_GPU_COMPUTE_CAP
                RESULT_VARIABLE CUFFTDX_NVSMI_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            STRING(REGEX REPLACE "\n.*" "" CUFFTDX_GPU_COMPUTE_CAP
                "${CUFFTDX_GPU_COMPUTE_CAP}")
            IF (CUFFTDX_NVSMI_RESULT EQUAL 0 AND
                CUFFTDX_GPU_COMPUTE_CAP MATCHES "^[0-9]+\\.[0-9]+$")
                STRING(REPLACE "." "" CUFFTDX_ARCH_2DIGIT
                    "${CUFFTDX_GPU_COMPUTE_CAP}")
            ELSE()
                SET(CUFFTDX_ARCH_2DIGIT "86")
            ENDIF()
        ENDIF()
        MATH(EXPR CUFFTDX_TARGET_SM "${CUFFTDX_ARCH_2DIGIT} * 10")
    ENDIF()

    MARK_AS_ADVANCED(CUFFTDX_INCLUDE_DIR)
    MARK_AS_ADVANCED(CUFFTDX_CUTLASS_INCLUDE_DIR)
ENDIF()
