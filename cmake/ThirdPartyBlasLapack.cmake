########################################################################
#
# ThirdParty configuration for Nektar++
#
# BLAS/LAPACK
#
########################################################################

# Test for if BLAS is in compiler.
INCLUDE(CheckFunctionExists)

CHECK_FUNCTION_EXISTS(dgemm_ HAVE_DGEMM)

IF (HAVE_DGEMM)
    MESSAGE(STATUS "Found BLAS/LAPACK: built in")
    SET(BLAS_LAPACK_BUILTIN ON)
ELSE()
    SET(BLAS_LAPACK_BUILTIN OFF)

    # NVIDIA Performance Libraries are used by default when found. This has to
    # be decided before NEKTAR_USE_SYSTEM_BLAS_LAPACK, which defaults to ON
    # only when no other BLAS/LAPACK has been selected.
    SET(NVPL_FOUND FALSE)
    SET(NVPL_THIRDPARTY_TARGETS "")
    IF(UNIX AND NOT APPLE)
        # FindNVPL looks for an installed NVPL, never the ThirdParty copy.
        IF(NVPL_BLAS MATCHES "^${TPDIST}/")
            UNSET(NVPL_LAPACK CACHE)
            UNSET(NVPL_BLAS CACHE)
            UNSET(NVPL_BLAS_CORE CACHE)
            UNSET(NVPL_INCLUDE_DIR CACHE)
        ENDIF()
        INCLUDE(FindNVPL)
    ENDIF()
    IF(NVPL_FOUND)
        MESSAGE(STATUS "Found NVPL: ${NVPL_BLAS}")
        # Turn NVPL on in a build directory configured before it was found.
        IF(DEFINED NEKTAR_NVPL_DETECTED AND NOT NEKTAR_NVPL_DETECTED)
            SET(NEKTAR_USE_NVPL ON CACHE BOOL
                "Use the NVIDIA Performance Libraries (NVPL) for BLAS and Lapack support." FORCE)
        ENDIF()
    ENDIF()
    SET(NEKTAR_NVPL_DETECTED ${NVPL_FOUND} CACHE INTERNAL "NVPL was found")

    # NVPL can also be downloaded from NVIDIA. Either option turns the other on:
    # a ThirdParty copy is used, and is downloaded whenever NVPL is requested
    # but not installed. Disable both to stop using NVPL.
    IF(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        SET(NVPL_ARCH_SUPPORTED ON)
    ELSE()
        SET(NVPL_ARCH_SUPPORTED OFF)
    ENDIF()
    CMAKE_DEPENDENT_OPTION(THIRDPARTY_BUILD_NVPL
        "Download NVPL BLAS and LAPACK from NVIDIA (NVPL licence applies)." OFF
        "NVPL_ARCH_SUPPORTED" OFF)
    IF(THIRDPARTY_BUILD_NVPL)
        SET(NEKTAR_USE_NVPL ON CACHE BOOL
            "Use the NVIDIA Performance Libraries (NVPL) for BLAS and Lapack support." FORCE)
    ENDIF()

    # BLAS Support
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_NVPL
        "Use the NVIDIA Performance Libraries (NVPL) for BLAS and Lapack support." ${NVPL_FOUND}
        "UNIX; NOT APPLE; NOT NEKTAR_USE_OPENBLAS; NOT NEKTAR_USE_MKL; NOT NEKTAR_USE_ACML; NOT NEKTAR_USE_FLEXIBLAS" OFF)

    IF(NEKTAR_USE_NVPL AND NOT NVPL_FOUND AND NVPL_ARCH_SUPPORTED)
        MESSAGE(STATUS "NVPL not installed: enabling THIRDPARTY_BUILD_NVPL")
        SET(THIRDPARTY_BUILD_NVPL ON CACHE BOOL
            "Download NVPL BLAS and LAPACK from NVIDIA (NVPL licence applies)." FORCE)
    ENDIF()
    IF(NEKTAR_USE_NVPL AND THIRDPARTY_BUILD_NVPL)
        INCLUDE(ThirdPartyNVPL)
    ENDIF()

    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_SYSTEM_BLAS_LAPACK
        "Use the system provided blas and lapack libraries" ON
        "UNIX; NOT APPLE; NOT NEKTAR_USE_OPENBLAS; NOT NEKTAR_USE_MKL; NOT NEKTAR_USE_ACML; NOT NEKTAR_USE_ACCELERATE_FRAMEWORK; NOT NEKTAR_USE_FLEXIBLAS; NOT NEKTAR_USE_NVPL" OFF)
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_OPENBLAS
        "Use OpenBLAS library as a substitute to native BLAS." OFF
        "NOT NEKTAR_USE_SYSTEM_BLAS_LAPACK" OFF)
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_FLEXIBLAS
        "Use FlexiBLAS library as a substitute to native BLAS." OFF
        "NOT NEKTAR_USE_SYSTEM_BLAS_LAPACK" OFF)
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_ACML
        "Use the AMD Core Math Library (ACML) for BLAS and Lapack support." OFF
        "NOT NEKTAR_USE_SYSTEM_BLAS_LAPACK" OFF)
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_MKL
        "Use the Intel Math Kernel Library (MKL) for BLAS and Lapack support." OFF
        "NOT NEKTAR_USE_SYSTEM_BLAS_LAPACK" OFF)
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_ACCELERATE_FRAMEWORK
        "Use the Mac Accelerate Framework for BLAS and Lapack support." ON
        "NOT NEKTAR_USE_SYSTEM_BLAS_LAPACK; APPLE" OFF)
    CMAKE_DEPENDENT_OPTION(NEKTAR_USE_WIN32_LAPACK
        "Use Win32 Lapack provided with the Third Party Distribution."
        ON "NOT NEKTAR_USE_SYSTEM_BLAS_LAPACK; WIN32" OFF)

    INCLUDE(FindBlasLapack)

    IF(LAPACK_FOUND)
        MESSAGE(STATUS "Found BLAS and LAPACK: ${BLAS_LAPACK}")
        SET(BUILD_BLAS_LAPACK OFF)
    ELSE()
        IF(CMAKE_Fortran_COMPILER)
            SET(BUILD_BLAS_LAPACK ON)
        ELSE()
            MESSAGE(SEND_ERROR "No BLAS/LAPACK installation found and cannot self build")
        ENDIF()
    ENDIF()

    OPTION(THIRDPARTY_BUILD_BLAS_LAPACK "Build BLAS and LAPACK libraries from ThirdParty."
        ${BUILD_BLAS_LAPACK})

    IF(THIRDPARTY_BUILD_BLAS_LAPACK)
        SET(LAPACK_VERSION 3.12.0)

        INCLUDE(ExternalProject)
        THIRDPARTY_LIBRARY(BLAS_LAPACK SHARED blas lapack DESCRIPTION "BLAS and LAPACK libraries")

        EXTERNALPROJECT_ADD(
            lapack-${LAPACK_VERSION}
            PREFIX ${TPSRC}
            URL ${TPURL}/lapack-3.12.0.tar.gz
            URL_MD5 "c1b38bef123584d86a1bd8000784a7b2"
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            SOURCE_DIR ${TPSRC}/lapack-${LAPACK_VERSION}
            BINARY_DIR ${TPBUILD}/lapack-${LAPACK_VERSION}
            TMP_DIR ${TPBUILD}/lapack-${LAPACK_VERSION}-tmp
            INSTALL_DIR ${TPDIST}
            BUILD_BYPRODUCTS ${BLAS_LAPACK}
	    PATCH_COMMAND ${LAPACK_PATCH_COMMAND}
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
            ${NEKTAR_EXTERNAL_PROJECT_CMAKE_GENERATOR_ARGS}
            -DCMAKE_Fortran_COMPILER:FILEPATH=${CMAKE_Fortran_COMPILER}
            -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
            -DCMAKE_INSTALL_LIBDIR:PATH=${TPDIST}/lib
            -DBUILD_SHARED_LIBS:STRING=ON
            -DBUILD_TESTING:STRING=OFF
            ${TPSRC}/lapack-${LAPACK_VERSION}
            )

        MESSAGE(STATUS "Build BLAS/LAPACK: ${BLAS_LAPACK}")

        ADD_DEPENDENCIES(thirdparty lapack-${LAPACK_VERSION})
    ENDIF()
ENDIF()
