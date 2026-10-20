########################################################################
#
# ThirdParty configuration for Nektar++
#
# NVIDIA Performance Libraries (NVPL) BLAS and LAPACK
#
# NVPL is distributed by NVIDIA as binary archives under its own licence, so
# the ThirdParty build downloads them from NVIDIA rather than from the
# Nektar++ mirror, checks them against the checksums NVIDIA publishes, and
# unpacks them into ${TPDIST}. There is nothing to configure or compile.
#
# Sets NVPL_FOUND, NVPL_LAPACK, NVPL_BLAS, NVPL_BLAS_CORE, NVPL_INCLUDE_DIR
# and NVPL_THIRDPARTY_TARGETS.
#
########################################################################

SET(NVPL_URL https://developer.download.nvidia.com/compute/nvpl/redist)

# Component versions and checksums of the NVPL 25.11 release, from
# ${NVPL_URL}/redistrib_25.11.json.
SET(NVPL_COMPONENTS blas lapack common)
SET(NVPL_blas_VERSION 0.5.0.1)
SET(NVPL_blas_SHA256
    c51fc9cc908cca7c6aa39e72c492c2bfcc9472b542b109b6bcfa5d143f9d780e)
SET(NVPL_lapack_VERSION 0.3.2)
SET(NVPL_lapack_SHA256
    3e4d9969223a52ab011d1bf38156cbe31fba0e4f12c3b87729393972a16daecf)
SET(NVPL_common_VERSION 0.3.4)
SET(NVPL_common_SHA256
    c68891dd293df0faf2ae3cebfeae69c567cf784c98006351abcde2a34fc387df)

SET(NVPL_LAPACK ${TPDIST}/lib/libnvpl_lapack_lp64_seq.so
    CACHE FILEPATH "NVPL LAPACK library" FORCE)
SET(NVPL_BLAS ${TPDIST}/lib/libnvpl_blas_lp64_seq.so
    CACHE FILEPATH "NVPL BLAS library" FORCE)
SET(NVPL_BLAS_CORE ${TPDIST}/lib/libnvpl_blas_core.so
    CACHE FILEPATH "NVPL BLAS core library" FORCE)
SET(NVPL_INCLUDE_DIR ${TPDIST}/include
    CACHE PATH "NVPL include directory" FORCE)

SET(NVPL_BYPRODUCTS_blas ${NVPL_BLAS} ${NVPL_BLAS_CORE})
SET(NVPL_BYPRODUCTS_lapack ${NVPL_LAPACK})
SET(NVPL_BYPRODUCTS_common "")

INCLUDE(ExternalProject)

SET(NVPL_THIRDPARTY_TARGETS "")
FOREACH(comp ${NVPL_COMPONENTS})
    SET(name nvpl_${comp}-${NVPL_${comp}_VERSION})
    SET(archive nvpl_${comp}-linux-sbsa-${NVPL_${comp}_VERSION}-archive.tar.xz)

    # Each archive holds include/ and/or lib/ below a single top-level
    # directory.
    EXTERNALPROJECT_ADD(
        ${name}
        PREFIX ${TPSRC}
        URL ${NVPL_URL}/nvpl_${comp}/linux-sbsa/${archive}
        URL_HASH SHA256=${NVPL_${comp}_SHA256}
        STAMP_DIR ${TPBUILD}/stamp
        DOWNLOAD_DIR ${TPSRC}
        SOURCE_DIR ${TPSRC}/${name}
        BINARY_DIR ${TPBUILD}/${name}
        TMP_DIR ${TPBUILD}/${name}-tmp
        INSTALL_DIR ${TPDIST}
        BUILD_BYPRODUCTS ${NVPL_BYPRODUCTS_${comp}}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        UPDATE_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -DSRC:PATH=<SOURCE_DIR>
            -DDST:PATH=${TPDIST}
            -P ${PROJECT_SOURCE_DIR}/cmake/scripts/install-nvpl-archive.cmake
    )

    LIST(APPEND NVPL_THIRDPARTY_TARGETS ${name})
    ADD_DEPENDENCIES(thirdparty ${name})
ENDFOREACH()

SET(NVPL_FOUND TRUE)
MESSAGE(STATUS "Build NVPL: ${NVPL_BLAS}")
MESSAGE(STATUS "NVPL is downloaded from NVIDIA and is subject to the NVIDIA "
    "Performance Libraries licence (see ${TPSRC}/nvpl_blas-"
    "${NVPL_blas_VERSION}/LICENSE).")

MARK_AS_ADVANCED(NVPL_LAPACK)
MARK_AS_ADVANCED(NVPL_BLAS)
MARK_AS_ADVANCED(NVPL_BLAS_CORE)
MARK_AS_ADVANCED(NVPL_INCLUDE_DIR)

UNSET(name)
UNSET(archive)
UNSET(comp)
