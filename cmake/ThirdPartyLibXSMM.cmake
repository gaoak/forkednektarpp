########################################################################
#
# ThirdParty configuration for Nektar++
#
# libxsmm
#
########################################################################

# First search for system libxsmm installs.
FIND_PATH(LIBXSMM_INCLUDE_DIR libxsmm.h)
FIND_LIBRARY(LIBXSMM_LIBRARY NAMES "xsmm")

# If we have our library then don't build libxsmm.
IF (LIBXSMM_INCLUDE_DIR AND LIBXSMM_LIBRARY)
    SET(BUILD_LIBXSMM OFF)
ELSE()
    SET(BUILD_LIBXSMM ON)
ENDIF ()

OPTION(THIRDPARTY_BUILD_LIBXSMM
        "Build libxsmm library from ThirdParty." ${BUILD_LIBXSMM})

IF (THIRDPARTY_BUILD_LIBXSMM)
    INCLUDE(ExternalProject)

    THIRDPARTY_LIBRARY(LIBXSMM_LIBRARY STATIC xsmm DESCRIPTION "libxsmm library")

    FIND_PROGRAM(MAKE_EXECUTABLE NAMES gmake make mingw32-make REQUIRED)
    EXTERNALPROJECT_ADD(
            libxsmm-1.17
            PREFIX ${TPSRC}
            URL https://github.com/libxsmm/libxsmm/archive/refs/tags/1.17.tar.gz
            URL_MD5 bef3b02f8837b0eed1ea334045da0524
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            SOURCE_DIR ${TPSRC}/libxsmm-1.17
            BINARY_DIR ${TPBUILD}/libxsmm-1.17
            TMP_DIR ${TPBUILD}/libxsmm-1.17-tmp
            INSTALL_DIR ${TPDIST}
            BUILD_BYPRODUCTS ${LIBXSMM_LIBRARY}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ${MAKE_EXECUTABLE} --silent RPM_OPT_FLAGS=-Wno-deprecated-declarations -C <SOURCE_DIR>
            INSTALL_COMMAND ${MAKE_EXECUTABLE} RPM_OPT_FLAGS=-Wno-deprecated-declarations -C <SOURCE_DIR> PREFIX=<INSTALL_DIR> install
    )

    SET(LIBXSMM_INCLUDE_DIR ${TPDIST}/include CACHE FILEPATH
            "libxsmm include" FORCE)
    MESSAGE(STATUS "Build libxsmm: ${LIBXSMM_LIBRARY}")
    SET(LIBXSMM_CONFIG_INCLUDE_DIR ${TPINC})

ELSE()
    ADD_CUSTOM_TARGET(libxsmm-1.17 ALL)
    MESSAGE(STATUS "Found libxsmm: ${LIBXSMM_LIBRARY}")
    SET(LIBXSMM_CONFIG_INCLUDE_DIR ${LIBXSMM_INCLUDE_DIR})
ENDIF (THIRDPARTY_BUILD_LIBXSMM)

INCLUDE_DIRECTORIES(${LIBXSMM_INCLUDE_DIR})

MARK_AS_ADVANCED(LIBXSMM_INCLUDE_DIR)
MARK_AS_ADVANCED(LIBXSMM_LIBRARY)
MARK_AS_ADVANCED(LIBXSMM_CONFIG_INCLUDE_DIR)
