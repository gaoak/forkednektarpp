########################################################################
#
# ThirdParty configuration for Nektar++
#
# libxsmm
#
########################################################################

IF (NEKTAR_ENABLE_SIMD)
    # libxsmm 2.x is needed for ARM64 but is slower than 1.17 on x86_64.
    IF (_SYSTEM_PROCESSOR STREQUAL "x86_64")
        SET(LIBXSMM_VERSION 1.17)
        SET(LIBXSMM_MD5 bef3b02f8837b0eed1ea334045da0524)
    ELSE()
        SET(LIBXSMM_VERSION 2.1.0)
        SET(LIBXSMM_MD5 89282f395b2ac5874b789195101d3f50)
    ENDIF()

    # First search for system libxsmm installs.
    FIND_PATH(LIBXSMM_INCLUDE_DIR libxsmm.h PATH_SUFFIXES libxsmm)
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

        IF (LIBXSMM_VERSION VERSION_GREATER_EQUAL 2.0)
            UNSET(PATCH CACHE)
            FIND_PROGRAM(PATCH patch)
            IF(NOT PATCH)
                MESSAGE(FATAL_ERROR
                    "'patch' tool for modifying files not found. Cannot build libxsmm.")
            ENDIF()
            MARK_AS_ADVANCED(PATCH)
            SET(LIBXSMM_PATCH_COMMAND ${PATCH} -p1 < ${PROJECT_SOURCE_DIR}/cmake/thirdparty-patches/libxsmm-2.1.0-apple-cpuid.patch)

            # libxsmm 2.x installs its headers under include/libxsmm.
            SET(LIBXSMM_TP_INCLUDE_DIR ${TPDIST}/include/libxsmm)
        ELSE()
            SET(LIBXSMM_PATCH_COMMAND "")
            SET(LIBXSMM_TP_INCLUDE_DIR ${TPDIST}/include)
        ENDIF()

        THIRDPARTY_LIBRARY(LIBXSMM_LIBRARY STATIC xsmm DESCRIPTION "libxsmm library")

        FIND_PROGRAM(LIBXSMM_MAKE_EXECUTABLE NAMES gmake make mingw32-make REQUIRED)
        MARK_AS_ADVANCED(LIBXSMM_MAKE_EXECUTABLE)

        EXTERNALPROJECT_ADD(
                libxsmm-${LIBXSMM_VERSION}
                PREFIX ${TPSRC}
                URL ${TPURL}/libxsmm-${LIBXSMM_VERSION}.tar.gz
                URL_MD5 ${LIBXSMM_MD5}
                STAMP_DIR ${TPBUILD}/stamp
                DOWNLOAD_DIR ${TPSRC}
                SOURCE_DIR ${TPSRC}/libxsmm-${LIBXSMM_VERSION}
                BINARY_DIR ${TPBUILD}/libxsmm-${LIBXSMM_VERSION}
                TMP_DIR ${TPBUILD}/libxsmm-${LIBXSMM_VERSION}-tmp
                INSTALL_DIR ${TPDIST}
                BUILD_BYPRODUCTS ${LIBXSMM_LIBRARY}
                PATCH_COMMAND ${LIBXSMM_PATCH_COMMAND}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ${LIBXSMM_MAKE_EXECUTABLE} --silent RPM_OPT_FLAGS=-Wno-deprecated-declarations -C <SOURCE_DIR>
                # CLEAN=0 stops the install from wiping PREFIX, which is shared with
                # the other ThirdParty libraries.
                INSTALL_COMMAND ${LIBXSMM_MAKE_EXECUTABLE} RPM_OPT_FLAGS=-Wno-deprecated-declarations CLEAN=0 -C <SOURCE_DIR> PREFIX=<INSTALL_DIR> install
        )

        SET(LIBXSMM_INCLUDE_DIR ${LIBXSMM_TP_INCLUDE_DIR} CACHE FILEPATH
                "libxsmm include" FORCE)
        MESSAGE(STATUS "Build libxsmm: ${LIBXSMM_LIBRARY}")
        SET(LIBXSMM_CONFIG_INCLUDE_DIR ${TPINC})

        INCLUDE_DIRECTORIES(${LIBXSMM_INCLUDE_DIR})
    ELSE()
        ADD_CUSTOM_TARGET(libxsmm-${LIBXSMM_VERSION} ALL)
        MESSAGE(STATUS "Found libxsmm: ${LIBXSMM_LIBRARY}")
        SET(LIBXSMM_CONFIG_INCLUDE_DIR ${LIBXSMM_INCLUDE_DIR})

        INCLUDE_DIRECTORIES(${LIBXSMM_INCLUDE_DIR})
    ENDIF (THIRDPARTY_BUILD_LIBXSMM)

    ADD_DEPENDENCIES(thirdparty libxsmm-${LIBXSMM_VERSION})

    MARK_AS_ADVANCED(LIBXSMM_INCLUDE_DIR)
    MARK_AS_ADVANCED(LIBXSMM_LIBRARY)
    MARK_AS_ADVANCED(LIBXSMM_CONFIG_INCLUDE_DIR)
ENDIF()
