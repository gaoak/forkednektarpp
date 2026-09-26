########################################################################
#
# ThirdParty configuration for Nektar++
#
# Scotch partitioner
#
########################################################################

IF (NOT WIN32)
    OPTION(NEKTAR_USE_SCOTCH
        "Use Scotch library for performing mesh partitioning." ON)
ENDIF(NOT WIN32)

IF (NEKTAR_USE_SCOTCH)
    IF (NEKTAR_USE_MPI)
        FIND_PACKAGE(Scotch 5 COMPONENTS ptscotch)
    ELSE()
        FIND_PACKAGE(Scotch 5)
    ENDIF()

    IF (SCOTCH_FOUND)
        SET(BUILD_SCOTCH OFF)
    ELSE()
        SET(BUILD_SCOTCH ON)
    ENDIF ()

    CMAKE_DEPENDENT_OPTION(THIRDPARTY_BUILD_SCOTCH
        "Build Scotch library from ThirdParty" ${BUILD_SCOTCH}
        "NEKTAR_USE_SCOTCH" OFF)

    ADD_DEFINITIONS(-DNEKTAR_USE_SCOTCH)

    IF (THIRDPARTY_BUILD_SCOTCH)
        INCLUDE(ExternalProject)

        # Note that scotch is compiled in the source-tree, so we unpack the
        # source code in the ThirdParty builds directory.
        SET(SCOTCH_SRC ${TPBUILD}/scotch-7.0.1/src)

        # Determine the build target and compiler to use for scotch.
        # We use the normal C compiler by default. If MPI is being used and is
        # not built into the normal compiler, we use the MPI-specified compiler.
        IF (NEKTAR_USE_MPI)
            SET(BUILD_PTSCOTCH ON)
        ELSE ()
            SET(BUILD_PTSCOTCH OFF)
        ENDIF ()

        IF (NEKTAR_USE_METIS)
            SET(INSTALL_METIS_HEADERS OFF)
        ELSE ()
            SET(INSTALL_METIS_HEADERS ON)
        ENDIF ()


        SET(SCOTCH_BUILD_BYPRODUCTS
            ${TPDIST}/include/scotch.h
            ${TPDIST}/include/scotchf.h
            ${TPDIST}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}scotch${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${TPDIST}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}scotcherr${CMAKE_STATIC_LIBRARY_SUFFIX}
        )
        IF (NEKTAR_USE_MPI)
            LIST(APPEND SCOTCH_BUILD_BYPRODUCTS
                ${TPDIST}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ptscotch${CMAKE_STATIC_LIBRARY_SUFFIX}
                ${TPDIST}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ptscotcherr${CMAKE_STATIC_LIBRARY_SUFFIX}
            )
        ENDIF()

        INCLUDE(ExternalProject)
        EXTERNALPROJECT_ADD(
            scotch-7.0.1
            PREFIX ${TPSRC}
            URL ${TPURL}/scotch-v7.0.1.tar.gz 
            URL_MD5 "34f1d3d2bb82c385b161befdccfaea45"
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            SOURCE_DIR ${TPSRC}/scotch-7.0.1
            BINARY_DIR ${TPBUILD}/scotch-7.0.1
            TMP_DIR ${TPBUILD}/scotch-7.0.1-tmp
            INSTALL_DIR ${TPDIST}
            BUILD_BYPRODUCTS ${SCOTCH_BUILD_BYPRODUCTS}
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                ${NEKTAR_EXTERNAL_PROJECT_CMAKE_GENERATOR_ARGS}
                -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
		"-DCMAKE_C_FLAGS:STRING=-w -O3 -fPIC -Wno-free-nonheap-object -Wno-unused-result -D_FORTIFY_SOURCE=2 -DSCOTCH_PTHREAD_NUMBER=1"
                -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                -DCMAKE_INSTALL_LIBDIR=lib
                -DINSTALL_METIS_HEADERS=${INSTALL_METIS_HEADERS}
                -DBUILD_PTSCOTCH=${BUILD_PTSCOTCH}
                -Wno-dev
                ${TPSRC}/scotch-7.0.1
        )

        THIRDPARTY_LIBRARY(SCOTCH_LIBRARY STATIC scotch
            DESCRIPTION "Scotch library")
        THIRDPARTY_LIBRARY(SCOTCHERR_LIBRARY STATIC scotcherr
            DESCRIPTION "Scotch error library")
        THIRDPARTY_LIBRARY(PTSCOTCH_LIBRARY STATIC ptscotch;scotch
            DESCRIPTION "PT-Scotch library")
        THIRDPARTY_LIBRARY(PTSCOTCHERR_LIBRARY STATIC ptscotcherr
            DESCRIPTION "PT-Scotch error library")
        SET(SCOTCH_INCLUDE_DIR ${TPDIST}/include CACHE FILEPATH
            "Scotch include directory" FORCE)
        SET(SCOTCH_LIBRARY_DIR ${TPDIST}/lib CACHE FILEPATH
            "Scotch library directory" FORCE)
        IF(NEKTAR_USE_MPI)
            MESSAGE(STATUS "Build PT-Scotch: ${PTSCOTCH_LIBRARY}")
        ELSE()
            MESSAGE(STATUS "Build Scotch: ${SCOTCH_LIBRARY}")
        ENDIF()
        SET(SCOTCH_CONFIG_INCLUDE_DIR ${TPINC})
    ELSE (THIRDPARTY_BUILD_SCOTCH)
        ADD_CUSTOM_TARGET(scotch-7.0.1 ALL)
        SET(SCOTCH_CONFIG_INCLUDE_DIR ${SCOTCH_INCLUDE_DIR})
    ENDIF (THIRDPARTY_BUILD_SCOTCH)

    ADD_DEPENDENCIES(thirdparty scotch-7.0.1)

    INCLUDE_DIRECTORIES(SYSTEM ${SCOTCH_INCLUDE_DIR} ${PTSCOTCH_INCLUDE_DIR})

    MARK_AS_ADVANCED(SCOTCH_LIBRARY)
    MARK_AS_ADVANCED(SCOTCHERR_LIBRARY)
    MARK_AS_ADVANCED(SCOTCH_LIBRARY_DIR)
    MARK_AS_ADVANCED(SCOTCH_INCLUDE_DIR)
    MARK_AS_ADVANCED(PTSCOTCH_LIBRARY)
    MARK_AS_ADVANCED(PTSCOTCHERR_LIBRARY)
ENDIF()
