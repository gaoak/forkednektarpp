########################################################################
#
# ThirdParty configuration for Nektar++
#
# MAGMA library
#
########################################################################

IF (NEKTAR_USE_MAGMA)
    IF (NOT (${NEKTAR_ENABLE_DEVICE} STREQUAL "CUDA" OR ${NEKTAR_ENABLE_DEVICE} STREQUAL "HIP"))
        MESSAGE(FATAL_ERROR "ThirdParty MAGMA compilation only available for CUDA and HIP devive backend")
    ENDIF()
    ADD_DEFINITIONS(-DNEKTAR_USE_MAGMA)

    FIND_PATH(MAGMA_INCLUDE_DIR magma.h)
    FIND_LIBRARY(MAGMA_LIBRARY NAMES "magma")

    # If we have our library then don't build magma.
    IF (MAGMA_LIBRARY AND MAGMA_INCLUDE_DIR)
        SET(BUILD_MAGMA OFF)
    ELSE()
        SET(BUILD_MAGMA ON)
    ENDIF ()

    OPTION(THIRDPARTY_BUILD_MAGMA
        "Build MAGMA library from ThirdParty" ${BUILD_MAGMA})

    IF(THIRDPARTY_BUILD_MAGMA)
        INCLUDE(ExternalProject)

        THIRDPARTY_LIBRARY(MAGMA_LIBRARY SHARED magma DESCRIPTION "MAGMA library")
        FIND_PROGRAM(MAKE_EXECUTABLE NAMES gmake make mingw32-make REQUIRED)
        EXECUTE_PROCESS(COMMAND mkdir -p ${TPBUILD}/MAGMA_CONFIG)
        IF (${NEKTAR_ENABLE_DEVICE} STREQUAL "CUDA")
            SET(MAGMA_ENABLE_CUDA ON)
            SET(MAGMA_ENABLE_HIP OFF)
            EXECUTE_PROCESS(COMMAND echo -e "BACKEND=cuda\nFORT=false\nGPU_TARGET=${NEKTAR_DEVICE_ARCH}" OUTPUT_FILE "${TPBUILD}/MAGMA_CONFIG/make.inc")
        ELSEIF (${NEKTAR_ENABLE_DEVICE} STREQUAL "HIP")
            SET(MAGMA_ENABLE_CUDA OFF)
            SET(MAGMA_ENABLE_HIP ON)
            EXECUTE_PROCESS(COMMAND echo -e "BACKEND=hip\nFORT=false\nGPU_TARGET=${NEKTAR_DEVICE_ARCH}" OUTPUT_FILE "${TPBUILD}/MAGMA_CONFIG/make.inc")
        ENDIF()
        EXTERNALPROJECT_ADD(
                magma-v2.10.0
                PREFIX ${TPSRC}
                GIT_REPOSITORY https://github.com/icl-utk-edu/magma.git
                GIT_TAG v2.10.0
                STAMP_DIR ${TPBUILD}/stamp
                DOWNLOAD_DIR ${TPSRC}
                SOURCE_DIR ${TPSRC}/magma-v2.10.0
                BINARY_DIR ${TPBUILD}/magma-v2.10.0
                TMP_DIR ${TPBUILD}/magma-v2.10.0-tmp
                INSTALL_DIR ${TPDIST}
                BUILD_BYPRODUCTS ${MAGMA_LIBRARY}
                CONFIGURE_COMMAND cp ${TPBUILD}/MAGMA_CONFIG/make.inc <SOURCE_DIR> && ${MAKE_EXECUTABLE} -C <SOURCE_DIR> generate
                BUILD_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR> -G ${CMAKE_GENERATOR}
                        -B <BINARY_DIR>
                        -DMAGMA_ENABLE_CUDA=${MAGMA_ENABLE_CUDA}
                        -DMAGMA_ENABLE_HIP=${MAGMA_ENABLE_HIP}
                        -DGPU_TARGET=${NEKTAR_DEVICE_ARCH}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
                UPDATE_COMMAND ""
        )

        SET(MAGMA_INCLUDE_DIR ${TPDIST}/include CACHE FILEPATH "magma include" FORCE)
        MESSAGE(STATUS "Build magma: ${MAGMA_LIBRARY}")
        SET(MAGMA_CONFIG_INCLUDE_DIR ${TPINC})

        INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIR})
    ELSE()
        ADD_CUSTOM_TARGET(magma-v2.10.0 ALL)
        MESSAGE(STATUS "Found magma: ${MAGMA_LIBRARY}")
        SET(MAGMA_CONFIG_INCLUDE_DIR ${MAGMA_INCLUDE_DIR})
    ENDIF()

    ADD_DEPENDENCIES(thirdparty magma-v2.10.0)

    MARK_AS_ADVANCED(MAGMA_LIBRARY)
    MARK_AS_ADVANCED(MAGMA_INCLUDE_DIR)
    MARK_AS_ADVANCED(MAGMA_CONFIG_INCLUDE_DIR)
ENDIF()
