########################################################################
#
# ThirdParty configuration for Nektar++
#
# MAGMA library
#
########################################################################

IF (NEKTAR_USE_MAGMA)
    ADD_DEFINITIONS(-DNEKTAR_USE_MAGMA)

    FIND_PATH(MAGMA_INCLUDE_DIR magma.h)
    FIND_LIBRARY(MAGMA_LIBRARY NAMES "magma")

    # If we have our library then don't build libmagma.
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
                libmagma
                PREFIX ${TPSRC}
                GIT_REPOSITORY https://github.com/icl-utk-edu/magma.git
                GIT_TAG v2.10.0
                STAMP_DIR ${TPBUILD}/stamp
                DOWNLOAD_DIR ${TPSRC}
                SOURCE_DIR ${TPSRC}/libmagma
                BINARY_DIR ${TPBUILD}/libmagma
                TMP_DIR ${TPBUILD}/libmagma-tmp
                INSTALL_DIR ${TPDIST}
                BUILD_BYPRODUCTS ${MAGMA_LIBRARY}
                CONFIGURE_COMMAND cp ${TPBUILD}/MAGMA_CONFIG/make.inc ${TPSRC}/libmagma/ && ${MAKE_EXECUTABLE} -C ${TPSRC}/libmagma generate
                BUILD_COMMAND ${CMAKE_COMMAND} ${TPSRC}/libmagma -G ${CMAKE_GENERATOR}
                        -B ${TPBUILD}/libmagma
                        -DMAGMA_ENABLE_CUDA=${MAGMA_ENABLE_CUDA}
                        -DMAGMA_ENABLE_HIP=${MAGMA_ENABLE_HIP}
                        -DGPU_TARGET=${NEKTAR_DEVICE_ARCH}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
                UPDATE_COMMAND ""
        )

        SET(MAGMA_INCLUDE_DIR ${TPDIST}/include CACHE FILEPATH "libmagma include" FORCE)
        MESSAGE(STATUS "Build libmagma: ${MAGMA_LIBRARY}")
        SET(MAGMA_CONFIG_INCLUDE_DIR ${TPINC})

        INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIR})
    ELSE()
        ADD_CUSTOM_TARGET(libmagma ALL)
        MESSAGE(STATUS "Found libmagma: ${MAGMA_LIBRARY}")
        SET(MAGMA_CONFIG_INCLUDE_DIR ${MAGMA_INCLUDE_DIR})
    ENDIF()

    ADD_DEPENDENCIES(thirdparty libmagma)

    MARK_AS_ADVANCED(MAGMA_LIBRARY)
    MARK_AS_ADVANCED(MAGMA_INCLUDE_DIR)
ENDIF()
