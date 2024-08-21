#
# NektarKokkos.cmake
#
# Sets up cmake variables needed for using Kokkos in Nektar++
#

SET(KOKKOS_MIN_VERSION "4.3")

ADD_DEFINITIONS(-DNEKTAR_ENABLE_KOKKOS)

# Check if Kokkos library is already built
SET(Kokkos_DIR ${TPBUILD}/kokkos-4.3.00)
FIND_PACKAGE(Kokkos ${KOKKOS_MIN_VERSION})
UNSET(Kokkos_DIR CACHE)

# Pre-build Kokkos library during configuration, if necessary
IF (NOT Kokkos_FOUND)
    EXECUTE_PROCESS(COMMAND rm -rf ${TPSRC}/kokkos-4.3.00)
    EXECUTE_PROCESS(COMMAND rm -rf ${TPBUILD}/kokkos-4.3.00)
    EXECUTE_PROCESS(COMMAND git clone -b release-candidate-4.3.00 https://github.com/kokkos/kokkos ${TPSRC}/kokkos-4.3.00)
    IF (NEKTAR_ENABLE_KOKKOS STREQUAL "Serial")
        EXECUTE_PROCESS(COMMAND 
                    cmake ${TPSRC}/kokkos-4.3.00 
                    -B ${TPBUILD}/kokkos-4.3.00
                    -DCMAKE_CXX_COMPILER=g++
                    -DCMAKE_CXX_FLAGS=-fPIC
                    -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                    -DCMAKE_INSTALL_LIBDIR:PATH=${TPDIST}/lib)
    ELSEIF (NEKTAR_ENABLE_KOKKOS STREQUAL "CUDA")
        EXECUTE_PROCESS(COMMAND 
                    cmake ${TPSRC}/kokkos-4.3.00 
                    -B ${TPBUILD}/kokkos-4.3.00
                    -DCMAKE_CXX_COMPILER=g++
                    -DCMAKE_CXX_FLAGS=-fPIC
                    -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                    -DCMAKE_INSTALL_LIBDIR:PATH=${TPDIST}/lib
                    -DKokkos_ENABLE_CUDA=ON 
                    -DKokkos_ENABLE_CUDA_LAMBDA=ON 
                    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON)
    ENDIF()
    EXECUTE_PROCESS(COMMAND make install -j8 -C ${TPBUILD}/kokkos-4.3.00)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSING_LATEST_KOKKOS")
    SET(Kokkos_DIR ${TPBUILD}/kokkos-4.3.00)
    FIND_PACKAGE(Kokkos ${KOKKOS_MIN_VERSION} REQUIRED)
    UNSET(Kokkos_DIR CACHE)
ENDIF()

SET(NEKTAR_KOKKOS_DEPENDS Kokkos::kokkos)
