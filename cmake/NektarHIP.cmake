#
# NektarHIP.cmake
#
# Sets up cmake variables needed for using HIP in Nektar++
#

OPTION(NEKTAR_ENABLE_HIP "Enable HIP execution" OFF)

IF (NEKTAR_ENABLE_HIP)
    IF(NOT HIP_MIN_VERSION)
        SET(HIP_MIN_VERSION "5.7")
    ENDIF()

    ADD_DEFINITIONS(-DNEKTAR_ENABLE_HIP)

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSING_LATEST_KOKKOS -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -D__HIP_ARCH_GFX908__ -I/opt/rocm-5.7.0/include -I/opt/rocm-5.7.0/llvm/bin/../lib/clang/14.0.0 -I/opt/rocm-5.7.0/hsa/include' \
    LDFLAGS='-ldl -L/opt/rocm-5.7.0/lib -lamdhip64")

    FIND_PACKAGE(HIP ${HIP_MIN_VERSION} REQUIRED)

    SET(NEKTAR_HIP_DEPENDS HIP::HIP)
ENDIF (NEKTAR_ENABLE_HIP)
