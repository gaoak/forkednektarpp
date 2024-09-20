#
# NektarSYCL.cmake
#
# Sets up cmake variables needed for using SYCL in Nektar++
#

ADD_DEFINITIONS(-DNEKTAR_ENABLE_SYCL)

IF (NEKTAR_ENABLE_SYCL STREQUAL "Default")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wno-enum-constexpr-conversion")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
    ADD_DEFINITIONS(-DSYCL_ENABLE_CUDA)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-enum-constexpr-conversion ")
ENDIF()

FIND_PACKAGE(IntelSYCL REQUIRED)
