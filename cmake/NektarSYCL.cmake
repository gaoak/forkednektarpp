#
# NektarSYCL.cmake
#
# Sets up cmake variables needed for using SYCL in Nektar++
#

ADD_DEFINITIONS(-DNEKTAR_ENABLE_SYCL)

IF (NEKTAR_ENABLE_SYCL STREQUAL "Default")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
ENDIF()

FIND_PACKAGE(IntelSYCL REQUIRED)
