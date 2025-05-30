#
# NektarIntelSYCL.cmake
#
# Sets up cmake variables needed for using SYCL in Nektar++
#

ADD_DEFINITIONS(-DNEKTAR_ENABLE_SYCL)

IF (USE_SYCL_BUILTIN_REDUCER)
    ADD_DEFINITIONS(-DUSE_SYCL_BUILTIN_REDUCER)
ENDIF()

IF (NEKTAR_ENABLE_SYCL STREQUAL "Default")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=x86_64 -Wno-enum-constexpr-conversion")
    ADD_DEFINITIONS(-DSYCL_ENABLE_CPU)
    SET(BOOST_MIN_VERSION "1.76.0")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
    ADD_DEFINITIONS(-DSYCL_ENABLE_CUDA)
    IF (NEKTAR_DEVICE_ARCH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvidia_gpu_${NEKTAR_DEVICE_ARCH} -Wno-enum-constexpr-conversion ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-enum-constexpr-conversion ")
    ENDIF()
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(BOOST_MIN_VERSION "1.76.0")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "HIP")
    ADD_DEFINITIONS(-DSYCL_ENABLE_HIP)
    IF (NEKTAR_DEVICE_ARCH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amd_gpu_${NEKTAR_DEVICE_ARCH} -Wno-enum-constexpr-conversion ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Wno-enum-constexpr-conversion ")
    ENDIF()
    SET(BOOST_MIN_VERSION "1.82.0")
ELSEIF (NEKTAR_ENABLE_SYCL STREQUAL "Intel")
    ADD_DEFINITIONS(-DSYCL_ENABLE_INTEL)
    IF (NEKTAR_DEVICE_ARCH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_${NEKTAR_DEVICE_ARCH} -Wno-enum-constexpr-conversion ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen -Wno-enum-constexpr-conversion ")
    ENDIF()
ENDIF()

FIND_PACKAGE(IntelSYCL REQUIRED)
FIND_PACKAGE(MKL CONFIG REQUIRED)
SET(NEKTAR_SYCL_DEPENDS
  MKL::MKL_SYCL
  )
