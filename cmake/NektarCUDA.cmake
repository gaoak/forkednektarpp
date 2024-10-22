#
# NektarCUDA.cmake
#
# Sets up cmake variables needed for using Cuda in Nektar++
#

IF(CMAKE_VERSION VERSION_LESS "3.18.0")
    MESSAGE(FATAL_ERROR "CUDA compilation requires CMake version >= 3.18")
ENDIF()

IF(NOT CUDA_MIN_VERSION)
    SET(CUDA_MIN_VERSION "11.1")
ENDIF()

ADD_DEFINITIONS(-DNEKTAR_ENABLE_CUDA)

IF (NOT DEFINED CMAKE_CUDA_STANDARD)
    SET(CMAKE_CUDA_STANDARD 17)
    SET(CMAKE_CUDA_STANDARD_REQUIRED ON)
ENDIF()

ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CUDA>:$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler="/permissive-">>)

SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

IF (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    IF(CMAKE_VERSION VERSION_LESS "3.18.0")
        MESSAGE("Please consider to switch to CMake 3.24.0")
        ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_60,code=sm_60 
        -gencode arch=compute_61,code=sm_61 
        -gencode arch=compute_62,code=sm_62 
        -gencode arch=compute_70,code=sm_70 
        -gencode arch=compute_72,code=sm_72 
        -gencode arch=compute_75,code=sm_75 
        -gencode arch=compute_80,code=sm_80 
        -gencode arch=compute_86,code=sm_86 
        >)

    ELSEIF(CMAKE_VERSION VERSION_LESS "3.24.0")
        MESSAGE("Please consider to switch to CMake 3.24.0")
        SET(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)
    ELSE()
        SET(CMAKE_CUDA_ARCHITECTURES native)
    ENDIF()
ENDIF()

ENABLE_LANGUAGE(CUDA)
SET(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr")
FIND_PACKAGE(CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED)
INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
SET(NEKTAR_CUDA_DEPENDS
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cublas
  )
