#
# NektarCUDA.cmake
#
# Sets up cmake variables needed for using CUDA in Nektar++
#

IF(CMAKE_VERSION VERSION_LESS "3.18.0")
    MESSAGE(FATAL_ERROR "CUDA compilation requires CMake version >= 3.18")
ENDIF()

SET(BOOST_MIN_VERSION "1.76.0")

IF(NOT CUDA_MIN_VERSION)
    SET(CUDA_MIN_VERSION "11.1")
ENDIF()

IF (NOT DEFINED CMAKE_CUDA_STANDARD)
    SET(CMAKE_CUDA_STANDARD 17)
    SET(CMAKE_CUDA_STANDARD_REQUIRED ON)
ENDIF()

ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CUDA>:$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler="/permissive-">>)
ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>)
ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=550>)

SET(CUDA_SEPARABLE_COMPILATION ON)

IF (NEKTAR_DEVICE_ARCH)
    STRING(REPLACE "_" ";" ARCH ${NEKTAR_DEVICE_ARCH})
    LIST(GET ARCH 1 ARCH)
    ADD_COMPILE_DEFINITIONS(__NEKTAR_DEVICE_ARCH__="compute_${ARCH}")
    SET(CMAKE_CUDA_ARCHITECTURES ${ARCH})
ELSE()
    IF(CMAKE_VERSION VERSION_LESS "3.24.0")
        MESSAGE("Please consider to switch to CMake 3.24.0")
        SET(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 90)
    ELSE()
        SET(CMAKE_CUDA_ARCHITECTURES native)
    ENDIF()
ENDIF()

ENABLE_LANGUAGE(CUDA)
SET(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr")
FIND_PACKAGE(CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED)
INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

OPTION(NEKTAR_USE_CUFFT_STATIC
    "Link cuFFT statically to enable store-callback support in NekDeviceFFT"
    OFF)
MARK_AS_ADVANCED(NEKTAR_USE_CUFFT_STATIC)
IF(NEKTAR_USE_CUFFT_STATIC)
    SET(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    SET(CUFFT_LIBRARY CUDA::cufft_static CUDA::culibos)
    MESSAGE(STATUS "NekDeviceFFT: static cuFFT linkage enabled (callbacks supported)")

    IF(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
        SET(CUFFT_ARCH_FLAGS "-arch=native")
    ELSE()
        SET(CUFFT_ARCH_FLAGS "")
        FOREACH(_a ${CMAKE_CUDA_ARCHITECTURES})
            LIST(APPEND CUFFT_ARCH_FLAGS
                "-gencode" "arch=compute_${_a},code=sm_${_a}")
        ENDFOREACH()
    ENDIF()
ELSE()
    SET(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    SET(CUFFT_LIBRARY CUDA::cufft)
ENDIF()

OPTION(NEKTAR_USE_CUFFTDX "Use cuFFTDx" OFF)
MARK_AS_ADVANCED(NEKTAR_USE_CUFFTDX)
IF(NEKTAR_USE_CUFFTDX)
    FIND_PATH(CUFFTDX_INCLUDE_DIR cufftdx/cufftdx.hpp
              HINTS "$ENV{CUFFTDX_HOME}/include"
                    "${TPSRC}/cufftdx-25.12/nvidia/mathdx/25.12/include"
                    /opt/nvidia/mathdx/24.01/include
                    /opt/nvidia/mathdx/24.04/include
                    /opt/nvidia/mathdx/25.12/include)

    IF (NOT CUFFTDX_INCLUDE_DIR)
        INCLUDE(FetchContent)
        FetchContent_Declare(
            cufftdx-25.12
            URL https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-25.12.1-cuda13.tar.gz 
            SOURCE_DIR "${TPSRC}/cufftdx-25.12"
        )

        FetchContent_Populate(cufftdx-25.12)

        SET(CUFFTDX_INCLUDE_DIR ${TPSRC}/cufftdx-25.12/nvidia/mathdx/25.12/include/ CACHE FILEPATH
            "FFTW include" FORCE)

        MESSAGE(STATUS "Installing CUFFTDX: ${TPSRC}/cufftdx-25.12/")
    ELSE()
        MESSAGE(STATUS "Found CUFFTDX: ${CUFFTDX_INCLUDE_DIR}")
    ENDIF()

    MARK_AS_ADVANCED(CUFFTDX_INCLUDE_DIR)

    # Override by passing -DCUFFTDX_TARGET_SM=<value> on the cmake command line.
    IF(NOT CUFFTDX_TARGET_SM)
        LIST(GET CMAKE_CUDA_ARCHITECTURES 0 _cufftdx_arch_raw)
        IF(_cufftdx_arch_raw MATCHES "^[0-9]+$")
            SET(_cufftdx_arch_2digit "${_cufftdx_arch_raw}")
        ELSE()
            EXECUTE_PROCESS(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE _gpu_compute_cap
                RESULT_VARIABLE _nvsmi_rc
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            STRING(REGEX REPLACE "\n.*" "" _gpu_compute_cap "${_gpu_compute_cap}")
            IF(_nvsmi_rc EQUAL 0 AND _gpu_compute_cap MATCHES "^[0-9]+\\.[0-9]+$")
                STRING(REPLACE "." "" _cufftdx_arch_2digit "${_gpu_compute_cap}")
            ELSE()
                SET(_cufftdx_arch_2digit "86")
            ENDIF()
        ENDIF()
        MATH(EXPR CUFFTDX_TARGET_SM "${_cufftdx_arch_2digit} * 10")
    ENDIF()
ENDIF()

SET(NEKTAR_CUDA_DEPENDS
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)
SET(CUBLAS_LIBRARY CUDA::cublas)
SET(CUSPARSE_LIBRARY CUDA::cusparse)
SET(CUSOLVER_LIBRARY CUDA::cusolver)
SET(CURAND_LIBRARY CUDA::curand)
