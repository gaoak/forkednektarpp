#
# NektarIntelSYCL.cmake
#
# Sets up cmake variables needed for using SYCL in Nektar++
#

ADD_DEFINITIONS(-D__DPCPP_COMPILER)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icpx needed -Wno-enum-constexpr-conversion while it was built on Clang
    # 16/17, where an out-of-range enum value in a constant expression was a
    # warning promoted to an error. Clang has since dropped that diagnostic:
    # on the Clang 21 behind icpx 2025.2 the flag is simply unknown, so
    # passing it unconditionally puts -Wunknown-warning-option on every single
    # compile. Probe for it instead, which keeps both generations quiet.
    #
    # The probe needs -Werror=unknown-warning-option to mean anything: Clang
    # accepts an unrecognised -Wno-* silently unless some other diagnostic is
    # emitted, so without it the check would pass everywhere.
    INCLUDE(CheckCXXCompilerFlag)
    SET(CMAKE_REQUIRED_FLAGS "-Werror=unknown-warning-option")
    CHECK_CXX_COMPILER_FLAG("-Wno-enum-constexpr-conversion"
        NEKTAR_HAVE_WNO_ENUM_CONSTEXPR_CONVERSION)
    UNSET(CMAKE_REQUIRED_FLAGS)
    IF (NEKTAR_HAVE_WNO_ENUM_CONSTEXPR_CONVERSION)
        SET(EXTRA_FLAGS -Wno-enum-constexpr-conversion)
    ENDIF()
ENDIF()

IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CPU")
    IF( CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # Using intel icpx compiler.
        IF (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=x86_64 ${EXTRA_FLAGS} ")
        ELSE()
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ${EXTRA_FLAGS} ")
        ENDIF()
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=native_cpu ${EXTRA_FLAGS} ")
        ADD_DEFINITIONS(-D__DPCPP_COMPILER_NATIVE_CPU)
    ENDIF()
    ADD_DEFINITIONS(-DSYCL_ENABLE_CPU)
    SET(BOOST_MIN_VERSION "1.76.0")
ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA")
    ADD_DEFINITIONS(-DSYCL_ENABLE_CUDA)
    IF (NEKTAR_DEVICE_ARCH)
        IF(NOT NEKTAR_DEVICE_ARCH MATCHES "^sm_[0-9]+$")
            MESSAGE(FATAL_ERROR "For Nvidia GPU, NEKTAR_DEVICE_ARCH must be specified as sm_XX, got '${NEKTAR_DEVICE_ARCH}'")
        ENDIF()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvidia_gpu_${NEKTAR_DEVICE_ARCH} ${EXTRA_FLAGS} ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda ${EXTRA_FLAGS} ")
    ENDIF()
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(BOOST_MIN_VERSION "1.82.0")
ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
    ADD_DEFINITIONS(-DSYCL_ENABLE_HIP)
    IF(NOT HIP_CXX_COMPILER)
        FIND_PATH(HIP_CXX_COMPILER clang++ HINTS ENV HIPROOT ENV ROCM_PATH)
    ENDIF()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${NEKTAR_DEVICE_ARCH} ${EXTRA_FLAGS} ")
    SET(BOOST_MIN_VERSION "1.82.0")
ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-Intel")
    ADD_DEFINITIONS(-DSYCL_ENABLE_INTEL)
    IF (NEKTAR_DEVICE_ARCH)
        # Ahead-of-time: the compiler hands the SPIR-V to ocloc to get native
        # GEN code. ocloc ships with the Intel GPU driver stack, NOT with the
        # compiler, so it can easily be missing on an otherwise complete
        # oneAPI install. Without this check the build fails much later with a
        # bare "llvm-foreach: No such file or directory", which says nothing
        # about the real cause.
        FIND_PROGRAM(NEKTAR_OCLOC ocloc)
        IF (NOT NEKTAR_OCLOC)
            MESSAGE(FATAL_ERROR
                "NEKTAR_DEVICE_ARCH=${NEKTAR_DEVICE_ARCH} requests "
                "ahead-of-time compilation for an Intel GPU, which needs "
                "the 'ocloc' tool. It was not found on PATH. Install it "
                "with the GPU driver stack (on Debian or Ubuntu: the "
                "intel-ocloc package), or leave NEKTAR_DEVICE_ARCH empty "
                "to compile kernels just-in-time for spir64 instead.")
        ENDIF()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_${NEKTAR_DEVICE_ARCH} ${EXTRA_FLAGS} ")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64 ${EXTRA_FLAGS} ")
    ENDIF()
    SET(BOOST_MIN_VERSION "1.82.0")
ENDIF()

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    FIND_PACKAGE(IntelSYCL REQUIRED)
ENDIF()

# On an Intel device the expression operator has no compile-time form of a
# session's expressions, so it builds its kernels at runtime from generated
# SYCL source (ExpressionDeviceGenericSYCLHelper.hpp) and needs
# sycl_ext_oneapi_kernel_compiler. The extension is absent in oneAPI 2024.x and
# present from 2025.x, so this constrains the compiler, not just the target.
#
# SYCL-Intel is the only back-end that needs it: SYCL-CUDA and SYCL-HIP compile
# through NVRTC and hipRTC instead, which is why they link CUDA::nvrtc and
# hiprtc, while SYCL-CPU and SYCL-SIMSYCL evaluate the expressions on the host.
#
# The extension is probed rather than inferred from a version number, so an
# implementation that gains it later is accepted without changing this.
IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-Intel")
    INCLUDE(CheckCXXSourceCompiles)
    CHECK_CXX_SOURCE_COMPILES("
        #include <sycl/sycl.hpp>
        #if !defined(SYCL_EXT_ONEAPI_KERNEL_COMPILER)
        #error sycl_ext_oneapi_kernel_compiler is unavailable
        #endif
        int main() { return 0; }"
        NEKTAR_HAVE_SYCL_KERNEL_COMPILER)

    IF (NOT NEKTAR_HAVE_SYCL_KERNEL_COMPILER)
        MESSAGE(FATAL_ERROR
            "NEKTAR_ENABLE_DEVICE=SYCL-Intel compiles its expression "
            "kernels at runtime, which needs a SYCL implementation "
            "providing sycl_ext_oneapi_kernel_compiler. The current "
            "compiler (${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}) does not define "
            "SYCL_EXT_ONEAPI_KERNEL_COMPILER. With Intel oneAPI this "
            "needs the 2025.x series or newer; 2024.x does not "
            "provide it.")
    ENDIF()
ENDIF()

IF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA")
    FIND_PACKAGE(CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED)
    SET(NEKTAR_SYCL_DEPENDS ${NEKTAR_SYCL_DEPENDS} CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
    IF(WIN32)
        SET(ROCM_ROOT
            "$ENV{HIP_PATH}"
            CACHE PATH
            "Root directory of the ROCm installation"
        )
    ELSE()
        SET(ROCM_ROOT
            "/opt/rocm"
            CACHE PATH
            "Root directory of the ROCm installation"
        )
    ENDIF()
    LIST(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
    FIND_PACKAGE(HIP REQUIRED)
    FIND_LIBRARY(HIPRTC_LIB hiprtc HINTS ${HIP_PATH}/lib /opt/rocm/lib)
    SET(NEKTAR_SYCL_DEPENDS ${NEKTAR_SYCL_DEPENDS} hip::host ${HIPRTC_LIB})
ENDIF()
