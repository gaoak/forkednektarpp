#
# NektarDeviceArchLimits.cmake
#
# Defines the device limits the launch heuristics read at compile time.
#

# Look up ARCH in the entries "arch:sharedMemoryPerBlock:
# sharedMemoryPerMultiprocessor:maxThreadsPerMultiprocessor" that follow it and,
# if listed, define them for host and device code, with NAME naming the
# architecture in messages. The values are as the CUDA or HIP device properties
# report them; an unlisted architecture queries them at run time.
FUNCTION(NEKTAR_DEVICE_ARCH_LIMITS ARCH NAME)
    FOREACH(ENTRY ${ARGN})
        STRING(REPLACE ":" ";" FIELDS ${ENTRY})
        LIST(GET FIELDS 0 ENTRY_ARCH)
        IF(ENTRY_ARCH STREQUAL ARCH)
            LIST(GET FIELDS 1 SHMEM_PER_BLOCK)
            LIST(GET FIELDS 2 SHMEM_PER_MP)
            LIST(GET FIELDS 3 THREADS_PER_MP)
            ADD_COMPILE_DEFINITIONS(
                NEKTAR_DEVICE_ARCH_NAME="${NAME}"
                NEKTAR_DEVICE_SHARED_MEMORY_PER_BLOCK=${SHMEM_PER_BLOCK}
                NEKTAR_DEVICE_SHARED_MEMORY_PER_MP=${SHMEM_PER_MP}
                NEKTAR_DEVICE_MAX_THREADS_PER_MP=${THREADS_PER_MP})
            RETURN()
        ENDIF()
    ENDFOREACH()
    MESSAGE(STATUS "No compile-time device limits for ${NAME}; "
                   "the shared-memory workspace is chosen at run time")
ENDFUNCTION()

# Limits of the NVIDIA architecture DEVICE_ARCH, given as sm_XX or sm_XXa; the
# architecture-specific variant shares its base's limits.
FUNCTION(NEKTAR_CUDA_ARCH_LIMITS DEVICE_ARCH)
    STRING(REGEX REPLACE "^sm_([0-9]+)a?$" "\\1" ARCH ${DEVICE_ARCH})
    NEKTAR_DEVICE_ARCH_LIMITS(${ARCH} ${DEVICE_ARCH}
        "60:49152:65536:2048"
        "61:49152:98304:2048"
        "62:49152:65536:2048"
        "70:49152:98304:2048"
        "72:49152:98304:2048"
        "75:49152:65536:1024"
        "80:49152:167936:2048"
        "86:49152:102400:1536"
        "87:49152:167936:1536"
        "89:49152:102400:1536"
        "90:49152:233472:2048"
        "100:49152:233472:2048"
        "103:49152:233472:2048"
        "120:49152:102400:1536"
        "121:49152:102400:1536")
ENDFUNCTION()

# Limits of the AMD architecture DEVICE_ARCH, given as gfxXXX.
FUNCTION(NEKTAR_AMD_ARCH_LIMITS DEVICE_ARCH)
    NEKTAR_DEVICE_ARCH_LIMITS(${DEVICE_ARCH} ${DEVICE_ARCH}
        "gfx906:65536:65536:2560"
        "gfx908:65536:65536:2560"
        "gfx90a:65536:65536:2048"
        "gfx940:65536:65536:2048"
        "gfx941:65536:65536:2048"
        "gfx942:65536:65536:2048"
        "gfx950:163840:163840:2048"
        "gfx1030:65536:65536:2048"
        "gfx1100:65536:65536:2048"
        "gfx1101:65536:65536:2048"
        "gfx1102:65536:65536:2048"
        "gfx1200:65536:65536:2048"
        "gfx1201:65536:65536:2048")
ENDFUNCTION()
