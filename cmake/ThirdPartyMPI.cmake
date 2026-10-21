########################################################################
#
# ThirdParty configuration for Nektar++
#
# MPI
#
########################################################################

OPTION(NEKTAR_USE_MPI "Use MPI for parallelisation." OFF)

CMAKE_DEPENDENT_OPTION(THIRDPARTY_BUILD_GSMPI
    "Build GSMPI if needed" ON
    "NEKTAR_USE_MPI" OFF)

IF( NEKTAR_USE_MPI )
    # First check to see if our compiler has MPI built in to avoid linking
    # libraries etc.
    INCLUDE (CheckIncludeFiles)
    INCLUDE (CheckFunctionExists)
    CHECK_INCLUDE_FILES  (mpi.h    HAVE_MPI_H)
    CHECK_FUNCTION_EXISTS(MPI_Send HAVE_MPI_SEND)

    SET(MPI_BUILTIN OFF CACHE INTERNAL
        "Determines whether MPI is built into the compiler")
    IF (NOT "${HAVE_MPI_H}" OR NOT "${HAVE_MPI_SEND}")
        # Enable version detection to identify if MPICH or OPENMPI
        SET(MPI_DETERMINE_LIBRARY_VERSION YES)
        FIND_PACKAGE(MPI REQUIRED)

        INCLUDE_DIRECTORIES(SYSTEM ${MPI_CXX_INCLUDE_PATH} )
        MESSAGE(STATUS "Found MPI: ${MPI_CXX_LIBRARIES}")
    ELSE()
        SET(MPI_BUILTIN ON)
        MESSAGE(STATUS "Found MPI: built in")
	FIND_PROGRAM(HAVE_APRUN aprun)
	IF (HAVE_APRUN)
	    # Probably on Cray
            SET(MPIEXEC "aprun" CACHE STRING "MPI job launching command")
	    SET(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING
                "MPI job launcher flag to specify number of processes")
	ELSE()
            SET(MPIEXEC "mpirun" CACHE STRING "MPI job launching command")
	    SET(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING
                "MPI job launcher flag to specify number of processes")
	ENDIF()
	MARK_AS_ADVANCED(MPIEXEC)
	MARK_AS_ADVANCED(MPIEXEC_NUMPROC_FLAG)
	UNSET(HAVE_APRUN CACHE)
    ENDIF()

    # Add compiler definition for MPI type. This is needed by the Tester
    # when defining the hostfile to ensure it is formatted correctly.
    IF ("${MPI_CXX_LIBRARY_VERSION_STRING}" MATCHES "MPICH")
	    ADD_DEFINITIONS(-DNEKTAR_MPI_TYPE=1)
    ELSE()
	    ADD_DEFINITIONS(-DNEKTAR_MPI_TYPE=0)
    ENDIF()

    ADD_DEFINITIONS(-DNEKTAR_USE_MPI)

    # Stream-ordered device communication through the MPIX_Stream enqueue
    # extension of MPICH (>= 4.1, built with CUDA or HIP support). The SYCL
    # back-ends reach the CUDA or HIP stream behind their queues.
    IF (NEKTAR_ENABLE_DEVICE STREQUAL "CUDA" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "HIP" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
        OPTION(NEKTAR_USE_MPIX_STREAM
            "Enqueue device communication on streams with MPIX_Stream." OFF)
        MARK_AS_ADVANCED(NEKTAR_USE_MPIX_STREAM)

        IF (NEKTAR_USE_MPIX_STREAM)
            INCLUDE(CheckCXXSymbolExists)
            SET(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})
            SET(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
            CHECK_CXX_SYMBOL_EXISTS(MPIX_Allreduce_enqueue mpi.h
                HAVE_MPIX_ALLREDUCE_ENQUEUE)
            UNSET(CMAKE_REQUIRED_INCLUDES)
            UNSET(CMAKE_REQUIRED_LIBRARIES)

            IF (NOT HAVE_MPIX_ALLREDUCE_ENQUEUE)
                MESSAGE(FATAL_ERROR "NEKTAR_USE_MPIX_STREAM requires an MPI "
                    "providing MPIX_Allreduce_enqueue (MPICH >= 4.1).")
            ENDIF()

            ADD_DEFINITIONS(-DNEKTAR_USE_MPIX_STREAM)
            MESSAGE(STATUS "Using MPIX_Stream for device communication")
        ENDIF()
    ENDIF()

    # Stream-ordered device communication through the collective
    # communication library of the device: NCCL (CUDA), RCCL (HIP) or oneCCL
    # (Intel, with DPC++). MPI still sets up the communicators and carries
    # everything else.
    IF (NEKTAR_ENABLE_DEVICE STREQUAL "CUDA" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "HIP" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP" OR
        NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-Intel")
        OPTION(NEKTAR_USE_CCL
            "Run device communication with NCCL, RCCL or oneCCL." OFF)
        MARK_AS_ADVANCED(NEKTAR_USE_CCL)

        IF (NEKTAR_USE_CCL)
            IF (NEKTAR_USE_MPIX_STREAM)
                MESSAGE(FATAL_ERROR "NEKTAR_USE_CCL and "
                    "NEKTAR_USE_MPIX_STREAM cannot be enabled together.")
            ENDIF()

            # Besides the system paths, the library is looked for under
            # CCL_ROOT, then the environment variables its installation or the
            # vendor's environment scripts set, then its usual install
            # locations. Setting CCL_INCLUDE_DIR and CCL_LIBRARY skips the
            # search.
            SET(CCL_ROOT "" CACHE PATH
                "Install prefix of NCCL, RCCL or oneCCL, searched first.")
            MARK_AS_ADVANCED(CCL_ROOT)

            IF (NEKTAR_ENABLE_DEVICE MATCHES "CUDA$")
                SET(CCL_NAME "NCCL")
                SET(CCL_HEADER nccl.h)
                SET(CCL_LIBNAME nccl)
                SET(CCL_ENV NCCL_ROOT NCCL_HOME NCCL_DIR NCCL_PATH CUDA_HOME
                    CUDA_PATH)
                # NCCL alongside the CUDA toolkit, then the copies the NVIDIA
                # HPC SDK ships with its communication libraries, per CUDA
                # version (comm_libs/<version>/nccl) or not. The SDK's prefix
                # is NVHPC_ROOT, or the part before cuda/ or comm_libs/ in the
                # paths of the CUDA toolkit, MPI or compilers it provides.
                SET(CCL_DEFAULTS ${CUDAToolkit_LIBRARY_ROOT})
                SET(CCL_NVHPC_ROOTS)
                IF (DEFINED ENV{NVHPC_ROOT})
                    LIST(APPEND CCL_NVHPC_ROOTS $ENV{NVHPC_ROOT})
                ENDIF()
                FOREACH (CCL_PATH ${CUDAToolkit_LIBRARY_ROOT}
                         ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES}
                         ${CMAKE_CXX_COMPILER} ${CMAKE_C_COMPILER})
                    IF (CCL_PATH MATCHES "^(.*)/(cuda|comm_libs)/")
                        LIST(APPEND CCL_NVHPC_ROOTS ${CMAKE_MATCH_1})
                    ENDIF()
                ENDFOREACH()
                IF (CCL_NVHPC_ROOTS)
                    LIST(REMOVE_DUPLICATES CCL_NVHPC_ROOTS)
                ENDIF()
                FOREACH (CCL_NVHPC ${CCL_NVHPC_ROOTS})
                    SET(CCL_CUDA_VERSION
                        ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
                    FILE(GLOB CCL_NVHPC_NCCL ${CCL_NVHPC}/comm_libs/*/nccl)
                    LIST(SORT CCL_NVHPC_NCCL)
                    LIST(REVERSE CCL_NVHPC_NCCL)
                    LIST(APPEND CCL_DEFAULTS
                        ${CCL_NVHPC}/comm_libs/${CCL_CUDA_VERSION}/nccl
                        ${CCL_NVHPC_NCCL} ${CCL_NVHPC}/comm_libs/nccl)
                ENDFOREACH()
            ELSEIF (NEKTAR_ENABLE_DEVICE MATCHES "HIP$")
                SET(CCL_NAME "RCCL")
                SET(CCL_HEADER rccl/rccl.h)
                SET(CCL_LIBNAME rccl)
                SET(CCL_ENV RCCL_ROOT RCCL_HOME ROCM_PATH ROCM_HOME HIP_PATH)
                # The default ROCm, then the versioned ones, newest first.
                FILE(GLOB CCL_DEFAULTS /opt/rocm-*)
                LIST(SORT CCL_DEFAULTS)
                LIST(REVERSE CCL_DEFAULTS)
                LIST(INSERT CCL_DEFAULTS 0 /opt/rocm)
            ELSE()
                SET(CCL_NAME "oneCCL")
                IF (NOT NEKTAR_SYCL_COMPILER STREQUAL "DPC++")
                    MESSAGE(FATAL_ERROR "NEKTAR_USE_CCL with SYCL-Intel "
                        "requires DPC++, whose SYCL objects oneCCL takes.")
                ENDIF()
                SET(CCL_HEADER oneapi/ccl.hpp)
                SET(CCL_LIBNAME ccl)
                # CCL_ROOT is set by the oneAPI environment scripts.
                SET(CCL_ENV CCL_ROOT ONECCL_ROOT)
                SET(CCL_DEFAULTS)
                IF (DEFINED ENV{ONEAPI_ROOT})
                    LIST(APPEND CCL_DEFAULTS $ENV{ONEAPI_ROOT}/ccl/latest)
                ENDIF()
                LIST(APPEND CCL_DEFAULTS /opt/intel/oneapi/ccl/latest)
            ENDIF()

            SET(CCL_HINTS ${CCL_ROOT})
            FOREACH (CCL_VAR ${CCL_ENV})
                IF (DEFINED ENV{${CCL_VAR}})
                    LIST(APPEND CCL_HINTS $ENV{${CCL_VAR}})
                ENDIF()
            ENDFOREACH()
            LIST(APPEND CCL_HINTS ${CCL_DEFAULTS})
            IF (CCL_HINTS)
                LIST(REMOVE_DUPLICATES CCL_HINTS)
            ENDIF()

            # Older oneCCL installs keep the SYCL build in cpu_gpu_dpcpp.
            FIND_PATH(CCL_INCLUDE_DIR ${CCL_HEADER} HINTS ${CCL_HINTS}
                PATH_SUFFIXES include include/cpu_gpu_dpcpp)
            FIND_LIBRARY(CCL_LIBRARY ${CCL_LIBNAME} HINTS ${CCL_HINTS}
                PATH_SUFFIXES lib lib64 lib/cpu_gpu_dpcpp)
            MARK_AS_ADVANCED(CCL_INCLUDE_DIR CCL_LIBRARY)

            IF (NOT CCL_INCLUDE_DIR OR NOT CCL_LIBRARY)
                MESSAGE(FATAL_ERROR "NEKTAR_USE_CCL requires ${CCL_NAME}, "
                    "which was not found: set CCL_ROOT to its install prefix, "
                    "or CCL_INCLUDE_DIR and CCL_LIBRARY.")
            ENDIF()

            # The point-to-point and group operations of oneCCL carry the
            # exchanges, and are recent.
            IF (CCL_NAME STREQUAL "oneCCL")
                INCLUDE(CheckCXXSourceCompiles)
                SET(CMAKE_REQUIRED_INCLUDES ${CCL_INCLUDE_DIR})
                SET(CMAKE_REQUIRED_LIBRARIES ${CCL_LIBRARY})
                CHECK_CXX_SOURCE_COMPILES("
                    #include <oneapi/ccl.hpp>
                    int main()
                    {
                        ccl::group_start();
                        ccl::group_end();
                        const ccl::pt2pt_attr &attr = ccl::default_pt2pt_attr;
                        (void)attr;
                        return 0;
                    }" NEKTAR_HAVE_ONECCL_PT2PT)
                UNSET(CMAKE_REQUIRED_INCLUDES)
                UNSET(CMAKE_REQUIRED_LIBRARIES)

                IF (NOT NEKTAR_HAVE_ONECCL_PT2PT)
                    MESSAGE(FATAL_ERROR "NEKTAR_USE_CCL requires a oneCCL "
                        "with point-to-point and group operations, such as "
                        "2021.17.")
                ENDIF()
            ENDIF()

            INCLUDE_DIRECTORIES(SYSTEM ${CCL_INCLUDE_DIR})
            ADD_DEFINITIONS(-DNEKTAR_USE_CCL)
            MESSAGE(STATUS "Using ${CCL_NAME} for device communication: "
                "${CCL_LIBRARY} (headers in ${CCL_INCLUDE_DIR})")
        ENDIF()
    ENDIF()

    IF (THIRDPARTY_BUILD_GSMPI)
        INCLUDE(ExternalProject)
        THIRDPARTY_LIBRARY(GSMPI_LIBRARY STATIC gsmpi DESCRIPTION "GSMPI Library")
        THIRDPARTY_LIBRARY(XXT_LIBRARY STATIC xxt DESCRIPTION "XXT Library")
        MARK_AS_ADVANCED(GSMPI_LIBRARY)
        MARK_AS_ADVANCED(XXT_LIBRARY)

    	IF(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
            UNSET(PATCH CACHE)
            FIND_PROGRAM(PATCH patch)
            IF(NOT PATCH)
	        MESSAGE(FATAL_ERROR
	            "'patch' tool for modifying files not found. Cannot build gsmpi.")
            ENDIF()
	    SET(GSMPI_PATCH_COMMAND ${PATCH} -p1 -f < ${PROJECT_SOURCE_DIR}/cmake/thirdparty-patches/gsmpi-1.2.1_2-cmake4.0.patch)
        ENDIF()

        EXTERNALPROJECT_ADD(
            gsmpi-1.2.1_2
            URL ${TPURL}/gsmpi-1.2.1_2.tar.bz2
            URL_MD5 3690b4a658324cd5ad17cf8f6115fb5d
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            SOURCE_DIR ${TPSRC}/gsmpi-1.2.1_2
            BINARY_DIR ${TPBUILD}/gsmpi-1.2.1_2
            TMP_DIR ${TPBUILD}/gsmpi-1.2.1_2-tmp
            INSTALL_DIR ${TPDIST}
            BUILD_BYPRODUCTS ${GSMPI_LIBRARY} ${XXT_LIBRARY}
	    PATCH_COMMAND ${GSMPI_PATCH_COMMAND}
            CONFIGURE_COMMAND
                ${CMAKE_COMMAND}
                ${NEKTAR_EXTERNAL_PROJECT_CMAKE_GENERATOR_ARGS}
                -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
                -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                ${TPSRC}/gsmpi-1.2.1_2
        )
        MESSAGE(STATUS "Build GSMPI: ${GSMPI_LIBRARY}")
        MESSAGE(STATUS "Build XXT: ${XXT_LIBRARY}")

        ADD_DEPENDENCIES(thirdparty gsmpi-1.2.1_2)
    ELSE (THIRDPARTY_BUILD_GSMPI)
        MESSAGE(FATAL_ERROR "Must build GSMPI and XXT")
    ENDIF (THIRDPARTY_BUILD_GSMPI)
ENDIF( NEKTAR_USE_MPI )
