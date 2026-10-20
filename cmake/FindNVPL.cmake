# Find the NVIDIA Performance Libraries (NVPL) BLAS and LAPACK, using the
# sequential LP64 interface. Sets NVPL_FOUND, NVPL_LAPACK, NVPL_BLAS,
# NVPL_BLAS_CORE and NVPL_INCLUDE_DIR. The CMake package config is used when
# available; otherwise the libraries are searched for directly, so that
# installations without the config files (or only on LD_LIBRARY_PATH, as set
# by environment modules) are also found.

SET(NVPL_HINTS $ENV{nvpl_ROOT} $ENV{NVPL_ROOT} $ENV{NVPL_HOME} $ENV{NVPL_DIR})

# Search again for anything found by an earlier configure that has since gone
# (e.g. an unloaded HPC SDK).
FOREACH(var NVPL_LAPACK NVPL_BLAS NVPL_BLAS_CORE NVPL_INCLUDE_DIR)
    IF(${var} AND NOT EXISTS "${${var}}")
        UNSET(${var} CACHE)
    ENDIF()
ENDFOREACH()

# NVPL is bundled with the NVIDIA HPC SDK on aarch64. Locate the SDK from its
# environment module (NVHPC_ROOT) or from the nvc/nvc++ compiler in use
# (<root>/compilers/bin), and add every directory below it holding NVPL BLAS.
SET(NVHPC_ROOTS $ENV{NVHPC_ROOT})
FOREACH(NVHPC_COMPILER ${CMAKE_CXX_COMPILER} ${CMAKE_C_COMPILER})
    IF(NVHPC_COMPILER MATCHES "/compilers/bin/nvc")
        GET_FILENAME_COMPONENT(NVHPC_COMPILER ${NVHPC_COMPILER} REALPATH)
        GET_FILENAME_COMPONENT(NVHPC_COMPILER_ROOT
            ${NVHPC_COMPILER}/../../.. ABSOLUTE)
        LIST(APPEND NVHPC_ROOTS ${NVHPC_COMPILER_ROOT})
    ENDIF()
ENDFOREACH()
FOREACH(NVHPC_ROOT_DIR ${NVHPC_ROOTS})
    FILE(GLOB NVHPC_NVPL_LIBS
        ${NVHPC_ROOT_DIR}/*/lib*/libnvpl_blas_lp64_seq.so
        ${NVHPC_ROOT_DIR}/*/*/lib*/libnvpl_blas_lp64_seq.so
        ${NVHPC_ROOT_DIR}/*/*/*/lib*/libnvpl_blas_lp64_seq.so
        ${NVHPC_ROOT_DIR}/*/*/*/*/lib*/libnvpl_blas_lp64_seq.so)
    FOREACH(NVHPC_NVPL_LIB ${NVHPC_NVPL_LIBS})
        GET_FILENAME_COMPONENT(NVHPC_NVPL_LIB ${NVHPC_NVPL_LIB} DIRECTORY)
        GET_FILENAME_COMPONENT(NVHPC_NVPL_PREFIX ${NVHPC_NVPL_LIB} DIRECTORY)
        LIST(APPEND NVPL_HINTS ${NVHPC_NVPL_PREFIX})
    ENDFOREACH()
ENDFOREACH()

FIND_PACKAGE(nvpl QUIET COMPONENTS blas lapack HINTS ${NVPL_HINTS})

STRING(REPLACE ":" ";" NVPL_LIB_ENV "$ENV{LD_LIBRARY_PATH}:$ENV{LIBRARY_PATH}")
STRING(REPLACE ":" ";" NVPL_INC_ENV "$ENV{CPATH}:$ENV{CPLUS_INCLUDE_PATH}")

FIND_LIBRARY(NVPL_LAPACK
    NAMES nvpl_lapack_lp64_seq
    HINTS ${nvpl_lapack_LIBRARY_DIR} ${NVPL_HINTS} ${NVPL_LIB_ENV}
    PATH_SUFFIXES lib lib64
)
FIND_LIBRARY(NVPL_BLAS
    NAMES nvpl_blas_lp64_seq
    HINTS ${nvpl_blas_LIBRARY_DIR} ${NVPL_HINTS} ${NVPL_LIB_ENV}
    PATH_SUFFIXES lib lib64
)

IF (NVPL_BLAS)
    GET_FILENAME_COMPONENT(NVPL_BLAS_DIR ${NVPL_BLAS} DIRECTORY)
ENDIF()

FIND_LIBRARY(NVPL_BLAS_CORE
    NAMES nvpl_blas_core
    HINTS ${NVPL_BLAS_DIR} ${nvpl_blas_LIBRARY_DIR}
)
FIND_PATH(NVPL_INCLUDE_DIR
    NAMES nvpl_blas_cblas.h
    HINTS ${nvpl_blas_INCLUDE_DIR} ${NVPL_BLAS_DIR}/../include ${NVPL_HINTS}
          ${NVPL_INC_ENV}
    PATH_SUFFIXES include
)

IF (NVPL_LAPACK AND NVPL_BLAS AND NVPL_INCLUDE_DIR)
    SET(NVPL_FOUND TRUE)
ELSE()
    SET(NVPL_FOUND FALSE)
ENDIF()

MARK_AS_ADVANCED(NVPL_LAPACK)
MARK_AS_ADVANCED(NVPL_BLAS)
MARK_AS_ADVANCED(NVPL_BLAS_CORE)
MARK_AS_ADVANCED(NVPL_INCLUDE_DIR)
