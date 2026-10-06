IF (NEKTAR_USE_MKL)
    # Deal with MKL due to issues with FindBLAS VENDOR variable.
    INCLUDE (FindMKL)
ELSEIF(NEKTAR_USE_WIN32_LAPACK)
    INCLUDE (FindWin32Lapack)
ELSE()
    IF(NEKTAR_USE_SYSTEM_BLAS_LAPACK)
        SET(TEST_ENV $ENV{LAPACK_DIR})
        IF(NOT DEFINED LAPACK_DIR AND DEFINED TEST_ENV)
	       SET(LAPACK_DIR $ENV{LAPACK_DIR})
        ENDIF()

        SET(TEST_ENV $ENV{BLAS_DIR})
        IF(NOT DEFINED BLAS_DIR AND DEFINED TEST_ENV)
	       SET(BLAS_DIR $ENV{BLAS_DIR})
        ENDIF()

        SET(VENDOR Generic)
    ELSEIF(NEKTAR_USE_OPENBLAS)
        SET(TEST_ENV $ENV{OPENBLAS_HOME})
        IF(NOT DEFINED LAPACK_DIR AND DEFINED TEST_ENV)
	       SET(LAPACK_DIR $ENV{OPENBLAS_HOME})
        ENDIF()

        SET(VENDOR OpenBLAS)
    ELSEIF(NEKTAR_USE_FLEXIBLAS)
        SET(VENDOR FlexiBLAS)
    ELSEIF(NEKTAR_USE_ACCELERATE_FRAMEWORK)
        SET(VENDOR Apple)
    ELSEIF(NEKTAR_USE_ACML)
        SET(VENDOR ACML)
    ENDIF()

    IF(VENDOR)
        SET(BLA_VENDOR ${VENDOR})
        find_package(LAPACK QUIET)
        SET(BLAS_LAPACK ${LAPACK_LIBRARIES} CACHE INTERNAL "BLAS/LAPACK location")

        # A specific vendor was requested: do not silently fall back to the
        # ThirdParty reference BLAS/LAPACK, which is a different library.
        IF(NOT LAPACK_FOUND AND NOT VENDOR STREQUAL "Generic")
            MESSAGE(FATAL_ERROR "Could not find ${VENDOR} BLAS/LAPACK. "
                "Check that it is installed and visible to CMake, or disable "
                "the corresponding NEKTAR_USE_* option.")
        ENDIF()
    ENDIF()
ENDIF()
