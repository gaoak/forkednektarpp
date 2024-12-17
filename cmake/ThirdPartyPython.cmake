########################################################################
#
# ThirdParty configuration for Nektar++
#
# Python interfaces
#
########################################################################

IF (NEKTAR_BUILD_PYTHON)
    IF (NOT DEFINED NEKTAR_PYTHON_EXECUTABLE)
        SET(NEKTAR_PYTHON_EXECUTABLE "" CACHE INTERNAL "")
    ENDIF()

    # If someone is using a newer variable name (from FindPython.cmake), set
    # this instead to the older variable name from PythonInterp.
    IF (DEFINED Python_EXECUTABLE)
        SET(PYTHON_EXECUTABLE ${Python_EXECUTABLE} CACHE INTERNAL "")
    ENDIF()

    IF (NOT PYTHON_EXECUTABLE STREQUAL NEKTAR_PYTHON_EXECUTABLE)
        unset(PYTHON_INCLUDE_DIR CACHE)
        unset(PYTHON_LIBRARY CACHE)
        unset(PYTHON_LIBRARY_DEBUG CACHE)
        unset(PYTHON_LIBRARIES CACHE)
        unset(PYTHONLIBS_VERSION_STRING CACHE)
        unset(BOOST_PYTHON_LIB CACHE)
        unset(BOOST_NUMPY_LIB CACHE)
    ENDIF()

    FIND_PACKAGE(PythonInterp REQUIRED)
    FIND_PACKAGE(PythonLibsNew REQUIRED)

    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DWITH_PYTHON)

    MESSAGE(STATUS "Searching for Python:")
    MESSAGE(STATUS "-- Found interpreter: ${PYTHON_EXECUTABLE}")
    MESSAGE(STATUS "-- Found development library: ${PYTHON_LIBRARIES}")

    # Build directory for binary installations
    SET(NEKPY_BASE_DIR "${CMAKE_BINARY_DIR}/python" CACHE INTERNAL "")
    FILE(MAKE_DIRECTORY ${NEKPY_BASE_DIR})
    FILE(MAKE_DIRECTORY ${NEKPY_BASE_DIR}/NekPy)

    CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/python/setup.py.in ${NEKPY_BASE_DIR}/setup.py)
    CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/python/setup.cfg.in ${NEKPY_BASE_DIR}/setup.cfg)
    CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/python/pyproject.toml.in ${NEKPY_BASE_DIR}/pyproject.toml)

    ADD_CUSTOM_TARGET(nekpy-install-user
        DEPENDS _MultiRegions
        COMMAND ${PYTHON_EXECUTABLE} -m pip install --user .
        WORKING_DIRECTORY ${NEKPY_BASE_DIR})

    ADD_CUSTOM_TARGET(nekpy-install-dev
        DEPENDS _MultiRegions
        COMMAND ${PYTHON_EXECUTABLE} -m pip install --user -e .
        WORKING_DIRECTORY ${NEKPY_BASE_DIR})

    ADD_CUSTOM_TARGET(nekpy-install-system
        DEPENDS _MultiRegions
        COMMAND ${PYTHON_EXECUTABLE} -m pip install .
        WORKING_DIRECTORY ${NEKPY_BASE_DIR})

    EXTERNALPROJECT_ADD(
        pybind11
        PREFIX ${TPSRC}
        URL ${TPURL}/pybind11-sh-3.0.0.zip
        URL_MD5 4583fbe042df34723c96cf51368d6cef
        STAMP_DIR ${TPBUILD}/stamp
        DOWNLOAD_DIR ${TPSRC}
        SOURCE_DIR ${TPSRC}/pybind11
        BINARY_DIR ${TPBUILD}/pybind11
        TMP_DIR ${TPBUILD}/pybind11-tmp
        INSTALL_DIR ${TPDIST}
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -G ${CMAKE_GENERATOR}
            -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
            -DPYBIND11_TEST=OFF
            ${TPSRC}/pybind11)

     # Add third-party include to include path.
     INCLUDE_DIRECTORIES(${TPDIST}/include)

     ADD_DEPENDENCIES(thirdparty pybind11)

     FILE(WRITE ${NEKPY_BASE_DIR}/NekPy/__init__.py "# Adjust dlopen flags to avoid OpenMPI issues
try:
    import DLFCN as dl
    import sys
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
except ImportError:
    pass")
ENDIF()
