#
# install-simsycl-deps.cmake
#
# SimSYCL pulls libenvpp with FetchContent, and libenvpp pulls fmt. Both are
# built as shared libraries, since SimSYCL itself has to be one (see
# ThirdPartySimSYCL.cmake), and libsimsycl.so records each in DT_NEEDED - but
# whether they are installed at all is up to each dependency's own CMake.
# Anything the install left behind in the FetchContent build trees is copied
# next to libsimsycl.so here, so that everything the loader needs sits in the
# one directory the Nektar RPATH names.
#
# Run with -DBINARY_DIR=<SimSYCL build dir> -DDEST=<ThirdParty dist lib dir>.
#

FILE(GLOB SIMSYCL_DEP_LIBS
    ${BINARY_DIR}/_deps/*/lib*.so*
    ${BINARY_DIR}/_deps/*/lib*.dylib*)

FOREACH(lib ${SIMSYCL_DEP_LIBS})
    GET_FILENAME_COMPONENT(name ${lib} NAME)
    IF (NOT EXISTS ${DEST}/${name})
        MESSAGE(STATUS "SimSYCL: installing dependency ${name}")
        FILE(COPY ${lib} DESTINATION ${DEST})
    ENDIF()
ENDFOREACH()
