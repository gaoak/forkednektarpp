function(add_component_sources HEADER_VAR SOURCE_VAR DIR)
    file(GLOB HEADER_FILES
            CONFIGURE_DEPENDS
            "${DIR}/*.h"
            "${DIR}/*.hpp"
    )

    file(GLOB SOURCE_FILES
            CONFIGURE_DEPENDS
            "${DIR}/*.cpp"
            "${DIR}/*.cxx"
            "${DIR}/*.cc"
    )

    if(NOT HEADER_FILES AND NOT SOURCE_FILES)
        message(WARNING
                "add_component_sources: no files found in directory: ${DIR}"
        )
    endif()

    set(RELATIVE_HEADER_FILES)
    foreach(FILE_PATH ${HEADER_FILES})
        file(RELATIVE_PATH REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${FILE_PATH}")
        list(APPEND RELATIVE_HEADER_FILES "${REL_PATH}")
    endforeach()

    set(${HEADER_VAR}
            ${${HEADER_VAR}}
            ${RELATIVE_HEADER_FILES}
            PARENT_SCOPE
    )

    set(${SOURCE_VAR}
            ${${SOURCE_VAR}}
            ${SOURCE_FILES}
            PARENT_SCOPE
    )
endfunction()


IF(NEKTAR_BUILD_REDESIGN)
    # Set empty sources and top-level headers
    set(SOLVER_CORE_SOURCES)
    set(SOLVER_CORE_HEADERS
        SolverCore.hpp
        SolverCoreDeclspec.h
    )

    # Automatically add all headers and sources in given directory
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Core")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Driver")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/EquationSystems")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Filters")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Forcing")

    # Debug: Print any detected headers and sources
#    message(STATUS "SolverCore headers: ${SOLVER_CORE_HEADERS}")
#    message(STATUS "SolverCore sources: ${SOLVER_CORE_SOURCES}")

    add_definitions(-DSOLVER_CORE_EXPORTS)

    ADD_NEKTAR_LIBRARY(SolverCore
        SOURCES ${SOLVER_CORE_SOURCES}
        HEADERS ${SOLVER_CORE_HEADERS}
        DEPENDS MultiRegions Operators
        SUMMARY "Nektar++ SolverCore library"
        DESCRIPTION "Minimal standalone solver core for redesign solvers.")
ENDIF(NEKTAR_BUILD_REDESIGN)
