IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    # Macro
    MACRO(SET_ALL_SOLVER_TEST solver session)
        FOREACH (ExecSpace IN LISTS ExecSpaces)
            # Add SumFac test for all execution spaces
            SET(Implementation SumFac)
            ADD_TEST(NAME ${solver}_${session}_${ExecSpace}_${Implementation}
                 COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
                  --opExecSpace=${ExecSpace} --opImpl=${Implementation})

            # Check if Device or not and set implementation for available algorithm
            IF( ("${ExecSpace}" STREQUAL "Serial") OR ("${ExecSpace}" STREQUAL "AVX") )
                SET(Implementation StdMat)
            ELSEIF( ("${ExecSpace}" STREQUAL "Device") )
                SET(Implementation SumFacTOP)
            ENDIF()

            # Add test for either Serial/AVX + StdMat or Device + SumFacTOP
            ADD_TEST(
                    NAME ${solver}_${session}_${ExecSpace}_${Implementation}
                    COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
                    --opExecSpace=${ExecSpace} --opImpl=${Implementation})
        ENDFOREACH ()
    ENDMACRO()

    # Reduced variant of SET_ALL_SOLVER_TEST for sessions where the varied
    # parameter (basis type, mesh, dimension, ...) is orthogonal to
    # opExecSpace/opImpl dispatch: a single Serial+SumFac smoke test is
    # enough to catch a regression in that parameter, since execspace/impl
    # correctness is already covered by the full matrix on a representative
    # sibling session. See TEST_SUITE_NOTES.md section 9.
    MACRO(SET_SERIAL_SUMFAC_SOLVER_TEST solver session)
        ADD_TEST(NAME ${solver}_${session}_Serial_SumFac
             COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
              --opExecSpace=Serial --opImpl=SumFac)
    ENDMACRO()

ENDIF()

MACRO(SOLVER_COMPONENT_NAME name var)
    # Lower-case solver name
    STRING(TOLOWER ${name} ${var})
    # Then replace ending "solver" with "-solver"
    STRING(REPLACE "solver" "-solver" ${var} ${${var}})
ENDMACRO()

MACRO(ADD_SOLVER_EXECUTABLE name)
    CMAKE_PARSE_ARGUMENTS(SOLVER "" "COMPONENT" "SOURCES;LIBRARY_SOURCES;DEPENDS" ${ARGN} )

    IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
        SET(SOLVER_TARGET_DEPENDS SolverUtils SolverCore)
    ELSE()
        SET(SOLVER_TARGET_DEPENDS SolverUtils)
    ENDIF()
    IF (SOLVER_DEPENDS)
        SET(SOLVER_TARGET_DEPENDS ${SOLVER_DEPENDS})
    ENDIF()

    IF (SOLVER_LIBRARY_SOURCES)
        ADD_LIBRARY(${name}-obj OBJECT ${SOLVER_LIBRARY_SOURCES})
        TARGET_LINK_LIBRARIES(${name}-obj PRIVATE ${NEKTAR_SYCL_DEPENDS})
        SET_PROPERTY(TARGET ${name}-obj PROPERTY POSITION_INDEPENDENT_CODE ON)
        ADD_DEPENDENCIES(${name}-obj ${SOLVER_TARGET_DEPENDS})
        LIST(APPEND SOLVER_SOURCES $<TARGET_OBJECTS:${name}-obj>)
        SET_COMMON_PROPERTIES(${name}-obj)
    ENDIF()

    IF (NOT SOLVER_COMPONENT)
        SOLVER_COMPONENT_NAME(${name} SOLVER_COMPONENT)
    ENDIF()

    ADD_NEKTAR_EXECUTABLE(${name}
            COMPONENT ${SOLVER_COMPONENT} DEPENDS ${SOLVER_TARGET_DEPENDS}
            SOURCES ${SOLVER_SOURCES})
ENDMACRO()

IF (NEKTAR_BUILD_SOLVER_LIBS)
    MACRO(ADD_SOLVER_LIBRARY name)
        CMAKE_PARSE_ARGUMENTS(SOLVERLIB "" "" "SOURCES;HEADERS;DEPENDS" ${ARGN} )

        # If an object library was already built using ADD_SOLVER_EXECUTABLE, reuse its objects
        IF (TARGET ${name}-obj)
            SET(SOLVERLIB_SRCS_USED $<TARGET_OBJECTS:${name}-obj>)
        ELSE()
            SET(SOLVERLIB_SRCS_USED ${SOLVERLIB_SOURCES})
        ENDIF()

        # Append "Lib" suffix to target name, otherwise it clashes with the executable (final output will still be called ${name})
        ADD_NEKTAR_LIBRARY(${name}
                TARGET_SUFFIX Lib
                LINK_MODE PRIVATE
                SOURCES ${SOLVERLIB_SRCS_USED}
                HEADERS ${SOLVERLIB_HEADERS}
                DEPENDS ${SOLVERLIB_DEPENDS}
                SUMMARY "Nektar++ ${name} solver library"
                DESCRIPTION "This library provides functionality used by the ${name} solver.")
    ENDMACRO()
ENDIF()

MACRO(ADD_SOLVER_UTILITY name solver)
    CMAKE_PARSE_ARGUMENTS(UTIL "USELIBRARY" "COMPONENT" "SOURCES" ${ARGN})

    IF (NOT UTIL_COMPONENT)
        SOLVER_COMPONENT_NAME(${solver} UTIL_COMPONENT)
    ENDIF()

    IF (UTIL_USELIBRARY)
        ADD_NEKTAR_EXECUTABLE(${name}
                COMPONENT ${UTIL_COMPONENT} DEPENDS SolverUtils
                SOURCES ${UTIL_SOURCES} $<TARGET_OBJECTS:${solver}-obj>)
    ELSE()
        ADD_NEKTAR_EXECUTABLE(${name}
                COMPONENT ${UTIL_COMPONENT} DEPENDS SolverUtils
                SOURCES ${UTIL_SOURCES})
    ENDIF()
ENDMACRO()

# Generate list of available subdirectories
FILE(GLOB dir_list "*")
FOREACH(dir ${dir_list})
    IF(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
        ADD_SUBDIRECTORY(${dir})
    ENDIF(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
ENDFOREACH(dir ${dir_list})
