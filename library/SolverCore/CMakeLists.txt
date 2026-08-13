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


IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    # Set empty sources and top-level headers
    set(SOLVER_CORE_SOURCES
        SolverCore.cpp
    )
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

    # Set SolverCore operators
    FILE(GLOB children Advection/* Advection/AdvectionWeakDG/* Diffusion/* Diffusion/DiffusionIP/*)

    SET(OPNAMESPACE "Nektar::SolverCore::detail")
    FOREACH(child ${children})
        MESSAGE("Child : " "${child}")
        ADD_OPERATOR(${child} OPNAMESPACE SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES)
    ENDFOREACH()

    # Set compilation for Riemann solvers
    SET(SolverTypes Upwind)
    SET(name RiemannSolver)
    SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers)
    SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/SolverCore/RiemannSolvers/RiemannOperatorFactoryDec.cpp.in)

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)
    SET(OPERATOR ${name})
    FOREACH (SolverType IN LISTS SolverTypes)
        # Loop through each possible execution space.
        FOREACH (ExecSpace IN LISTS ExecSpaces)

            # Set file extension.
            SET(HEADER_EXT hpp)
            IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                SET(SOURCE_EXT cu)
            ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                SET(SOURCE_EXT hip)
            ELSE()
                SET(SOURCE_EXT cpp)
            ENDIF()

            # Loop through each possible data type.
            FOREACH (TData IN LISTS DataTypes)
                # Set up the name used for the .cpp declaration file.
                SET(FactoryDeclName src/${name}${SolverType}${ExecSpace}${TData}.${SOURCE_EXT})

                # Get the source and header files for this operator,
                # execution space, and implementation.
                SET(CURRENT_HEADERS ${name}${SolverType}_${ExecSpace}_HEADERS)
                SET(CURRENT_SOURCES ${name}${SolverType}_${ExecSpace}_SOURCES)
                    
                # Get the headers matching the name and execution space
                FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}${SolverType}${ExecSpace}*.hpp)

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}OpImpl.hpp")
                SET(IMPL_HEADER "#include \"SolverCore/${include_file}\"")

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${SolverType}SolverKernels.hpp")
                SET(KERNEL_HEADER "#include \"SolverCore/${include_file}\"")

                # Create the cpp file from the implementation file.
                CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                # Add the respective cpp file.
                SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
                SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})

            ENDFOREACH()
        ENDFOREACH()
    ENDFOREACH()

    SET(SOLVER_CORE_HEADERS
        ${SOLVER_CORE_HEADERS}
        ${${NAME_HEADERS}}
    )
    SET(SOLVER_CORE_SOURCES
        ${SOLVER_CORE_SOURCES}
        ${${NAME_SOURCES}}
    )
    add_definitions(-DSOLVER_CORE_EXPORTS)

    ADD_NEKTAR_LIBRARY(SolverCore
        SOURCES ${SOLVER_CORE_SOURCES}
        HEADERS ${SOLVER_CORE_HEADERS}
        DEPENDS MultiRegions Operators
        SUMMARY "Nektar++ SolverCore library"
        DESCRIPTION "Minimal standalone solver core for device support solvers.")

    IF(NEKTAR_BUILD_UNIT_TESTS)
        ADD_SUBDIRECTORY(RiemannSolvers/UnitTests)
    ENDIF()
ENDIF(NEKTAR_ENABLE_DEVICE_SUPPORT)
