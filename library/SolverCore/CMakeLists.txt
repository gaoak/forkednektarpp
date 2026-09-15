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
    # Macro for each operator.
    MACRO(ADD_TIME_OPERATOR dir)

        # Extract operator name form directory if it exists.
        IF(IS_DIRECTORY ${dir})
            GET_FILENAME_COMPONENT(name "${dir}" NAME)
        ELSE()
            CONTINUE()
        ENDIF()

        SET(NAME_HEADERS ${name}_HEADERS)
        SET(NAME_SOURCES ${name}_SOURCES)
        SET(OPERATOR ${name})

        # These operator have an additional argument for the FieldState
        # which requires two delcarations.
        IF (${name} STREQUAL "IMEXdirk")
             SET(CONFIG_FILE TimeOps/IMEXdirk/IMEXdirkOpFactoryDec.cpp.in)
        ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
             SET(CONFIG_FILE TimeOps/SDC/SDCOpFactoryDec.cpp.in)
        ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
             SET(CONFIG_FILE TimeOps/GEM/GEMOpFactoryDec.cpp.in)
        ELSE() 
             SET(CONFIG_FILE TimeOps/TimeOpFactoryDec.cpp.in)
        ENDIF()

        # Set timeintegration order
        SET(Variants Dummy)
        IF (${name} STREQUAL "RungeKutta")
            LIST(APPEND Variants SSP)
        ELSEIF (${name} STREQUAL "DIRK")
            LIST(APPEND Variants _ES)
        ELSEIF (${name} STREQUAL "CNAB")
            LIST(APPEND Variants Modified)
        ENDIF()

        FOREACH (Variant IN LISTS Variants)
            # Set timeintegration order
            IF (${name} STREQUAL "RungeKutta") 
                IF ("${Variant}" STREQUAL "Dummy")
                    SET(TimeIntegrationOrders 1 2 3 4 5)
                ELSEIF ("${Variant}" STREQUAL "SSP")
                    SET(TimeIntegrationOrders 1 2 3)
                ENDIF()
            ELSEIF (${name} STREQUAL "DIRK")
                IF ("${Variant}" STREQUAL "Dummy")
                    SET(TimeIntegrationOrders 1 2 3)
                ELSEIF ("${Variant}" STREQUAL "_ES")
                    SET(TimeIntegrationOrders 2 3 4)
                ENDIF()
            ELSEIF (${name} STREQUAL "IMEXdirk")
                SET(TimeIntegrationImpStages 1 1 1 2 2 2 3 4)
                SET(TimeIntegrationExpStages 1 2 2 2 3 3 4 4)
                SET(TimeIntegrationOrders 1 1 2 2 2 3 3 3)
            ELSEIF (${name} STREQUAL "CNAB")
                SET(TimeIntegrationOrders 2)
            ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
                SET(TimeIntegrationOrders -1)
            ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
                SET(TimeIntegrationOrders -1)
            ELSE()
                SET(TimeIntegrationOrders 1 2 3 4)
            ENDIF()

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

                IF ("${Variant}" STREQUAL "Dummy")
                    SET(Variant "")
                ENDIF()

                # Loop through each possible data type.
                FOREACH (TData IN LISTS DataTypes)
                    LIST(LENGTH TimeIntegrationOrders NINDEX)
                    MATH(EXPR NINDEX "${NINDEX} - 1")
                    FOREACH (INDEX RANGE ${NINDEX})
                        LIST(GET TimeIntegrationOrders ${INDEX} IntOrder)
                        # Set up the name used for the .cpp declaration file.
                        IF (${name} STREQUAL "IMEXdirk")
                            LIST(GET TimeIntegrationImpStages ${INDEX} ImpStage)
                            LIST(GET TimeIntegrationExpStages ${INDEX} ExpStage)
                            SET(FactoryDeclName src/${name}${Variant}${ExecSpace}${TData}${ImpStage}${ExpStage}Order${IntOrder}.${SOURCE_EXT})
                        ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
                            SET(FactoryDeclName src/${name}${ExecSpace}${TData}.${SOURCE_EXT})
                        ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
                            SET(FactoryDeclName src/${name}${ExecSpace}${TData}.${SOURCE_EXT})
                        ELSE()
                            SET(FactoryDeclName src/${name}${Variant}${ExecSpace}${TData}Order${IntOrder}.${SOURCE_EXT})
                        ENDIF()

                        # Get the source and header files for this operator,
                        # execution space, and implementation.
                        SET(CURRENT_HEADERS ${name}_${ExecSpace}_HEADERS)
                        SET(CURRENT_SOURCES ${name}_${ExecSpace}_SOURCES)

                        # Get the headers matching the name and execution space
                        FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                                ${dir}/${name}${ExecSpace}*.hpp)

                        # Found a impl header.
                        IF(EXISTS "${dir}/${name}OpImpl.hpp")
                            FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                                    "${dir}/${name}OpImpl.hpp")
                            SET(IMPL_HEADER "#include \"SolverCore/${include_file}\"")
                            MESSAGE("Adding operator with a ${ExecSpace} execution space: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")
                        ELSE()
                            CONTINUE() # This avoid creating a *.cpp file
                        ENDIF()

                        # Create the cpp file from the implementation file.
                        CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                        # Add the respective cpp file.
                        SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
                        SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                        SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})
                    ENDFOREACH()
                ENDFOREACH()
            ENDFOREACH()
        ENDFOREACH()

        # Add this operator's implemenations to the global operator source
        # and header files.
        SET(SOLVER_CORE_HEADERS
            ${SOLVER_CORE_HEADERS}
            ${${NAME_HEADERS}}
        )
        SET(SOLVER_CORE_SOURCES
            ${SOLVER_CORE_SOURCES}
            ${${NAME_SOURCES}}
        )
    ENDMACRO()

    # Set empty sources and top-level headers
    set(SOLVER_CORE_SOURCES
        SolverCore.cpp
        TimeOps/TimeOp.cpp
    )
    set(SOLVER_CORE_HEADERS
        SolverCore.hpp
        TimeOps/TimeOp.hpp
        SolverCoreDeclspec.h
    )

    # Automatically add all headers and sources in given directory
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Core")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Driver")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/EquationSystems")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Filters")
    add_component_sources(SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Forcing")

    # Set SolverCore operators
    FILE(GLOB children Advection/* Advection/AdvectionWeakDG/* Diffusion/* Diffusion/DiffusionIP/* MeanRemoval NormOps/* CFL/*
         GlobalLinSysOps/LinearSystems/* GlobalLinSysOps/LinearSolvers/* PreconOps/*)

    SET(OPNAMESPACE "Nektar::SolverCore::detail")
    FOREACH(child ${children})
        # MESSAGE("Child : " "${child}")
        ADD_OPERATOR(${child} OPNAMESPACE SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES)
    ENDFOREACH()

    # Set compilation for MeanRemoval operator
    ADD_BLOCK_OPERATOR("${CMAKE_CURRENT_SOURCE_DIR}/MeanRemoval" OPNAMESPACE SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES)

    # Set compilation for the Norm and CFL block operators
    FILE(GLOB children NormOps/* CFL/*)
    FOREACH(child ${children})
        ADD_BLOCK_OPERATOR(${child} OPNAMESPACE SOLVER_CORE_HEADERS SOLVER_CORE_SOURCES)
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

    # Loop through TimeIntegration directory which contain operators and add
    # their source and header files.
    FILE(GLOB children TimeOps/*)
    FOREACH(child ${children})
        ADD_TIME_OPERATOR(${child})
    ENDFOREACH()
    FILE(GLOB children TimeOps/SDC/*)
    FOREACH(child ${children})
        ADD_TIME_OPERATOR(${child})
    ENDFOREACH()
    FILE(GLOB children TimeOps/GEM/*)
    FOREACH(child ${children})
        ADD_TIME_OPERATOR(${child})
    ENDFOREACH()

    add_definitions(-DSOLVER_CORE_EXPORTS)

    ADD_NEKTAR_LIBRARY(SolverCore
        SOURCES ${SOLVER_CORE_SOURCES}
        HEADERS ${SOLVER_CORE_HEADERS}
        DEPENDS MultiRegions Operators
        SUMMARY "Nektar++ SolverCore library"
        DESCRIPTION "Minimal standalone solver core for device support solvers.")

    IF(NEKTAR_BUILD_UNIT_TESTS)
        ADD_SUBDIRECTORY(RiemannSolvers/UnitTests)
        ADD_SUBDIRECTORY(MeanRemoval/UnitTests)
    ENDIF()
ENDIF(NEKTAR_ENABLE_DEVICE_SUPPORT)
