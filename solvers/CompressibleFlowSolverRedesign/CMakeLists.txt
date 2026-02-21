CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_RE
    "Build the Compressible Flow Solver using redesign framework." ON
    "NEKTAR_BUILD_SOLVERS" OFF)

IF (NOT NEKTAR_BUILD_SOLVERS)
    SET(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_RE OFF CACHE INTERNAL "")
ENDIF()

IF(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_RE)

    # Default execution spaces.
    SET(ExecSpaces Serial)

    # Optional AVX execution space.
    IF(NEKTAR_ENABLE_SIMD)
        SET(ExecSpaces ${ExecSpaces} AVX)
    ENDIF()

    # Optional Device execution space.
    IF(NEKTAR_ENABLE_DEVICE)
        SET(ExecSpaces ${ExecSpaces} Device)
    ENDIF()

    # set default data types
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
    SET(DataTypes double)
    ENDIF()
    IF(NEKTAR_ENABLE_SINGLE_PRECISION)
        SET(DataTypes ${DataTypes} float)
    ENDIF()

    # Macro for each operator.
    MACRO(ADD_OPERATOR dir)

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
        SET(CONFIG_FILE OperatorFactoryDec.cpp.in)

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

            # Set execution space.
            IF("${ExecSpace}" STREQUAL "Serial")
                SET(EXEC_MEM_SPACE_TAG NektarSpaces::Serial)
            ELSEIF("${ExecSpace}" STREQUAL "AVX")
                SET(EXEC_MEM_SPACE_TAG NektarSpaces::AVX)
            ELSEIF("${ExecSpace}" STREQUAL "Device")
                SET(EXEC_MEM_SPACE_TAG NektarSpaces::Device)
            ENDIF()

            # Loop through each possible data type.
            FOREACH (TData IN LISTS DataTypes)
                # Set up the name used for the .cpp declaration file.
                SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})

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
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                # No implementation, skip.
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

        # Add this operator's implemenations to the global operator source
        # and header files.
        SET(OPERATORS_HEADERS
            ${OPERATORS_HEADERS}
            ${${NAME_HEADERS}}
            )
        SET(OPERATORS_SOURCES
            ${OPERATORS_SOURCES}
            ${${NAME_SOURCES}}
            )
    ENDMACRO()

    # Macro for single operator.
    MACRO(ADD_SINGLE_OPERATOR dir opname)

        SET(FactoryDeclName "${opname}${ExecSpace}${TData}.${SOURCE_EXT}")
        CONFIGURE_FILE(${dir}/${opname}FactoryDec.cpp.in
                        ${FactoryDeclName})
        SET(OPERATORS_SOURCES  ${OPERATORS_SOURCES}  ${FactoryDeclName})
        MESSAGE("Adding operator with a ${ExecSpace} implementation: "
                        "${FactoryDeclName}")
    ENDMACRO()

    # Loop through all of the directories which contain operators and add
    # their source and header files.
    FILE(GLOB children RiemannSolvers/*)

    FOREACH(child ${children})
    MESSAGE("Child : " "${child}")
        ADD_OPERATOR(${child})
    ENDFOREACH()
    MESSAGE("Sources 0 : " "${OPERATORS_SOURCES}")

    # add operators manually 

    #Serial double cases
    SET(TData double)

    SET(EXEC_MEM_SPACE_TAG NektarSpaces::Serial)
    SET(ExecSpace Serial)
    SET(SOURCE_EXT cpp)

    #AVX double cases
    IF (AVX IN_LIST ExecSpaces)
        SET(EXEC_MEM_SPACE_TAG NektarSpaces::AVX)
        SET(ExecSpace AVX)
        SET(SOURCE_EXT cpp)
    ENDIF()

    #Device double cases
    IF (Device IN_LIST ExecSpaces)
        SET(EXEC_MEM_SPACE_TAG NektarSpaces::Device)
        SET(ExecSpace Device)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

    ENDIF()

    MESSAGE("Sources 1 : " "${OPERATORS_SOURCES}")

    LINK_LIBRARIES(Operators)

    # Make list visible to subdirectories
    SET(RIEMANN_FACTORY_SOURCES "${OPERATORS_SOURCES}" CACHE INTERNAL "Generated Riemann factory sources")

    IF(NEKTAR_BUILD_UNIT_TESTS)
      ADD_SUBDIRECTORY(RiemannSolvers/UnitTests)
    ENDIF()

ENDIF(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_RE)
