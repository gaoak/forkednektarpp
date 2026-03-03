IF(NEKTAR_BUILD_REDESIGN)
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

        # Set default data types
        IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        SET(DataTypes double)
        ENDIF()
        IF(NEKTAR_ENABLE_SINGLE_PRECISION)
            SET(DataTypes ${DataTypes} float)
        ENDIF()

        # Set compilation for Riemann solvers
        SET(SolverTypes Average AUSM0 AUSM1 AUSM2 AUSM3 HLL HLLC LaxFriedrichs Roe)

        SET(name CompressibleSolver)
        SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers)

        SET(NAME_HEADERS ${name}_HEADERS)
        SET(NAME_SOURCES ${name}_SOURCES)
        SET(OPERATOR ${name})

        # These operator have an additional argument for the FieldState
        # which requires two delcarations.
        SET(CONFIG_FILE ${dir}/OperatorFactoryDec.cpp.in)

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
                    SET(FactoryDeclName ${name}${SolverType}${ExecSpace}${TData}.${SOURCE_EXT})

                    # Get the source and header files for this operator,
                    # execution space, and implementation.
                    SET(CURRENT_HEADERS ${name}${SolverType}_${ExecSpace}_HEADERS)
                    SET(CURRENT_SOURCES ${name}${SolverType}_${ExecSpace}_SOURCES)
                        
                    # Get the headers matching the name and execution space
                    FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                        ${dir}/${name}${SolverType}${ExecSpace}*.hpp)

                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}OpImpl.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")

                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${SolverType}SolverKernels.hpp")
                    SET(KERNEL_HEADER "#include \"${include_file}\"")

                    # Create the cpp file from the implementation file.
                    CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                    # Add the respective cpp file.
                    SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
                    SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                    SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})

                ENDFOREACH()
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

        # add operators manually 
        LINK_LIBRARIES(Operators)

        # Make list visible to subdirectories
        SET(RIEMANN_FACTORY_SOURCES "${OPERATORS_SOURCES}" CACHE INTERNAL "Generated Riemann factory sources")

        IF(NEKTAR_BUILD_UNIT_TESTS)
          ADD_SUBDIRECTORY(RiemannSolvers/UnitTests)
        ENDIF()

        ADD_SUBDIRECTORY(RiemannSolvers/Profilers)

    ENDIF(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_RE)
ENDIF(NEKTAR_BUILD_REDESIGN)
