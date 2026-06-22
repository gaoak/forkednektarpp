IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
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

        # Set Equation of State types 
        SET(EoSTypes IdealGasEoS VanDerWaalsEoS)

        ################################################################
        # Volume Flux operators
        SET(CFEVolFluxes  EulerVolumeFlux DiffusionCFEVolFlux DiffusionCFETraceFlux)

        FOREACH (name IN LISTS CFEVolFluxes)
            SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/${name})
            SET(NAME_HEADERS ${name}_HEADERS)
            SET(NAME_SOURCES ${name}_SOURCES)
            SET(OPERATOR ${name})  #Operator parameter for boiler plate code

            # These operator have an additional argument for Equation of state and
            # SovlerTypes which requires more  delcarations.
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign/VolumeFluxFactoryDec.cpp.in)

            FOREACH (EoSType IN LISTS EoSTypes)
                # Loop through each possible execution space.
                FOREACH (ExecSpace IN LISTS ExecSpaces)

                    # Set file extension.
                    SET(HEADER_EXT hpp)
                    IF("${ExecSpace}" STREQUAL "Device"
                           AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                            SET(SOURCE_EXT cu)
                    ELSEIF("${ExecSpace}" STREQUAL "Device"
                          AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                          SET(SOURCE_EXT hip)
                    ELSE()
                          SET(SOURCE_EXT cpp)
                    ENDIF()

                    # Loop through each possible data type.
                    FOREACH (TData IN LISTS DataTypes)
                        # Set up the name used for the .cpp declaration file.
                        SET(FactoryDeclName
                        ${name}${SolverType}${EoSType}${ExecSpace}${TData}.${SOURCE_EXT})

                        # Get the source and header files for this operator,
                        # execution space, and implementation.
                        SET(CURRENT_HEADERS ${name}_${ExecSpace}_HEADERS)
                        SET(CURRENT_SOURCES ${name}_${ExecSpace}_SOURCES)
                        
                        # Get the headers matching the name and execution space
                        FILE(GLOB ${CURRENT_HEADERS} RELATIVE
                        ${CMAKE_CURRENT_SOURCE_DIR}
                        ${dir}/${name}${SolverType}${ExecSpace}*.hpp)

                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                            "${dir}/${name}OpImpl.hpp")
                        SET(IMPL_HEADER "#include \"${include_file}\"")

                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                            "${dir}/SolverKernels.hpp")
                        SET(KERNEL_HEADER "#include \"${include_file}\"")

                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                            "${dir}/../EquationOfState/${EoSType}.hpp")
                        SET(EOS_HEADER "#include \"${include_file}\"")
                        SET(EOS_TYPE ${EoSType}) 
                    
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

        # Make list visible to subdirectories
        SET(VOLUE_FACTORY_SOURCES "${OPERATORS_SOURCES}" CACHE INTERNAL "Generated Volume flux factory sources")
        ENDFOREACH()

        ################################################################
        # Trace Flux operators
        SET(name CompressibleSolver)
        SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers)
        SET(NAME_HEADERS ${name}_HEADERS)
        SET(NAME_SOURCES ${name}_SOURCES)
        SET(OPERATOR ${name})  #Operator parameter for boiler plate code

        # Set compilation for Riemann solvers
        SET(SolverTypes  Average AUSM0 AUSM1 AUSM2 AUSM3 HLL HLLC LaxFriedrichs Roe)

        # These operator have an additional argument for Equation of state and
        # SovlerTypes which requires more  delcarations.
        SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign/RiemannSolvers/RiemannOperatorFactoryDec.cpp.in)

        FOREACH (EoSType IN LISTS EoSTypes)
           FOREACH (SolverType IN LISTS SolverTypes)
               # Loop through each possible execution space.
               FOREACH (ExecSpace IN LISTS ExecSpaces)
        
                   # Set file extension.
                   SET(HEADER_EXT hpp)
                   IF("${ExecSpace}" STREQUAL "Device"
                                     AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                       SET(SOURCE_EXT cu)
                   ELSEIF("${ExecSpace}" STREQUAL "Device"
                                         AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                       SET(SOURCE_EXT hip)
                   ELSE()
                       SET(SOURCE_EXT cpp)
                   ENDIF()

                   # Loop through each possible data type.
                   FOREACH (TData IN LISTS DataTypes)
                        # Set up the name used for the .cpp declaration file.
                        SET(FactoryDeclName
                             ${name}${SolverType}${EoSType}${ExecSpace}${TData}.${SOURCE_EXT})

                        # Get the source and header files for this operator,
                        # execution space, and implementation.
                        SET(CURRENT_HEADERS ${name}${SolverType}_${ExecSpace}_HEADERS)
                        SET(CURRENT_SOURCES ${name}${SolverType}_${ExecSpace}_SOURCES)
                        
                        # Get the headers matching the name and execution space
                        FILE(GLOB ${CURRENT_HEADERS} RELATIVE
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${dir}/${name}${SolverType}${ExecSpace}*.hpp)

                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                            "${dir}/${name}OpImpl.hpp")
                        SET(IMPL_HEADER "#include \"${include_file}\"")

                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                            "${dir}/${SolverType}SolverKernels.hpp")
                        SET(KERNEL_HEADER "#include \"${include_file}\"")

                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                            "${dir}/../EquationOfState/${EoSType}.hpp")
                        SET(EOS_HEADER "#include \"${include_file}\"")
                        SET(EOS_TYPE ${EoSType}) 
                    
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
        SET(OPERATORS_HEADERS
            ${OPERATORS_HEADERS}
            ${${NAME_HEADERS}}
            )
        SET(OPERATORS_SOURCES
            ${OPERATORS_SOURCES}
            ${${NAME_SOURCES}}
            )

        # Make list visible to subdirectories
        SET(RIEMANN_FACTORY_SOURCES "${OPERATORS_SOURCES}" CACHE INTERNAL "Generated Riemann factory sources")

        IF(NEKTAR_BUILD_UNIT_TESTS)
          ADD_SUBDIRECTORY(RiemannSolvers/UnitTests)
        ENDIF()


        ################################################################
        # Macro for each operator.
        MACRO(ADD_OPERATOR dir)

            IF(IS_DIRECTORY ${dir})
                GET_FILENAME_COMPONENT(name "${dir}" NAME)
                GET_FILENAME_COMPONENT(abs_dir "${dir}" ABSOLUTE)
            ELSE()
                CONTINUE()
            ENDIF()

            SET(NAME_HEADERS ${name}_HEADERS)
            SET(NAME_SOURCES ${name}_SOURCES)
            SET(OPERATOR ${name})

            # These operator have an additional argument for the FieldState
            # which requires two delcarations.
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/Operators/Common/OpFactoryDec.cpp.in)

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
                    SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})

                    # Get the source and header files for this operator,
                    # execution space, and implementation.
                    SET(CURRENT_HEADERS ${name}_${ExecSpace}_HEADERS)
                    SET(CURRENT_SOURCES ${name}_${ExecSpace}_SOURCES)
                        
                    # Get the headers matching the name and execution space
                    FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                        ${abs_dir}/${name}${ExecSpace}*.hpp)

                    # Found a impl header.
                    IF(EXISTS "${abs_dir}/${name}OpImpl.hpp")
                        SET(IMPL_HEADER "#include \"${abs_dir}/${name}OpImpl.hpp\"")
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

        # Loop through all of the directories which contain operators and add
        # their source and header files.
        FILE(GLOB children)

        FOREACH(child ${children})
        MESSAGE("Child : " "${child}")
            ADD_OPERATOR(${child})
        ENDFOREACH()

        ADD_SUBDIRECTORY(RiemannSolvers/Profilers)

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

        ADD_SOLVER_EXECUTABLE(CompressibleFlowSolverRedesign
        DEPENDS SolverCore Operators
        SOURCES
        CompressibleFlowSolverRedesign.cpp
        EquationSystems/EulerCFE.cpp
        EquationSystems/NavierStokesCFE.cpp
        ${OPERATORS_SOURCES}
    )

    SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign IsentropicVortex)
    SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign TGVFlow3D_Hex8)

    TARGET_INCLUDE_DIRECTORIES(CompressibleFlowSolverRedesign PRIVATE
    "${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign")

    IF(NEKTAR_BUILD_SOLVER_LIBS)
        MESSAGE(WARNING, "Solver library build not set up for RedesignSolver - skipping.")
    ENDIF()

ENDIF(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_RE)
ENDIF(NEKTAR_ENABLE_DEVICE_SUPPORT)
