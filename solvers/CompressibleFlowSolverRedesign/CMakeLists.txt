IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_DEVICE_SUPPORT
        "Build the Compressible Flow Solver using redesign framework." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NOT NEKTAR_BUILD_SOLVERS)
        SET(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_DEVICE_SUPPORT OFF CACHE INTERNAL "")
    ENDIF()
ENDIF()

IF(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_DEVICE_SUPPORT)

    # Set Equation of State types 
    SET(EoSTypes IdealGasEoS VanDerWaalsEoS)

    ################################################################
    # Operators templated on the equation of state but not on a Riemann
    # solver. Entries are paths relative to this directory, so an operator
    # may sit in a subdirectory; the operator name is the basename, matching
    # what ADD_OPERATOR does, so nesting does not change the key an operator
    # registers under.
    SET(CFEVolFluxes  EulerVolumeFlux
                      DiffusionCFEVolFlux
                      BndCondOps/BndCondPressureOutflowCFE
                      BndCondOps/BndCondWallCFE)

    FOREACH (relPath IN LISTS CFEVolFluxes)
        SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/${relPath})
        GET_FILENAME_COMPONENT(name "${relPath}" NAME)
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
                    src/${name}${SolverType}${EoSType}${ExecSpace}${TData}.${SOURCE_EXT})

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

                    # Anchored at the solver root rather than relative to the
                    # operator, so an operator in a subdirectory still finds
                    # the equation of state headers.
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${CMAKE_CURRENT_SOURCE_DIR}/EquationOfState/${EoSType}.hpp")
                    SET(EOS_HEADER "#include \"${include_file}\"")
                    SET(EOS_TYPE ${EoSType}) 
                
                    # Create the cpp file from the implementation file.
                    CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                    # Add the respective cpp file.
                    SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                    SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})

                ENDFOREACH()
            ENDFOREACH()
        ENDFOREACH()

        # Add this operator's implemenations to the global operator source
        # and header files.
        SET(CFS_OPERATORS_SOURCES
            ${CFS_OPERATORS_SOURCES}
            ${${NAME_SOURCES}}
            )

        # Make list visible to subdirectories
        SET(VOLUE_FACTORY_SOURCES "${CFS_OPERATORS_SOURCES}" CACHE INTERNAL "Generated Volume flux factory sources")
    ENDFOREACH()


        ################################################################
        # Trace Flux operators
        SET(OpList AdvTraceFluxCFE AdvDiffTraceFluxCFE)

	FOREACH (name IN LISTS OpList)

	    SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/${name})
            SET(dirR ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSolvers)

            SET(NAME_HEADERS ${name}_HEADERS)
            SET(NAME_SOURCES ${name}_SOURCES)
            SET(OPERATOR ${name})      #Operator parameter for boiler plate code
        
            #boiler plate code for cpp file
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign/RiemannSolvers/RiemannOperatorFactoryDec.cpp.in)

            # Set compilation for Riemann solvers
            SET(SolverTypes  Average AUSM0 AUSM1 AUSM2 AUSM3 HLL HLLC LaxFriedrichs Roe)

            # These operator have an additional argument for Equation of state and
            # SovlerTypes which requires more  delcarations.

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
                               src/${name}${SolverType}${EoSType}${ExecSpace}${TData}.${SOURCE_EXT})

                            # Get the source and header files for this operator,
                            # execution space, and implementation.
                            SET(CURRENT_HEADERS ${name}${SolverType}_${ExecSpace}_HEADERS)
                            SET(CURRENT_SOURCES ${name}${SolverType}_${ExecSpace}_SOURCES)
                        
                            # Get the headers matching the name and execution space
                            FILE(GLOB ${CURRENT_HEADERS} RELATIVE
                               ${CMAKE_CURRENT_SOURCE_DIR}
                               ${dir}/${name}${SolverType}${ExecSpace}*.hpp)

                            SET(IMPL_HEADER "#include \"${dir}/${name}OpImpl.hpp\"")
                            SET(KERNEL_HEADER "#include \"${dirR}/${SolverType}SolverKernels.hpp\"")

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
            SET(CFS_OPERATORS_SOURCES
                ${CFS_OPERATORS_SOURCES}
                ${${NAME_SOURCES}}
                 )

            # Make list visible to subdirectories
            SET(RIEMANN_FACTORY_SOURCES "${CFS_OPERATORS_SOURCES}" CACHE INTERNAL "Generated Riemann factory sources")
  	 ENDFOREACH()
        ################################################################

    # Loop through all of the directories which contain operators and add
    # their source and header files.
    #
    # Boundary condition operators live under BndCondOps, mirroring the
    # library's own Operators/BndCondOps grouping, and are picked up by
    # wildcard: a new one is registered by creating its directory, with no
    # change needed here. ADD_OPERATOR takes the directory's basename, so
    # nesting does not affect the name an operator is registered under.
    FILE(GLOB children BndCondOps/*)

    # Operators templated on the equation of state are generated by the
    # CFEVolFluxes loop above, which knows how to spell their extra template
    # argument. Without this they would be registered twice, and the
    # declaration from here would not compile.
    FOREACH(eosOperator IN LISTS CFEVolFluxes)
        LIST(REMOVE_ITEM children
             "${CMAKE_CURRENT_SOURCE_DIR}/${eosOperator}")
    ENDFOREACH()

    SET(OPNAMESPACE "Nektar::detail")
    FOREACH(child ${children})
        ADD_OPERATOR(${child} OPNAMESPACE CFS_OPERATORS_HEADERS CFS_OPERATORS_SOURCES)
    ENDFOREACH()

    ADD_SUBDIRECTORY(RiemannSolvers/UnitTests)
    # The Riemann profiler predates the flux framework and is not built.
    # ADD_SUBDIRECTORY(RiemannSolvers/Profilers)

    ADD_SOLVER_EXECUTABLE(CompressibleFlowSolverRedesign
        DEPENDS SolverCore Operators
        SOURCES
        CompressibleFlowSolverRedesign.cpp
        EquationSystems/EulerCFE.cpp
        EquationSystems/NavierStokesCFE.cpp
        ${CFS_OPERATORS_SOURCES}
    )

    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign IsentropicVortex)
        IF (NEKTAR_USE_MPI)
            SET_HOST_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign IsentropicVortex_par)
        ENDIF()
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign Couette_WeakDG_IP_MODIFIED_Adiabatic)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign Couette_WeakDG_IP_MODIFIED_Isothermal)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign Couette_WeakDG_IP_MODIFIED_Isothermal_TagTwall)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign ChanFlow3D_PressureOutflow)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign EnforceEntropyVelocity_2D_mix)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign EnforceEntropyPressure_2D_mix)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign EnforceEntropyTotalEnthalpy_2D_mix)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign SlipWall_ExtrapOrder0_2D_mix)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign RiemannInvariant_2D_mix)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign SquareDomain_Euler_2D_AxialFlow)
        SET_SUMFAC_SOLVER_TEST(CompressibleFlowSolverRedesign ChanFlow3D_Euler_PressureOutflow)
        SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign TGVFlow3D_Hex8)
        SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign TGVFlow3D_Hex8_PerFaceRot)
        SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign TGV_8x8x8_Tet_ortho_P0)
        SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign Euler1D_DensityPulse)
        SET_ALL_SOLVER_TEST(CompressibleFlowSolverRedesign Euler1D)
    ENDIF()

    TARGET_INCLUDE_DIRECTORIES(CompressibleFlowSolverRedesign PRIVATE
    "${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign")

    IF(NEKTAR_BUILD_SOLVER_LIBS)
        MESSAGE(WARNING, "Solver library build not set up for RedesignSolver - skipping.")
    ENDIF()
ENDIF(NEKTAR_SOLVER_COMPRESSIBLE_FLOW_DEVICE_SUPPORT)
