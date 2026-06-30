IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_ADR_RE
        "Build the ADR Solver." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NOT NEKTAR_BUILD_SOLVERS)
        SET(NEKTAR_SOLVER_ADR_RE OFF CACHE INTERNAL "")
    ENDIF()

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

    # Set default data types.
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        SET(DataTypes double)
    ENDIF()
    IF(NEKTAR_ENABLE_SINGLE_PRECISION)
        SET(DataTypes ${DataTypes} float)
    ENDIF()

    ################################################################
    # ADR-specific operators.
    MACRO(ADD_ADR_OPERATOR dir)

            IF(IS_DIRECTORY ${dir})
                GET_FILENAME_COMPONENT(name "${dir}" NAME)
                GET_FILENAME_COMPONENT(abs_dir "${dir}" ABSOLUTE)
            ELSE()
                CONTINUE()
            ENDIF()

            SET(NAME_HEADERS ${name}_HEADERS)
            SET(NAME_SOURCES ${name}_SOURCES)
            SET(OPERATOR ${name})
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/Operators/Common/OpFactoryDec.cpp.in)

            FOREACH(ExecSpace IN LISTS ExecSpaces)
                SET(HEADER_EXT hpp)
                IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                    SET(SOURCE_EXT cu)
                ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                    SET(SOURCE_EXT hip)
                ELSE()
                    SET(SOURCE_EXT cpp)
                ENDIF()

                FOREACH(TData IN LISTS DataTypes)
                    SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})
                    SET(CURRENT_HEADERS ${name}_${ExecSpace}_HEADERS)
                    SET(CURRENT_SOURCES ${name}_${ExecSpace}_SOURCES)

                    FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                        ${abs_dir}/${name}${ExecSpace}*.hpp)

                    IF(EXISTS "${abs_dir}/${name}OpImpl.hpp")
                        FILE(RELATIVE_PATH include_file ${CMAKE_SOURCE_DIR}/solvers
                            "${abs_dir}/${name}OpImpl.hpp")
                        SET(IMPL_HEADER "#include \"${include_file}\"")
                    ELSE()
                        CONTINUE()
                    ENDIF()

                    CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                    SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
                    SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                    SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})
                ENDFOREACH()
            ENDFOREACH()

            SET(OPERATORS_HEADERS
                ${OPERATORS_HEADERS}
                ${${NAME_HEADERS}}
                )
            SET(OPERATORS_SOURCES
                ${OPERATORS_SOURCES}
                ${${NAME_SOURCES}}
                )
    ENDMACRO()

    FILE(GLOB ADR_OPERATOR_DIRS
        LinearAdvVolumeFlux
        DiffusionScalarIPVolFlux
        DiffusionScalarIPTraceFlux)
    FOREACH(child ${ADR_OPERATOR_DIRS})
        ADD_ADR_OPERATOR(${child})
    ENDFOREACH()

    SET(ADR_OPERATOR_SOURCES "${OPERATORS_SOURCES}" CACHE INTERNAL
        "Generated ADR operator factory sources")

    IF (NEKTAR_SOLVER_ADR_RE)

        # Macro
        MACRO(SET_ALL_SOLVER_TEST solver session)
            FOREACH (ExecSpace IN LISTS ExecSpaces)
                # Add SumFac test for all execution spaces
                SET(Implementation SumFac)
                ADD_TEST(
                        NAME ${solver}_${session}_${ExecSpace}_${Implementation}
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

        ADD_SOLVER_EXECUTABLE(ADRSolverRedesign
            DEPENDS SolverCore Operators
            SOURCES
            ADRSolverRedesign.cpp
            EquationSystems/SteadyADR.cpp
            EquationSystems/UnsteadyADR.cpp
            ${OPERATORS_SOURCES}
        )

        IF(NEKTAR_BUILD_SOLVER_LIBS)
            MESSAGE(WARNING, "Solver library build not set up for ADRSolverRedesign - skipping.")
        ENDIF()

        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion1D_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion2D_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion3D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion3D_3C)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion3D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion3D_3C)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyDiffusion1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyDiffusion2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyDiffusion3D)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff3D)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction3D)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff1D_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff2D_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_3C)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_hex)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_tet)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_prism)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_tetprism)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_Tet_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_Prism_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_TetPrism_3C)

        # SDC time-integration
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_ExplicitSDCGaussLobattoLegendre)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_ExplicitSDCGaussRadauLegendre)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_ExplicitSDCGaussGaussLegendre)

        # 2D discontinuous advection
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_deformed_GLL_LAGRANGE_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_deformed_GLL_LAGRANGE_SEM_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_deformed_MODIFIED_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_GLL_LAGRANGE_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_GLL_LAGRANGE_SEM_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_MODIFIED_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_reg_def_MODIFIED_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_SEM_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_MODIFIED_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_regular_GLL_LAGRANGE_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_regular_GLL_LAGRANGE_SEM_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_regular_MODIFIED_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_10x10)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_SEM_10x10)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_MODIFIED_10x10)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection_m12_DG_periodic)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_SSPRK2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_SSPRK3)

        # 3D discontinous advection
        # SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection3D_m10_DG_hex_periodic_nodal) # Failing due to memery on device
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection3D_m12_DG_hex_periodic)
        # face rotation test
        # SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection3D_DG_prism_varP) # Failing due to workspace size

        # 2D explicit diffusion (ScalarIP)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ExDiffusion_2D_IP_hybrid_m3_periodic)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ExDiffusion_2D_IP_hybrid_m3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ExDiffusion_2D_IP_hybrid_m8)

    ENDIF()
ENDIF(NEKTAR_ENABLE_DEVICE_SUPPORT)
