IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_ADR_DEVICE_SUPPORT
        "Build the ADR Solver." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NOT NEKTAR_BUILD_SOLVERS)
        SET(NEKTAR_SOLVER_ADR_DEVICE_SUPPORT OFF CACHE INTERNAL "")
    ENDIF()
ENDIF()

IF(NEKTAR_SOLVER_ADR_DEVICE_SUPPORT)
    ################################################################
    # ADR-specific operators.
    FILE(GLOB ADR_OPERATOR_DIRS
        LinearAdvVolumeFlux
        DiffusionScalarIPVolFlux
        DiffusionScalarIPTraceFlux
    )

    SET(OPNAMESPACE "Nektar::detail")
    FOREACH(child ${ADR_OPERATOR_DIRS})
        ADD_OPERATOR(${child} OPNAMESPACE ADR_OPERATORS_HEADERS ADR_OPERATORS_SOURCES)
    ENDFOREACH()

    SET(OPERATOR_UNIT_TEST_DEPENDS Operators SolverCore)

    IF(BOOST_TEST_DYN_LINK)
        SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
            ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    ENDIF()

    ADD_NEKTAR_EXECUTABLE(test_advectionweakdg
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SolverUtils
        SOURCES UnitTests/test_advectionweakdg.cpp ${ADR_OPERATORS_SOURCES})

    SET_UNIT_TEST(AdvectionWeakDG_serial_stdmat test_advectionweakdg Serial StdMat FALSE)
    SET_UNIT_TEST(AdvectionWeakDG_serial_sumfac test_advectionweakdg Serial SumFac FALSE)
    IF(NEKTAR_ENABLE_SIMD)
        SET_UNIT_TEST(AdvectionWeakDG_avx_stdmat test_advectionweakdg AVX StdMat FALSE)
        SET_UNIT_TEST(AdvectionWeakDG_avx_sumfac test_advectionweakdg AVX SumFac FALSE)
    ENDIF()
    IF(NEKTAR_ENABLE_DEVICE)
        SET_UNIT_TEST(AdvectionWeakDG_device_stdmat test_advectionweakdg Device StdMat FALSE)
        SET_UNIT_TEST(AdvectionWeakDG_device_sumfac test_advectionweakdg Device SumFac FALSE)
        SET_UNIT_TEST(AdvectionWeakDG_device_sumfactop test_advectionweakdg Device SumFacTOP FALSE)
    ENDIF()

    ADD_SOLVER_EXECUTABLE(ADRSolverRedesign
        DEPENDS SolverCore Operators
        SOURCES
        ADRSolverRedesign.cpp
        EquationSystems/SteadyADR.cpp
        EquationSystems/UnsteadyADR.cpp
        ${ADR_OPERATORS_SOURCES}
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

    SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection1D_TimeDependentDirichlet)
    SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_TimeDependentDirichlet)
    SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection3D_TimeDependentDirichlet)

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
ENDIF (NEKTAR_SOLVER_ADR_DEVICE_SUPPORT)
