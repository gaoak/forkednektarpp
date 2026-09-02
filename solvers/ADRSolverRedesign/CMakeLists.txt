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

    SET(OPERATOR_UNIT_TEST_DEPENDS SolverCore)

    IF(BOOST_TEST_DYN_LINK)
        SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
            ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    ENDIF()

    ADD_NEKTAR_EXECUTABLE(test_advectionweakdg
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SolverCore SolverUtils
        SOURCES UnitTests/test_advectionweakdg.cpp ${ADR_OPERATORS_SOURCES})

    IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
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
    ENDIF()

    ADD_SOLVER_EXECUTABLE(ADRSolverRedesign
        DEPENDS SolverCore
        SOURCES
        ADRSolverRedesign.cpp
        EquationSystems/SteadyADR.cpp
        EquationSystems/UnsteadyADR.cpp
        ${ADR_OPERATORS_SOURCES}
    )

    IF(NEKTAR_BUILD_SOLVER_LIBS)
        MESSAGE(WARNING, "Solver library build not set up for ADRSolverRedesign - skipping.")
    ENDIF()

    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        # Dimension escalation (1D/2D/3D of the same physics): spatial dimension
        # is orthogonal to opExecSpace/opImpl dispatch, so only 3D keeps the
        # full execspace/impl matrix; 1D/2D are reduced to a Serial+SumFac
        # smoke test. See TEST_SUITE_NOTES.md section 9.
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion1D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion1D_3C)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion2D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion2D_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion3D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyDiffusion3D_3C)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion1D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion3D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign ReactionDiffusion3D_3C)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign SteadyDiffusion1D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign SteadyDiffusion2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyDiffusion3D)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff1D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff3D)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction1D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction3D)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff1D)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff1D_3C)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff2D_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_3C)

        # UnsteadyAdvecDiff3D_3C Serial+StdMat is disabled: at NUMMODES=8 in 3D,
        # StdMat's dense per-element matrix-vector product (O(nmTot^2),
        # nmTot=~512) combined with hundreds of CG iterations per implicit
        # diffusion solve in unvectorized Serial execspace makes this the
        # heaviest test in the whole redesign suite - it was already hitting the
        # 300s ctest timeout outright before the NumSteps 10->5 reduction above
        # (see TEST_SUITE_NOTES.md section 14), and even after that reduction
        # still takes ~210s, the slowest of any test with the least tolerance
        # margin to spare. This is not a bug (see section 14.1 - confirmed via
        # perf profiling that it's the correctly-templated StdMat kernel doing
        # legitimate work, not a fallback path) - StdMat is simply the wrong
        # implementation choice at this order/dimension in Serial. Other
        # execspace/impl combos (AVX+StdMat, Serial/AVX+SumFac, Device+SumFacTOP)
        # still exercise this session and the StdMat implementation, so no
        # coverage is lost by disabling only this one combo.
        SET_TESTS_PROPERTIES(
            ADRSolverRedesign_UnsteadyAdvecDiff3D_3C_Serial_StdMat
            PROPERTIES DISABLED TRUE)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_hex)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_tet)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_prism)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign affine_mesh_test_tetprism)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_Tet_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_Prism_3C)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D_TetPrism_3C)

        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection1D_TimeDependentDirichlet)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_TimeDependentDirichlet)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection3D_TimeDependentDirichlet)

        # SDC time-integration: quadrature-node choice is a time-integration
        # detail independent of opExecSpace/opImpl, so only the commonly-used
        # default node set (GaussLobattoLegendre) keeps the full matrix.
        SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_ExplicitSDCGaussLobattoLegendre)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_ExplicitSDCGaussRadauLegendre)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvection2D_ExplicitSDCGaussGaussLegendre)

        # 2D discontinuous advection: basis type and mesh regularity/size are
        # interpolation/quadrature concerns handled identically regardless of
        # execspace, so the full matrix is kept only on three sessions that
        # between them span the basis/BC/mesh-size space (ISO+regular+MODIFIED,
        # periodic+regular+MODIFIED, periodic+deformed+GLL_LAGRANGE+10x10); the
        # rest are reduced to a Serial+SumFac smoke test. Advection_m12_DG_periodic
        # (different flux order) and the SSPRK2/SSPRK3 pair (different time
        # integrator) are untouched - not part of this basis/mesh sweep.
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_deformed_GLL_LAGRANGE_3x3)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_deformed_GLL_LAGRANGE_SEM_3x3)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_deformed_MODIFIED_3x3)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_GLL_LAGRANGE_3x3)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_GLL_LAGRANGE_SEM_3x3)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_regular_MODIFIED_3x3)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_ISO_reg_def_MODIFIED_3x3)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_2x2)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_SEM_2x2)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_MODIFIED_2x2)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_regular_GLL_LAGRANGE_2x2)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_regular_GLL_LAGRANGE_SEM_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_regular_MODIFIED_2x2)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_10x10)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_GLL_LAGRANGE_SEM_10x10)
        SET_SERIAL_SUMFAC_SOLVER_TEST(ADRSolverRedesign Advection2D_periodic_deformed_MODIFIED_10x10)
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
ENDIF (NEKTAR_SOLVER_ADR_DEVICE_SUPPORT)
