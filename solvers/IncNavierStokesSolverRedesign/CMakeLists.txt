IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_DEVICE_SUPPORT
        "Build the IncNavierStokesSolver using redesign operators." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NOT NEKTAR_BUILD_SOLVERS)
        SET(NEKTAR_SOLVER_INCNAVIERSTOKES_DEVICE_SUPPORT OFF CACHE INTERNAL "")
    ENDIF()
ENDIF()

IF (NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_DEVICE_SUPPORT)
    # IncNS-specific operators.
    SET(OPNAMESPACE "Nektar::detail")
    FOREACH(child IN ITEMS
            ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditions/HOPressureBndCond
            ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditions/HOutflowBndCond)
        ADD_OPERATOR(${child} OPNAMESPACE INCNS_OPERATORS_HEADERS
            INCNS_OPERATORS_SOURCES)
    ENDFOREACH()

    IF(NEKTAR_BUILD_UNIT_TESTS AND NEKTAR_ENABLE_DOUBLE_PRECISION)
        SET(INCNS_UNIT_TEST_DEPENDS SolverCore MultiRegions SolverUtils)
        IF(BOOST_TEST_DYN_LINK)
            LIST(APPEND INCNS_UNIT_TEST_DEPENDS
                ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        ENDIF()

        ADD_NEKTAR_EXECUTABLE(TestHOPressureBndCond
            COMPONENT unit-test
            DEPENDS ${INCNS_UNIT_TEST_DEPENDS}
            SOURCES UnitTests/TestHOPressureBndCond.cpp
                    ${HOPressureBndCond_SOURCES})

        ADD_NEKTAR_EXECUTABLE(TestHOutflowBndCond
            COMPONENT unit-test
            DEPENDS ${INCNS_UNIT_TEST_DEPENDS}
            SOURCES UnitTests/TestHOutflowBndCond.cpp
                    ${HOutflowBndCond_SOURCES})

        # The pressure and velocity kernels at single points, without a
        # session.
        ADD_TEST(NAME OperatorsHOutflowBndCond_host_reference
            COMMAND $<TARGET_FILE:TestHOutflowBndCond>
                --detect_memory_leaks=0)

        # The pressure condition on a region of two elements, through the
        # operator, which checks the extraction and the block layout of the
        # boundary data.
        FUNCTION(ADD_HOUTFLOW_EXTRACTION_UNIT_TEST testname execspace implementation)
            ADD_TEST(
                NAME Operators${testname}
                COMMAND $<TARGET_FILE:TestHOutflowBndCond>
                    --detect_memory_leaks=0
                    --run_test=TestHOutflowBndCond/build_boundary_element_data_manufactured_multi_element
                    -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests/HOutflowBndCondExtraction2D.xml)
        ENDFUNCTION()

        ADD_HOUTFLOW_EXTRACTION_UNIT_TEST(
            HOutflowBndCond_extraction_serial_stdmat Serial StdMat)
        ADD_HOUTFLOW_EXTRACTION_UNIT_TEST(
            HOutflowBndCond_extraction_serial_sumfac Serial SumFac)
        IF(NEKTAR_ENABLE_SIMD)
            ADD_HOUTFLOW_EXTRACTION_UNIT_TEST(
                HOutflowBndCond_extraction_avx_sumfac AVX SumFac)
        ENDIF()
        IF(NEKTAR_ENABLE_DEVICE)
            ADD_HOUTFLOW_EXTRACTION_UNIT_TEST(
                HOutflowBndCond_extraction_device_sumfactop Device SumFacTOP)
        ENDIF()

        # The pressure and velocity conditions applied to fields at the
        # interleave width of the execution space and at width 1, on two
        # regions meeting at a corner of one element.
        FUNCTION(ADD_HOUTFLOW_INTERLEAVED_UNIT_TEST testname execspace implementation)
            ADD_TEST(
                NAME Operators${testname}
                COMMAND $<TARGET_FILE:TestHOutflowBndCond>
                    --detect_memory_leaks=0
                    --run_test=TestHOutflowBndCond/apply_interleaved_matches_width_one
                    -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests/HOutflowBndCondCorner2D.xml)
        ENDFUNCTION()

        ADD_HOUTFLOW_INTERLEAVED_UNIT_TEST(
            HOutflowBndCond_interleaved_serial_stdmat Serial StdMat)
        ADD_HOUTFLOW_INTERLEAVED_UNIT_TEST(
            HOutflowBndCond_interleaved_serial_sumfac Serial SumFac)
        IF(NEKTAR_ENABLE_SIMD)
            ADD_HOUTFLOW_INTERLEAVED_UNIT_TEST(
                HOutflowBndCond_interleaved_avx_sumfac AVX SumFac)
        ENDIF()
        IF(NEKTAR_ENABLE_DEVICE)
            ADD_HOUTFLOW_INTERLEAVED_UNIT_TEST(
                HOutflowBndCond_interleaved_device_sumfactop Device SumFacTOP)
        ENDIF()

        FUNCTION(ADD_HOPRESSURE_UNIT_TEST testname execspace implementation)
            SET(session HOPressureBndCond2D.xml)
            IF(ARGC GREATER 3)
                SET(session ${ARGV3})
            ENDIF()
            ADD_TEST(
                NAME Operators${testname}
                COMMAND $<TARGET_FILE:TestHOPressureBndCond>
                    --detect_memory_leaks=0 -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests/${session})
        ENDFUNCTION()

        FUNCTION(ADD_HOPRESSURE_SHAPE_UNIT_TESTS testprefix execspace implementation)
            ADD_HOPRESSURE_UNIT_TEST(${testprefix}_tri ${execspace}
                ${implementation} HOPressureBndCond2D_Tri.xml)
            ADD_HOPRESSURE_UNIT_TEST(${testprefix}_hex ${execspace}
                ${implementation} HOPressureBndCond3D_Hex.xml)
            ADD_HOPRESSURE_UNIT_TEST(${testprefix}_tet ${execspace}
                ${implementation} HOPressureBndCond3D_Tet.xml)
            ADD_HOPRESSURE_UNIT_TEST(${testprefix}_prism ${execspace}
                ${implementation} HOPressureBndCond3D_Prism.xml)
        ENDFUNCTION()

        FUNCTION(ADD_HOPRESSURE_KOVASZNAY_UNIT_TEST testname execspace implementation)
            ADD_TEST(
                NAME Operators${testname}
                COMMAND $<TARGET_FILE:TestHOPressureBndCond>
                    --detect_memory_leaks=0
                    --run_test=TestHOPressureBndCond/update_then_apply_kovasznay_boundary_flux
                    -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/KovasznayFlowHOPBC2D.xml)
            # The boundary acceleration of a Dirichlet velocity that the
            # boundary expansion of the small time step case cannot represent.
            ADD_TEST(
                NAME Operators${testname}_discrete_boundary_velocity
                COMMAND $<TARGET_FILE:TestHOPressureBndCond>
                    --detect_memory_leaks=0
                    --run_test=TestHOPressureBndCond/update_then_apply_kovasznay_discrete_boundary_velocity
                    -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/KovasznayFlowHOPBC2D.xml
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/KovasznayFlowHOPBC2D_SmallTimeStep.xml)
        ENDFUNCTION()

        # A region whose velocity is Dirichlet in one component and Neumann
        # in the other: the Neumann component takes no acceleration, and its
        # flux is formed from the explicit contribution.
        FUNCTION(ADD_HOPRESSURE_NEUMANN_VELOCITY_UNIT_TEST testname execspace
                 implementation)
            ADD_TEST(
                NAME Operators${testname}_neumann_velocity
                COMMAND $<TARGET_FILE:TestHOPressureBndCond>
                    --detect_memory_leaks=0
                    --run_test=TestHOPressureBndCond/update_then_apply_neumann_velocity_boundary_flux
                    -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests/HOPressureBndCondNeumannVelocity2D.xml)
            # The flux of the weak pressure form on the same region.
            ADD_TEST(
                NAME Operators${testname}_neumann_velocity_weak
                COMMAND $<TARGET_FILE:TestHOPressureBndCond>
                    --detect_memory_leaks=0
                    --run_test=TestHOPressureBndCond/update_weak_then_apply_neumann_velocity_boundary_flux
                    -- ${execspace} ${implementation}
                    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests/HOPressureBndCondNeumannVelocity2D.xml)
        ENDFUNCTION()

        ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_serial_stdmat
            Serial StdMat)
        ADD_HOPRESSURE_NEUMANN_VELOCITY_UNIT_TEST(
            HOPressureBndCond_serial_stdmat Serial StdMat)
        ADD_HOPRESSURE_NEUMANN_VELOCITY_UNIT_TEST(
            HOPressureBndCond_serial_sumfac Serial SumFac)
        ADD_HOPRESSURE_SHAPE_UNIT_TESTS(HOPressureBndCond_serial_stdmat
            Serial StdMat)
        ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_serial_sumfac
            Serial SumFac)
        ADD_HOPRESSURE_SHAPE_UNIT_TESTS(HOPressureBndCond_serial_sumfac
            Serial SumFac)
        ADD_HOPRESSURE_KOVASZNAY_UNIT_TEST(
            HOPressureBndCond_kovasznay_serial_stdmat Serial StdMat)

        # The selection of the boundary regions (USERDEFINEDTYPE="H") is
        # host code, so Serial/StdMat covers it.
        ADD_TEST(
            NAME OperatorsHOPressureBndCond_selection_serial_stdmat
            COMMAND $<TARGET_FILE:TestHOPressureBndCond>
                --detect_memory_leaks=0
                --run_test=TestHOPressureBndCond/update_then_apply_selection_respects_hop_tag
                -- Serial StdMat
                ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests/HOPressureBndCondSelection2D.xml)
        IF(NEKTAR_ENABLE_SIMD)
            ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_avx_stdmat
                AVX StdMat)
            ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_avx_sumfac
                AVX SumFac)
            ADD_HOPRESSURE_SHAPE_UNIT_TESTS(HOPressureBndCond_avx_sumfac
                AVX SumFac)
            ADD_HOPRESSURE_NEUMANN_VELOCITY_UNIT_TEST(
                HOPressureBndCond_avx_sumfac AVX SumFac)
            ADD_HOPRESSURE_KOVASZNAY_UNIT_TEST(
                HOPressureBndCond_kovasznay_avx_sumfac AVX SumFac)
        ENDIF()
        IF(NEKTAR_ENABLE_DEVICE)
            ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_device_stdmat
                Device StdMat)
            ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_device_sumfac
                Device SumFac)
            ADD_HOPRESSURE_UNIT_TEST(HOPressureBndCond_device_sumfactop
                Device SumFacTOP)
            ADD_HOPRESSURE_SHAPE_UNIT_TESTS(HOPressureBndCond_device_sumfactop
                Device SumFacTOP)
            ADD_HOPRESSURE_KOVASZNAY_UNIT_TEST(
                HOPressureBndCond_kovasznay_device_sumfactop
                Device SumFacTOP)
            ADD_HOPRESSURE_NEUMANN_VELOCITY_UNIT_TEST(
                HOPressureBndCond_device_stdmat Device StdMat)
            ADD_HOPRESSURE_NEUMANN_VELOCITY_UNIT_TEST(
                HOPressureBndCond_device_sumfactop Device SumFacTOP)
        ENDIF()
    ENDIF()

    # Default timestepping schemes.
    SET(Formulations linearimplicit semiimplicit)

    # Macro
    MACRO(SET_ALL_INC_SOLVER_TEST solver session)
        FOREACH (ExecSpace IN LISTS ExecSpaces)
            # Formulation is a time-integration choice, largely orthogonal to
            # opExecSpace/opImpl dispatch, so it isn't swept at every
            # combination. Serial always covers both formulations at both
            # implementations (cheap, catches formulation-specific bugs).
            # AVX/Device only add linearimplicit at their SumFac-family
            # implementation - AVX+SumFac, Device+SumFacTOP - because
            # linearimplicit exercises device/vector mechanics that
            # semiimplicit doesn't use; their other implementation
            # (AVX+StdMat, Device+SumFac) only needs semiimplicit to catch
            # execspace/impl-specific regressions. See TEST_SUITE_NOTES.md
            # section 9.

            # --- SumFac test for all execution spaces ---
            SET(Implementation SumFac)
            IF ( ("${ExecSpace}" STREQUAL "Serial") OR ("${ExecSpace}" STREQUAL "AVX") )
                SET(FormulationsToTest ${Formulations})
            ELSE()
                SET(FormulationsToTest semiimplicit)
            ENDIF()
            FOREACH (Formulation IN LISTS FormulationsToTest)
                ADD_TEST(
                        NAME ${solver}_${session}_${Formulation}_${ExecSpace}_${Implementation}
                        COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
            ENDFOREACH ()

            # --- StdMat (Serial/AVX) or SumFacTOP (Device) test ---
            IF( ("${ExecSpace}" STREQUAL "Serial") OR ("${ExecSpace}" STREQUAL "AVX") )
                SET(Implementation StdMat)
            ELSEIF( ("${ExecSpace}" STREQUAL "Device") )
                SET(Implementation SumFacTOP)
            ENDIF()
            IF ( ("${ExecSpace}" STREQUAL "Serial") OR ("${ExecSpace}" STREQUAL "Device") )
                SET(FormulationsToTest ${Formulations})
            ELSE()
                SET(FormulationsToTest semiimplicit)
            ENDIF()
            FOREACH (Formulation IN LISTS FormulationsToTest)
                ADD_TEST(
                        NAME ${solver}_${session}_${Formulation}_${ExecSpace}_${Implementation}
                        COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
            ENDFOREACH ()
        ENDFOREACH ()
    ENDMACRO()

    ADD_SOLVER_EXECUTABLE(IncNavierStokesSolverRedesign
        DEPENDS SolverCore
        SOURCES
            IncNavierStokesSolverRedesign.cpp
            EquationSystems/VelocityCorrectionScheme.cpp
            ${INCNS_OPERATORS_SOURCES}
    )

    IF(NEKTAR_BUILD_SOLVER_LIBS)
        MESSAGE(WARNING, "Solver library build not set up for IncNavierStokesSolverRedesign - skipping.")
    ENDIF()

    FOREACH(Session IN ITEMS
            UniformFlow2D
            CouetteFlow2D
            ChannelFlow2D
            ChannelFlowHOPBC2D
            # Manufactured flow on the faceted-cylinder mesh, with non-zero
            # advection, body forcing and CurlCurl contributions to the
            # higher-order pressure boundary condition.
            CylinderFlowHOPBC2D
            # Steady exact flow with non-zero advection and spatially varying
            # higher-order pressure boundary data.
            KovasznayFlowHOPBC2D
            # Small time step with a Dirichlet velocity that the boundary
            # expansion cannot represent: the higher-order pressure boundary
            # acceleration must use the discrete boundary velocity.
            KovasznayFlowHOPBC2D_SmallTimeStep
            # The second-order IMEX startup and IMEXdirk stages, whose
            # explicit coefficients do not sum to the implicit coefficient:
            # the higher-order pressure boundary velocity and the
            # linear-implicit advection must carry the weight of the
            # explicit contribution.
            KovasznayFlowHOPBC2D_IMEXOrder2
            KovasznayFlowHOPBC2D_IMEXdirk
            # The weak pressure form, with the predictor, advection and
            # forcing kept off the higher-order pressure boundary flux.
            KovasznayFlowHOPBC2D_WeakPressure
            ChannelFlowHOPBC2D_WeakPressure
            KovasznayFlowHOPBC2D_IMEXOrder2_WeakPressure
            KovasznayFlowHOPBC2D_IMEXdirk_WeakPressure
            KovasznayFlow2D
            UniformFlow3D_Hex
            CouetteFlow3D_Hex
            ChannelFlow3D_Hex
            TGVFlow3D_Hex8
            UniformFlow3D_Tet
            CouetteFlow3D_Tet
            CouetteFlow3D_Prism
            ChannelFlow3D_Tet
        )
        IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
            SET_ALL_INC_SOLVER_TEST(IncNavierStokesSolverRedesign ${Session})
        ENDIF()
    ENDFOREACH()

    # Two partitions, of which only one reaches the pressure Dirichlet
    # boundary. The timeout reports a deadlock as a failure.
    IF(NEKTAR_USE_MPI AND NEKTAR_ENABLE_DOUBLE_PRECISION)
        ADD_TEST(
            NAME IncNavierStokesSolverRedesign_ChannelFlow2D_OutletPressure_MPI
            COMMAND Tester
                ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D_OutletPressure_MPI.tst
                -v --opExecSpace=Serial --opImpl=SumFac
                "--solver-options=-I Formulation=semiimplicit")
        SET_TESTS_PROPERTIES(
            IncNavierStokesSolverRedesign_ChannelFlow2D_OutletPressure_MPI
            PROPERTIES PROCESSORS 2 TIMEOUT 120)
    ENDIF()

    # A file initial condition that is discontinuous and off the Dirichlet
    # velocity, averaged and given the Dirichlet values (the default
    # ProjectInitialConditions), on every execution space and on two
    # processes. The _Galerkin test projects the same file by the global
    # mass matrix solve.
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        FOREACH(ExecSpace IN LISTS ExecSpaces)
            IF("${ExecSpace}" STREQUAL "Device")
                SET(Implementation SumFacTOP)
            ELSE()
                SET(Implementation SumFac)
            ENDIF()
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_ChannelFlow2D_FileInitialCondition_semiimplicit_${ExecSpace}_${Implementation}
                COMMAND Tester
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D_FileInitialCondition.tst
                    -v --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                    "--solver-options=-I Formulation=semiimplicit")
        ENDFOREACH()
        ADD_TEST(
            NAME IncNavierStokesSolverRedesign_ChannelFlow2D_FileInitialCondition_Galerkin
            COMMAND Tester
                ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D_FileInitialCondition_Galerkin.tst
                -v --opExecSpace=Serial --opImpl=SumFac
                "--solver-options=-I Formulation=semiimplicit")
        IF(NEKTAR_USE_MPI)
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_ChannelFlow2D_FileInitialCondition_MPI
                COMMAND Tester
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D_FileInitialCondition_MPI.tst
                    -v --opExecSpace=Serial --opImpl=SumFac
                    "--solver-options=-I Formulation=semiimplicit")
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_ChannelFlow2D_FileInitialCondition_MPI
                PROPERTIES PROCESSORS 2 TIMEOUT 120)
        ENDIF()
    ENDIF()

    # TGVFlow3D_Hex8 Serial+StdMat is disabled for now: at this element
    # order (NUMMODES=7) it hits a pre-existing slow non-templated-operator
    # fallback in Serial execspace dispatch, unrelated to StdMat evaluation
    # itself (see the comment in Tests/TGVFlow3D_Hex8.tst). Even with the
    # NumSteps/IterativeSolverTolerance runtime reductions validated in
    # TEST_SUITE_NOTES.md section 12 (which give a clean ~2x speedup and
    # <35s everywhere else), this combo still takes ~300s in Debug - right
    # at/over a typical ctest --timeout 300. Re-enable once the underlying
    # fallback is fixed; other execspace/impl combos still exercise the
    # TGVFlow3D_Hex8 session and StdMat implementation, so no coverage is
    # lost by disabling only this one combo.
    # Guarded by NEKTAR_ENABLE_DOUBLE_PRECISION to match SET_ALL_INC_SOLVER_TEST
    # above - in a single-precision build no solver tests are registered, so
    # SET_TESTS_PROPERTIES would otherwise fail with "Can not find test".
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        SET_TESTS_PROPERTIES(
            IncNavierStokesSolverRedesign_TGVFlow3D_Hex8_linearimplicit_Serial_StdMat
            IncNavierStokesSolverRedesign_TGVFlow3D_Hex8_semiimplicit_Serial_StdMat
            PROPERTIES DISABLED TRUE)
    ENDIF()

    # Mixed 3D higher-order pressure boundary condition regressions, on the
    # main host and device implementations, covering hexahedron, prism and
    # tetrahedron boundary faces.
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        FOREACH(HOPBCMixedSession IN ITEMS
                ChannelFlowHOPBC3D_Mixed
                ChannelFlowHOPBC3D_TetPrism)
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_${HOPBCMixedSession}_linearimplicit_Serial_SumFac
                COMMAND Tester
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${HOPBCMixedSession}.tst -v
                    --opExecSpace=Serial --opImpl=SumFac
                    "--solver-options=-I Formulation=linearimplicit")

            IF(NEKTAR_ENABLE_DEVICE)
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_${HOPBCMixedSession}_linearimplicit_Device_SumFacTOP
                    COMMAND Tester
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${HOPBCMixedSession}.tst -v
                        --opExecSpace=Device --opImpl=SumFacTOP
                        "--solver-options=-I Formulation=linearimplicit")
            ENDIF()
        ENDFOREACH()
    ENDIF()

    # The mixed hex/prism channel with higher-order pressure regions whose
    # velocity is not Dirichlet in every component: an all-Neumann inflow,
    # whose normal lies along a Neumann component, and two slip side walls.
    # Both formulations at the SumFac-family implementation of every
    # execution space.
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        FOREACH(ExecSpace IN LISTS ExecSpaces)
            IF("${ExecSpace}" STREQUAL "Device")
                SET(Implementation SumFacTOP)
            ELSE()
                SET(Implementation SumFac)
            ENDIF()
            FOREACH(Formulation IN LISTS Formulations)
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_ChannelFlowHOPBC3D_NeumannVelocity_${Formulation}_${ExecSpace}_${Implementation}
                    COMMAND Tester
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlowHOPBC3D_NeumannVelocity.tst
                        -v --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
            ENDFOREACH()
        ENDFOREACH()
    ENDIF()

    # Dong's high-order outflow condition on the Kovasznay mesh: a steady
    # exact solution, through the second-order IMEX startup and in the weak
    # pressure form too, and an outlet with a sign-changing normal velocity.
    IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
        SET(HOutflowCases Serial:StdMat)
        IF(NEKTAR_ENABLE_SIMD)
            LIST(APPEND HOutflowCases AVX:SumFac)
        ENDIF()
        IF(NEKTAR_ENABLE_DEVICE)
            LIST(APPEND HOutflowCases Device:SumFacTOP)
        ENDIF()
        FOREACH(HOutflowCase IN LISTS HOutflowCases)
            STRING(REPLACE ":" ";" HOutflowCase ${HOutflowCase})
            LIST(GET HOutflowCase 0 ExecSpace)
            LIST(GET HOutflowCase 1 Implementation)
            FOREACH(Formulation IN LISTS Formulations)
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_HOutflowKovasznay2D_${Formulation}_${ExecSpace}_${Implementation}
                    COMMAND Tester
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOutflowKovasznay2D.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_HOutflowKovasznay2D_IMEXOrder2_${Formulation}_${ExecSpace}_${Implementation}
                    COMMAND Tester
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOutflowKovasznay2D_IMEXOrder2.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_HOutflowKovasznay2D_WeakPressure_${Formulation}_${ExecSpace}_${Implementation}
                    COMMAND Tester
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOutflowKovasznay2D_WeakPressure.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
            ENDFOREACH()
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOutflowBackflow2D_semiimplicit_${ExecSpace}_${Implementation}
                COMMAND Tester
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOutflowBackflow2D.tst -v
                    --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                    "--solver-options=-I Formulation=semiimplicit")
        ENDFOREACH()
    ENDIF()

    # Tests for time-order convergence
    # Note the test should be independent of implementation, hence, we test only for Execution Space
    IF(NEKTAR_BUILD_PYTHON)
        FIND_PACKAGE(Python3 COMPONENTS Interpreter REQUIRED)
        IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_TimeIntegrationOrders123_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow3D_Hex.xml
                        --op-exec-space Serial
                        --op-impl SumFac)
            # check_time_order.py internally fans out its cases across a
            # thread pool of concurrent solver subprocesses (default cap: 4,
            # see --max-workers), so this test uses more than one core's
            # worth of resources despite being a single ctest slot; tell
            # ctest's own -j scheduler so it doesn't stack other tests on
            # top of it.
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_TimeIntegrationOrders123_Serial_SumFac
                PROPERTIES PROCESSORS 4)

            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_ForcedMMS2D_TriQuad_IMEXdirkOrders23_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_imexdirk_forced_mms_2d.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --formulations linearimplicit semiimplicit
                        --orders 2 3)
            # check_imexdirk_forced_mms_2d.py fans its (order, formulation)
            # cases out across a thread pool too (default cap: 4, see
            # --max-workers) - same reasoning as
            # IncNavierStokesSolverRedesign_TimeIntegrationOrders123_* above.
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_ForcedMMS2D_TriQuad_IMEXdirkOrders23_Serial_SumFac
                PROPERTIES PROCESSORS 4)

            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOPBC_TimeIntegrationOrders23_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_hopbc_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOPressureTimeConvergence2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --formulations semiimplicit linearimplicit)
            # check_hopbc_time_order.py fans its (order, formulation) cases
            # out across a thread pool too (default cap: 4, see
            # --max-workers) - same reasoning as
            # IncNavierStokesSolverRedesign_TimeIntegrationOrders123_* above.
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_HOPBC_TimeIntegrationOrders23_Serial_SumFac
                PROPERTIES TIMEOUT 120 PROCESSORS 4)

            # The same manufactured flow in the weak pressure form, whose
            # higher-order pressure flux holds the viscous term and the
            # boundary acceleration only.
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOPBC_WeakPressure_TimeIntegrationOrders23_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_hopbc_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOPressureTimeConvergence2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --formulations semiimplicit
                        --pressure-forms weak)
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_HOPBC_WeakPressure_TimeIntegrationOrders23_Serial_SumFac
                PROPERTIES TIMEOUT 120 PROCESSORS 4)

            # The linear-implicit weak form takes the explicit terms of its
            # pressure step, including the curl-curl velocity of the
            # higher-order pressure flux, at the latest level, which keeps
            # large time steps stable but limits that form to second order in
            # time, so only order 2 is checked.
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOPBC_WeakPressure_LinearImplicit_TimeIntegrationOrder2_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_hopbc_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOPressureTimeConvergence2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --orders 2
                        --formulations linearimplicit
                        --pressure-forms weak)
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_HOPBC_WeakPressure_LinearImplicit_TimeIntegrationOrder2_Serial_SumFac
                PROPERTIES TIMEOUT 120 PROCESSORS 4)

            # An unsteady flow whose advection changes in time, in both
            # pressure forms.
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOPBC_Advective_TimeIntegrationOrders23_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_hopbc_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOPressureTimeConvergence2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --manufactured advective
                        --formulations semiimplicit
                        --pressure-forms strong weak)
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_HOPBC_Advective_TimeIntegrationOrders23_Serial_SumFac
                PROPERTIES TIMEOUT 120 PROCESSORS 4)

            # The linear-implicit velocity solve advects with the extrapolated
            # velocity u*, which the IMEXdirk startup stages combine from
            # stage values, accurate to first order only, so this
            # formulation is second order in time on this flow for J = 3 as
            # well, and is checked against the threshold of order 2.
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOPBC_Advective_LinearImplicit_TimeIntegrationOrders23_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_hopbc_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOPressureTimeConvergence2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --manufactured advective
                        --formulations linearimplicit
                        --pressure-forms strong
                        --min-rate 1.7)
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_HOPBC_Advective_LinearImplicit_TimeIntegrationOrders23_Serial_SumFac
                PROPERTIES TIMEOUT 120 PROCESSORS 4)

            # The linear-implicit weak form takes the advection and the
            # curl-curl velocity of its pressure step at the latest level,
            # for stability at large time steps, which leaves an O(dt) error
            # in the pressure near the higher-order pressure regions: its
            # velocity is second order and its pressure tends to first order
            # (rates of 1.6 and 1.5 for J = 2 and 3 at these steps, falling
            # under refinement), so the pressure is checked only against a
            # fall to first order.
            ADD_TEST(
                NAME IncNavierStokesSolverRedesign_HOPBC_Advective_WeakPressure_LinearImplicit_TimeIntegrationOrders23_Serial_SumFac
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_hopbc_time_order.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/HOPressureTimeConvergence2D.xml
                        --op-exec-space Serial
                        --op-impl SumFac
                        --manufactured advective
                        --formulations linearimplicit
                        --pressure-forms weak
                        --min-rate 1.7
                        --min-pressure-rate 1.1)
            SET_TESTS_PROPERTIES(
                IncNavierStokesSolverRedesign_HOPBC_Advective_WeakPressure_LinearImplicit_TimeIntegrationOrders23_Serial_SumFac
                PROPERTIES TIMEOUT 120 PROCESSORS 4)
        ENDIF()

        # Optional AVX execution space.
        IF(NEKTAR_ENABLE_SIMD)
            IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_TimeIntegrationOrders123_AVX_SumFac
                    COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_time_order.py
                            --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                            --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow3D_Hex.xml
                            --op-exec-space AVX
                            --op-impl SumFac)
                SET_TESTS_PROPERTIES(
                    IncNavierStokesSolverRedesign_TimeIntegrationOrders123_AVX_SumFac
                    PROPERTIES PROCESSORS 4)

                ADD_TEST(
                        NAME IncNavierStokesSolverRedesign_ForcedMMS2D_TriQuad_IMEXdirkOrders23_AVX_SumFac
                        COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_imexdirk_forced_mms_2d.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D.xml
                        --op-exec-space AVX
                        --op-impl SumFac
                        --formulations linearimplicit semiimplicit
                        --orders 2 3)
                SET_TESTS_PROPERTIES(
                    IncNavierStokesSolverRedesign_ForcedMMS2D_TriQuad_IMEXdirkOrders23_AVX_SumFac
                    PROPERTIES PROCESSORS 4)
            ENDIF()
        ENDIF()

        # Optional Device execution space.
        IF(NEKTAR_ENABLE_DEVICE)
            IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
                ADD_TEST(
                    NAME IncNavierStokesSolverRedesign_TimeIntegrationOrders123_Device_SumFacTOP
                    COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_time_order.py
                            --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                            --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow3D_Hex.xml
                            --op-exec-space Device
                            --op-impl SumFacTOP)
                SET_TESTS_PROPERTIES(
                    IncNavierStokesSolverRedesign_TimeIntegrationOrders123_Device_SumFacTOP
                    PROPERTIES PROCESSORS 4)

                ADD_TEST(
                        NAME IncNavierStokesSolverRedesign_ForcedMMS2D_TriQuad_IMEXdirkOrders23_Device_SumFac
                        COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/check_imexdirk_forced_mms_2d.py
                        --solver $<TARGET_FILE:IncNavierStokesSolverRedesign>
                        --session ${CMAKE_CURRENT_SOURCE_DIR}/Tests/ChannelFlow2D.xml
                        --op-exec-space Device
                        --op-impl SumFac
                        --formulations linearimplicit semiimplicit
                        --orders 2 3)
                SET_TESTS_PROPERTIES(
                    IncNavierStokesSolverRedesign_ForcedMMS2D_TriQuad_IMEXdirkOrders23_Device_SumFac
                    PROPERTIES PROCESSORS 4)
            ENDIF()
        ENDIF()
    ENDIF(NEKTAR_BUILD_PYTHON)
ENDIF (NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_DEVICE_SUPPORT)
