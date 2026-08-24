IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_DEVICE_SUPPORT
        "Build the IncNavierStokesSolver using redesign operators." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NOT NEKTAR_BUILD_SOLVERS)
        SET(NEKTAR_SOLVER_INCNAVIERSTOKES_DEVICE_SUPPORT OFF CACHE INTERNAL "")
    ENDIF()
ENDIF()

IF (NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_DEVICE_SUPPORT)
    # Default timestepping schemes.
    SET(Formulations linearimplicit semiimplicit)

    # Macro
    MACRO(SET_ALL_INC_SOLVER_TEST solver session)
        FOREACH (Formulation IN LISTS Formulations)
            FOREACH (ExecSpace IN LISTS ExecSpaces)
                # Add SumFac test for all execution spaces
                SET(Implementation SumFac)
                ADD_TEST(
                        NAME ${solver}_${session}_${Formulation}_${ExecSpace}_${Implementation}
                        COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")

                # Check if Device or not and set implementation for available algorithm
                IF( ("${ExecSpace}" STREQUAL "Serial") OR ("${ExecSpace}" STREQUAL "AVX") )
                    SET(Implementation StdMat)
                ELSEIF( ("${ExecSpace}" STREQUAL "Device") )
                    SET(Implementation SumFacTOP)
                ENDIF()

                # Add test for either Serial/AVX + StdMat or Device + SumFacTOP
                ADD_TEST(
                        NAME ${solver}_${session}_${Formulation}_${ExecSpace}_${Implementation}
                        COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${session}.tst -v
                        --opExecSpace=${ExecSpace} --opImpl=${Implementation}
                        "--solver-options=-I Formulation=${Formulation}")
            ENDFOREACH ()
        ENDFOREACH ()
    ENDMACRO()

    ADD_SOLVER_EXECUTABLE(IncNavierStokesSolverRedesign
        DEPENDS SolverCore Operators
        SOURCES
            IncNavierStokesSolverRedesign.cpp
            EquationSystems/VelocityCorrectionScheme.cpp
    )

    IF(NEKTAR_BUILD_SOLVER_LIBS)
        MESSAGE(WARNING, "Solver library build not set up for IncNavierStokesSolverRedesign - skipping.")
    ENDIF()

    FOREACH(Session IN ITEMS
            UniformFlow2D
            CouetteFlow2D
            ChannelFlow2D
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
            ENDIF()
        ENDIF()
    ENDIF(NEKTAR_BUILD_PYTHON)
ENDIF (NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_DEVICE_SUPPORT)
