IF(NEKTAR_BUILD_REDESIGN)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_REDESIGN
        "Build the ADR Solver." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NEKTAR_SOLVER_REDESIGN)
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

        ADD_SOLVER_EXECUTABLE(ADRSolverRedesign SOURCES
            ADRSolverRedesign.cpp
            EquationSystems/UnsteadyDiffusion.cpp
	    EquationSystems/SteadyADR.cpp
	    EquationSystems/UnsteadyADR.cpp
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

        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff1DBaseFlow)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff2DBaseFlow)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiff3DBaseFlow)

        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction1D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction2D)
        SET_ALL_SOLVER_TEST(ADRSolverRedesign SteadyAdvecDiffReaction3D)

        # SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff1D)
        # SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff2D)
        # SET_ALL_SOLVER_TEST(ADRSolverRedesign UnsteadyAdvecDiff3D)

    ENDIF()
ENDIF(NEKTAR_BUILD_REDESIGN)
