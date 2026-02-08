IF(NEKTAR_BUILD_REDESIGN)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_REDESIGN
        "Build the Redesign Solver." ON
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

        ADD_SOLVER_EXECUTABLE(RedesignSolver SOURCES
            RedesignSolver.cpp
            EquationSystems/UnsteadyDiffusion.cpp
            EquationSystems/Poisson.cpp
        )

        IF(NEKTAR_BUILD_SOLVER_LIBS)
            MESSAGE(WARNING, "Solver library build not set up for RedesignSolver - skipping.")
        ENDIF()

        SET_ALL_SOLVER_TEST(RedesignSolver ImplicitDiffusion1D)
        SET_ALL_SOLVER_TEST(RedesignSolver ImplicitDiffusion2D)
        SET_ALL_SOLVER_TEST(RedesignSolver ImplicitDiffusion3D)
        SET_ALL_SOLVER_TEST(RedesignSolver SteadyDiffusion1D)
        SET_ALL_SOLVER_TEST(RedesignSolver SteadyDiffusion2D)
        SET_ALL_SOLVER_TEST(RedesignSolver SteadyDiffusion3D)
    #    SET_ALL_SOLVER_TEST(RedesignSolver ImplicitDiffusion3D_3C)
    #    SET_ALL_SOLVER_TEST(RedesignSolver ReactionDiffusion1D)
    #    SET_ALL_SOLVER_TEST(RedesignSolver ReactionDiffusion2D)
    #    SET_ALL_SOLVER_TEST(RedesignSolver ReactionDiffusion3D)
    #    SET_ALL_SOLVER_TEST(RedesignSolver ReactionDiffusion3D_3C)
    ENDIF()
ENDIF(NEKTAR_SOLVER_REDESIGN)
