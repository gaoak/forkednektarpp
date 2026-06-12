IF(NEKTAR_BUILD_REDESIGN)
    CMAKE_DEPENDENT_OPTION(NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_RE
        "Build the IncNavierStokesSolver using redesign operators." ON
        "NEKTAR_BUILD_SOLVERS" OFF)

    IF (NEKTAR_SOLVER_INCNAVIERSTOKESSOLVER_RE)
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
            SET_ALL_SOLVER_TEST(IncNavierStokesSolverRedesign ${Session})
        ENDFOREACH()

    ENDIF()
ENDIF(NEKTAR_BUILD_REDESIGN)
