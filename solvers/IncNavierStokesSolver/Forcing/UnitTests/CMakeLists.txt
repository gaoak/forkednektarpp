SET(IncNSForcingUnitTestsSources
    main.cpp
    TestForcingMovingBody.cpp
)

ADD_SOLVER_UTILITY(IncNSForcingUnitTests IncNavierStokesSolver
    SOURCES ${IncNSForcingUnitTestsSources} USELIBRARY)
ADD_TEST(NAME IncNSForcingUnitTests COMMAND IncNSForcingUnitTests --detect_memory_leaks=0)
