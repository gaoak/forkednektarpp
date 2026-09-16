SET(UnitTestSources
    main.cpp
    ScaledMatrixUnitTests.cpp
    BlockMatrixUnitTests.cpp
    TestLinearSystem.cpp
    TestNekMatrix.cpp
    TestNekPoint.cpp
    TestNekVector.cpp
    TestNekSharedArray.cpp
    Memory/TestNekMemoryManager.cpp
    util.cpp
)

ADD_DEFINITIONS(-DENABLE_NEKTAR_EXCEPTIONS -DNEKTAR_UNIT_TESTS)

ADD_NEKTAR_EXECUTABLE(UnitTests
    COMPONENT unit-test DEPENDS StdRegions SOURCES ${UnitTestSources})

ADD_TEST(NAME UnitTests COMMAND UnitTests --detect_memory_leaks=0)

SUBDIRS(LibUtilities StdRegions LocalRegions Collections MultiRegions SpatialDomains NekMesh)

if (NEKTAR_ENABLE_DEVICE_SUPPORT)
    SUBDIRS(Operators)
    SUBDIRS(SolverCore)
endif()
