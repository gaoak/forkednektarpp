SET(LibUtilitiesUnitTestSources
    TestConsistentObjectAccess.cpp
    TestRawType.cpp
    TestInterpreter.cpp
    TestSharedArray.cpp
    TestCheckedCast.cpp
    TestRealComparison.cpp
    TestInterpCoeff.cpp
    TestPolylib.cpp
    TestCrystalRouter.cpp
    TestEntityResolver.cpp
    main.cpp
    ../util.cpp
)

SET(UnitTestSources ${PrecompiledHeaderSources} main.cpp)

SET(UnitTestHeaders
    LibUtilitiesUnitTestsPrecompiledHeader.h
    ../util.h
)

ADD_DEFINITIONS(-DENABLE_NEKTAR_EXCEPTIONS)
ADD_NEKTAR_EXECUTABLE(LibUtilitiesUnitTests
    COMPONENT unit-test DEPENDS LibUtilities SOURCES ${LibUtilitiesUnitTestSources})
ADD_TEST(NAME LibUtilitiesUnitTests COMMAND LibUtilitiesUnitTests --detect_memory_leaks=0)

# EntityResolver's MPI transports can only be exercised under mpirun at more
# than one rank, so this one lives outside the (serial) Boost test binary and is
# driven by the Tester. The rank counts (2, 3, 5, 8) live as sibling <test>
# elements in a single .tst file, and include non-powers of two, where the
# crystal router's odd splits are.
IF(NEKTAR_USE_MPI)
    ADD_NEKTAR_EXECUTABLE(EntityResolverMPITest
        COMPONENT unit-test DEPENDS LibUtilities
        SOURCES EntityResolverMPITest.cpp)

    ADD_NEKTAR_TEST(EntityResolverMPI)
ENDIF()

SUBDIRS(LinearAlgebra)
SUBDIRS(VmathTimer)
