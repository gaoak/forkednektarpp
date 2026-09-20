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

IF(NEKTAR_ENABLE_DEVICE)
    SET(DeviceFFTUnitTestSources
        TestDeviceFFT.cpp
        main.cpp
        ../util.cpp
    )

    ADD_NEKTAR_EXECUTABLE(DeviceFFTUnitTests
                          COMPONENT unit-test DEPENDS LibUtilities
                          SOURCES ${DeviceFFTUnitTestSources})

    ADD_TEST(NAME DeviceFFTUnitTests COMMAND DeviceFFTUnitTests --detect_memory_leaks=0)
ENDIF()

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

# Math and Comm unit tests. These exercise the LibUtilities math kernels and
# device-aware communicators, neither of which is built without device support.
IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    SET(LIBUTILITIES_UNIT_TEST_DEPENDS MultiRegions)

    IF(BOOST_TEST_DYN_LINK)
        SET(LIBUTILITIES_UNIT_TEST_DEPENDS ${LIBUTILITIES_UNIT_TEST_DEPENDS}
            ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    ENDIF()

    ADD_NEKTAR_EXECUTABLE(TestMath
        COMPONENT unit-test DEPENDS ${LIBUTILITIES_UNIT_TEST_DEPENDS} SOURCES TestMath.cpp)
    ADD_NEKTAR_EXECUTABLE(TestComm
        COMPONENT unit-test DEPENDS ${LIBUTILITIES_UNIT_TEST_DEPENDS} SOURCES TestComm.cpp)
    IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        ADD_NEKTAR_EXECUTABLE(TestParallelReduce
            COMPONENT unit-test DEPENDS ${LIBUTILITIES_UNIT_TEST_DEPENDS} SOURCES TestParallelReduce.cu)
    ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
        ADD_NEKTAR_EXECUTABLE(TestParallelReduce
            COMPONENT unit-test DEPENDS ${LIBUTILITIES_UNIT_TEST_DEPENDS} SOURCES TestParallelReduce.hip)
    ELSE()
        ADD_NEKTAR_EXECUTABLE(TestParallelReduce
            COMPONENT unit-test DEPENDS ${LIBUTILITIES_UNIT_TEST_DEPENDS} SOURCES TestParallelReduce.cpp)
    ENDIF()

    # The fixtures in TestMath.hpp and TestParallelReduce.hpp open
    # "run/segment.xml" relative to the working directory of the test, which
    # is this directory in the build tree. The meshes themselves are shared
    # by every unit test and live in one place, so copy them from there.
    FILE(COPY ${CMAKE_SOURCE_DIR}/library/UnitTests/run DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    # Registers one test per execution space. Named rather than overriding the
    # SET_UNIT_TEST of cmake/NektarCommon.cmake, which the sibling Operators
    # unit tests use with a different signature.
    MACRO(SET_LIBUTILITIES_UNIT_TEST testname name execspace mpi_flag)
        IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
            SET(name-ext "${name}-g")
        ELSEIF (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
            SET(name-ext "${name}-ms")
        ELSEIF (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            SET(name-ext "${name}-rg")
        ELSE()
            SET(name-ext "${name}")
        ENDIF()

        IF (${mpi_flag} AND NEKTAR_USE_MPI)
            IF (MSVC)
                ADD_TEST(
                  NAME LibUtilities${testname}_MPI
                  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${name-ext}
                          --detect_memory_leaks=0 -- ${execspace})
            ELSE()
                ADD_TEST(
                  NAME LibUtilities${testname}_MPI
                  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/${name-ext}
                          --detect_memory_leaks=0 -- ${execspace})
            ENDIF()
        ELSE()
            ADD_TEST(
              NAME LibUtilities${testname}
              COMMAND ${name-ext} --detect_memory_leaks=0 -- ${execspace})
        ENDIF()
    ENDMACRO()

    IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
    # Comm
    IF (NEKTAR_USE_MPI)
        SET_LIBUTILITIES_UNIT_TEST(CommHost TestComm HostSpace TRUE)
        SET_LIBUTILITIES_UNIT_TEST(CommDevice TestComm DeviceSpace TRUE)
    ENDIF()

    # Math
    SET_LIBUTILITIES_UNIT_TEST(MathSerial TestMath Serial FALSE)
    IF (NEKTAR_ENABLE_SIMD)
        SET_LIBUTILITIES_UNIT_TEST(MathAVX TestMath AVX FALSE)
    ENDIF()
    IF(NEKTAR_ENABLE_DEVICE)
        SET_LIBUTILITIES_UNIT_TEST(MathDevice TestMath Device FALSE)
    ENDIF()

    # Parallel reduce
    SET_LIBUTILITIES_UNIT_TEST(ParallelReduceSerial TestParallelReduce Serial FALSE)
    IF (NEKTAR_ENABLE_SIMD)
        SET_LIBUTILITIES_UNIT_TEST(ParallelReduceAVX TestParallelReduce AVX FALSE)
    ENDIF()
    IF(NEKTAR_ENABLE_DEVICE)
        SET_LIBUTILITIES_UNIT_TEST(ParallelReduceDevice TestParallelReduce Device FALSE)
    ENDIF()
    ENDIF()
ENDIF()

SUBDIRS(LinearAlgebra)
SUBDIRS(VmathTimer)
