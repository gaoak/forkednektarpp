SET(LibUtilitiesUnitTestSources
    TestConsistentObjectAccess.cpp
    TestRawType.cpp
    TestInterpreter.cpp
    TestSharedArray.cpp
    TestCheckedCast.cpp
    TestRealComparison.cpp
    TestInterpCoeff.cpp
    TestPolylib.cpp
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

IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
	SET(DeviceFFTUnitTestSources
					TestDeviceFFT.cpp
					main.cpp
					../util.cpp
	)

	ADD_NEKTAR_EXECUTABLE(DeviceFFTUnitTests
					COMPONENT unit-test DEPENDS LibUtilities
					SOURCES ${DeviceFFTUnitTestSources})

	TARGET_LINK_LIBRARIES(DeviceFFTUnitTests PRIVATE ${NEKTAR_CUFFT_TARGET} CUDA::cudart)

	ADD_TEST(NAME DeviceFFTUnitTests COMMAND DeviceFFTUnitTests --detect_memory_leaks=0)
ENDIF()

SUBDIRS(LinearAlgebra)
SUBDIRS(VmathTimer)
