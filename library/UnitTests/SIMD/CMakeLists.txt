SET(VecDataTestSources
    main.cpp
    TestSimdLibDouble.cpp
    TestSimdLibSingle.cpp
)

ADD_DEFINITIONS(-DENABLE_NEKTAR_EXCEPTIONS)
ADD_NEKTAR_EXECUTABLE(VecDataUnitTests
    COMPONENT unit-test DEPENDS tinysimd SOURCES ${VecDataTestSources})

ADD_DEPENDENCIES(VecDataUnitTests boost)
IF (NEKTAR_USE_SIMSYCL)
    ADD_DEPENDENCIES(VecDataUnitTests simsycl-${SIMSYCL_VERSION})
ENDIF()

ADD_TEST(NAME VecDataUnitTests COMMAND VecDataUnitTests --detect_memory_leaks=0)
