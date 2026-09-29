SET(SpatialDomainsUnitTestsSources
    main.cpp
    TestMovement.cpp
    TestMakeOrder.cpp
    TestPyrInverseMap.cpp
    TestFaceOrientation.cpp
)

ADD_DEFINITIONS(-DENABLE_NEKTAR_EXCEPTIONS)
ADD_NEKTAR_EXECUTABLE(SpatialDomainsUnitTests
    COMPONENT unit-test DEPENDS SpatialDomains SOURCES ${SpatialDomainsUnitTestsSources})
ADD_TEST(NAME SpatialDomainsUnitTests COMMAND SpatialDomainsUnitTests --detect_memory_leaks=0)
