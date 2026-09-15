SET(RIEMANN_UNIT_TEST_DEPENDS SolverCore)

IF(BOOST_TEST_DYN_LINK)
    SET(RIEMANN_UNIT_TEST_DEPENDS ${RIEMANN_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(RiemannOpUnitTests 
    COMPONENT unit-test DEPENDS ${RIEMANN_UNIT_TEST_DEPENDS} SOURCES TestRiemann.cpp)
 
# Macro
MACRO(SET_UNIT_TEST testname name execspace)
    IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
        SET(name-ext "${name}-g")
    ELSEIF (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        SET(name-ext "${name}-ms")
    ELSEIF (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        SET(name-ext "${name}-rg")
    ELSE()
        SET(name-ext "${name}")
    ENDIF()

    ADD_TEST(
          NAME SolverCore${testname} 
          COMMAND ${name-ext} --detect_memory_leaks=0 -- ${execspace})
ENDMACRO()

IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
SET_UNIT_TEST(RiemannOpUnitTestsSerial RiemannOpUnitTests Serial)
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(RiemannOpUnitTestsAVX RiemannOpUnitTests AVX)
ENDIF()
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(RiemannOpUnitTestsDevice RiemannOpUnitTests Device)
ENDIF()
ENDIF()

# The test fixtures open their session files as "run/<mesh>.xml", relative
# to the working directory of the test. The meshes themselves are shared by
# every unit test and live in one place, so copy them from there.
FILE(COPY ${CMAKE_SOURCE_DIR}/library/UnitTests/run DESTINATION ${CMAKE_BINARY_DIR}/library/SolverCore/RiemannSolvers/UnitTests/)
