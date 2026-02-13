SET(RIEMANN_UNIT_TEST_DEPENDS Operators)

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
          NAME SolverUtilsOps${testname} 
          COMMAND ${name-ext} --detect_memory_leaks=0 -- ${execspace})
ENDMACRO()

SET_UNIT_TEST(RiemannOpUnitTestsSerial RiemannOpUnitTests Serial)
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(RiemannOpUnitTestsAVX RiemannOpUnitTests AVX)
ENDIF()
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(RiemannOpUnitTestsDevice RiemannOpUnitTests Device)
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/Operators/SolverUtilsOps/RiemannSolvers/UnitTests/)
