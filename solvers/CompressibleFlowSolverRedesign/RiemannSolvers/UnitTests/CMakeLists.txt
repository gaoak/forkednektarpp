SET(RIEMANN_UNIT_TEST_DEPENDS SolverCore)

IF(BOOST_TEST_DYN_LINK)
    SET(RIEMANN_UNIT_TEST_DEPENDS ${RIEMANN_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(RiemannCompOpUnitTests 
    COMPONENT unit-test DEPENDS ${RIEMANN_UNIT_TEST_DEPENDS} SOURCES TestRiemann.cpp ${RIEMANN_FACTORY_SOURCES})
    
TARGET_INCLUDE_DIRECTORIES(RiemannCompOpUnitTests PRIVATE
    "${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign"
)

IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
SET_UNIT_TEST(CompressibleFlowSolverRiemannOpUnitTestsSerial RiemannCompOpUnitTests Serial none FALSE)
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(CompressibleFlowSolverRiemannOpUnitTestsAVX RiemannCompOpUnitTests AVX none FALSE)
ENDIF()
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(CompressibleFlowSolverRiemannOpUnitTestsDevice RiemannCompOpUnitTests Device none FALSE)
ENDIF()
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/solvers/CompressibleFlowSolverRedesign/RiemannSolvers/UnitTests/)
