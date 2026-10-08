SET(SOLVERCORE_UNIT_TEST_DEPENDS SolverCore)

IF(BOOST_TEST_DYN_LINK)
    SET(SOLVERCORE_UNIT_TEST_DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(TestRungeKutta
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestRungeKutta.cpp)
ADD_NEKTAR_EXECUTABLE(TestAdamsBashforth
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestAdamsBashforth.cpp)
ADD_NEKTAR_EXECUTABLE(TestDIRK
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestDIRK.cpp)
ADD_NEKTAR_EXECUTABLE(TestAdamsMoulton
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestAdamsMoulton.cpp)
ADD_NEKTAR_EXECUTABLE(TestBDF
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestBDF.cpp)
ADD_NEKTAR_EXECUTABLE(TestCNAB
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestCNAB.cpp)
ADD_NEKTAR_EXECUTABLE(TestIMEX
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestIMEX.cpp)
ADD_NEKTAR_EXECUTABLE(TestIMEXdirk
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestIMEXdirk.cpp)
ADD_NEKTAR_EXECUTABLE(TestExplicitSDC
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestExplicitSDC.cpp)
ADD_NEKTAR_EXECUTABLE(TestImplicitSDC
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestImplicitSDC.cpp)
ADD_NEKTAR_EXECUTABLE(TestIMEXSDC
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestIMEXSDC.cpp)
ADD_NEKTAR_EXECUTABLE(TestExplicitGEM
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestExplicitGEM.cpp)
ADD_NEKTAR_EXECUTABLE(TestImplicitGEM
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestImplicitGEM.cpp)
ADD_NEKTAR_EXECUTABLE(TestIMEXGEM
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestIMEXGEM.cpp)
ADD_NEKTAR_EXECUTABLE(TestNorm
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestNormOps.cpp)
ADD_NEKTAR_EXECUTABLE(TestMaxStdVelocity
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestMaxStdVelocity.cpp)
ADD_NEKTAR_EXECUTABLE(TestMeanRemoval
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestMeanRemoval.cpp)
ADD_NEKTAR_EXECUTABLE(TestFieldFileIO
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestFieldFileIO.cpp)
ADD_NEKTAR_EXECUTABLE(TestNullPrecon
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestNullPrecon.cpp)
ADD_NEKTAR_EXECUTABLE(TestDiagPrecon
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestDiagPrecon.cpp)
ADD_NEKTAR_EXECUTABLE(TestFwdTrans
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestFwdTrans.cpp)
ADD_NEKTAR_EXECUTABLE(TestPoissonSolve
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestPoissonSolve.cpp)
ADD_NEKTAR_EXECUTABLE(TestHelmSolve
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestHelmSolve.cpp)
ADD_NEKTAR_EXECUTABLE(TestLinearADRSolve
    COMPONENT unit-test DEPENDS ${SOLVERCORE_UNIT_TEST_DEPENDS} SOURCES TestLinearADRSolve.cpp)

# Registers one test per execution space. Named rather than overriding the
# SET_UNIT_TEST of cmake/NektarCommon.cmake, which the sibling Operators unit
# tests use with a different signature.
MACRO(SET_SOLVERCORE_UNIT_TEST testname name execspace)
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

# As SET_SOLVERCORE_UNIT_TEST, but for tests that also select an operator
# implementation: the preconditioners are configured from a Helmholtz operator,
# whose StdMat/SumFac/SumFacTOP implementations are each worth covering.
MACRO(SET_SOLVERCORE_UNIT_TEST_IMPL testname name execspace implementation)
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
          COMMAND ${name-ext} --detect_memory_leaks=0 -- ${execspace} ${implementation})
ENDMACRO()

IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
SET_SOLVERCORE_UNIT_TEST(RungeKuttaSerial TestRungeKutta Serial)
SET_SOLVERCORE_UNIT_TEST(AdamsBashforthSerial TestAdamsBashforth Serial)
SET_SOLVERCORE_UNIT_TEST(DIRKSerial TestDIRK Serial)
SET_SOLVERCORE_UNIT_TEST(AdamsMoultonSerial TestAdamsMoulton Serial)
SET_SOLVERCORE_UNIT_TEST(BDFSerial TestBDF Serial)
SET_SOLVERCORE_UNIT_TEST(CNABSerial TestCNAB Serial)
SET_SOLVERCORE_UNIT_TEST(IMEXSerial TestIMEX Serial)
SET_SOLVERCORE_UNIT_TEST(IMEXdirkSerial TestIMEXdirk Serial)
SET_SOLVERCORE_UNIT_TEST(ExplicitSDCSerial TestExplicitSDC Serial)
SET_SOLVERCORE_UNIT_TEST(ImplicitSDCSerial TestImplicitSDC Serial)
SET_SOLVERCORE_UNIT_TEST(IMEXSDCSerial TestIMEXSDC Serial)
SET_SOLVERCORE_UNIT_TEST(ExplicitGEMSerial TestExplicitGEM Serial)
SET_SOLVERCORE_UNIT_TEST(ImplicitGEMSerial TestImplicitGEM Serial)
SET_SOLVERCORE_UNIT_TEST(IMEXGEMSerial TestIMEXGEM Serial)
SET_SOLVERCORE_UNIT_TEST(NormSerial TestNorm Serial)
SET_SOLVERCORE_UNIT_TEST(MaxStdVelocitySerial TestMaxStdVelocity Serial)
SET_SOLVERCORE_UNIT_TEST(MeanRemovalSerial TestMeanRemoval Serial)
SET_SOLVERCORE_UNIT_TEST(FieldFileIOSerial TestFieldFileIO Serial)
SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_serial_stdmat TestNullPrecon Serial StdMat)
SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_serial_sumfac TestNullPrecon Serial SumFac)
SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_serial_stdmat TestDiagPrecon Serial StdMat)
SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_serial_sumfac TestDiagPrecon Serial SumFac)
SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_serial_stdmat TestFwdTrans Serial StdMat)
SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_serial_sumfac TestFwdTrans Serial SumFac)
SET_SOLVERCORE_UNIT_TEST_IMPL(PoissonSolve_serial_stdmat TestPoissonSolve Serial StdMat)
SET_SOLVERCORE_UNIT_TEST_IMPL(PoissonSolve_serial_sumfac TestPoissonSolve Serial SumFac)
SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_serial_stdmat TestHelmSolve Serial StdMat)
SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_serial_sumfac TestHelmSolve Serial SumFac)
SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_serial_stdmat TestLinearADRSolve Serial StdMat)
SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_serial_sumfac TestLinearADRSolve Serial SumFac)
IF (NEKTAR_USE_MPI)
    # SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_serial TestHelmSolve Serial SumFac)
ENDIF()
IF (NEKTAR_ENABLE_SIMD)
    SET_SOLVERCORE_UNIT_TEST(RungeKuttaAVX TestRungeKutta AVX)
    SET_SOLVERCORE_UNIT_TEST(AdamsBashforthAVX TestAdamsBashforth AVX)
    SET_SOLVERCORE_UNIT_TEST(DIRKAVX TestDIRK AVX)
    SET_SOLVERCORE_UNIT_TEST(AdamsMoultonAVX TestAdamsMoulton AVX)
    SET_SOLVERCORE_UNIT_TEST(BDFAVX TestBDF AVX)
    SET_SOLVERCORE_UNIT_TEST(CNABAVX TestCNAB AVX)
    SET_SOLVERCORE_UNIT_TEST(IMEXAVX TestIMEX AVX)
    SET_SOLVERCORE_UNIT_TEST(IMEXdirkAVX TestIMEXdirk AVX)
    SET_SOLVERCORE_UNIT_TEST(ExplicitSDCAVX TestExplicitSDC AVX)
    SET_SOLVERCORE_UNIT_TEST(ImplicitSDCAVX TestImplicitSDC AVX)
    SET_SOLVERCORE_UNIT_TEST(IMEXSDCAVX TestIMEXSDC AVX)
    SET_SOLVERCORE_UNIT_TEST(ExplicitGEMAVX TestExplicitGEM AVX)
    SET_SOLVERCORE_UNIT_TEST(ImplicitGEMAVX TestImplicitGEM AVX)
    SET_SOLVERCORE_UNIT_TEST(IMEXGEMAVX TestIMEXGEM AVX)
    SET_SOLVERCORE_UNIT_TEST(NormAVX TestNorm AVX)
    SET_SOLVERCORE_UNIT_TEST(MaxStdVelocityAVX TestMaxStdVelocity AVX)
    SET_SOLVERCORE_UNIT_TEST(MeanRemovalAVX TestMeanRemoval AVX)
    SET_SOLVERCORE_UNIT_TEST(FieldFileIOAVX TestFieldFileIO AVX)
    SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_avx_stdmat TestNullPrecon AVX StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_avx_sumfac TestNullPrecon AVX SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_avx_stdmat TestDiagPrecon AVX StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_avx_sumfac TestDiagPrecon AVX SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_avx_stdmat TestFwdTrans AVX StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_avx_sumfac TestFwdTrans AVX SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_avx_stdmat TestHelmSolve AVX StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_avx_sumfac TestHelmSolve AVX SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_avx_stdmat TestLinearADRSolve AVX StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_avx_sumfac TestLinearADRSolve AVX SumFac)
ENDIF()
IF(NEKTAR_ENABLE_DEVICE)
    SET_SOLVERCORE_UNIT_TEST(RungeKuttaDevice TestRungeKutta Device)
    SET_SOLVERCORE_UNIT_TEST(AdamsBashforthDevice TestAdamsBashforth Device)
    SET_SOLVERCORE_UNIT_TEST(DIRKDevice TestDIRK Device)
    SET_SOLVERCORE_UNIT_TEST(AdamsMoultonDevice TestAdamsMoulton Device)
    SET_SOLVERCORE_UNIT_TEST(BDFDevice TestBDF Device)
    SET_SOLVERCORE_UNIT_TEST(CNABDevice TestCNAB Device)
    SET_SOLVERCORE_UNIT_TEST(IMEXDevice TestIMEX Device)
    SET_SOLVERCORE_UNIT_TEST(IMEXdirkDevice TestIMEXdirk Device)
    SET_SOLVERCORE_UNIT_TEST(ExplicitSDCDevice TestExplicitSDC Device)
    SET_SOLVERCORE_UNIT_TEST(ImplicitSDCDevice TestImplicitSDC Device)
    SET_SOLVERCORE_UNIT_TEST(IMEXSDCDevice TestIMEXSDC Device)
    SET_SOLVERCORE_UNIT_TEST(ExplicitGEMDevice TestExplicitGEM Device)
    SET_SOLVERCORE_UNIT_TEST(ImplicitGEMDevice TestImplicitGEM Device)
    SET_SOLVERCORE_UNIT_TEST(IMEXGEMDevice TestIMEXGEM Device)
    SET_SOLVERCORE_UNIT_TEST(NormDevice TestNorm Device)
    SET_SOLVERCORE_UNIT_TEST(MaxStdVelocityDevice TestMaxStdVelocity Device)
    SET_SOLVERCORE_UNIT_TEST(MeanRemovalDevice TestMeanRemoval Device)
    SET_SOLVERCORE_UNIT_TEST(FieldFileIODevice TestFieldFileIO Device)
    SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_device_stdmat TestNullPrecon Device StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_device_sumfac TestNullPrecon Device SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(NullPrecon_device_sumfactop TestNullPrecon Device SumFacTOP)
    SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_device_stdmat TestDiagPrecon Device StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_device_sumfac TestDiagPrecon Device SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(DiagPrecon_device_sumfactop TestDiagPrecon Device SumFacTOP)
    SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_device_stdmat TestFwdTrans Device StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_device_sumfac TestFwdTrans Device SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(FwdTrans_device_sumfactop TestFwdTrans Device SumFacTOP)
    SET_SOLVERCORE_UNIT_TEST_IMPL(PoissonSolve_device_stdmat TestPoissonSolve Device StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(PoissonSolve_device_sumfac TestPoissonSolve Device SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(PoissonSolve_device_sumfactop TestPoissonSolve Device SumFacTOP)
    SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_device_stdmat TestHelmSolve Device StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_device_sumfac TestHelmSolve Device SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(HelmSolve_device_sumfactop TestHelmSolve Device SumFacTOP)
    SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_device_stdmat TestLinearADRSolve Device StdMat)
    SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_device_sumfac TestLinearADRSolve Device SumFac)
    SET_SOLVERCORE_UNIT_TEST_IMPL(LinearADRSolve_device_sumfactop TestLinearADRSolve Device SumFacTOP)
ENDIF()
ENDIF()

# The test fixtures open their session files as "run/<mesh>.xml", relative
# to the working directory of the test. The meshes themselves are shared by
# every unit test and live in one place, so copy them from there.
FILE(COPY ${CMAKE_SOURCE_DIR}/library/UnitTests/run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/SolverCore/)
