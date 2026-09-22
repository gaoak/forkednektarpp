# These tests can all likely be derived from the same cpp.in file
# AssmbScatr                      Nodepend - Seg, Quad, etc
# BwdTrans                        Nodepend - Seg, Quad, etc
# FwdTransBC                      Nodepend - Seg, Quad, only 0D, 1D and 2D
# Helmholtz                       Nodepend - Seg, Quad, etc
# Laplacian                       Nodepend - Seg, Quad, etc
# IProductWRTBase                 Nodepend - Seg, Quad, etc
# IProductWRTDerivBase            Nodepend - Seg, Quad, etc
# LinAdvDiffReaction              Nodepend - Seg, Quad, etc
# Mass                            Nodepend - Seg, Quad, etc
# MultiplyByElmtInvMass           Nodepend - Seg, Quad, etc
# PhysDeriv                       Nodepend - Seg, Quad, etc
# Advection                       Nodepend - Seg, Quad, etc
# Divergence                      Nodepend - Seg, Quad, etc
# CurlCurl                        Nodepend - Seg, Quad, etc
# PhysInterp1DScaled              Nodepend - Seg, Quad, etc
# PhysGalerkinProject1DScaled     Nodepend - Seg, Quad, etc
# AdvectionDealias                Nodepend - Seg, Quad, etc
# PhysTraceExtract                Nodepend - Seg, Quad, etc
# IProductWRTPhysTrace            Nodepend - Seg, Quad, etc
# IProductWRTPhysNormalDerivTrace Nodepend - Seg, Quad, etc
# Expression                      Nodepend - Helmholtz1D_Seg, etc.

# Dirichlet                       only output - Nodepend - Helmholtz1D_Seg, etc.
# Neumann                         only output - Nodepend - Helmholtz1D_Seg, etc.
# Robin                           only output - Nodepend - Helmholtz2D_AllBCs
# NOTE: TestRobin is not built - its executable and test lines are commented
#       out below.

# Optimization                    backend/impl selection - optimization.xml

SET(OPERATOR_UNIT_TEST_DEPENDS SolverCore)

IF(BOOST_TEST_DYN_LINK)
    SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(TestOptimization
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestOptimization.cpp)
ADD_NEKTAR_EXECUTABLE(TestAssmbScatr
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestAssmbScatr.cpp)
ADD_NEKTAR_EXECUTABLE(TestDirichlet
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestDirichlet.cpp)
ADD_NEKTAR_EXECUTABLE(TestNeumann
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestNeumann.cpp)
#ADD_NEKTAR_EXECUTABLE(TestRobin
#    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestRobin.cpp)
ADD_NEKTAR_EXECUTABLE(TestMultiplyByElmtInvMass
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestMultiplyByElmtInvMass.cpp)
ADD_NEKTAR_EXECUTABLE(TestExpression
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestExpression.cpp)
ADD_NEKTAR_EXECUTABLE(TestBwdTrans
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestBwdTrans.cpp)
ADD_NEKTAR_EXECUTABLE(TestFwdTransBC
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestFwdTransBC.cpp)
ADD_NEKTAR_EXECUTABLE(TestHelmholtz
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestHelmholtz.cpp)
ADD_NEKTAR_EXECUTABLE(TestLaplacian
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestLaplacian.cpp)
ADD_NEKTAR_EXECUTABLE(TestIProductWRTBase
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestIProductWRTBase.cpp)
ADD_NEKTAR_EXECUTABLE(TestIProductWRTDerivBase
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestIProductWRTDerivBase.cpp)
ADD_NEKTAR_EXECUTABLE(TestLinAdvDiffReaction
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestLinAdvDiffReaction.cpp)
ADD_NEKTAR_EXECUTABLE(TestMass
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestMass.cpp)
ADD_NEKTAR_EXECUTABLE(TestPhysDeriv
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestPhysDeriv.cpp)
ADD_NEKTAR_EXECUTABLE(TestAdvection
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestAdvection.cpp)
ADD_NEKTAR_EXECUTABLE(TestDivergence
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestDivergence.cpp)
ADD_NEKTAR_EXECUTABLE(TestCurlCurl
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestCurlCurl.cpp)
ADD_NEKTAR_EXECUTABLE(TestPhysInterp1DScaled
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestPhysInterp1DScaled.cpp)
ADD_NEKTAR_EXECUTABLE(TestPhysGalerkinProject1DScaled
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestPhysGalerkinProject1DScaled.cpp)
ADD_NEKTAR_EXECUTABLE(TestAdvectionDealias
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestAdvectionDealias.cpp)
ADD_NEKTAR_EXECUTABLE(TestIProductWRTPhysTrace
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestIProductWRTPhysTrace.cpp)
ADD_NEKTAR_EXECUTABLE(TestIProductWRTPhysNormalDerivTrace
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestIProductWRTPhysNormalDerivTrace.cpp)
ADD_NEKTAR_EXECUTABLE(TestPhysTraceExtract
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES TestPhysTraceExtract.cpp)

IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
SET_UNIT_TEST(Optimization TestOptimization none none FALSE)

# Serial
SET_UNIT_TEST(AssmbScatr_serial TestAssmbScatr Serial none FALSE)
SET_UNIT_TEST(Dirichlet_serial TestDirichlet Serial none FALSE)
SET_UNIT_TEST(Neumann_serial TestNeumann Serial SumFac FALSE)
#SET_UNIT_TEST(Robin_serial TestRobin Serial none FALSE)
IF (NEKTAR_USE_MPI)
    SET_UNIT_TEST(AssmbScatr_serial TestAssmbScatr Serial none TRUE)
    SET_UNIT_TEST(Dirichlet_serial TestDirichlet Serial none TRUE)
ENDIF()

# Serial generic
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_generic TestMultiplyByElmtInvMass Serial Generic FALSE)
SET_UNIT_TEST(FwdTransBC_serial_generic TestFwdTransBC Serial Generic FALSE)
SET_UNIT_TEST(Expression_serial_generic TestExpression Serial Generic FALSE)

# Serial StdMat
SET_UNIT_TEST(IProductWRTPhysTrace_serial_stdmat TestIProductWRTPhysTrace Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_serial_stdmat TestIProductWRTPhysNormalDerivTrace Serial StdMat FALSE)
SET_UNIT_TEST(PhysTraceExtract_serial_stdmat TestPhysTraceExtract Serial StdMat FALSE)
SET_UNIT_TEST(BwdTrans_serial_stdmat TestBwdTrans Serial StdMat FALSE)
SET_UNIT_TEST(Helmholtz_serial_stdmat TestHelmholtz Serial StdMat FALSE)
SET_UNIT_TEST(Laplacian_serial_stdmat TestLaplacian Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_stdmat TestIProductWRTBase Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_stdmat TestIProductWRTDerivBase Serial StdMat FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_stdmat TestLinAdvDiffReaction Serial StdMat FALSE)
SET_UNIT_TEST(Mass_serial_stdmat TestMass Serial StdMat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat TestPhysDeriv Serial StdMat FALSE)
SET_UNIT_TEST(Advection_serial_stdmat TestAdvection Serial StdMat FALSE)
SET_UNIT_TEST(Divergence_serial_stdmat TestDivergence Serial StdMat FALSE)
SET_UNIT_TEST(CurlCurl_serial_stdmat TestCurlCurl Serial StdMat FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_stdmat TestPhysInterp1DScaled Serial StdMat FALSE)
SET_UNIT_TEST(PhysGalerkinProject1DScaled_serial_stdmat TestPhysGalerkinProject1DScaled Serial StdMat FALSE)
SET_UNIT_TEST(AdvectionDealias_serial_stdmat TestAdvectionDealias Serial StdMat FALSE)

# Serial SumFac
SET_UNIT_TEST(IProductWRTPhysTrace_serial_sumfac TestIProductWRTPhysTrace Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_serial_sumfac TestIProductWRTPhysNormalDerivTrace Serial SumFac FALSE)
SET_UNIT_TEST(PhysTraceExtract_serial_sumfac TestPhysTraceExtract Serial SumFac FALSE)
SET_UNIT_TEST(BwdTrans_serial_sumfac TestBwdTrans Serial SumFac FALSE)
SET_UNIT_TEST(Helmholtz_serial_sumfac TestHelmholtz Serial SumFac FALSE)
SET_UNIT_TEST(Laplacian_serial_sumfac TestLaplacian Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_sumfac TestIProductWRTBase Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_sumfac TestIProductWRTDerivBase Serial SumFac FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_sumfac TestLinAdvDiffReaction Serial SumFac FALSE)
SET_UNIT_TEST(Mass_serial_sumfac TestMass Serial SumFac FALSE)
SET_UNIT_TEST(PhysDeriv_serial_sumfac TestPhysDeriv Serial SumFac FALSE)
SET_UNIT_TEST(Advection_serial_sumfac TestAdvection Serial SumFac FALSE)
SET_UNIT_TEST(Divergence_serial_sumfac TestDivergence Serial SumFac FALSE)
SET_UNIT_TEST(CurlCurl_serial_sumfac TestCurlCurl Serial SumFac FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_sumfac TestPhysInterp1DScaled Serial SumFac FALSE)
SET_UNIT_TEST(PhysGalerkinProject1DScaled_serial_sumfac TestPhysGalerkinProject1DScaled Serial SumFac FALSE)
SET_UNIT_TEST(AdvectionDealias_serial_sumfac TestAdvectionDealias Serial SumFac FALSE)

# AVX
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(AssmbScatr_avx TestAssmbScatr AVX none FALSE)
    SET_UNIT_TEST(Dirichlet_avx  TestDirichlet AVX none FALSE)
    SET_UNIT_TEST(Neumann_avx TestNeumann AVX SumFac FALSE)
    IF (NEKTAR_USE_MPI)
        # SET_UNIT_TEST(AssmbScatr_avx TestAssmbScatr AVX none TRUE)
    ENDIF()
ENDIF()

# AVX generic
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(MultiplyByElmtInvMass_avx_generic TestMultiplyByElmtInvMass AVX Generic FALSE)
    SET_UNIT_TEST(FwdTransBC_avx_generic TestFwdTransBC AVX Generic FALSE)
    SET_UNIT_TEST(Expression_avx_generic TestExpression AVX Generic FALSE)
ENDIF()

# AVX StdMat
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(IProductWRTPhysTrace_avx_stdmat TestIProductWRTPhysTrace AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_avx_stdmat TestIProductWRTPhysNormalDerivTrace AVX StdMat FALSE)
    SET_UNIT_TEST(PhysTraceExtract_avx_stdmat TestPhysTraceExtract AVX StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_avx_stdmat TestBwdTrans AVX StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_avx_stdmat TestHelmholtz AVX StdMat FALSE)
    SET_UNIT_TEST(Laplacian_avx_stdmat TestLaplacian AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_stdmat TestIProductWRTBase AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_stdmat TestIProductWRTDerivBase AVX StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_stdmat TestLinAdvDiffReaction AVX StdMat FALSE)
    SET_UNIT_TEST(Mass_avx_stdmat TestMass AVX StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_stdmat TestPhysDeriv AVX StdMat FALSE)
    SET_UNIT_TEST(Advection_avx_stdmat TestAdvection AVX StdMat FALSE)
    SET_UNIT_TEST(Divergence_avx_stdmat TestDivergence AVX StdMat FALSE)
    SET_UNIT_TEST(CurlCurl_avx_stdmat TestCurlCurl AVX StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_stdmat TestPhysInterp1DScaled Serial StdMat FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_avx_stdmat TestPhysGalerkinProject1DScaled AVX StdMat FALSE)
    SET_UNIT_TEST(AdvectionDealias_avx_stdmat TestAdvectionDealias AVX StdMat FALSE)
ENDIF()

# AVX SumFac
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(IProductWRTPhysTrace_avx_sumfac TestIProductWRTPhysTrace AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_avx_sumfac TestIProductWRTPhysNormalDerivTrace AVX SumFac FALSE)
    SET_UNIT_TEST(PhysTraceExtract_avx_sumfac TestPhysTraceExtract AVX SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_avx_sumfac TestBwdTrans AVX SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_avx_sumfac TestHelmholtz AVX SumFac FALSE)
    SET_UNIT_TEST(Laplacian_avx_sumfac TestLaplacian AVX SumFac FALSE)    
    SET_UNIT_TEST(IProductWRTBase_avx_sumfac TestIProductWRTBase AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_sumfac TestIProductWRTDerivBase AVX SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_sumfac TestLinAdvDiffReaction AVX SumFac FALSE)
    SET_UNIT_TEST(Mass_avx_sumfac TestMass AVX SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_sumfac TestPhysDeriv AVX SumFac FALSE)
    SET_UNIT_TEST(Advection_avx_sumfac TestAdvection AVX SumFac FALSE)
    SET_UNIT_TEST(Divergence_avx_sumfac TestDivergence AVX SumFac FALSE)
    SET_UNIT_TEST(CurlCurl_avx_sumfac TestCurlCurl AVX SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_sumfac TestPhysInterp1DScaled AVX SumFac FALSE)
    SET_UNIT_TEST(AdvectionDealias_avx_sumfac TestAdvectionDealias AVX SumFac FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_avx_sumfac TestPhysGalerkinProject1DScaled AVX SumFac FALSE)
ENDIF()


# Device
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(AssmbScatr_device TestAssmbScatr Device none FALSE)
    SET_UNIT_TEST(Dirichlet_device TestDirichlet Device none FALSE)
    SET_UNIT_TEST(Neumann_device TestNeumann Device SumFac FALSE)
    IF (NEKTAR_USE_MPI)
        # Disabled: intermittently hangs in mesh partitioning (MeshPartition::
        # PartitionGraph) under CI, unrelated to the Device execution space --
        # AssmbScatr_serial_MPI/Dirichlet_serial_MPI already exercise the same
        # partitioning code path with 2 MPI ranks.
        # SET_UNIT_TEST(AssmbScatr_device TestAssmbScatr Device none TRUE)
        # SET_UNIT_TEST(Dirichlet_device TestDirichlet Device none TRUE)
    ENDIF()
ENDIF()

# Device generic
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(MultiplyByElmtInvMass_device_generic TestMultiplyByElmtInvMass Device Generic FALSE)
    SET_UNIT_TEST(FwdTransBC_device_generic TestFwdTransBC Device Generic FALSE)
    SET_UNIT_TEST(Expression_device_generic TestExpression Device Generic FALSE)
ENDIF()

# Device StdMat
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(IProductWRTPhysTrace_device_stdmat TestIProductWRTPhysTrace Device StdMat FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_device_stdmat TestIProductWRTPhysNormalDerivTrace Device StdMat FALSE)
    SET_UNIT_TEST(PhysTraceExtract_device_stdmat TestPhysTraceExtract Device StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_device_stdmat TestBwdTrans Device StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_device_stdmat TestHelmholtz Device StdMat FALSE)
    SET_UNIT_TEST(Laplacian_device_stdmat TestLaplacian Device StdMat FALSE)    
    SET_UNIT_TEST(IProductWRTBase_device_stdmat TestIProductWRTBase Device StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_stdmat TestIProductWRTDerivBase Device StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_stdmat TestLinAdvDiffReaction Device StdMat FALSE)
    SET_UNIT_TEST(Mass_device_stdmat TestMass Device StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_device_stdmat TestPhysDeriv Device StdMat FALSE)
    SET_UNIT_TEST(Advection_device_stdmat TestAdvection Device StdMat FALSE)
    SET_UNIT_TEST(Divergence_device_stdmat TestDivergence Device StdMat FALSE)
    SET_UNIT_TEST(CurlCurl_device_stdmat TestCurlCurl Device StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_stdmat TestPhysInterp1DScaled Device StdMat FALSE)
    SET_UNIT_TEST(AdvectionDealias_device_stdmat TestAdvectionDealias Device StdMat FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_device_stdmat TestPhysGalerkinProject1DScaled Device StdMat FALSE)
ENDIF()

# Device SumFac
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(IProductWRTPhysTrace_device_sumfac TestIProductWRTPhysTrace Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_device_sumfac TestIProductWRTPhysNormalDerivTrace Device SumFac FALSE)
    SET_UNIT_TEST(PhysTraceExtract_device_sumfac TestPhysTraceExtract Device SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfac TestBwdTrans Device SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfac TestHelmholtz Device SumFac FALSE)
    SET_UNIT_TEST(Laplacian_device_sumfac TestLaplacian Device SumFac FALSE)    
    SET_UNIT_TEST(IProductWRTBase_device_sumfac TestIProductWRTBase Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac TestIProductWRTDerivBase Device SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfac TestLinAdvDiffReaction Device SumFac FALSE)
    SET_UNIT_TEST(Mass_device_sumfac TestMass Device SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac TestPhysDeriv Device SumFac FALSE)
    SET_UNIT_TEST(Advection_device_sumfac TestAdvection Device SumFac FALSE)
    SET_UNIT_TEST(Divergence_device_sumfac TestDivergence Device SumFac FALSE)
    SET_UNIT_TEST(CurlCurl_device_sumfac TestCurlCurl Device SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfac TestPhysInterp1DScaled Device SumFac FALSE)
    SET_UNIT_TEST(AdvectionDealias_device_sumfac TestAdvectionDealias Device SumFac FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_device_sumfac TestPhysGalerkinProject1DScaled Device SumFac FALSE)
ENDIF()

# Device SumFacTOP
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(IProductWRTPhysTrace_device_sumfactop TestIProductWRTPhysTrace Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_device_sumfactop TestIProductWRTPhysNormalDerivTrace Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysTraceExtract_device_sumfactop TestPhysTraceExtract Device SumFacTOP FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfactop TestBwdTrans Device SumFacTOP FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfactop TestHelmholtz Device SumFacTOP FALSE)
    SET_UNIT_TEST(Laplacian_device_sumfactop TestLaplacian Device SumFacTOP FALSE)    
    SET_UNIT_TEST(IProductWRTBase_device_sumfactop TestIProductWRTBase Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfactop TestIProductWRTDerivBase Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfactop TestLinAdvDiffReaction Device SumFacTOP FALSE)
    SET_UNIT_TEST(Mass_device_sumfactop TestMass Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfactop TestPhysDeriv Device SumFacTOP FALSE)
    SET_UNIT_TEST(Advection_device_sumfactop TestAdvection Device SumFacTOP FALSE)
    SET_UNIT_TEST(Divergence_device_sumfactop TestDivergence Device SumFacTOP FALSE)
    SET_UNIT_TEST(CurlCurl_device_sumfactop TestCurlCurl Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfactop TestPhysInterp1DScaled Device SumFacTOP FALSE)
    SET_UNIT_TEST(AdvectionDealias_device_sumfactop TestAdvectionDealias Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_device_sumfactop TestPhysGalerkinProject1DScaled Device SumFacTOP FALSE)
ENDIF()

ENDIF()

# The test fixtures open their session files as "run/<mesh>.xml", relative
# to the working directory of the test. The meshes themselves are shared by
# every unit test and live in one place, so copy them from there.
FILE(COPY ${CMAKE_SOURCE_DIR}/library/UnitTests/run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
