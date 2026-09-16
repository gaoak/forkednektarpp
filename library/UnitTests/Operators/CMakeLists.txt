# These tests can all likely be derived from the same cpp.in file
# AddTraceIntegral       Nodepend ReConfigure() - Seg, Quad, etc
# AssmbScatr             Nodepend - Seg, Quad, etc
# BwdTran                Nodepend - Seg, Quad, etc
# FwdTransBC             Nodepend - Seg, Quad, only 0D, 1D and 2D
# Helmholtz              Nodepend - Seg, Quad, etc
# Laplacian              Nodepend - Seg, Quad, etc
# IdentityCoeff          Nodepend - Seg, Quad, etc
# IdentityPhys           Nodepend - Seg, Quad, etc
# IProductWRTBase        Nodepend - Seg, Quad, etc
# IProductWRTDerivBase   Nodepend - Seg, Quad, etc
# LinAdvDiffReaction     Nodepend - Seg, Quad, etc
# Mass                   Nodepend - Seg, Quad, etc
# MultiplyByElmtInvMass  Nodepend - Seg, Quad, etc
# PhysDeriv              Nodepend - Seg, Quad, etc
# Advection              Nodepend - Seg, Quad, etc
# Divergence             Nodepend - Seg, Quad, etc
# CurlCurl               Nodepend - Seg, Quad, etc
# PhysInterp1D           Nodepend - Seg, Quad, etc
# Expression             Nodepend - Helmholtz1D_Seg, etc.

# These tests may require an individual  cpp.in file

# FwdTrans               create operator - DiagPrecon - Helmholtz1D_Seg, etc.
# HelmSolve              create operator - DiagPrecon - Helmholtz1D_Seg, etc.
# PoissonSolve           create operator - DiagPrecon - Poisson1D_Seg, etc.
# LinearADRSolve         create operator - DiagPrecon - Helmholtz1D_Seg, etc.

# Dirichlet              only output - Nodepend - Helmholtz1D_Seg, etc.
# Neumann                only output - Nodepend - Helmholtz1D_Seg, etc.

SET(OPERATOR_UNIT_TEST_DEPENDS SolverCore)

IF(BOOST_TEST_DYN_LINK)
    SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(test_optimization
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_optimization.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet.cpp)
ADD_NEKTAR_EXECUTABLE(test_neumann
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann.cpp)
#ADD_NEKTAR_EXECUTABLE(test_robin
#    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_robin.cpp)
ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass.cpp)
ADD_NEKTAR_EXECUTABLE(test_expression
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_expression.cpp)
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral.cpp)
ADD_NEKTAR_EXECUTABLE(test_bwdtrans
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtransbc
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtransbc.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz.cpp)
ADD_NEKTAR_EXECUTABLE(test_laplacian
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_laplacian.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase.cpp)
ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv.cpp)
ADD_NEKTAR_EXECUTABLE(test_advection
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_advection.cpp)
ADD_NEKTAR_EXECUTABLE(test_divergence
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_divergence.cpp)
ADD_NEKTAR_EXECUTABLE(test_curlcurl
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_curlcurl.cpp)
ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled.cpp)
ADD_NEKTAR_EXECUTABLE(test_physgalerkinproject1dscaled
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physgalerkinproject1dscaled.cpp)
ADD_NEKTAR_EXECUTABLE(test_advectiondealias
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_advectiondealias.cpp)
ADD_NEKTAR_EXECUTABLE(test_getfwdbwdtracephys
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_getfwdbwdtracephys.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtTrace
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtTrace.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtphysnormderivtrace
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtphysnormderivtrace.cpp)
ADD_NEKTAR_EXECUTABLE(test_phystraceextract
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_phystraceextract.cpp)

IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
SET_UNIT_TEST(Optimization test_optimization none none FALSE)

# Serial
SET_UNIT_TEST(AssmbScatr_serial test_assmbscatr Serial none FALSE)
SET_UNIT_TEST(Dirichlet_serial test_dirichlet Serial none FALSE)
SET_UNIT_TEST(Neumann_serial test_neumann Serial SumFac FALSE)
#SET_UNIT_TEST(Robin_serial test_robin Serial none FALSE)
SET_UNIT_TEST(Getfwdbwdtracephys_serial test_getfwdbwdtracephys Serial none FALSE)
IF (NEKTAR_USE_MPI)
    SET_UNIT_TEST(AssmbScatr_serial test_assmbscatr Serial none TRUE)
    SET_UNIT_TEST(Dirichlet_serial test_dirichlet Serial none TRUE)
ENDIF()

# Serial generic
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_generic test_multiplybyelmtinvmass Serial Generic FALSE)
SET_UNIT_TEST(FwdTransBC_serial_generic test_fwdtransbc Serial Generic FALSE)
SET_UNIT_TEST(Expression_serial_generic test_expression Serial Generic FALSE)

# Serial StdMat
SET_UNIT_TEST(IProductWRTPhysTrace_serial_stdmat test_ipwrtTrace Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_serial_stdmat test_ipwrtphysnormderivtrace Serial StdMat FALSE)
SET_UNIT_TEST(PhysTraceExtract_serial_stdmat test_phystraceextract Serial StdMat FALSE)
SET_UNIT_TEST(AddTraceIntegral_serial_stdmat test_addtraceintegral Serial StdMat FALSE)
SET_UNIT_TEST(BwdTrans_serial_stdmat test_bwdtrans Serial StdMat FALSE)
SET_UNIT_TEST(Helmholtz_serial_stdmat test_helmholtz Serial StdMat FALSE)
SET_UNIT_TEST(Laplacian_serial_stdmat test_laplacian Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_stdmat test_ipwrtbase Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_stdmat test_ipwrtderivbase Serial StdMat FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_stdmat test_linadvdiffreaction Serial StdMat FALSE)
SET_UNIT_TEST(Mass_serial_stdmat test_mass Serial StdMat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat test_physderiv Serial StdMat FALSE)
SET_UNIT_TEST(Advection_serial_stdmat test_advection Serial StdMat FALSE)
SET_UNIT_TEST(Divergence_serial_stdmat test_divergence Serial StdMat FALSE)
SET_UNIT_TEST(CurlCurl_serial_stdmat test_curlcurl Serial StdMat FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_stdmat test_physinterp1dscaled Serial StdMat FALSE)
SET_UNIT_TEST(PhysGalerkinProject1DScaled_serial_stdmat test_physgalerkinproject1dscaled Serial StdMat FALSE)
SET_UNIT_TEST(AdvectionDealias_serial_stdmat test_advectiondealias Serial StdMat FALSE)

# Serial SumFac
SET_UNIT_TEST(IProductWRTPhysTrace_serial_sumfac test_ipwrtTrace Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_serial_sumfac test_ipwrtphysnormderivtrace Serial SumFac FALSE)
SET_UNIT_TEST(PhysTraceExtract_serial_sumfac test_phystraceextract Serial SumFac FALSE)
SET_UNIT_TEST(AddTraceIntegral_serial_sumfac test_addtraceintegral Serial SumFac FALSE)
SET_UNIT_TEST(BwdTrans_serial_sumfac test_bwdtrans Serial SumFac FALSE)
SET_UNIT_TEST(Helmholtz_serial_sumfac test_helmholtz Serial SumFac FALSE)
SET_UNIT_TEST(Laplacian_serial_sumfac test_laplacian Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_sumfac test_ipwrtderivbase Serial SumFac FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_sumfac test_linadvdiffreaction Serial SumFac FALSE)
SET_UNIT_TEST(Mass_serial_sumfac test_mass Serial SumFac FALSE)
SET_UNIT_TEST(PhysDeriv_serial_sumfac test_physderiv Serial SumFac FALSE)
SET_UNIT_TEST(Advection_serial_sumfac test_advection Serial SumFac FALSE)
SET_UNIT_TEST(Divergence_serial_sumfac test_divergence Serial SumFac FALSE)
SET_UNIT_TEST(CurlCurl_serial_sumfac test_curlcurl Serial SumFac FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_sumfac test_physinterp1dscaled Serial SumFac FALSE)
SET_UNIT_TEST(PhysGalerkinProject1DScaled_serial_sumfac test_physgalerkinproject1dscaled Serial SumFac FALSE)
SET_UNIT_TEST(AdvectionDealias_serial_sumfac test_advectiondealias Serial SumFac FALSE)

# AVX
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(AssmbScatr_avx test_assmbscatr AVX none FALSE)
    SET_UNIT_TEST(Dirichlet_avx  test_dirichlet AVX none FALSE)
    SET_UNIT_TEST(Neumann_avx test_neumann AVX SumFac FALSE)
    SET_UNIT_TEST(Getfwdbwdtracephys_avx test_getfwdbwdtracephys AVX none FALSE)
    IF (NEKTAR_USE_MPI)
        # SET_UNIT_TEST(AssmbScatr_avx test_assmbscatr AVX none TRUE)
    ENDIF()
ENDIF()

# AVX generic
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(MultiplyByElmtInvMass_avx_generic test_multiplybyelmtinvmass AVX Generic FALSE)
    SET_UNIT_TEST(FwdTransBC_avx_generic test_fwdtransbc AVX Generic FALSE)
    SET_UNIT_TEST(Expression_avx_generic test_expression AVX Generic FALSE)
ENDIF()

# AVX StdMat
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(IProductWRTPhysTrace_avx_stdmat test_ipwrtTrace AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_avx_stdmat test_ipwrtphysnormderivtrace AVX StdMat FALSE)
    SET_UNIT_TEST(PhysTraceExtract_avx_stdmat test_phystraceextract AVX StdMat FALSE)
    SET_UNIT_TEST(AddTraceIntegral_avx_stdmat test_addtraceintegral AVX StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_avx_stdmat test_bwdtrans AVX StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_avx_stdmat test_helmholtz AVX StdMat FALSE)
    SET_UNIT_TEST(Laplacian_avx_stdmat test_laplacian AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_stdmat test_ipwrtbase AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_stdmat test_ipwrtderivbase AVX StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_stdmat test_linadvdiffreaction AVX StdMat FALSE)
    SET_UNIT_TEST(Mass_avx_stdmat test_mass AVX StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_stdmat test_physderiv AVX StdMat FALSE)
    SET_UNIT_TEST(Advection_avx_stdmat test_advection AVX StdMat FALSE)
    SET_UNIT_TEST(Divergence_avx_stdmat test_divergence AVX StdMat FALSE)
    SET_UNIT_TEST(CurlCurl_avx_stdmat test_curlcurl AVX StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_stdmat test_physinterp1dscaled Serial StdMat FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_avx_stdmat test_physgalerkinproject1dscaled AVX StdMat FALSE)
    SET_UNIT_TEST(AdvectionDealias_avx_stdmat test_advectiondealias AVX StdMat FALSE)
ENDIF()

# AVX SumFac
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(IProductWRTPhysTrace_avx_sumfac test_ipwrtTrace AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_avx_sumfac test_ipwrtphysnormderivtrace AVX SumFac FALSE)
    SET_UNIT_TEST(PhysTraceExtract_avx_sumfac test_phystraceextract AVX SumFac FALSE)
    SET_UNIT_TEST(AddTraceIntegral_avx_sumfac test_addtraceintegral AVX SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_avx_sumfac test_bwdtrans AVX SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_avx_sumfac test_helmholtz AVX SumFac FALSE)
    SET_UNIT_TEST(Laplacian_avx_sumfac test_laplacian AVX SumFac FALSE)    
    SET_UNIT_TEST(IProductWRTBase_avx_sumfac test_ipwrtbase AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_sumfac test_ipwrtderivbase AVX SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_sumfac test_linadvdiffreaction AVX SumFac FALSE)
    SET_UNIT_TEST(Mass_avx_sumfac test_mass AVX SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_sumfac test_physderiv AVX SumFac FALSE)
    SET_UNIT_TEST(Advection_avx_sumfac test_advection AVX SumFac FALSE)
    SET_UNIT_TEST(Divergence_avx_sumfac test_divergence AVX SumFac FALSE)
    SET_UNIT_TEST(CurlCurl_avx_sumfac test_curlcurl AVX SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_sumfac test_physinterp1dscaled AVX SumFac FALSE)
    SET_UNIT_TEST(AdvectionDealias_avx_sumfac test_advectiondealias AVX SumFac FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_avx_sumfac test_physgalerkinproject1dscaled AVX SumFac FALSE)
ENDIF()


# Device
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(AssmbScatr_device test_assmbscatr Device none FALSE)
    SET_UNIT_TEST(Dirichlet_device test_dirichlet Device none FALSE)
    SET_UNIT_TEST(Neumann_device test_neumann Device SumFac FALSE)
    SET_UNIT_TEST(Getfwdbwdtracephys_device test_getfwdbwdtracephys Device none FALSE)
    IF (NEKTAR_USE_MPI)
        # Disabled: intermittently hangs in mesh partitioning (MeshPartition::
        # PartitionGraph) under CI, unrelated to the Device execution space --
        # AssmbScatr_serial_MPI/Dirichlet_serial_MPI already exercise the same
        # partitioning code path with 2 MPI ranks.
        # SET_UNIT_TEST(AssmbScatr_device test_assmbscatr Device none TRUE)
        # SET_UNIT_TEST(Dirichlet_device test_dirichlet Device none TRUE)
    ENDIF()
ENDIF()

# Device generic
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(MultiplyByElmtInvMass_device_generic test_multiplybyelmtinvmass Device Generic FALSE)
    SET_UNIT_TEST(FwdTransBC_device_generic test_fwdtransbc Device Generic FALSE)
    SET_UNIT_TEST(Expression_device_generic test_expression Device Generic FALSE)
ENDIF()

# Device StdMat
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_device_stdmat test_ipwrtphysnormderivtrace Device StdMat FALSE)
    SET_UNIT_TEST(PhysTraceExtract_device_stdmat test_phystraceextract Device StdMat FALSE)
    SET_UNIT_TEST(AddTraceIntegral_device_stdmat test_addtraceintegral Device StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_device_stdmat test_bwdtrans Device StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_device_stdmat test_helmholtz Device StdMat FALSE)
    SET_UNIT_TEST(Laplacian_device_stdmat test_laplacian Device StdMat FALSE)    
    SET_UNIT_TEST(IProductWRTBase_device_stdmat test_ipwrtbase Device StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_stdmat test_ipwrtderivbase Device StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_stdmat test_linadvdiffreaction Device StdMat FALSE)
    SET_UNIT_TEST(Mass_device_stdmat test_mass Device StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_device_stdmat test_physderiv Device StdMat FALSE)
    SET_UNIT_TEST(Advection_device_stdmat test_advection Device StdMat FALSE)
    SET_UNIT_TEST(Divergence_device_stdmat test_divergence Device StdMat FALSE)
    SET_UNIT_TEST(CurlCurl_device_stdmat test_curlcurl Device StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_stdmat test_physinterp1dscaled Device StdMat FALSE)
    SET_UNIT_TEST(AdvectionDealias_device_stdmat test_advectiondealias Device StdMat FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_device_stdmat test_physgalerkinproject1dscaled Device StdMat FALSE)
ENDIF()

# Device SumFac
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(IProductWRTPhysTrace_device_sumfac test_ipwrtTrace Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTPhysNormalDerivTrace_device_sumfac test_ipwrtphysnormderivtrace Device SumFac FALSE)
    SET_UNIT_TEST(PhysTraceExtract_device_sumfac test_phystraceextract Device SumFac FALSE)
    SET_UNIT_TEST(AddTraceIntegral_device_sumfac test_addtraceintegral Device SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfac test_bwdtrans Device SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfac test_helmholtz Device SumFac FALSE)
    SET_UNIT_TEST(Laplacian_device_sumfac test_laplacian Device SumFac FALSE)    
    SET_UNIT_TEST(IProductWRTBase_device_sumfac test_ipwrtbase Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac test_ipwrtderivbase Device SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfac test_linadvdiffreaction Device SumFac FALSE)
    SET_UNIT_TEST(Mass_device_sumfac test_mass Device SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac test_physderiv Device SumFac FALSE)
    SET_UNIT_TEST(Advection_device_sumfac test_advection Device SumFac FALSE)
    SET_UNIT_TEST(Divergence_device_sumfac test_divergence Device SumFac FALSE)
    SET_UNIT_TEST(CurlCurl_device_sumfac test_curlcurl Device SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfac test_physinterp1dscaled Device SumFac FALSE)
    SET_UNIT_TEST(AdvectionDealias_device_sumfac test_advectiondealias Device SumFac FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_device_sumfac test_physgalerkinproject1dscaled Device SumFac FALSE)
ENDIF()

# Device SumFacTOP
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(AddTraceIntegral_device_sumfactop test_addtraceintegral Device SumFacTOP FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfactop test_bwdtrans Device SumFacTOP FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfactop test_helmholtz Device SumFacTOP FALSE)
    SET_UNIT_TEST(Laplacian_device_sumfactop test_laplacian Device SumFacTOP FALSE)    
    SET_UNIT_TEST(IProductWRTBase_device_sumfactop test_ipwrtbase Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfactop test_ipwrtderivbase Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfactop test_linadvdiffreaction Device SumFacTOP FALSE)
    SET_UNIT_TEST(Mass_device_sumfactop test_mass Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfactop test_physderiv Device SumFacTOP FALSE)
    SET_UNIT_TEST(Advection_device_sumfactop test_advection Device SumFacTOP FALSE)
    SET_UNIT_TEST(Divergence_device_sumfactop test_divergence Device SumFacTOP FALSE)
    SET_UNIT_TEST(CurlCurl_device_sumfactop test_curlcurl Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfactop test_physinterp1dscaled Device SumFacTOP FALSE)
    SET_UNIT_TEST(AdvectionDealias_device_sumfactop test_advectiondealias Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysGalerkinProject1DScaled_device_sumfactop test_physgalerkinproject1dscaled Device SumFacTOP FALSE)
ENDIF()

ENDIF()

# The test fixtures open their session files as "run/<mesh>.xml", relative
# to the working directory of the test. The meshes themselves are shared by
# every unit test and live in one place, so copy them from there.
FILE(COPY ${CMAKE_SOURCE_DIR}/library/UnitTests/run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
