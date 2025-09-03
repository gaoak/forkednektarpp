# These tests can all likely be derived from the same cpp.in file
# AddTraceIntegral       Nodepend ReConfigure() - Seg, Quad, etc
# AssmbScatr             Nodepend - Seg, Quad, etc
# BwdTran                Nodepend - Seg, Quad, etc
# Helmholtz              Nodepend - Seg, Quad, etc
# IdentityCoeff          Nodepend - Seg, Quad, etc
# IdentityPhys           Nodepend - Seg, Quad, etc
# IProductWRTBase        Nodepend - Seg, Quad, etc
# IProductWRTDerivBase   Nodepend - Seg, Quad, etc
# LinAdvDiffReaction     Nodepend - Seg, Quad, etc
# Mass                   Nodepend - Seg, Quad, etc
# MultiplyByElmtInvMass  Nodepend - Seg, Quad, etc
# NullPrecon             Nodepend - Seg, Quad, etc
# PhysDeriv              Nodepend - Seg, Quad, etc
# PhysInterp1D           Nodepend - Seg, Quad, etc
# Expression             Nodepend - Helmholtz1D_Seg, etc.

# These tests may require an individual  cpp.in file
# DiagPrecon             create operator - Helmholtz - Helmholtz1D_Seg, etc.

# FwdTrans               create operator - DiagPrecon - Helmholtz1D_Seg, etc.
# HelmSolve              create operator - DiagPrecon - Helmholtz1D_Seg, etc.

# Dirichlet              only output - Nodepend - Helmholtz1D_Seg, etc.
# Neumann                only output - Nodepend - Helmholtz1D_Seg, etc.

SET(OPERATOR_UNIT_TEST_DEPENDS Operators)

IF(BOOST_TEST_DYN_LINK)
    SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(test_math
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet.cpp)
ADD_NEKTAR_EXECUTABLE(test_neumann
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann.cpp)
ADD_NEKTAR_EXECUTABLE(test_runge_kutta
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_runge_kutta.cpp)
ADD_NEKTAR_EXECUTABLE(test_runge_kutta_ssp
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_runge_kutta_ssp.cpp)
ADD_NEKTAR_EXECUTABLE(test_adams_bashforth
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_adams_bashforth.cpp)
ADD_NEKTAR_EXECUTABLE(test_cnab
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_cnab.cpp)
ADD_NEKTAR_EXECUTABLE(test_mcnab
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mcnab.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirk
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirk.cpp)
ADD_NEKTAR_EXECUTABLE(test_esdirk
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_esdirk.cpp)
ADD_NEKTAR_EXECUTABLE(test_adams_moulton
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_adams_moulton.cpp)
ADD_NEKTAR_EXECUTABLE(test_bdf
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bdf.cpp)
ADD_NEKTAR_EXECUTABLE(test_imex
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_imex.cpp)

ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass.cpp)
ADD_NEKTAR_EXECUTABLE(test_expression
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_expression.cpp)

ADD_NEKTAR_EXECUTABLE(test_addtraceintegral
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral.cpp)
ADD_NEKTAR_EXECUTABLE(test_bwdtrans
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans.cpp)
ADD_NEKTAR_EXECUTABLE(test_diagprecon
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtrans
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_conjgrad
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_conjgrad.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_gmres
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_gmres.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase.cpp)
ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass.cpp)
ADD_NEKTAR_EXECUTABLE(test_nullprecon
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv.cpp)
ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled.cpp)

# CUDA SumFacMat
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_device_sumfacmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_device_sumfacmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_device_sumfacmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_device_sumfacmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_device_sumfacmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_device_sumfacmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv_device_sumfacmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_device_sumfacmat.cpp)
ENDIF()

# Macro
MACRO(SET_UNIT_TEST testname name execspace implementation mpi_flag)
    IF (${mpi_flag})
        ADD_TEST(
          NAME Operators${testname}MPI
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/${name} --detect_memory_leaks=0 -- ${execspace} ${implementation}
      )
    ELSE()
        ADD_TEST(
          NAME Operators${testname} 
          COMMAND ${name} --detect_memory_leaks=0 -- ${execspace} ${implementation})
    ENDIF()
ENDMACRO()

# Serial
SET_UNIT_TEST(Math_serial test_math Serial none  FALSE)
SET_UNIT_TEST(AssmbScatr_serial test_assmbscatr Serial none FALSE)
SET_UNIT_TEST(Dirichlet_serial test_dirichlet Serial none FALSE)
SET_UNIT_TEST(Neumann_serial test_neumann Serial none FALSE)
SET_UNIT_TEST(RungeKutta_serial test_runge_kutta Serial none FALSE)
SET_UNIT_TEST(RungeKuttaSSP_serial test_runge_kutta_ssp Serial none FALSE)
SET_UNIT_TEST(AdamsBashforth_serial test_adams_bashforth Serial none FALSE)
SET_UNIT_TEST(CNAB_serial test_cnab Serial none FALSE)
SET_UNIT_TEST(MCNAB_serial test_mcnab Serial none FALSE)
SET_UNIT_TEST(DIRK_serial test_dirk Serial none FALSE)
SET_UNIT_TEST(ESDIRK_serial test_esdirk Serial none FALSE)
SET_UNIT_TEST(AdamsMoulton_serial test_adams_moulton Serial none FALSE)
SET_UNIT_TEST(BDF_serial test_bdf Serial none FALSE)
SET_UNIT_TEST(IMEX_serial test_imex Serial none FALSE)

# Serial generic
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_generic test_multiplybyelmtinvmass Serial Generic FALSE)
SET_UNIT_TEST(expression_serial_generic test_expression Serial Generic FALSE)

# Serial StdMat
SET_UNIT_TEST(AddTraceIntegral_serial_stdmat test_addtraceintegral Serial StdMat FALSE)
SET_UNIT_TEST(BwdTrans_serial_stdmat test_bwdtrans Serial StdMat FALSE)
SET_UNIT_TEST(DiagPrecon_serial_stdmat test_diagprecon Serial StdMat FALSE)
SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans Serial StdMat FALSE)
SET_UNIT_TEST(Helmholtz_serial_stdmat test_helmholtz Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_conjgrad_serial_stdmat test_helmsolve_conjgrad Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_gmres_serial_stdmat test_helmsolve_gmres Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_stdmat test_ipwrtbase Serial StdMat FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_stdmat test_ipwrtderivbase Serial StdMat FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_stdmat test_linadvdiffreaction Serial StdMat FALSE)
SET_UNIT_TEST(Mass_serial_stdmat test_mass Serial StdMat FALSE)
SET_UNIT_TEST(NullPrecon_serial_stdmat test_nullprecon Serial StdMat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat test_physderiv Serial StdMat FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_stdmat test_physinterp1dscaled Serial StdMat FALSE)

# Serial SumFac
SET_UNIT_TEST(AddTraceIntegral_serial_sumfac test_addtraceintegral Serial SumFac FALSE)
SET_UNIT_TEST(BwdTrans_serial_sumfac test_bwdtrans Serial SumFac FALSE)
SET_UNIT_TEST(DiagPrecon_serial_sumfac test_diagprecon Serial SumFac FALSE)
SET_UNIT_TEST(FwdTrans_serial_sumfac test_fwdtrans Serial SumFac FALSE)
SET_UNIT_TEST(Helmholtz_serial_sumfac test_helmholtz Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_conjgrad_serial_sumfac test_helmsolve_conjgrad Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_gmres_serial_sumfac test_helmsolve_gmres Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_sumfac test_ipwrtderivbase Serial SumFac FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_sumfac test_linadvdiffreaction Serial SumFac FALSE)
SET_UNIT_TEST(Mass_serial_sumfac test_mass Serial SumFac FALSE)
SET_UNIT_TEST(NullPrecon_serial_sumfac test_nullprecon Serial SumFac FALSE)
SET_UNIT_TEST(PhysDeriv_serial_sumfac test_physderiv Serial SumFac FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_sumfac test_physinterp1dscaled Serial SumFac FALSE)

# AVX
IF (${NEKTAR_ENABLE_SIMD})
    SET_UNIT_TEST(Math_avx test_math AVX none  FALSE)
    SET_UNIT_TEST(AssmbScatr_avx test_assmbscatr AVX none FALSE)
    SET_UNIT_TEST(Dirichlet_avx  test_dirichlet AVX none FALSE)
    SET_UNIT_TEST(Neumann_avx test_neumann AVX none FALSE)
    SET_UNIT_TEST(RungeKutta_avx test_runge_kutta AVX none FALSE)
    SET_UNIT_TEST(RungeKuttaSSP_avx test_runge_kutta_ssp AVX none FALSE)
    SET_UNIT_TEST(AdamsBashforth_avx test_adams_bashforth AVX none FALSE)
    SET_UNIT_TEST(CNAB_avx test_cnab AVX none FALSE)
    SET_UNIT_TEST(MCNAB_avx test_mcnab AVX none FALSE)
    SET_UNIT_TEST(DIRK_avx test_dirk AVX none FALSE)
    SET_UNIT_TEST(ESDIRK_avx test_esdirk AVX none FALSE)
    SET_UNIT_TEST(AdamsMoulton_avx test_adams_moulton AVX none FALSE)
    SET_UNIT_TEST(BDF_avx test_bdf AVX none FALSE)
    SET_UNIT_TEST(IMEX_avx test_imex AVX none FALSE)
ENDIF()

# AVX generic
IF (${NEKTAR_ENABLE_SIMD})
    # SET_UNIT_TEST(MultiplyByElmtInvMass_avx_generic test_multiplybyelmtinvmass AVX Generic FALSE)
    # SET_UNIT_TEST(expression_avx_generic test_expression AVX Generic FALSE)
ENDIF()

# AVX StdMat
IF (${NEKTAR_ENABLE_SIMD})
    # SET_UNIT_TEST(AddTraceIntegral_avx_stdmat test_addtraceintegral AVX StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_avx_stdmat test_bwdtrans AVX StdMat FALSE)
    SET_UNIT_TEST(DiagPrecon_avx_stdmat test_diagprecon AVX StdMat FALSE)
    # SET_UNIT_TEST(FwdTrans_avx_stdmat AVX StdMat test_fwdtrans AVX StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_avx_stdmat test_helmholtz AVX StdMat FALSE)
    # SET_UNIT_TEST(HelmSolve_conjgrad_avx_stdmat test_helmsolve_conjgrad AVX StdMat FALSE)
    # SET_UNIT_TEST(HelmSolve_gmres_avx_stdmat test_helmsolve_gmres AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_stdmat test_ipwrtbase AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_stdmat test_ipwrtderivbase AVX StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_stdmat test_linadvdiffreaction AVX StdMat FALSE)
    SET_UNIT_TEST(Mass_avx_stdmat test_mass AVX StdMat FALSE)
    # SET_UNIT_TEST(NullPrecon_avx_stdmat test_nullprecon AVX StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_stdmat test_physderiv AVX StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_stdmat test_physinterp1dscaled Serial StdMat FALSE)
ENDIF()

# AVX SumFac
IF (${NEKTAR_ENABLE_SIMD})
    # SET_UNIT_TEST(AddTraceIntegral_avx_sumfac test_addtraceintegral AVX SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_avx_sumfac test_bwdtrans AVX SumFac FALSE)
    SET_UNIT_TEST(DiagPrecon_avx_sumfac test_diagprecon AVX SumFac FALSE)
    # SET_UNIT_TEST(FwdTrans_avx_sumfac Serial StdMat test_fwdtrans AVX SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_avx_sumfac test_helmholtz AVX SumFac FALSE)
    # SET_UNIT_TEST(HelmSolve_conjgrad_avx_sumfac test_helmsolve_conjgrad AVX SumFac FALSE)
    # SET_UNIT_TEST(HelmSolve_gmres_avx_sumfa test_helmsolve_gmres AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_sumfac test_ipwrtbase AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_sumfac test_ipwrtderivbase AVX SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_sumfac test_linadvdiffreaction AVX SumFac FALSE)
    SET_UNIT_TEST(Mass_avx_sumfac test_mass AVX SumFac FALSE)
    # SET_UNIT_TEST(NullPrecon_avx_sumfac test_nullprecon AVX SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_sumfac test_physderiv AVX SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_sumfac test_physinterp1dscaled AVX SumFac FALSE)
ENDIF()

# Device
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    SET_UNIT_TEST(Math_device test_math Device none  FALSE)
    SET_UNIT_TEST(AssmbScatr_device test_assmbscatr Device none FALSE)
    SET_UNIT_TEST(Dirichlet_device test_dirichlet Device none FALSE)
    SET_UNIT_TEST(Neumann_device test_neumann Device none FALSE)
    SET_UNIT_TEST(RungeKutta_device test_runge_kutta Device none FALSE)
    SET_UNIT_TEST(RungeKuttaSSP_device test_runge_kutta_ssp Device none FALSE)
    SET_UNIT_TEST(AdamsBashforth_device test_adams_bashforth Device none FALSE)
    SET_UNIT_TEST(CNAB_device test_cnab Device none FALSE)
    SET_UNIT_TEST(MCNAB_device test_mcnab Device none FALSE)
    SET_UNIT_TEST(DIRK_device test_dirk Device none FALSE)
    SET_UNIT_TEST(ESDIRK_device test_esdirk Device none FALSE)
    SET_UNIT_TEST(AdamsMoulton_device test_adams_moulton Device none FALSE)
    SET_UNIT_TEST(BDF_device test_bdf Device none FALSE)
    SET_UNIT_TEST(IMEX_device test_imex Device none FALSE)
ENDIF()

# Device generic
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    IF(NOT NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
        SET_UNIT_TEST(MultiplyByElmtInvMass_device_generic test_multiplybyelmtinvmass Device Generic FALSE)
        # SET_UNIT_TEST(expression_device_generic test_expression Device Generic FALSE)
    ENDIF()
ENDIF()

# Device StdMat
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    IF(NOT NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
        SET_UNIT_TEST(AddTraceIntegral_device_stdmat test_addtraceintegral Device StdMat FALSE)
        SET_UNIT_TEST(BwdTrans_device_stdmat test_bwdtrans Device StdMat FALSE)
        SET_UNIT_TEST(DiagPrecon_device_stdmat test_diagprecon Device StdMat FALSE)
        SET_UNIT_TEST(FwdTrans_device_stdmat test_fwdtrans Device StdMat FALSE)
        SET_UNIT_TEST(Helmholtz_device_stdmat test_helmholtz Device StdMat FALSE)
        SET_UNIT_TEST(HelmSolve_conjgrad_device_stdmat test_helmsolve_conjgrad Device StdMat FALSE)
        SET_UNIT_TEST(HelmSolve_gmres_device_stdmat test_helmsolve_gmres Device StdMat FALSE)
        SET_UNIT_TEST(IProductWRTBase_device_stdmat test_ipwrtbase Device StdMat FALSE)
        SET_UNIT_TEST(IProductWRTDerivBase_device_stdmat test_ipwrtderivbase Device StdMat FALSE)
        SET_UNIT_TEST(LinAdvDiffReaction_device_stdmat test_linadvdiffreaction Device StdMat FALSE)
        SET_UNIT_TEST(Mass_device_stdmat test_mass Device StdMat FALSE)
        SET_UNIT_TEST(NullPrecon_device_stdmat test_nullprecon Device StdMat FALSE)
        SET_UNIT_TEST(PhysDeriv_device_stdmat test_physderiv Device StdMat FALSE)
        SET_UNIT_TEST(PhysInterp1DScaled_device_stdmat test_physinterp1dscaled Device StdMat FALSE)
    ENDIF()
ENDIF()

# Device SumFac
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    #SET_UNIT_TEST(AddTraceIntegral_device_sumfac test_addtraceintegral Device SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfac test_bwdtrans Device SumFac FALSE)
    # SET_UNIT_TEST(DiagPrecon_device_sumfac test_diagprecon Device SumFac FALSE)
    # SET_UNIT_TEST(FwdTrans_device_sumfac Serial StdMat test_fwdtrans Device SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfac test_helmholtz Device SumFac FALSE)
    # SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfac test_helmsolve_conjgrad Device SumFac FALSE)
    # SET_UNIT_TEST(HelmSolve_gmres_device_sumfac AVX StdMat test_helmsolve_gmres Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfac test_ipwrtbase Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac test_ipwrtderivbase Device SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfac test_linadvdiffreaction Device SumFac FALSE)
    SET_UNIT_TEST(Mass_device_sumfac test_mass Device SumFac FALSE)
    # SET_UNIT_TEST(NullPrecon_device_sumfac test_nullprecon Device SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac test_physderiv Device SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfac test_physinterp1dscaled Device SumFac FALSE)
ENDIF()

# Device SumFacTOP
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    SET_UNIT_TEST(AddTraceIntegral_device_sumfactop test_addtraceintegral Device SumFacTOP FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfactop test_bwdtrans Device SumFacTOP FALSE)
    SET_UNIT_TEST(DiagPrecon_device_sumfactop test_diagprecon Device SumFacTOP FALSE)
    SET_UNIT_TEST(FwdTrans_device_sumfactop test_fwdtrans Device SumFacTOP FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfactop test_helmholtz Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfactop test_helmsolve_conjgrad Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_device_sumfactop test_helmsolve_gmres Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfactop test_ipwrtbase Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfactop test_ipwrtderivbase Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfactop test_linadvdiffreaction Device SumFacTOP FALSE)
    SET_UNIT_TEST(Mass_device_sumfactop test_mass Device SumFacTOP FALSE)
    SET_UNIT_TEST(NullPrecon_device_sumfactop test_nullprecon Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfactop test_physderiv Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfactop test_physinterp1dscaled Device SumFacTOP FALSE)
ENDIF()

# CUDA SumFacMat
IF(NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(BwdTrans_device_sumfacmat test_bwdtrans_device_sumfacmat Device SumFacMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfacmat test_ipwrtbase_device_sumfacmat Device SumFacMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfacmat test_ipwrtderivbase_device_sumfacmat Device SumFacMat FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfacmat test_physderiv_device_sumfacmat Device SumFacMat FALSE)
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
