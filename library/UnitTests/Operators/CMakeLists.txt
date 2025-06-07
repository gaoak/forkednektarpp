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

# Serial
ADD_NEKTAR_EXECUTABLE(test_math_serial
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_serial.cpp MathSerialKernelsLauncher.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr_serial
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_serial.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet_serial
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_serial.cpp)
ADD_NEKTAR_EXECUTABLE(test_neumann_serial
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann_serial.cpp)

# Serial generic
ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_serial_generic
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass_serial_generic.cpp)
ADD_NEKTAR_EXECUTABLE(test_expression_serial_generic
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_expression_serial_generic.cpp)

# Serial StdMat
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_bwdtrans_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_diagprecon_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtrans_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_conjgrad_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_conjgrad_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_gmres_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_gmres_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_nullprecon_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction_serial_stdmat.cpp)

# Serial SumFac
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_bwdtrans_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_serial_sumfac.cpp)
ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_serial_sumfac.cpp)

# AVX SumFac
IF (${NEKTAR_ENABLE_SIMD_AVX2} OR ${NEKTAR_ENABLE_SIMD_AVX512})
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_math_avx
            COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_avx.cpp MathAVXKernelsLauncher.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_avx_sumfac.cpp)
ENDIF()

# Device
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_math_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_device.cpp MathCUDAKernelsLauncher.cu)
ELSEIF(NEKTAR_ENABLE_HIP)
    ADD_NEKTAR_EXECUTABLE(test_math_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_device.cpp MathHIPKernelsLauncher.hip)
ELSEIF(NEKTAR_ENABLE_SYCL)
    ADD_NEKTAR_EXECUTABLE(test_math_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_device.cpp MathSYCLKernelsLauncher.cpp)
ELSEIF(NEKTAR_ENABLE_DEVICEONHOST)
    ADD_NEKTAR_EXECUTABLE(test_math_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_device.cpp MathDeviceOnHostKernelsLauncher.cpp)
ENDIF()

IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_device.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_device.cpp)
    ADD_NEKTAR_EXECUTABLE(test_neumann_device
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann_device.cpp)
ENDIF()

# Device generic
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_device_generic
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass_device_generic.cpp)
ENDIF()

# Device StdMat
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_device_stdmat.cpp)
    ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction_device_stdmat
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction_device_stdmat.cpp)
ENDIF()

# Device SumFac
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    #ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_device_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_device_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_diagprecon_device_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_device_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_fwdtrans_device_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_device_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_helmsolve_conjgrad_device_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_conjgrad_device_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_helmsolve_gmres_device_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_gmres_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_device_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_nullprecon_device_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_device_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction_device_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction_device_sumfac.cpp)
ENDIF()

# Device SumFacQP
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_conjgrad_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_conjgrad_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_gmres_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_gmres_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_device_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_linadvdiffreaction_device_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linadvdiffreaction_device_sumfac_qp.cpp)
ENDIF()

# CUDA SumFacCUBLAS
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_device_sumfac_cublas
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_device_sumfac_cublas.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_device_sumfac_cublas
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_device_sumfac_cublas.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_device_sumfac_cublas
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_device_sumfac_cublas.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv_device_sumfac_cublas
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_device_sumfac_cublas.cpp)
ENDIF()

# Macro
MACRO(SET_UNIT_TEST testname name mpi_flag)
    IF (${mpi_flag})
        ADD_TEST(
          NAME Operators${testname}MPI
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/${name} --detect_memory_leaks=0
      )
    ELSE()
        ADD_TEST(
          NAME Operators${testname}
          COMMAND ${name} --detect_memory_leaks=0
      )
    ENDIF()
ENDMACRO()

# Serial
SET_UNIT_TEST(Math_serial test_math_serial FALSE)
SET_UNIT_TEST(AssmbScatr_serial test_assmbscatr_serial FALSE)
SET_UNIT_TEST(Dirichlet_serial test_dirichlet_serial FALSE)
SET_UNIT_TEST(Neumann_serial test_neumann_serial FALSE)

# Serial generic
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_generic test_multiplybyelmtinvmass_serial_generic FALSE)
SET_UNIT_TEST(expression_serial_generic test_expression_serial_generic FALSE)

# Serial StdMat
SET_UNIT_TEST(AddTraceIntegral_serial_stdmat test_addtraceintegral_serial_stdmat FALSE)
SET_UNIT_TEST(BwdTrans_serial_stdmat test_bwdtrans_serial_stdmat FALSE)
SET_UNIT_TEST(DiagPrecon_serial_stdmat test_diagprecon_serial_stdmat FALSE)
SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans_serial_stdmat FALSE)
#IF (NEKTAR_USE_MPI)
#    SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans_serial_stdmat TRUE)
#ENDIF()
SET_UNIT_TEST(Helmholtz_serial_stdmat test_helmholtz_serial_stdmat FALSE)
SET_UNIT_TEST(HelmSolve_conjgrad_serial_stdmat test_helmsolve_conjgrad_serial_stdmat FALSE)
SET_UNIT_TEST(HelmSolve_gmres_serial_stdmat test_helmsolve_gmres_serial_stdmat FALSE)
#IF (NEKTAR_USE_MPI)
#    SET_UNIT_TEST(HelmSolve_conjgrad_serial_stdmat test_helmsolve_conjgrad_serial_stdmat TRUE)
#    SET_UNIT_TEST(HelmSolve_gmres_serial_stdmat test_helmsolve_gmres_serial_stdmat TRUE)
#ENDIF()
SET_UNIT_TEST(IProductWRTBase_serial_stdmat test_ipwrtbase_serial_stdmat FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_stdmat test_ipwrtderivbase_serial_stdmat FALSE)
SET_UNIT_TEST(Mass_serial_stdmat test_mass_serial_stdmat FALSE)
SET_UNIT_TEST(NullPrecon_serial_stdmat test_nullprecon_serial_stdmat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat test_physderiv_serial_stdmat FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_stdmat test_physinterp1dscaled_serial_stdmat FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_stdmat test_linadvdiffreaction_serial_stdmat FALSE)

# Serial SumFac
SET_UNIT_TEST(AddTraceIntegral_serial_sumfac test_addtraceintegral_serial_sumfac FALSE)
SET_UNIT_TEST(BwdTrans_serial_sumfac test_bwdtrans_serial_sumfac FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase_serial_sumfac FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_sumfac test_ipwrtderivbase_serial_sumfac FALSE)
SET_UNIT_TEST(Mass_serial_sumfac test_mass_serial_sumfac FALSE)
SET_UNIT_TEST(PhysDeriv_serial_sumfac test_physderiv_serial_sumfac FALSE)
SET_UNIT_TEST(Helmholtz_serial_sumfac test_helmholtz_serial_sumfac FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_sumfac test_physinterp1dscaled_serial_sumfac FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_sumfac test_linadvdiffreaction_serial_sumfac FALSE)

# AVX SumFac
IF (NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
    #SET_UNIT_TEST(AddTraceIntegral_avx_sumfac test_addtraceintegral_avx_sumfac FALSE)
    SET_UNIT_TEST(Math_avx test_math_avx FALSE)
    SET_UNIT_TEST(BwdTrans_avx_sumfac test_bwdtrans_avx_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_sumfac test_ipwrtbase_avx_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_sumfac test_ipwrtderivbase_avx_sumfac FALSE)
    SET_UNIT_TEST(Mass_avx_sumfac test_mass_avx_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_sumfac test_physderiv_avx_sumfac FALSE)
    SET_UNIT_TEST(Helmholtz_avx_sumfac test_helmholtz_avx_sumfac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_sumfac test_physinterp1dscaled_avx_sumfac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_sumfac test_linadvdiffreaction_avx_sumfac FALSE)
ENDIF()

# Device
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    SET_UNIT_TEST(Math_device test_math_device FALSE)
    SET_UNIT_TEST(AssmbScatr_device test_assmbscatr_device FALSE)
    SET_UNIT_TEST(Dirichlet_device test_dirichlet_device FALSE)
    SET_UNIT_TEST(Neumann_device test_neumann_device FALSE)
ENDIF()

# Device generic
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    IF(NOT NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
        SET_UNIT_TEST(MultiplyByElmtInvMass_device_generic test_multiplybyelmtinvmass_device_generic FALSE)
    ENDIF()
ENDIF()

# Device StdMat
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    IF(NOT NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
        SET_UNIT_TEST(BwdTrans_device_stdmat test_bwdtrans_device_stdmat FALSE)
        SET_UNIT_TEST(IProductWRTBase_device_stdmat test_ipwrtbase_device_stdmat FALSE)
        SET_UNIT_TEST(IProductWRTDerivBase_device_stdmat test_ipwrtderivbase_device_stdmat FALSE)
        SET_UNIT_TEST(PhysDeriv_device_stdmat test_physderiv_device_stdmat FALSE)
        SET_UNIT_TEST(Mass_device_stdmat test_mass_device_stdmat FALSE)
        SET_UNIT_TEST(Helmholtz_device_stdmat test_helmholtz_device_stdmat FALSE)
        SET_UNIT_TEST(PhysInterp1DScaled_device_stdmat test_physinterp1dscaled_device_stdmat FALSE)
        SET_UNIT_TEST(LinAdvDiffReaction_device_stdmat test_linadvdiffreaction_device_stdmat FALSE)
    ENDIF()
ENDIF()

# Device SumFac
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    #SET_UNIT_TEST(AddTraceIntegral_device_sumfac test_addtraceintegral_device_sumfac FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfac test_bwdtrans_device_sumfac FALSE)
    #SET_UNIT_TEST(DiagPrecon_device_sumfac test_diagprecon_device_sumfac FALSE)
    #SET_UNIT_TEST(FwdTrans_device_sumfac test_fwdtrans_device_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_device_sumfac test_fwdtrans_device_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_device_sumfac test_helmholtz_device_sumfac FALSE)
    #SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfac test_helmsolve_conjgrad_device_sumfac FALSE)
    #SET_UNIT_TEST(HelmSolve_gmres_device_sumfac test_helmsolve_gmres_device_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfac test_helmsolve_conjgrad_device_sumfac TRUE)
    #    SET_UNIT_TEST(HelmSolve_gmres_device_sumfac test_helmsolve_gmres_device_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_device_sumfac test_ipwrtbase_device_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac test_ipwrtderivbase_device_sumfac FALSE)
    SET_UNIT_TEST(Mass_device_sumfac test_mass_device_sumfac FALSE)
    #SET_UNIT_TEST(NullPrecon_device_sumfac test_nullprecon_device_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac test_physderiv_device_sumfac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfac test_physinterp1dscaled_device_sumfac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfac test_linadvdiffreaction_device_sumfac FALSE)
ENDIF()

# Device SumFacQP
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    SET_UNIT_TEST(AddTraceIntegral_device_sumfac_qp test_addtraceintegral_device_sumfac_qp FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfac_qp test_bwdtrans_device_sumfac_qp FALSE)
    SET_UNIT_TEST(DiagPrecon_device_sumfac_qp test_diagprecon_device_sumfac_qp FALSE)
    SET_UNIT_TEST(FwdTrans_device_sumfac_qp test_fwdtrans_device_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_device_sumfac_qp test_fwdtrans_device_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_device_sumfac_qp test_helmholtz_device_sumfac_qp FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfac_qp test_helmsolve_conjgrad_device_sumfac_qp FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_device_sumfac_qp test_helmsolve_gmres_device_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfac_qp test_helmsolve_conjgrad_device_sumfac_qp TRUE)
    #    SET_UNIT_TEST(HelmSolve_gmres_device_sumfac_qp test_helmsolve_gmres_device_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_device_sumfac_qp test_ipwrtbase_device_sumfac_qp FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac_qp test_ipwrtderivbase_device_sumfac_qp FALSE)
    SET_UNIT_TEST(Mass_device_sumfac_qp test_mass_device_sumfac_qp FALSE)
    SET_UNIT_TEST(NullPrecon_device_sumfac_qp test_nullprecon_device_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac_qp test_physderiv_device_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfac_qp test_physinterp1dscaled_device_sumfac_qp FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfac_qp test_linadvdiffreaction_device_sumfac_qp FALSE)
ENDIF()

# CUDA SumFacCUBLAS
IF(NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(BwdTrans_device_sumfac_cublas test_bwdtrans_device_sumfac_cublas FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfac_cublas test_ipwrtbase_device_sumfac_cublas FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac_cublas test_ipwrtderivbase_device_sumfac_cublas FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac_cublas test_physderiv_device_sumfac_cublas FALSE)
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
