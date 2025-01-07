# These tests can all likely be derived from the same cpp.in file
# AddTraceIntegral       Nodepend ReConfigure() - Seg, Quad, etc
# AssmbScatr             Nodepend - Seg, Quad, etc
# BwdTran                Nodepend - Seg, Quad, etc
# Helmholtz              Nodepend - Seg, Quad, etc
# IdentityCoeff          Nodepend - Seg, Quad, etc
# IdentityPhys           Nodepend - Seg, Quad, etc
# IProductWRTBase        Nodepend - Seg, Quad, etc
# IProductWRTDerivBase   Nodepend - Seg, Quad, etc
# Mass                   Nodepend - Seg, Quad, etc
# MultiplyByElmtInvMass  Nodepend - Seg, Quad, etc
# NullPrecon             Nodepend - Seg, Quad, etc
# PhysDeriv              Nodepend - Seg, Quad, etc

# These tests may require an individual  cpp.in file
# DiagPrecon             creaate operator - Helmholtz - Helmholtz1D_Seg, etc.

# FwdTrans               creaate operator - DiagPrecon - Helmholtz1D_Seg, etc.
# HelmSolve              creaate operator - DiagPrecon - Helmholtz1D_Seg, etc.

# Dirichlet              only output - Nodepend - Helmholtz1D_Seg, etc.
# Neumann                only output - Nodepend - Helmholtz1D_Seg, etc.

# NOTE: THE KOKKOS VERSION OF THE TEST SHOULD BECOME THE TEMPLATE FOR
# THE *.cpp.in FILE. DO NOT USE ANY OTHER TEST AS THE KOKKOS VERSION
# OF THE TEST HAS BEEN CLEANDED UP CONSIDERABLElY SO TO FOLLOW A
# COMMON FORMAT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


SET(OPERATOR_UNIT_TEST_DEPENDS Operators)

IF(BOOST_TEST_DYN_LINK)
    SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

# Serial StdMat
ADD_NEKTAR_EXECUTABLE(test_math_serial
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_serial.cpp MathSerialKernelsLauncher.cpp)
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_bwdtrans_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_diagprecon_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtrans_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_neumann_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_nullprecon_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_serial_stdmat.cpp)

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
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_avx_sumfac.cpp)
ENDIF()

# CUDA SumFac
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_math_cuda
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_cuda.cpp MathCUDAKernelsLauncher.cu)
    #ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_cuda_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_cuda_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_diagprecon_cuda_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_cuda_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_fwdtrans_cuda_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_cuda_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_helmsolve_cuda_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_neumann_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann_cuda_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_nullprecon_cuda_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_cuda_sumfac.cpp)
ENDIF()

# CUDA SumFacQP
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_cuda_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_cuda_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_cuda_sumfac_qp.cpp)
ENDIF()

# SYCL SumFac
IF(NEKTAR_ENABLE_SYCL)
    ADD_NEKTAR_EXECUTABLE(test_math_sycl
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_sycl.cpp MathSYCLKernelsLauncher.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_sycl_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_sycl_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_diagprecon_sycl_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_sycl_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_fwdtrans_sycl_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_sycl_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_helmsolve_sycl_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_neumann_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann_sycl_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_nullprecon_sycl_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_sycl_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_sycl_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_sycl_sumfac.cpp)
ENDIF()

# SYCL SumFacQP
IF(NEKTAR_ENABLE_SYCL)
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_sycl_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_sycl_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_sycl_sumfac_qp.cpp)
ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_sycl_sumfac_qp
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_sycl_sumfac.cpp)
ENDIF()

# Kokkos SumFac
IF(NEKTAR_ENABLE_KOKKOS)
    ADD_NEKTAR_EXECUTABLE(test_math_kokkos
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_kokkos.cpp MathKokkosKernelsLauncher.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_kokkos_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_kokkos_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_diagprecon_kokkos_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_kokkos_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_fwdtrans_kokkos_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_kokkos_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_helmsolve_kokkos_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_multiplybyelmtinvmass_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_neumann_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann_kokkos_sumfac.cpp)
    #ADD_NEKTAR_EXECUTABLE(test_nullprecon_kokkos_sumfac
    #    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_kokkos_sumfac.cpp)
ENDIF()

# Kokkos SumFacQP
IF(NEKTAR_ENABLE_KOKKOS)
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtderivbase_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_mass_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_mass_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_kokkos_sumfac_qp.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physinterp1dscaled_kokkos_sumfac_qp
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physinterp1dscaled_kokkos_sumfac_qp.cpp)
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

# Serial StdMat
SET_UNIT_TEST(Math_serial test_math_serial FALSE)
SET_UNIT_TEST(AssmbScatr_serial_stdmat test_assmbscatr_serial_stdmat FALSE)
SET_UNIT_TEST(AddTraceIntegral_serial_stdmat test_addtraceintegral_serial_stdmat FALSE)
SET_UNIT_TEST(BwdTrans_serial_stdmat test_bwdtrans_serial_stdmat FALSE)
SET_UNIT_TEST(DiagPrecon_serial_stdmat test_diagprecon_serial_stdmat FALSE)
SET_UNIT_TEST(Dirichlet_serial_stdmat test_dirichlet_serial_stdmat FALSE)
SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans_serial_stdmat FALSE)
#IF (NEKTAR_USE_MPI)
#    SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans_serial_stdmat TRUE)
#ENDIF()
SET_UNIT_TEST(Helmholtz_serial_stdmat test_helmholtz_serial_stdmat FALSE)
SET_UNIT_TEST(HelmSolve_serial_stdmat test_helmsolve_serial_stdmat FALSE)
#IF (NEKTAR_USE_MPI)
#    SET_UNIT_TEST(HelmSolve_serial_stdmat test_helmsolve_serial_stdmat TRUE)
#ENDIF()
SET_UNIT_TEST(IProductWRTBase_serial_stdmat test_ipwrtbase_serial_stdmat FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_stdmat test_ipwrtderivbase_serial_stdmat FALSE)
SET_UNIT_TEST(Mass_serial_stdmat test_mass_serial_stdmat FALSE)
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_stdmat test_multiplybyelmtinvmass_serial_stdmat FALSE)
SET_UNIT_TEST(Neumann_serial_stdmat test_neumann_serial_stdmat FALSE)
SET_UNIT_TEST(NullPrecon_serial_stdmat test_nullprecon_serial_stdmat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat test_physderiv_serial_stdmat FALSE)

# Serial SumFac
SET_UNIT_TEST(AddTraceIntegral_serial_sumfac test_addtraceintegral_serial_sumfac FALSE)
SET_UNIT_TEST(BwdTrans_serial_sumfac test_bwdtrans_serial_sumfac FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase_serial_sumfac FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_sumfac test_ipwrtderivbase_serial_sumfac FALSE)
SET_UNIT_TEST(Mass_serial_sumfac test_mass_serial_sumfac FALSE)
SET_UNIT_TEST(PhysDeriv_serial_sumfac test_physderiv_serial_sumfac FALSE)
SET_UNIT_TEST(Helmholtz_serial_sumfac test_helmholtz_serial_sumfac FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_sumfac test_physinterp1dscaled_serial_sumfac FALSE)

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
ENDIF()


# CUDA SumFac
IF(NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(Math_cuda test_math_cuda FALSE)
    #SET_UNIT_TEST(AddTraceIntegral_cuda_sumfac test_addtraceintegral_cuda_sumfac FALSE)
    SET_UNIT_TEST(AssmbScatr_cuda_sumfac test_assmbscatr_cuda_sumfac FALSE)
    SET_UNIT_TEST(BwdTrans_cuda_sumfac test_bwdtrans_cuda_sumfac FALSE)
    #SET_UNIT_TEST(DiagPrecon_cuda_sumfac test_diagprecon_cuda_sumfac FALSE)
    SET_UNIT_TEST(Dirichlet_cuda_sumfac test_dirichlet_cuda_sumfac FALSE)
    #SET_UNIT_TEST(FwdTrans_cuda_sumfac test_fwdtrans_cuda_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_cuda_sumfac test_fwdtrans_cuda_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_cuda_sumfac test_helmholtz_cuda_sumfac FALSE)
    #SET_UNIT_TEST(HelmSolve_cuda_sumfac test_helmsolve_cuda_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_cuda_sumfac test_helmsolve_cuda_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_cuda_sumfac test_ipwrtbase_cuda_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_cuda_sumfac test_ipwrtderivbase_cuda_sumfac FALSE)
    SET_UNIT_TEST(Mass_cuda_sumfac test_mass_cuda_sumfac FALSE)
    SET_UNIT_TEST(MultiplyByElmtInvMass_cuda_sumfac test_multiplybyelmtinvmass_cuda_sumfac FALSE)
    SET_UNIT_TEST(Neumann_cuda_sumfac test_neumann_cuda_sumfac FALSE)
    #SET_UNIT_TEST(NullPrecon_cuda_sumfac test_nullprecon_cuda_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_cuda_sumfac test_physderiv_cuda_sumfac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_cuda_sumfac test_physinterp1dscaled_cuda_sumfac FALSE)
ENDIF()

# CUDA SumFacQP
IF(NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(AddTraceIntegral_cuda_sumfac_qp test_addtraceintegral_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(BwdTrans_cuda_sumfac_qp test_bwdtrans_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(DiagPrecon_cuda_sumfac_qp test_diagprecon_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(FwdTrans_cuda_sumfac_qp test_fwdtrans_cuda_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_cuda_sumfac_qp test_fwdtrans_cuda_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_cuda_sumfac_qp test_helmholtz_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(HelmSolve_cuda_sumfac_qp test_helmsolve_cuda_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_cuda_sumfac_qp test_helmsolve_cuda_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_cuda_sumfac_qp test_ipwrtbase_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_cuda_sumfac_qp test_ipwrtderivbase_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(Mass_cuda_sumfac_qp test_mass_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(NullPrecon_cuda_sumfac_qp test_nullprecon_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysDeriv_cuda_sumfac_qp test_physderiv_cuda_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_cuda_sumfac_qp test_physinterp1dscaled_cuda_sumfac_qp FALSE)
ENDIF()

# SYCL SumFac
IF(NEKTAR_ENABLE_SYCL)
    SET_UNIT_TEST(Math_sycl test_math_sycl FALSE)
    #SET_UNIT_TEST(AddTraceIntegral_sycl_sumfac test_addtraceintegral_sycl_sumfac FALSE)
    SET_UNIT_TEST(AssmbScatr_sycl_sumfac test_assmbscatr_sycl_sumfac FALSE)
    SET_UNIT_TEST(BwdTrans_sycl_sumfac test_bwdtrans_sycl_sumfac FALSE)
    #SET_UNIT_TEST(DiagPrecon_sycl_sumfac test_diagprecon_sycl_sumfac FALSE)
    SET_UNIT_TEST(Dirichlet_sycl_sumfac test_dirichlet_sycl_sumfac FALSE)
    #SET_UNIT_TEST(FwdTrans_sycl_sumfac test_fwdtrans_sycl_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_sycl_sumfac test_fwdtrans_sycl_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_sycl_sumfac test_helmholtz_sycl_sumfac FALSE)
    #SET_UNIT_TEST(HelmSolve_sycl_sumfac test_helmsolve_sycl_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_sycl_sumfac test_helmsolve_sycl_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_sycl_sumfac test_ipwrtbase_sycl_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_sycl_sumfac test_ipwrtderivbase_sycl_sumfac FALSE)
    SET_UNIT_TEST(Mass_sycl_sumfac test_mass_sycl_sumfac FALSE)
    IF(NOT NEKTAR_ENABLE_SYCL STREQUAL "CUDA")
        SET_UNIT_TEST(MultiplyByElmtInvMass_sycl_sumfac test_multiplybyelmtinvmass_sycl_sumfac FALSE)
    ENDIF()
    SET_UNIT_TEST(Neumann_sycl_sumfac test_neumann_sycl_sumfac FALSE)
    #SET_UNIT_TEST(NullPrecon_sycl_sumfac test_nullprecon_sycl_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_sycl_sumfac test_physderiv_sycl_sumfac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_sycl_sumfac test_physinterp1dscaled_sycl_sumfac FALSE)
ENDIF()

# SYCL SumFacQP
IF(NEKTAR_ENABLE_SYCL)
    SET_UNIT_TEST(AddTraceIntegral_sycl_sumfac_qp test_addtraceintegral_sycl_sumfac_qp FALSE)
    SET_UNIT_TEST(BwdTrans_sycl_sumfac_qp test_bwdtrans_sycl_sumfac_qp FALSE)
    #SET_UNIT_TEST(DiagPrecon_sycl_sumfac_qp test_diagprecon_sycl_sumfac_qp FALSE)
    #SET_UNIT_TEST(FwdTrans_sycl_sumfac_qp test_fwdtrans_sycl_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_sycl_sumfac_qp test_fwdtrans_sycl_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_sycl_sumfac_qp test_helmholtz_sycl_sumfac_qp FALSE)
    #SET_UNIT_TEST(HelmSolve_sycl_sumfac_qp test_helmsolve_sycl_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_sycl_sumfac_qp test_helmsolve_sycl_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_sycl_sumfac_qp test_ipwrtbase_sycl_sumfac_qp FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_sycl_sumfac_qp test_ipwrtderivbase_sycl_sumfac_qp FALSE)
    SET_UNIT_TEST(Mass_sycl_sumfac_qp test_mass_sycl_sumfac_qp FALSE)
    SET_UNIT_TEST(NullPrecon_sycl_sumfac_qp test_nullprecon_sycl_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysDeriv_sycl_sumfac_qp test_physderiv_sycl_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_sycl_sumfac_qp test_physinterp1dscaled_sycl_sumfac_qp FALSE)
ENDIF()

# Kokkos SumFac
IF(NEKTAR_ENABLE_KOKKOS)
    SET_UNIT_TEST(Math_kokkos test_math_kokkos FALSE)
    #SET_UNIT_TEST(AddTraceIntegral_kokkos_sumfac test_addtraceintegral_kokkos_sumfac FALSE)
    SET_UNIT_TEST(AssmbScatr_kokkos_sumfac test_assmbscatr_kokkos_sumfac FALSE)
    SET_UNIT_TEST(BwdTrans_kokkos_sumfac test_bwdtrans_kokkos_sumfac FALSE)
    #SET_UNIT_TEST(DiagPrecon_kokkos_sumfac test_diagprecon_kokkos_sumfac FALSE)
    SET_UNIT_TEST(Dirichlet_kokkos_sumfac test_dirichlet_kokkos_sumfac FALSE)
    #SET_UNIT_TEST(FwdTrans_kokkos_sumfac test_fwdtrans_kokkos_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_kokkos_sumfac test_fwdtrans_kokkos_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_kokkos_sumfac test_helmholtz_kokkos_sumfac FALSE)
    #SET_UNIT_TEST(HelmSolve_kokkos_sumfac test_helmsolve_kokkos_sumfac FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_kokkos_sumfac test_helmsolve_kokkos_sumfac TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_kokkos_sumfac test_ipwrtbase_kokkos_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_kokkos_sumfac test_ipwrtderivbase_kokkos_sumfac FALSE)
    SET_UNIT_TEST(Mass_kokkos_sumfac test_mass_kokkos_sumfac FALSE)
    #SET_UNIT_TEST(MultiplyByElmtInvMass_kokkos_sumfac test_multiplybyelmtinvmass_kokkos_sumfac FALSE)
    SET_UNIT_TEST(Neumann_kokkos_sumfac test_neumann_kokkos_sumfac FALSE)
    #SET_UNIT_TEST(NullPrecon_kokkos_sumfac test_nullprecon_kokkos_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_kokkos_sumfac test_physderiv_kokkos_sumfac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_kokkos_sumfac test_physinterp1dscaled_kokkos_sumfac FALSE)
ENDIF()

# Kokkos SumFacQP
IF(NEKTAR_ENABLE_KOKKOS)
    SET_UNIT_TEST(AddTraceIntegral_kokkos_sumfac_qp test_addtraceintegral_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(BwdTrans_kokkos_sumfac_qp test_bwdtrans_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(DiagPrecon_kokkos_sumfac_qp test_diagprecon_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(FwdTrans_kokkos_sumfac_qp test_fwdtrans_kokkos_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(FwdTrans_kokkos_sumfac_qp test_fwdtrans_kokkos_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(Helmholtz_kokkos_sumfac_qp test_helmholtz_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(HelmSolve_kokkos_sumfac_qp test_helmsolve_kokkos_sumfac_qp FALSE)
    #IF (NEKTAR_USE_MPI)
    #    SET_UNIT_TEST(HelmSolve_kokkos_sumfac_qp test_helmsolve_kokkos_sumfac_qp TRUE)
    #ENDIF()
    SET_UNIT_TEST(IProductWRTBase_kokkos_sumfac_qp test_ipwrtbase_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_kokkos_sumfac_qp test_ipwrtderivbase_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(Mass_kokkos_sumfac_qp test_mass_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(NullPrecon_kokkos_sumfac_qp test_nullprecon_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysDeriv_kokkos_sumfac_qp test_physderiv_kokkos_sumfac_qp FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_kokkos_sumfac_qp test_physinterp1dscaled_kokkos_sumfac_qp FALSE)
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
