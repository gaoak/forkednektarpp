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

ADD_NEKTAR_EXECUTABLE(test_math_serial
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_serial.cpp MathSerialKernelsLauncher.cpp)
# Serial StdMat
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
ADD_NEKTAR_EXECUTABLE(test_identitycoeff_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_identitycoeff_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_identityphys_serial_stdmat
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_identityphys_serial_stdmat.cpp)
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
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_serial_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_serial_sumfac.cpp)

# These AVX test do not work
# AVX SumFac
IF (${NEKTAR_ENABLE_SIMD_AVX2} OR ${NEKTAR_ENABLE_SIMD_AVX512})
    ADD_NEKTAR_EXECUTABLE(test_math_avx
            COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_avx.cpp MathAVXKernelsLauncher.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_ipwrtbase_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_avx_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_avx_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_avx_sumfac.cpp)
ENDIF()

# CUDA
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_math_cuda
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_cuda.cpp MathCUDAKernelsLauncher.cu)
# CUDA SumFac
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identitycoeff_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_identitycoeff_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identityphys_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_identityphys_cuda_sumfac.cpp)
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
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_cuda_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_cuda_sumfac.cpp)
ENDIF()

# SYCL
IF(NEKTAR_ENABLE_SYCL)
    # SYCL SumFac
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_sycl_sumfac
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_sycl_sumfac.cpp)
ENDIF()

# Kokkos
IF(NEKTAR_ENABLE_KOKKOS)
    ADD_NEKTAR_EXECUTABLE(test_math_kokkos
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math_kokkos.cpp MathKokkosKernelsLauncher.cpp)
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_addtraceintegral_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bwdtrans_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_diagprecon_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_fwdtrans_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmholtz_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identitycoeff_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_identitycoeff_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identityphys_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_identityphys_kokkos_sumfac.cpp)
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
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_nullprecon_kokkos_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_kokkos_sumfac
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_physderiv_kokkos_sumfac.cpp)
ENDIF()

# Macro
MACRO(SET_UNIT_TEST testname name mpi_flag)
    IF (mpi_flag)
        ADD_TEST(
          NAME Operators${testname}MPI
          COMMAND mpirun -np 2 ${name} --detect_memory_leaks=0
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
SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans_serial_stdmat ${NEKTAR_USE_MPI})
SET_UNIT_TEST(Helmholtz_serial_stdmat test_helmholtz_serial_stdmat FALSE)
SET_UNIT_TEST(HelmSolve_serial_stdmat test_helmsolve_serial_stdmat ${NEKTAR_USE_MPI})
SET_UNIT_TEST(IProductWRTBase_serial_stdmat test_ipwrtbase_serial_stdmat FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_stdmat test_ipwrtderivbase_serial_stdmat FALSE)
SET_UNIT_TEST(IdentityCoeff_serial_stdmat test_identitycoeff_serial_stdmat FALSE)
SET_UNIT_TEST(IdentityPhys_serial_stdmat test_identityphys_serial_stdmat FALSE)
SET_UNIT_TEST(Mass_serial_stdmat test_mass_serial_stdmat FALSE)
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_stdmat test_multiplybyelmtinvmass_serial_stdmat FALSE)
SET_UNIT_TEST(Neumann_serial_stdmat test_neumann_serial_stdmat FALSE)
SET_UNIT_TEST(NullPrecon_serial_stdmat test_nullprecon_serial_stdmat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat test_physderiv_serial_stdmat FALSE)

# Serial SumFac
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase_serial_sumfac FALSE)

# AVX SumFac
IF (NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
    SET_UNIT_TEST(Math_avx test_math_avx FALSE)
    SET_UNIT_TEST(BwdTrans_avx_sumfac test_bwdtrans_avx_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_sumfac test_ipwrtbase_avx_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_sumfac test_physderiv_avx_sumfac FALSE)
    SET_UNIT_TEST(Helmholtz_avx_sumfac test_helmholtz_avx_sumfac FALSE)
ENDIF()

# CUDA SumFac
IF(NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(Math_cuda test_math_cuda FALSE)
    SET_UNIT_TEST(AddTraceIntegral_cuda_sumfac test_addtraceintegral_cuda_sumfac FALSE)
    SET_UNIT_TEST(AssmbScatr_cuda_sumfac test_assmbscatr_cuda_sumfac FALSE)
    SET_UNIT_TEST(BwdTrans_cuda_sumfac test_bwdtrans_cuda_sumfac FALSE)
    SET_UNIT_TEST(DiagPrecon_cuda_sumfac test_diagprecon_cuda_sumfac FALSE)
    SET_UNIT_TEST(Dirichlet_cuda_sumfac test_dirichlet_cuda_sumfac FALSE)
    SET_UNIT_TEST(FwdTrans_cuda_sumfac test_fwdtrans_cuda_sumfac FALSE)
    SET_UNIT_TEST(Helmholtz_cuda_sumfac test_helmholtz_cuda_sumfac FALSE)
    SET_UNIT_TEST(HelmSolve_cuda_sumfac test_helmsolve_cuda_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTBase_cuda_sumfac test_ipwrtbase_cuda_sumfac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_cuda_sumfac test_ipwrtderivbase_cuda_sumfac FALSE)
    SET_UNIT_TEST(IdentityCoeff_cuda_sumfac test_identitycoeff_cuda_sumfac FALSE)
    SET_UNIT_TEST(IdentityPhys_cuda_sumfac test_identityphys_cuda_sumfac FALSE)
    SET_UNIT_TEST(Mass_cuda_sumfac test_mass_cuda_sumfac FALSE)
    SET_UNIT_TEST(MultiplyByElmtInvMass_cuda_sumfac test_multiplybyelmtinvmass_cuda_sumfac FALSE)
    SET_UNIT_TEST(Neumann_cuda_sumfac test_neumann_cuda_sumfac FALSE)
    SET_UNIT_TEST(NullPrecon_cuda_sumfac test_nullprecon_cuda_sumfac FALSE)
    SET_UNIT_TEST(PhysDeriv_cuda_sumfac test_physderiv_cuda_sumfac FALSE)
ENDIF()

# SYCL SumFac
IF(NEKTAR_ENABLE_SYCL)
    SET_UNIT_TEST(BwdTrans_sycl_sumfac test_bwdtrans_sycl_sumfac FALSE)
ENDIF()

# Kokkos - if no Kokkos loops are serial
# These Kokkos tests do not work always with AVX probably because of
# interleaaving and slightly different results??? Once AVX is fully
# built try to run them.
IF((NOT ${NEKTAR_ENABLE_SIMD_AVX2}) AND (NOT ${NEKTAR_ENABLE_SIMD_AVX512}))
    IF(NEKTAR_ENABLE_KOKKOS)
        SET_UNIT_TEST(Math_kokkos test_math_kokkos FALSE)
        SET_UNIT_TEST(AddTraceIntegral_kokkos_sumfac test_addtraceintegral_kokkos_sumfac FALSE)
        SET_UNIT_TEST(Assmbscatr_kokkos_sumfac test_assmbscatr_kokkos_sumfac FALSE)
        SET_UNIT_TEST(BwdTrans_kokkos_sumfac test_bwdtrans_kokkos_sumfac FALSE)
        SET_UNIT_TEST(DiagPrecon_kokkos_sumfac test_diagprecon_kokkos_sumfac FALSE)
        SET_UNIT_TEST(Dirichlet_kokkos_sumfac test_dirichlet_kokkos_sumfac FALSE)
        SET_UNIT_TEST(FwdTrans_kokkos_sumfac test_fwdtrans_kokkos_sumfac FALSE)
        SET_UNIT_TEST(Helmholtz_kokkos_sumfac test_helmholtz_kokkos_sumfac FALSE)
        SET_UNIT_TEST(HelmSolve_kokkos_sumfac test_helmsolve_kokkos_sumfac FALSE)
        SET_UNIT_TEST(IdentityCoeff_kokkos_sumfac test_identitycoeff_kokkos_sumfac FALSE)
        SET_UNIT_TEST(IdentityPhys_kokkos_sumfac test_identityphys_kokkos_sumfac FALSE)
        SET_UNIT_TEST(IProductWRTBase_kokkos_sumfac test_ipwrtbase_kokkos_sumfac FALSE)
        SET_UNIT_TEST(IProductWRTDerivBase_kokkos_sumfac test_ipwrtderivbase_kokkos_sumfac FALSE)
        SET_UNIT_TEST(Mass_kokkos_sumfac test_mass_kokkos_sumfac FALSE)
        SET_UNIT_TEST(MultiplyByElmtInvMass_kokkos_sumfac test_multiplybyelmtinvmass_kokkos_sumfac FALSE)
        SET_UNIT_TEST(Neumann_kokkos_sumfac test_neumann_kokkos_sumfac FALSE)
        SET_UNIT_TEST(NullPrecon_kokkos_sumfac test_nullprecon_kokkos_sumfac FALSE)
        SET_UNIT_TEST(PhysDeriv_kokkos_sumfac test_physderiv_kokkos_sumfac FALSE)
    ENDIF()
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
