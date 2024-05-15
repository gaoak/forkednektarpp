# Serial StdMat
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_addtraceintegral_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_assmbscatr_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_bwdtrans_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_bwdtrans_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_diagprecon_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_diagprecon_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_dirichlet_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtrans_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_fwdtrans_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_helmholtz_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_helmsolve_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_identitycoeff_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_identitycoeff_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_identityphys_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_identityphys_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtderivbase_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_mass_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_multiplybyelmtinvmass_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_nullprecon_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_nullprecon_serial_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv_serial_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_physderiv_serial_stdmat.cpp)

# Serial SumFac
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_serial_sumfac
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase_serial_sumfac.cpp)

# These AVX test do not work 
# AVX SumFac
IF (NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_avx_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_bwdtrans_avx_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_avx_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase_avx_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_physderiv_avx_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_physderiv_avx_sumfac.cpp)    
ENDIF()

# CUDA
IF(NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_cudamathkernels
        COMPONENT unit-test DEPENDS Operators SOURCES test_cudamathkernels.cpp CUDAMathKernelsLauncher.cu)
# CUDA SumFac
    ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_cuda_sumfac
        COMPONENT unit-test DEPENDS Operators SOURCES test_addtraceintegral_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatr_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_assmbscatr_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_bwdtrans_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_diagprecon_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_diagprecon_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_dirichlet_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_dirichlet_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_fwdtrans_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_fwdtrans_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_helmholtz_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_helmholtz_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_helmsolve_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_helmsolve_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identitycoeff_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_identitycoeff_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_identityphys_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_identityphys_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtderivbase_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_mass_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_mass_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_multiplybyelmtinvmass_cuda_sumfac.cpp)
    ADD_NEKTAR_EXECUTABLE(test_neumann_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_neumann_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_nullprecon_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_nullprecon_cuda_sumfac.cpp)    
    ADD_NEKTAR_EXECUTABLE(test_physderiv_cuda_sumfac    
        COMPONENT unit-test DEPENDS Operators SOURCES test_physderiv_cuda_sumfac.cpp)    
ENDIF()

# Kokkos - if no Kokkos loops are serial
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_addtraceintegral_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_assmbscatr_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_dirichlet_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtrans_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_fwdtrans_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_helmholtz_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_identitycoeff_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_identitycoeff_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_identityphys_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_identityphys_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_multiplybyelmtinvmass_kokkos_stdmat.cpp)
ADD_NEKTAR_EXECUTABLE(test_neumann_kokkos_stdmat
    COMPONENT unit-test DEPENDS Operators SOURCES test_neumann_kokkos_stdmat.cpp)

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
SET_UNIT_TEST(NullPrecon_serial_stdmat test_nullprecon_serial_stdmat FALSE)
SET_UNIT_TEST(PhysDeriv_serial_stdmat test_physderiv_serial_stdmat FALSE)

# Serial SumFac
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase_serial_sumfac FALSE)

# These AVX test do not work 
# AVX SumFac
#IF (NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
#    SET_UNIT_TEST(BwdTrans_avx_sumfac test_bwdtrans_avx_sumfac FALSE)
#    SET_UNIT_TEST(IProductWRTBase_avx_sumfac test_ipwrtbase_avx_sumfac FALSE)
#    SET_UNIT_TEST(PhysDeriv_avx_sumfac test_physderiv_avx_sumfac FALSE)
#ENDIF()

# CUDA SumFac
IF(NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(CUDAMathKernels test_cudamathkernels FALSE)
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
    SET_UNIT_TEST(PhysDeriv_cuda_sumfac test_physderiv_cuda_sumfac FALSE)
    SET_UNIT_TEST(Mass_cuda_sumfac test_mass_cuda_sumfac FALSE)
    SET_UNIT_TEST(MultiplyByElmtInvMass_cuda_sumfac test_multiplybyelmtinvmass_cuda_sumfac FALSE)
    SET_UNIT_TEST(Neumann_cuda_sumfac test_neumann_cuda_sumfac FALSE)
    SET_UNIT_TEST(NullPrecon_cuda_sumfac test_nullprecon_cuda_sumfac FALSE)
ENDIF()

# Kokkos - if no Kokkos loops are serial
SET_UNIT_TEST(AddTraceIntegral_kokkos_stdmat test_addtraceintegral_kokkos_stdmat FALSE)
SET_UNIT_TEST(Helmholtz_kokkos_stdmat test_helmholtz_kokkos_stdmat FALSE)
SET_UNIT_TEST(IdentityCoeff_kokkos_stdmat test_identitycoeff_kokkos_stdmat FALSE)
SET_UNIT_TEST(IdentityPhys_kokkos_stdmat test_identityphys_kokkos_stdmat FALSE)
SET_UNIT_TEST(IProductWRTBase_kokkos_stdmat test_ipwrtbase_kokkos_stdmat FALSE)
SET_UNIT_TEST(MultiplyByElmtInvMass_kokkos_stdmat test_multiplybyelmtinvmass_kokkos_stdmat FALSE)

# These Kokkos test do not work probably because of interleaaving???
IF((NOT DEFINED NEKTAR_ENABLE_SIMD_AVX2) AND
   (NOT DEFINED NEKTAR_ENABLE_SIMD_AVX512))
    SET_UNIT_TEST(Assmbscatr_kokkos_stdmat test_assmbscatr_kokkos_stdmat FALSE)
    SET_UNIT_TEST(Dirichlet_kokkos_stdmat test_dirichlet_kokkos_stdmat FALSE)
    SET_UNIT_TEST(FwdTrans_kokkos_stdmat test_fwdtrans_kokkos_stdmat FALSE)
    SET_UNIT_TEST(Neumann_kokkos_stdmat test_neumann_kokkos_stdmat FALSE)
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
