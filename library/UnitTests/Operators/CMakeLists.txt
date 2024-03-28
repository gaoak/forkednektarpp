# StdMat
ADD_NEKTAR_EXECUTABLE(test_bwdtrans
    COMPONENT unit-test DEPENDS Operators SOURCES test_bwdtrans.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtbase
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase.cpp)
ADD_NEKTAR_EXECUTABLE(test_physderiv
    COMPONENT unit-test DEPENDS Operators SOURCES test_physderiv.cpp)
ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbase
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtderivbase.cpp)
ADD_NEKTAR_EXECUTABLE(test_identitycoeff
    COMPONENT unit-test DEPENDS Operators SOURCES test_identitycoeff.cpp)
ADD_NEKTAR_EXECUTABLE(test_identityphys
    COMPONENT unit-test DEPENDS Operators SOURCES test_identityphys.cpp)
ADD_NEKTAR_EXECUTABLE(test_mass
    COMPONENT unit-test DEPENDS Operators SOURCES test_mass.cpp)
ADD_NEKTAR_EXECUTABLE(test_assmbscatr
    COMPONENT unit-test DEPENDS Operators SOURCES test_assmbscatr.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet
    COMPONENT unit-test DEPENDS Operators SOURCES test_dirichlet.cpp)
ADD_NEKTAR_EXECUTABLE(test_nullprecon
    COMPONENT unit-test DEPENDS Operators SOURCES test_nullprecon.cpp)
ADD_NEKTAR_EXECUTABLE(test_diagprecon
    COMPONENT unit-test DEPENDS Operators SOURCES test_diagprecon.cpp)
ADD_NEKTAR_EXECUTABLE(test_fwdtrans
    COMPONENT unit-test DEPENDS Operators SOURCES test_fwdtrans.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmholtz
    COMPONENT unit-test DEPENDS Operators SOURCES test_helmholtz.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve
    COMPONENT unit-test DEPENDS Operators SOURCES test_helmsolve.cpp)
ADD_NEKTAR_EXECUTABLE(test_addtraceintegral
    COMPONENT unit-test DEPENDS Operators SOURCES test_addtraceintegral.cpp)
ADD_NEKTAR_EXECUTABLE(test_multiplybyelmtinvmass
    COMPONENT unit-test DEPENDS Operators SOURCES test_multiplybyelmtinvmass.cpp)

# SumFac
ADD_NEKTAR_EXECUTABLE(test_ipwrtbasesumfac
    COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbasesumfac.cpp)

# MatFree
IF (NEKTAR_ENABLE_SIMD)
    ADD_NEKTAR_EXECUTABLE(test_bwdtrans_matfree
        COMPONENT unit-test DEPENDS Operators SOURCES test_bwdtrans_matfree.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbase_matfree
        COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbase_matfree.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderiv_matfree
        COMPONENT unit-test DEPENDS Operators SOURCES test_physderiv_matfree.cpp)
ENDIF()

IF (NEKTAR_ENABLE_CUDA)
    ADD_NEKTAR_EXECUTABLE(test_cudamathkernels
        COMPONENT unit-test DEPENDS Operators SOURCES test_cudamathkernels.cpp CUDAMathKernelsLauncher.cu)
    ADD_NEKTAR_EXECUTABLE(test_bwdtranscuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_bwdtranscuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtbasecuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtbasecuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_physderivcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_physderivcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_ipwrtderivbasecuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_ipwrtderivbasecuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identitycoeffcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_identitycoeffcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_identityphyscuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_identityphyscuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_masscuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_masscuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_assmbscatrcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_assmbscatrcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_dirichletcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_dirichletcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_neumanncuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_neumanncuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_nullpreconcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_nullpreconcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_diagpreconcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_diagpreconcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_fwdtranscuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_fwdtranscuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmholtzcuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_helmholtzcuda.cpp)
    ADD_NEKTAR_EXECUTABLE(test_helmsolvecuda
        COMPONENT unit-test DEPENDS Operators SOURCES test_helmsolvecuda.cpp)
ENDIF()


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

# StdMat
SET_UNIT_TEST(BwdTrans test_bwdtrans FALSE)
SET_UNIT_TEST(IProductWRTBase test_ipwrtbase FALSE)
SET_UNIT_TEST(PhysDeriv test_physderiv FALSE)
SET_UNIT_TEST(IProductWRTDerivBase test_ipwrtderivbase FALSE)
SET_UNIT_TEST(IdentityCoeff test_identitycoeff FALSE)
SET_UNIT_TEST(IdentityPhys test_identityphys FALSE)
SET_UNIT_TEST(Mass test_mass FALSE)
SET_UNIT_TEST(AssmbScatr test_assmbscatr FALSE)
SET_UNIT_TEST(Dirichlet test_dirichlet FALSE)
SET_UNIT_TEST(NullPrecon test_nullprecon FALSE)
SET_UNIT_TEST(DiagPrecon test_diagprecon FALSE)
SET_UNIT_TEST(FwdTrans test_fwdtrans ${NEKTAR_USE_MPI})
SET_UNIT_TEST(Helmholtz test_helmholtz FALSE)
SET_UNIT_TEST(HelmSolve test_helmsolve ${NEKTAR_USE_MPI})
SET_UNIT_TEST(AddTraceIntegral test_addtraceintegral FALSE)
SET_UNIT_TEST(MultiplyByElmtInvMass test_multiplybyelmtinvmass FALSE)

# SumFac
SET_UNIT_TEST(IProductWRTBaseSumFac test_ipwrtbasesumfac FALSE)

# MatFree
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(BwdTransMatFree test_bwdtrans_matfree FALSE)
    SET_UNIT_TEST(IProductWRTBaseMatFree  test_ipwrtbase_matfree FALSE)
    SET_UNIT_TEST(PhysDerivMatFree test_physderiv_matfree FALSE)
ENDIF()

# CUDA
IF (NEKTAR_ENABLE_CUDA)
    SET_UNIT_TEST(CUDAMathKernels test_cudamathkernels FALSE)
    SET_UNIT_TEST(BwdTransCUDA test_bwdtranscuda FALSE)
    SET_UNIT_TEST(IProductWRTBaseCUDA test_ipwrtbasecuda FALSE)
    SET_UNIT_TEST(PhysDerivCUDA test_physderivcuda FALSE)
    SET_UNIT_TEST(IProductWRTDerivBaseCUDA test_ipwrtderivbasecuda FALSE)
    SET_UNIT_TEST(IdentityCoeffCUDA test_identitycoeffcuda FALSE)
    SET_UNIT_TEST(IdentityPhysCUDA test_identityphyscuda FALSE)
    SET_UNIT_TEST(MassCUDA test_masscuda FALSE)
    SET_UNIT_TEST(AssmbScatrCUDA test_assmbscatrcuda FALSE)
    SET_UNIT_TEST(DirichletCUDA test_dirichletcuda FALSE)
    SET_UNIT_TEST(NeumannCUDA test_neumanncuda FALSE)
    SET_UNIT_TEST(NullPreconCUDA test_nullpreconcuda FALSE)
    SET_UNIT_TEST(DiagPreconCUDA test_diagpreconcuda FALSE)
    SET_UNIT_TEST(FwdTransCUDA test_fwdtranscuda FALSE)
    SET_UNIT_TEST(HelmholtzCUDA test_helmholtzcuda FALSE)
    SET_UNIT_TEST(HelmSolveCUDA test_helmsolvecuda FALSE)
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)

