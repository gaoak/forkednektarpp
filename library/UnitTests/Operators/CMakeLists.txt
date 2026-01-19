# These tests can all likely be derived from the same cpp.in file
# AddTraceIntegral       Nodepend ReConfigure() - Seg, Quad, etc
# AssmbScatr             Nodepend - Seg, Quad, etc
# BwdTran                Nodepend - Seg, Quad, etc
# Helmholtz              Nodepend - Seg, Quad, etc
# Laplacian              Nodepend - Seg, Quad, etc
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
# PoissonSolve           create operator - DiagPrecon - Poisson1D_Seg, etc.
# LinearADRSolve         create operator - DiagPrecon - Helmholtz1D_Seg, etc.

# Dirichlet              only output - Nodepend - Helmholtz1D_Seg, etc.
# Neumann                only output - Nodepend - Helmholtz1D_Seg, etc.

SET(OPERATOR_UNIT_TEST_DEPENDS Operators)

IF(BOOST_TEST_DYN_LINK)
    SET(OPERATOR_UNIT_TEST_DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDIF()

ADD_NEKTAR_EXECUTABLE(test_optimization
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_optimization.cpp)
ADD_NEKTAR_EXECUTABLE(test_comm
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_comm.cpp)
ADD_NEKTAR_EXECUTABLE(test_math
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_math.cpp)
IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
    ADD_NEKTAR_EXECUTABLE(test_parallel_reduce
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_parallel_reduce.cu)
ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
    ADD_NEKTAR_EXECUTABLE(test_parallel_reduce
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_parallel_reduce.hip)
ELSE()
    ADD_NEKTAR_EXECUTABLE(test_parallel_reduce
        COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_parallel_reduce.cpp)
ENDIF()
ADD_NEKTAR_EXECUTABLE(test_assmbscatr
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_assmbscatr.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirichlet
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirichlet.cpp)
ADD_NEKTAR_EXECUTABLE(test_neumann
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_neumann.cpp)
ADD_NEKTAR_EXECUTABLE(test_runge_kutta
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_runge_kutta.cpp)
ADD_NEKTAR_EXECUTABLE(test_adams_bashforth
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_adams_bashforth.cpp)
ADD_NEKTAR_EXECUTABLE(test_dirk
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_dirk.cpp)
ADD_NEKTAR_EXECUTABLE(test_adams_moulton
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_adams_moulton.cpp)
ADD_NEKTAR_EXECUTABLE(test_bdf
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_bdf.cpp)
ADD_NEKTAR_EXECUTABLE(test_cnab
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_cnab.cpp)
ADD_NEKTAR_EXECUTABLE(test_imex
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_imex.cpp)
ADD_NEKTAR_EXECUTABLE(test_imexdirk
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_imexdirk.cpp)
ADD_NEKTAR_EXECUTABLE(test_explicit_sdc
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_explicit_sdc.cpp)
ADD_NEKTAR_EXECUTABLE(test_implicit_sdc
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_implicit_sdc.cpp)
ADD_NEKTAR_EXECUTABLE(test_imex_sdc
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_imex_sdc.cpp)
ADD_NEKTAR_EXECUTABLE(test_explicit_gem
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_explicit_gem.cpp)
ADD_NEKTAR_EXECUTABLE(test_implicit_gem
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_implicit_gem.cpp)
ADD_NEKTAR_EXECUTABLE(test_imex_gem
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_imex_gem.cpp)

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
ADD_NEKTAR_EXECUTABLE(test_laplacian
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_laplacian.cpp)
ADD_NEKTAR_EXECUTABLE(test_poissonsolve
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_poissonsolve.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_richardson
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_richardson.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_conjres
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_conjres.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_conjgrad
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_conjgrad.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_minres
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_minres.cpp)
ADD_NEKTAR_EXECUTABLE(test_helmsolve_gmres
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_helmsolve_gmres.cpp)
ADD_NEKTAR_EXECUTABLE(test_linearadrsolve_cgs
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linearadrsolve_cgs.cpp)
ADD_NEKTAR_EXECUTABLE(test_linearadrsolve_tfqmr
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linearadrsolve_tfqmr.cpp)
ADD_NEKTAR_EXECUTABLE(test_linearadrsolve_bicgstab
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linearadrsolve_bicgstab.cpp)
ADD_NEKTAR_EXECUTABLE(test_linearadrsolve_gmres
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linearadrsolve_gmres.cpp)
ADD_NEKTAR_EXECUTABLE(test_linearadrsolve_idrs
    COMPONENT unit-test DEPENDS ${OPERATOR_UNIT_TEST_DEPENDS} SOURCES test_linearadrsolve_idrs.cpp)
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
IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
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
    IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
        SET(name-ext "${name}-g")
    ELSEIF (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        SET(name-ext "${name}-ms")
    ELSEIF (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        SET(name-ext "${name}-rg")
    ELSE()
        SET(name-ext "${name}")
    ENDIF()

    IF (${mpi_flag} AND NEKTAR_USE_MPI)
        IF (MSVC)
            ADD_TEST(
              NAME Operators${testname}_MPI
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${name-ext}
                      --detect_memory_leaks=0 -- ${execspace} ${implementation}
            )
        ELSE()
            ADD_TEST(
              NAME Operators${testname}_MPI
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/${name-ext}
                      --detect_memory_leaks=0 -- ${execspace} ${implementation}
            )
        ENDIF()
    ELSE()
        ADD_TEST(
          NAME Operators${testname} 
          COMMAND ${name-ext} --detect_memory_leaks=0 -- ${execspace} ${implementation})
    ENDIF()
ENDMACRO()

IF (NEKTAR_ENABLE_DOUBLE_PRECISION) # Only enable unit tests for double precision, for now
SET_UNIT_TEST(Optimization test_optimization none none FALSE)

# Comm
IF (NEKTAR_USE_MPI)
    SET_UNIT_TEST(Comm_host test_comm HostSpace none TRUE)
    SET_UNIT_TEST(Comm_device test_comm DeviceSpace none TRUE)
ENDIF()

# Serial
SET_UNIT_TEST(Math_serial test_math Serial none FALSE)
SET_UNIT_TEST(Parallel_reduce_serial test_parallel_reduce Serial none FALSE)
SET_UNIT_TEST(AssmbScatr_serial test_assmbscatr Serial none FALSE)
SET_UNIT_TEST(Dirichlet_serial test_dirichlet Serial none FALSE)
SET_UNIT_TEST(Neumann_serial test_neumann Serial none FALSE)
SET_UNIT_TEST(RungeKutta_serial test_runge_kutta Serial none FALSE)
SET_UNIT_TEST(AdamsBashforth_serial test_adams_bashforth Serial none FALSE)
SET_UNIT_TEST(DIRK_serial test_dirk Serial none FALSE)
SET_UNIT_TEST(AdamsMoulton_serial test_adams_moulton Serial none FALSE)
SET_UNIT_TEST(BDF_serial test_bdf Serial none FALSE)
SET_UNIT_TEST(CNAB_serial test_cnab Serial none FALSE)
SET_UNIT_TEST(IMEX_serial test_imex Serial none FALSE)
SET_UNIT_TEST(IMEXdirk_serial test_imexdirk Serial none FALSE)
SET_UNIT_TEST(Explicit_SDC_serial test_explicit_sdc Serial none FALSE)
SET_UNIT_TEST(Implicit_SDC_serial test_implicit_sdc Serial none FALSE)
SET_UNIT_TEST(IMEX_SDC_serial test_imex_sdc Serial none FALSE)
SET_UNIT_TEST(Explicit_GEM_serial test_explicit_gem Serial none FALSE)
SET_UNIT_TEST(Implicit_GEM_serial test_implicit_gem Serial none FALSE)
SET_UNIT_TEST(IMEX_GEM_serial test_imex_gem Serial none FALSE)
IF (NEKTAR_USE_MPI)
    SET_UNIT_TEST(AssmbScatr_serial test_assmbscatr Serial none TRUE)
ENDIF()

# Serial generic
SET_UNIT_TEST(MultiplyByElmtInvMass_serial_generic test_multiplybyelmtinvmass Serial Generic FALSE)
SET_UNIT_TEST(expression_serial_generic test_expression Serial Generic FALSE)

# Serial StdMat
SET_UNIT_TEST(AddTraceIntegral_serial_stdmat test_addtraceintegral Serial StdMat FALSE)
SET_UNIT_TEST(BwdTrans_serial_stdmat test_bwdtrans Serial StdMat FALSE)
SET_UNIT_TEST(DiagPrecon_serial_stdmat test_diagprecon Serial StdMat FALSE)
SET_UNIT_TEST(FwdTrans_serial_stdmat test_fwdtrans Serial StdMat FALSE)
SET_UNIT_TEST(Helmholtz_serial_stdmat test_helmholtz Serial StdMat FALSE)
SET_UNIT_TEST(Laplacian_serial_stdmat test_laplacian Serial StdMat FALSE)
SET_UNIT_TEST(PoissonSolve_serial_stdmat test_poissonsolve Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_richardson_serial_stdmat test_helmsolve_richardson Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_conjres_serial_stdmat test_helmsolve_conjres Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_conjgrad_serial_stdmat test_helmsolve_conjgrad Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_minres_serial_stdmat test_helmsolve_minres Serial StdMat FALSE)
SET_UNIT_TEST(HelmSolve_gmres_serial_stdmat test_helmsolve_gmres Serial StdMat FALSE)
SET_UNIT_TEST(LinearADRSolve_cgs_serial_stdmat test_linearadrsolve_cgs Serial StdMat FALSE)
SET_UNIT_TEST(LinearADRSolve_tfqmr_serial_stdmat test_linearadrsolve_tfqmr Serial StdMat FALSE)
SET_UNIT_TEST(LinearADRSolve_bicgstab_serial_stdmat test_linearadrsolve_bicgstab Serial StdMat FALSE)
SET_UNIT_TEST(LinearADRSolve_gmres_serial_stdmat test_linearadrsolve_gmres Serial StdMat FALSE)
SET_UNIT_TEST(LinearADRSolve_idrs_serial_stdmat test_linearadrsolve_idrs Serial StdMat FALSE)
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
SET_UNIT_TEST(Laplacian_serial_sumfac test_laplacian Serial SumFac FALSE)
SET_UNIT_TEST(PoissonSolve_serial_sumfac test_poissonsolve Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_richardson_serial_sumfac test_helmsolve_richardson Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_conjres_serial_sumfac test_helmsolve_conjres Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_conjgrad_serial_sumfac test_helmsolve_conjgrad Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_minres_serial_sumfac test_helmsolve_minres Serial SumFac FALSE)
SET_UNIT_TEST(HelmSolve_gmres_serial_sumfac test_helmsolve_gmres Serial SumFac FALSE)
SET_UNIT_TEST(LinearADRSolve_cgs_serial_sumfac test_linearadrsolve_cgs Serial SumFac FALSE)
SET_UNIT_TEST(LinearADRSolve_tfqmr_serial_sumfac test_linearadrsolve_tfqmr Serial SumFac FALSE)
SET_UNIT_TEST(LinearADRSolve_bicgstab_serial_sumfac test_linearadrsolve_bicgstab Serial SumFac FALSE)
SET_UNIT_TEST(LinearADRSolve_gmres_serial_sumfac test_linearadrsolve_gmres Serial SumFac FALSE)
SET_UNIT_TEST(LinearADRSolve_idrs_serial_sumfac test_linearadrsolve_idrs Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTBase_serial_sumfac test_ipwrtbase Serial SumFac FALSE)
SET_UNIT_TEST(IProductWRTDerivBase_serial_sumfac test_ipwrtderivbase Serial SumFac FALSE)
SET_UNIT_TEST(LinAdvDiffReaction_serial_sumfac test_linadvdiffreaction Serial SumFac FALSE)
SET_UNIT_TEST(Mass_serial_sumfac test_mass Serial SumFac FALSE)
SET_UNIT_TEST(NullPrecon_serial_sumfac test_nullprecon Serial SumFac FALSE)
SET_UNIT_TEST(PhysDeriv_serial_sumfac test_physderiv Serial SumFac FALSE)
SET_UNIT_TEST(PhysInterp1DScaled_serial_sumfac test_physinterp1dscaled Serial SumFac FALSE)
IF (NEKTAR_USE_MPI)
    SET_UNIT_TEST(HelmSolve_richardson_serial test_helmsolve_richardson Serial SumFac TRUE)
    SET_UNIT_TEST(HelmSolve_conjres_serial test_helmsolve_conjres Serial SumFac TRUE)
    SET_UNIT_TEST(HelmSolve_conjgrad_serial test_helmsolve_conjgrad Serial SumFac TRUE)
    SET_UNIT_TEST(HelmSolve_minres_serial test_helmsolve_minres Serial SumFac TRUE)
    SET_UNIT_TEST(HelmSolve_gmres_serial test_helmsolve_gmres Serial SumFac TRUE)
ENDIF()

# AVX
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(Math_avx test_math AVX none FALSE)
    SET_UNIT_TEST(Parallel_reduce_avx test_parallel_reduce AVX none FALSE)
    SET_UNIT_TEST(AssmbScatr_avx test_assmbscatr AVX none FALSE)
    SET_UNIT_TEST(Dirichlet_avx  test_dirichlet AVX none FALSE)
    SET_UNIT_TEST(Neumann_avx test_neumann AVX none FALSE)
    SET_UNIT_TEST(RungeKutta_avx test_runge_kutta AVX none FALSE)
    SET_UNIT_TEST(AdamsBashforth_avx test_adams_bashforth AVX none FALSE)
    SET_UNIT_TEST(DIRK_avx test_dirk AVX none FALSE)
    SET_UNIT_TEST(AdamsMoulton_avx test_adams_moulton AVX none FALSE)
    SET_UNIT_TEST(BDF_avx test_bdf AVX none FALSE)
    SET_UNIT_TEST(CNAB_avx test_cnab AVX none FALSE)
    SET_UNIT_TEST(IMEX_avx test_imex AVX none FALSE)
    SET_UNIT_TEST(IMEXdirk_avx test_imexdirk AVX none FALSE)
    SET_UNIT_TEST(Explicit_SDC_avx test_explicit_sdc AVX none FALSE)
    SET_UNIT_TEST(Implicit_SDC_avx test_implicit_sdc AVX none FALSE)
    SET_UNIT_TEST(IMEX_SDC_avx test_imex_sdc AVX none FALSE)
    SET_UNIT_TEST(Explicit_GEM_avx test_explicit_gem AVX none FALSE)
    SET_UNIT_TEST(Implicit_GEM_avx test_implicit_gem AVX none FALSE)
    SET_UNIT_TEST(IMEX_GEM_avx test_imex_gem AVX none FALSE)
    IF (NEKTAR_USE_MPI)
        SET_UNIT_TEST(AssmbScatr_avx test_assmbscatr AVX none TRUE)
    ENDIF()
ENDIF()

# AVX generic
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(MultiplyByElmtInvMass_avx_generic test_multiplybyelmtinvmass AVX Generic FALSE)
    # SET_UNIT_TEST(expression_avx_generic test_expression AVX Generic FALSE)
ENDIF()

# AVX StdMat
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(AddTraceIntegral_avx_stdmat test_addtraceintegral AVX StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_avx_stdmat test_bwdtrans AVX StdMat FALSE)
    SET_UNIT_TEST(DiagPrecon_avx_stdmat test_diagprecon AVX StdMat FALSE)
    SET_UNIT_TEST(FwdTrans_avx_stdmat test_fwdtrans AVX StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_avx_stdmat test_helmholtz AVX StdMat FALSE)
    SET_UNIT_TEST(Laplacian_avx_stdmat test_laplacian AVX StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_richardson_avx_stdmat test_helmsolve_richardson AVX StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_conjres_avx_stdmat test_helmsolve_conjres AVX StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_avx_stdmat test_helmsolve_conjgrad AVX StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_minres_avx_stdmat test_helmsolve_minres AVX StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_avx_stdmat test_helmsolve_gmres AVX StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_cgs_avx_stdmat test_linearadrsolve_cgs AVX StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_tfqmr_avx_stdmat test_linearadrsolve_tfqmr AVX StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_bicgstab_avx_stdmat test_linearadrsolve_bicgstab AVX StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_gmres_avx_stdmat test_linearadrsolve_gmres AVX StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_idrs_avx_stdmat test_linearadrsolve_idrs AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_stdmat test_ipwrtbase AVX StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_stdmat test_ipwrtderivbase AVX StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_stdmat test_linadvdiffreaction AVX StdMat FALSE)
    SET_UNIT_TEST(Mass_avx_stdmat test_mass AVX StdMat FALSE)
    SET_UNIT_TEST(NullPrecon_avx_stdmat test_nullprecon AVX StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_stdmat test_physderiv AVX StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_stdmat test_physinterp1dscaled Serial StdMat FALSE)
ENDIF()

# AVX SumFac
IF (NEKTAR_ENABLE_SIMD)
    SET_UNIT_TEST(AddTraceIntegral_avx_sumfac test_addtraceintegral AVX SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_avx_sumfac test_bwdtrans AVX SumFac FALSE)
    SET_UNIT_TEST(DiagPrecon_avx_sumfac test_diagprecon AVX SumFac FALSE)
    SET_UNIT_TEST(FwdTrans_avx_sumfac test_fwdtrans AVX SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_avx_sumfac test_helmholtz AVX SumFac FALSE)
    SET_UNIT_TEST(Laplacian_avx_sumfac test_laplacian AVX SumFac FALSE)    
    SET_UNIT_TEST(HelmSolve_richardson_avx_sumfac test_helmsolve_richardson AVX SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_conjres_avx_sumfac test_helmsolve_conjres AVX SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_avx_sumfac test_helmsolve_conjgrad AVX SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_minres_avx_sumfac test_helmsolve_minres AVX SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_avx_sumfac test_helmsolve_gmres AVX SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_cgs_avx_sumfac test_linearadrsolve_cgs AVX SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_tfqmr_avx_sumfac test_linearadrsolve_tfqmr AVX SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_bicgstab_avx_sumfac test_linearadrsolve_bicgstab AVX SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_gmres_avx_sumfac test_linearadrsolve_gmres AVX SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_idrs_avx_sumfac test_linearadrsolve_idrs AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTBase_avx_sumfac test_ipwrtbase AVX SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_avx_sumfac test_ipwrtderivbase AVX SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_avx_sumfac test_linadvdiffreaction AVX SumFac FALSE)
    SET_UNIT_TEST(Mass_avx_sumfac test_mass AVX SumFac FALSE)
    SET_UNIT_TEST(NullPrecon_avx_sumfac test_nullprecon AVX SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_avx_sumfac test_physderiv AVX SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_avx_sumfac test_physinterp1dscaled AVX SumFac FALSE)
ENDIF()


# Device
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(Math_device test_math Device none FALSE)
    SET_UNIT_TEST(Parallel_reduce_device test_parallel_reduce Device none FALSE)
    SET_UNIT_TEST(AssmbScatr_device test_assmbscatr Device none FALSE)
    SET_UNIT_TEST(Dirichlet_device test_dirichlet Device none FALSE)
    SET_UNIT_TEST(Neumann_device test_neumann Device none FALSE)
    SET_UNIT_TEST(RungeKutta_device test_runge_kutta Device none FALSE)
    SET_UNIT_TEST(AdamsBashforth_device test_adams_bashforth Device none FALSE)
    SET_UNIT_TEST(DIRK_device test_dirk Device none FALSE)
    SET_UNIT_TEST(AdamsMoulton_device test_adams_moulton Device none FALSE)
    SET_UNIT_TEST(BDF_device test_bdf Device none FALSE)
    SET_UNIT_TEST(CNAB_device test_cnab Device none FALSE)
    SET_UNIT_TEST(IMEX_device test_imex Device none FALSE)
    SET_UNIT_TEST(IMEXdirk_device test_imexdirk Device none FALSE)
    SET_UNIT_TEST(Explicit_SDC_device test_explicit_sdc Device none FALSE)
    SET_UNIT_TEST(Implicit_SDC_device test_implicit_sdc Device none FALSE)
    SET_UNIT_TEST(IMEX_SDC_device test_imex_sdc Device none FALSE)
    SET_UNIT_TEST(Explicit_GEM_device test_explicit_gem Device none FALSE)
    SET_UNIT_TEST(Implicit_GEM_device test_implicit_gem Device none FALSE)
    SET_UNIT_TEST(IMEX_GEM_device test_imex_gem Device none FALSE)
    IF (NEKTAR_USE_MPI)
        IF (NOT NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA")
            SET_UNIT_TEST(AssmbScatr_device test_assmbscatr Device none TRUE)
        ENDIF()
    ENDIF()
ENDIF()

# Device generic
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(MultiplyByElmtInvMass_device_generic test_multiplybyelmtinvmass Device Generic FALSE)
    # SET_UNIT_TEST(expression_device_generic test_expression Device Generic FALSE)
ENDIF()

# Device StdMat
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(AddTraceIntegral_device_stdmat test_addtraceintegral Device StdMat FALSE)
    SET_UNIT_TEST(BwdTrans_device_stdmat test_bwdtrans Device StdMat FALSE)
    SET_UNIT_TEST(DiagPrecon_device_stdmat test_diagprecon Device StdMat FALSE)
    SET_UNIT_TEST(FwdTrans_device_stdmat test_fwdtrans Device StdMat FALSE)
    SET_UNIT_TEST(Helmholtz_device_stdmat test_helmholtz Device StdMat FALSE)
    SET_UNIT_TEST(Laplacian_device_stdmat test_laplacian Device StdMat FALSE)    
    SET_UNIT_TEST(PoissonSolve_device_stdmat test_poissonsolve Device StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_richardson_device_stdmat test_helmsolve_richardson Device StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_conjres_device_stdmat test_helmsolve_conjres Device StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_device_stdmat test_helmsolve_conjgrad Device StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_minres_device_stdmat test_helmsolve_minres Device StdMat FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_device_stdmat test_helmsolve_gmres Device StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_cgs_device_stdmat test_linearadrsolve_cgs Device StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_tfqmr_device_stdmat test_linearadrsolve_tfqmr Device StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_bicgstab_device_stdmat test_linearadrsolve_bicgstab Device StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_gmres_device_stdmat test_linearadrsolve_gmres Device StdMat FALSE)
    SET_UNIT_TEST(LinearADRSolve_idrs_device_stdmat test_linearadrsolve_idrs Device StdMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_stdmat test_ipwrtbase Device StdMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_stdmat test_ipwrtderivbase Device StdMat FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_stdmat test_linadvdiffreaction Device StdMat FALSE)
    SET_UNIT_TEST(Mass_device_stdmat test_mass Device StdMat FALSE)
    SET_UNIT_TEST(NullPrecon_device_stdmat test_nullprecon Device StdMat FALSE)
    SET_UNIT_TEST(PhysDeriv_device_stdmat test_physderiv Device StdMat FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_stdmat test_physinterp1dscaled Device StdMat FALSE)
ENDIF()

# Device SumFac
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(AddTraceIntegral_device_sumfac test_addtraceintegral Device SumFac FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfac test_bwdtrans Device SumFac FALSE)
    SET_UNIT_TEST(DiagPrecon_device_sumfac test_diagprecon Device SumFac FALSE)
    SET_UNIT_TEST(FwdTrans_device_sumfac test_fwdtrans Device SumFac FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfac test_helmholtz Device SumFac FALSE)
    SET_UNIT_TEST(Laplacian_device_sumfac test_laplacian Device SumFac FALSE)    
    SET_UNIT_TEST(HelmSolve_richardson_device_sumfac test_helmsolve_richardson Device SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_conjres_device_sumfac test_helmsolve_conjres Device SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfac test_helmsolve_conjgrad Device SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_minres_device_sumfac test_helmsolve_minres Device SumFac FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_device_sumfac test_helmsolve_gmres Device SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_cgs_device_sumfac test_linearadrsolve_cgs Device SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_tfqmr_device_sumfac test_linearadrsolve_tfqmr Device SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_bicgstab_device_sumfac test_linearadrsolve_bicgstab Device SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_gmres_device_sumfac test_linearadrsolve_gmres Device SumFac FALSE)
    SET_UNIT_TEST(LinearADRSolve_idrs_device_sumfac test_linearadrsolve_idrs Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfac test_ipwrtbase Device SumFac FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfac test_ipwrtderivbase Device SumFac FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfac test_linadvdiffreaction Device SumFac FALSE)
    SET_UNIT_TEST(Mass_device_sumfac test_mass Device SumFac FALSE)
    SET_UNIT_TEST(NullPrecon_device_sumfac test_nullprecon Device SumFac FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfac test_physderiv Device SumFac FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfac test_physinterp1dscaled Device SumFac FALSE)
ENDIF()

# Device SumFacTOP
IF(NEKTAR_ENABLE_DEVICE)
    SET_UNIT_TEST(AddTraceIntegral_device_sumfactop test_addtraceintegral Device SumFacTOP FALSE)
    SET_UNIT_TEST(BwdTrans_device_sumfactop test_bwdtrans Device SumFacTOP FALSE)
    SET_UNIT_TEST(DiagPrecon_device_sumfactop test_diagprecon Device SumFacTOP FALSE)
    SET_UNIT_TEST(FwdTrans_device_sumfactop test_fwdtrans Device SumFacTOP FALSE)
    SET_UNIT_TEST(Helmholtz_device_sumfactop test_helmholtz Device SumFacTOP FALSE)
    SET_UNIT_TEST(Laplacian_device_sumfactop test_laplacian Device SumFacTOP FALSE)    
    SET_UNIT_TEST(PoissonSolve_device_sumfactop test_poissonsolve Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_richardson_device_sumfactop test_helmsolve_richardson Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_conjres_device_sumfactop test_helmsolve_conjres Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_conjgrad_device_sumfactop test_helmsolve_conjgrad Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_minres_device_sumfactop test_helmsolve_minres Device SumFacTOP FALSE)
    SET_UNIT_TEST(HelmSolve_gmres_device_sumfactop test_helmsolve_gmres Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinearADRSolve_cgs_device_sumfactop test_linearadrsolve_cgs Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinearADRSolve_tfqmr_device_sumfactop test_linearadrsolve_tfqmr Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinearADRSolve_bicgstab_device_sumfactop test_linearadrsolve_bicgstab Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinearADRSolve_gmres_device_sumfactop test_linearadrsolve_gmres Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinearADRSolve_idrs_device_sumfactop test_linearadrsolve_idrs Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfactop test_ipwrtbase Device SumFacTOP FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfactop test_ipwrtderivbase Device SumFacTOP FALSE)
    SET_UNIT_TEST(LinAdvDiffReaction_device_sumfactop test_linadvdiffreaction Device SumFacTOP FALSE)
    SET_UNIT_TEST(Mass_device_sumfactop test_mass Device SumFacTOP FALSE)
    SET_UNIT_TEST(NullPrecon_device_sumfactop test_nullprecon Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfactop test_physderiv Device SumFacTOP FALSE)
    SET_UNIT_TEST(PhysInterp1DScaled_device_sumfactop test_physinterp1dscaled Device SumFacTOP FALSE)
ENDIF()

# CUDA SumFacMat
IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
    SET_UNIT_TEST(BwdTrans_device_sumfacmat test_bwdtrans_device_sumfacmat Device SumFacMat FALSE)
    SET_UNIT_TEST(IProductWRTBase_device_sumfacmat test_ipwrtbase_device_sumfacmat Device SumFacMat FALSE)
    SET_UNIT_TEST(IProductWRTDerivBase_device_sumfacmat test_ipwrtderivbase_device_sumfacmat Device SumFacMat FALSE)
    SET_UNIT_TEST(PhysDeriv_device_sumfacmat test_physderiv_device_sumfacmat Device SumFacMat FALSE)
ENDIF()
ENDIF()

# Copy XML file to build directory
FILE(COPY run DESTINATION ${CMAKE_BINARY_DIR}/library/UnitTests/Operators/)
