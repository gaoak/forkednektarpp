# Set switch limit
SET(SWITCH_DIR ${CMAKE_SOURCE_DIR}/library/LibUtilities/BasicUtils/Switch)
MESSAGE("SWITCH_DIR = ${SWITCH_DIR}")
CONFIGURE_FILE(${SWITCH_DIR}/SwitchLimits.h.in ${SWITCH_DIR}/SwitchLimits.h @ONLY)

SET(BasicConstHeaders
    ./BasicConst/NektarUnivConsts.hpp
    ./BasicConst/NektarUnivTypeDefs.hpp
    ./BasicConst/GitRevision.h
)

SET(BasicUtilsHeaders
    ./BasicUtils/ArrayPolicies.hpp
    ./BasicUtils/CheckedCast.hpp
    ./BasicUtils/ConsistentObjectAccess.hpp
    ./BasicUtils/CompressData.h
    ./BasicUtils/CsvIO.h
    ./BasicUtils/Equation.h
    ./BasicUtils/FieldIO.h
    ./BasicUtils/FieldIOXml.h
    ./BasicUtils/Filesystem.hpp
    ./BasicUtils/ErrorUtil.hpp
    ./BasicUtils/Interpolator.h
    ./BasicUtils/HashUtils.hpp
    ./BasicUtils/Log.hpp
    ./BasicUtils/NekManager.hpp
    ./BasicUtils/NekFactory.hpp
    ./BasicUtils/ParseUtils.h
    ./BasicUtils/Progressbar.hpp
    ./BasicUtils/PtsField.h
    ./BasicUtils/PtsIO.h
    ./BasicUtils/Timer.h
    ./BasicUtils/RawType.hpp
    ./BasicUtils/RealComparison.hpp
    ./BasicUtils/SessionReader.h
    ./BasicUtils/ShapeType.hpp
    ./BasicUtils/SharedArray.hpp
    ./BasicUtils/Smath.hpp
    ./BasicUtils/Thread.h
    ./BasicUtils/ThreadStd.h
    ./BasicUtils/Vmath.hpp
    ./BasicUtils/VmathArray.hpp
    ./BasicUtils/Smath.hpp
    ./BasicUtils/Metis.hpp
    ./BasicUtils/VtkUtil.hpp
    ./BasicUtils/Likwid.hpp
    ./BasicUtils/Utils/UtilsDeviceKernels.hpp
    ./BasicUtils/Utils/UtilsDeviceKernelsHelper.hpp
)

SET(BasicUtilsSources
    ./BasicUtils/CompressData.cpp
    ./BasicUtils/CsvIO.cpp
    ./BasicUtils/Equation.cpp
    ./BasicUtils/ErrorUtil.cpp
    ./BasicUtils/FieldIO.cpp
    ./BasicUtils/FieldIOXml.cpp
    ./BasicUtils/Interpolator.cpp
    ./BasicUtils/ParseUtils.cpp
    ./BasicUtils/PtsField.cpp
    ./BasicUtils/PtsIO.cpp
    ./BasicUtils/SessionReader.cpp
    ./BasicUtils/Thread.cpp
    ./BasicUtils/ThreadStd.cpp
    ./BasicUtils/Timer.cpp
    ./BasicUtils/Vmath.cpp
    ./BasicUtils/Utils/UtilsDeviceKernels.cpp
)

# HDF5
IF( NEKTAR_USE_HDF5 )
    SET(BasicUtilsHeaders ${BasicUtilsHeaders}
        ./BasicUtils/H5.h
        ./BasicUtils/FieldIOHdf5.h
        ./BasicUtils/PtsIOHdf5.h)
    SET(BasicUtilsSources ${BasicUtilsSources}
        ./BasicUtils/H5.cpp
        ./BasicUtils/FieldIOHdf5.cpp
        ./BasicUtils/PtsIOHdf5.cpp)
ENDIF( NEKTAR_USE_HDF5 )


SET(CommunicationHeaders
    ./Communication/Comm.h
    ./Communication/CommSerial.h
    ./Communication/CommDataType.h
    ./Communication/ConnectivityResolver.hpp
    ./Communication/SharedPayloadResolver.hpp
    ./Communication/GatherScatter.h
    ./Communication/GsLib.hpp
    ./Communication/SharedIdPlan.hpp
    ./Communication/Transport.hpp
    ./Communication/Transposition.h
    ./Communication/Xxt.hpp
)

SET(CommunicationSources
    ./Communication/Comm.cpp
    ./Communication/CommDataType.cpp
    ./Communication/CommSerial.cpp
    ./Communication/GatherScatter.cpp
    ./Communication/Transposition.cpp
)

IF(NEKTAR_USE_MPI)
    SET(CommunicationHeaders ${CommunicationHeaders}
        ./Communication/CommMpi.h)
    SET(CommunicationSources ${CommunicationSources}
        ./Communication/CommMpi.cpp)
ENDIF(NEKTAR_USE_MPI)

IF ( NEKTAR_USE_CWIPI )
    SET(CommunicationHeaders ${CommunicationHeaders}
        ./Communication/CommCwipi.h)
    SET(CommunicationSources ${CommunicationSources}
        ./Communication/CommCwipi.cpp)
ENDIF ()

SET(FFTHeaders
    ./FFT/NektarFFT.h
)

SET(FFTSources
    ./FFT/NektarFFT.cpp
)

IF(NEKTAR_USE_FFTW)
    SET(FFTHeaders ${FFTHeaders} ./FFT/NekFFTW.h)
    SET(FFTSources ${FFTSources} ./FFT/NekFFTW.cpp)
ENDIF(NEKTAR_USE_FFTW)

IF(NEKTAR_ENABLE_DEVICE)
    SET(FFTHeaders ${FFTHeaders}
        ./FFT/NekDeviceFFT.h
        ./FFT/DerivZDeviceFFT.h)
    # The helpers include the vendor FFT headers, so they are compiled with the
    # backend sources but not installed.
    IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        SET(FFTPrivateHeaders ./FFT/NekDeviceFFTHIPCUDAHelper.h)
        SET(FFTSources ${FFTSources}
            ./FFT/NekDeviceFFT.cu ./FFT/DerivZDeviceFFT.cu)
    ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
        SET(FFTPrivateHeaders ./FFT/NekDeviceFFTHIPCUDAHelper.h)
        SET(FFTSources ${FFTSources}
            ./FFT/NekDeviceFFT.hip ./FFT/DerivZDeviceFFT.hip)
    ELSEIF(NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
        SET(FFTPrivateHeaders ./FFT/NekDeviceFFTSYCLHelper.h)
        SET(FFTSources ${FFTSources}
            ./FFT/NekDeviceFFT.cpp ./FFT/DerivZDeviceFFT.cpp)
    ENDIF()
ENDIF()

SET(FoundationHeaders
    ./Foundations/Basis.h
    ./Foundations/BasisType.h
    ./Foundations/BLPoints.h
    ./Foundations/FoundationsFwd.hpp
    ./Foundations/Foundations.hpp
    ./Foundations/FourierPoints.h
    ./Foundations/FourierSingleModePoints.h
    ./Foundations/GaussPoints.h
    ./Foundations/Graph.h
    ./Foundations/InterpCoeff.h
    ./Foundations/Interp.h
    ./Foundations/ManagerAccess.h
    ./Foundations/NodalPrismEvenlySpaced.h
    ./Foundations/NodalPrismElec.h
    ./Foundations/NodalTetElecData.h
    ./Foundations/NodalTetElec.h
    ./Foundations/NodalTetEvenlySpaced.h
    ./Foundations/NodalTriElecData.h
    ./Foundations/NodalTriElec.h
    ./Foundations/NodalTriEvenlySpaced.h
    ./Foundations/NodalTriFeketeData.h
    ./Foundations/NodalTriFekete.h
    ./Foundations/NodalTriSPIData.h
    ./Foundations/NodalTriSPI.h
    ./Foundations/NodalTetSPI.h
    ./Foundations/NodalPrismSPI.h
    ./Foundations/NodalPyrEvenlySpaced.h
    ./Foundations/NodalTetSPIData.h
    ./Foundations/NodalQuadElec.h
    ./Foundations/NodalHexElec.h
    ./Foundations/NodalUtil.h
    ./Foundations/PhysGalerkinProject.h
    ./Foundations/Points.h
    ./Foundations/PointsType.h
    ./Foundations/PolyEPoints.h
)

SET(FoundationSources
    ./Foundations/Basis.cpp
    ./Foundations/BLPoints.cpp
    ./Foundations/FourierPoints.cpp
    ./Foundations/FourierSingleModePoints.cpp
    ./Foundations/GaussPoints.cpp
    ./Foundations/Graph.cpp
    ./Foundations/InterpCoeff.cpp
    ./Foundations/Interp.cpp
    ./Foundations/ManagerAccess.cpp
    ./Foundations/NodalPrismEvenlySpaced.cpp
    ./Foundations/NodalPrismElec.cpp
    ./Foundations/NodalTetElec.cpp
    ./Foundations/NodalTetEvenlySpaced.cpp
    ./Foundations/NodalTriElec.cpp
    ./Foundations/NodalTriEvenlySpaced.cpp
    ./Foundations/NodalTriFekete.cpp
    ./Foundations/NodalTriSPI.cpp
    ./Foundations/NodalTetSPI.cpp
    ./Foundations/NodalQuadElec.cpp
    ./Foundations/NodalHexElec.cpp
    ./Foundations/NodalPrismSPI.cpp
    ./Foundations/NodalPyrEvenlySpaced.cpp
    ./Foundations/NodalUtil.cpp
    ./Foundations/PhysGalerkinProject.cpp
    ./Foundations/Points.cpp
    ./Foundations/PolyEPoints.cpp
)

SET(InterpreterHeaders
    ./Interpreter/Interpreter.h
)

SET(InterpreterSources
    ./Interpreter/Interpreter.cpp
)

SET(LinearAlgebraHeaders
    ./LinearAlgebra/Arpack.hpp
    ./LinearAlgebra/Blas.hpp
    ./LinearAlgebra/BlockMatrix.hpp
    ./LinearAlgebra/CanGetRawPtr.hpp
    ./LinearAlgebra/ExplicitInstantiation.h
    ./LinearAlgebra/Lapack.hpp
    ./LinearAlgebra/MatrixBase.hpp
    ./LinearAlgebra/MatrixFuncs.h
    ./LinearAlgebra/MatrixOperations.hpp
    ./LinearAlgebra/MatrixStorageType.h
    ./LinearAlgebra/MatrixType.h
    ./LinearAlgebra/NekLinSys.hpp
    ./LinearAlgebra/NekLinSysIter.h
    ./LinearAlgebra/NekLinSysIterCG.h
    ./LinearAlgebra/NekLinSysIterCGLoc.h
    ./LinearAlgebra/NekLinSysIterEvsDirect.h
    ./LinearAlgebra/NekLinSysIterFixedpointJacobi.h
    ./LinearAlgebra/NekLinSysIterGMRES.h
    ./LinearAlgebra/NekLinSysIterGMRESLoc.h
    ./LinearAlgebra/NekLinSysIterEvsDirect.h
    ./LinearAlgebra/NekMatrixFwd.hpp
    ./LinearAlgebra/NekMatrix.hpp
    ./LinearAlgebra/NekNonlinSysIter.h
    ./LinearAlgebra/NekNonlinSysIterNewton.h
    ./LinearAlgebra/NekNonlinSysIterNewtonHookStep.h
    ./LinearAlgebra/NekNonlinSysIterNewtonTrustRegion.h
    ./LinearAlgebra/NekNonlinSysIterNewtonBacktrack.h
    ./LinearAlgebra/NekPoint.hpp
    ./LinearAlgebra/NekSys.h
    ./LinearAlgebra/NekTypeDefs.hpp
    ./LinearAlgebra/NekVectorFwd.hpp
    ./LinearAlgebra/NekVector.hpp
    ./LinearAlgebra/PointerWrapper.h
    ./LinearAlgebra/ScaledMatrix.hpp
    ./LinearAlgebra/StandardMatrix.hpp
    ./LinearAlgebra/TransF77.hpp
    ./LinearAlgebra/StorageSmvBsr.hpp
    ./LinearAlgebra/NistSparseDescriptors.hpp
    ./LinearAlgebra/SparseDiagBlkMatrix.hpp
    ./LinearAlgebra/SparseMatrix.hpp
    ./LinearAlgebra/SparseMatrixFwd.hpp
    ./LinearAlgebra/SparseUtils.hpp
)

SET(LinearAlgebraSources
    ./LinearAlgebra/Blas.cpp
    ./LinearAlgebra/BlockMatrix.cpp
    ./LinearAlgebra/MatrixBase.cpp
    ./LinearAlgebra/MatrixFuncs.cpp
    ./LinearAlgebra/MatrixOperations.cpp
    ./LinearAlgebra/MatrixVectorMultiplication.cpp
    ./LinearAlgebra/NekLinSysIter.cpp
    ./LinearAlgebra/NekLinSysIterGMRES.cpp
    ./LinearAlgebra/NekLinSysIterGMRESLoc.cpp
    ./LinearAlgebra/NekLinSysIterCG.cpp
    ./LinearAlgebra/NekLinSysIterCGLoc.cpp
    ./LinearAlgebra/NekLinSysIterEvsDirect.cpp
    ./LinearAlgebra/NekLinSysIterFixedpointJacobi.cpp
    ./LinearAlgebra/NekLinSysIterEvsDirect.cpp
    ./LinearAlgebra/NekNonlinSysIter.cpp
    ./LinearAlgebra/NekNonlinSysIterNewton.cpp
    ./LinearAlgebra/NekNonlinSysIterNewtonHookStep.cpp
    ./LinearAlgebra/NekNonlinSysIterNewtonTrustRegion.cpp
    ./LinearAlgebra/NekNonlinSysIterNewtonBacktrack.cpp
    ./LinearAlgebra/NekSys.cpp
    ./LinearAlgebra/NekVector.cpp
    ./LinearAlgebra/ScaledMatrix.cpp
    ./LinearAlgebra/StandardMatrix.cpp
    ./LinearAlgebra/SparseUtils.cpp
    ./LinearAlgebra/StorageSmvBsr.cpp
    ./LinearAlgebra/SparseDiagBlkMatrix.cpp
    ./LinearAlgebra/SparseMatrix.cpp
)

SET(MemoryHeaders
    ./Memory/ThreadSpecificPool.hpp
    ./Memory/NekMemoryManager.hpp
    ./Memory/ObjectPool.hpp
)

SET(MemorySources
    ./Memory/ThreadSpecificPool.cpp
)

SET(PolyLibHeaders
    ./Polylib/Polylib.h
)

SET(PolylibSources
    ./Polylib/Polylib.cpp
)

SET(TimeIntegrationHeaders
    ./TimeIntegration/AdamsBashforthTimeIntegrationSchemes.h
    ./TimeIntegration/AdamsMoultonTimeIntegrationSchemes.h
    ./TimeIntegration/BDFImplicitTimeIntegrationSchemes.h
    ./TimeIntegration/EulerTimeIntegrationSchemes.h
    ./TimeIntegration/EulerExponentialTimeIntegrationSchemes.h
    ./TimeIntegration/CNABTimeIntegrationScheme.h
    ./TimeIntegration/DIRKTimeIntegrationSchemes.h
    ./TimeIntegration/IMEXTimeIntegrationSchemes.h
    ./TimeIntegration/IMEXdirkTimeIntegrationSchemes.h
    ./TimeIntegration/IMEXTimeIntegrationSchemeSDC.h
    ./TimeIntegration/ImplicitTimeIntegrationSchemeSDC.h
    ./TimeIntegration/MCNABTimeIntegrationScheme.h
    ./TimeIntegration/RungeKuttaTimeIntegrationSchemes.h
    ./TimeIntegration/TimeIntegrationScheme.h
    ./TimeIntegration/NoSchemeTimeIntegrationScheme.h
    ./TimeIntegration/TimeIntegrationSchemeFIT.h
    ./TimeIntegration/TimeIntegrationSchemeGEM.h
    ./TimeIntegration/TimeIntegrationSchemeGLM.h
    ./TimeIntegration/TimeIntegrationSchemeSDC.h
    ./TimeIntegration/TimeIntegrationAlgorithmGLM.h
    ./TimeIntegration/TimeIntegrationSolutionGLM.h
)

SET(TimeIntegrationSources
    ./TimeIntegration/SchemeInitializer.cpp
    ./TimeIntegration/TimeIntegrationScheme.cpp
    ./TimeIntegration/NoSchemeTimeIntegrationScheme.cpp
    ./TimeIntegration/TimeIntegrationSchemeFIT.cpp
    ./TimeIntegration/TimeIntegrationSchemeGEM.cpp
    ./TimeIntegration/TimeIntegrationSchemeGLM.cpp
    ./TimeIntegration/TimeIntegrationSchemeSDC.cpp
    ./TimeIntegration/TimeIntegrationAlgorithmGLM.cpp
    ./TimeIntegration/TimeIntegrationSolutionGLM.cpp
)

SET(BaseHeaders
    LibUtilitiesDeclspec.h
)

IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    SET(DeviceSupportHeaders 
        ./Backends/DeviceProperties.hpp
        ./BasicUtils/Field/Block.hpp
        ./BasicUtils/Math/Math.hpp
        ./BasicUtils/Math/MathDeclspec.h
        ./BasicUtils/Math/MathHelperDef.hpp
        ./BasicUtils/DataWarehouse/BasisDataWarehouse.hpp
        ./BasicUtils/DataWarehouse/ModeIndexDataWarehouse.hpp
        ./LoopExecution/LoopExecution.hpp
        ./Memory/MemoryAlloc.hpp
    )
    SET(DeviceSupportSources 
        ./Backends/Backends.cpp
        ./Backends/DeviceProperties.cpp
        ./BasicUtils/Field/Block.cpp
        ./BasicUtils/DataWarehouse/BasisDataWarehouse.cpp
        ./BasicUtils/DataWarehouse/ModeIndexDataWarehouse.cpp
        ./LoopExecution/LoopExecution.cpp
    )
    IF (NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        SET(DeviceSupportHeaders
            ${DeviceSupportHeaders} 
            ./Backends/CUDAStream.hpp
        )
        SET(DeviceSupportSources 
            ${DeviceSupportSources} 
            ./Backends/CUDAStream.cpp
            ./BasicUtils/Field/BlockHelper.cu
            ./BasicUtils/Math/MathHelper.cu
            ./Memory/MemoryAlloc.cu
        )
    ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
        SET(DeviceSupportHeaders 
            ${DeviceSupportHeaders} 
            ./Backends/HIPStream.hpp
        )
        SET(DeviceSupportSources 
            ${DeviceSupportSources} 
            ./Backends/HIPStream.cpp
            ./BasicUtils/Field/BlockHelper.hip
            ./BasicUtils/Math/MathHelper.hip
            ./Memory/MemoryAlloc.hip
        )
    ELSEIF (NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
        SET(DeviceSupportHeaders 
            ${DeviceSupportHeaders} 
            ./Backends/SYCLQueue.hpp
        )
        SET(DeviceSupportSources 
            ${DeviceSupportSources} 
            ./Backends/SYCLQueue.cpp
            ./BasicUtils/Field/BlockHelper.cpp
            ./BasicUtils/Math/MathHelper.cpp
        )
    ELSE()
        SET(DeviceSupportSources 
            ${DeviceSupportSources} 
            ./BasicUtils/Field/BlockHelper.cpp
            ./BasicUtils/Math/MathHelper.cpp
        )
    ENDIF()
ENDIF()

SOURCE_GROUP("Basic Const" FILES ${BasicConstHeaders})
SOURCE_GROUP("Basic Utils Headers" FILES ${BasicUtilsHeaders})
SOURCE_GROUP("Basic Utils Sources" FILES ${BasicUtilsSources})
SOURCE_GROUP("Foundation Headers" FILES ${FoundationHeaders})
SOURCE_GROUP("Foundation Sources" FILES ${FoundationSources})
SOURCE_GROUP("Linear Algebra Headers" FILES ${LinearAlgebraHeaders})
SOURCE_GROUP("Linear Algebra Sources" FILES ${LinearAlgebraSources})
SOURCE_GROUP("Memory Headers" FILES ${MemoryHeaders})
SOURCE_GROUP("Memory Sources" FILES ${MemorySources})

SOURCE_GROUP("Polylib Headers" FILES ${PolyLibHeaders})
SOURCE_GROUP("Polylib Sources" FILES ${PolylibSources})

SOURCE_GROUP("Interpreter Headers" FILES ${InterpreterHeaders})
SOURCE_GROUP("Interpreter Sources" FILES ${InterpreterSources})

SOURCE_GROUP("Communication Headers" FILES ${CommunicationHeaders})
SOURCE_GROUP("Communication Sources" FILES ${CommunicationSources})

SOURCE_GROUP("TimeIntegration Headers" FILES ${TimeIntegrationHeaders})
SOURCE_GROUP("TimeIntegration Sources" FILES ${TimeIntegrationSources})

SOURCE_GROUP("FFT Headers" FILES ${FFTHeaders} ${FFTPrivateHeaders})
SOURCE_GROUP("FFT Sources" FILES ${FFTSources})

SOURCE_GROUP("Device Support Headers" FILES ${DeviceSupportHeaders})
SOURCE_GROUP("Device Support Sources" FILES ${DeviceSupportSources})

SOURCE_GROUP("Base" FILES ${BaseHeaders})

SET(LibUtilityHeaders
    ${MemoryHeaders}
    ${BasicUtilsHeaders}
    ${PolyLibHeaders}
    ${CommunicationHeaders}
    ${FoundationHeaders}
    ${CADSystemHeaders}
    ${LinearAlgebraHeaders}
    ${BasicConstHeaders}
    ${InterpreterHeaders}
    ${TimeIntegrationHeaders}
    ${FFTHeaders}
    ${DeviceSupportHeaders}
    ${BaseHeaders}
)

# Define the git branch and SHA1 hash if we are in a git repository. This is
# stored in an automatically generated file so that the source tree is not built
# on commit/update.
SET(GitRevisionFile "${CMAKE_CURRENT_BINARY_DIR}/GitRevision.cpp")
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/BasicConst/GitRevision.cpp.in
${GitRevisionFile})

SET(LibUtilitySources
    ${BasicUtilsSources}
    ${FoundationSources}
    ${CADSystemSources}
    ${PolylibSources}
    ${CommunicationSources}
    ${LinearAlgebraSources}
    ${InterpreterSources}
    ${TimeIntegrationSources}
    ${FFTSources}
    ${FFTPrivateHeaders}
    ${DeviceSupportSources}
    ${MemorySources}
    ${GitRevisionFile}
)

ADD_DEFINITIONS(-DLIB_UTILITIES_EXPORTS)

ADD_NEKTAR_LIBRARY(LibUtilities
    SOURCES ${LibUtilitySources}
    HEADERS ${LibUtilityHeaders}
    SUMMARY "Nektar++ LibUtilities library"
    DESCRIPTION "This library provides core routines including linear algebra and integration with ThirdParty libraries.")

# tinysimd
ADD_LIBRARY(tinysimd INTERFACE)
TARGET_INCLUDE_DIRECTORIES(tinysimd INTERFACE SimdLib)

ADD_DEPENDENCIES(LibUtilities boost tinyxml-2.6.2 zlib-1.2.9 tinysimd)

# Older gcc versions (6 <= version < 9) require libstdc++fs.
IF (DEFINED NEKTAR_FILESYSTEM_LIBRARY)
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${NEKTAR_FILESYSTEM_LIBRARY})
ENDIF()

### Libraries needed specifically by LibUtilities ###

# CWIPI
IF ( NEKTAR_USE_CWIPI )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${CWIPI_LIBRARY})
    ADD_DEPENDENCIES(LibUtilities cwipi-0.11.1)
ENDIF ()

# FFTW
IF( NEKTAR_USE_FFTW )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${FFTW_LIBRARY})
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${FFTWF_LIBRARY})
    ADD_DEPENDENCIES(LibUtilities fftw-3.2.2)
    ADD_DEPENDENCIES(LibUtilities fftwf-3.2.2)
ENDIF( NEKTAR_USE_FFTW )

# ARPACK
IF( NEKTAR_USE_ARPACK )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${ARPACK_LIBRARY})
    ADD_DEPENDENCIES(LibUtilities arpack-ng-1.0)
ENDIF( NEKTAR_USE_ARPACK )

# Boost
TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    optimized ${ZLIB_LIBRARIES} debug ${ZLIB_LIBRARIES_DEBUG}
)

# TinyXML
TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${TINYXML_LIBRARY})
IF( ${CMAKE_SYSTEM} MATCHES "Linux.*" )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC rt)
ENDIF()

# MPI
IF( NEKTAR_USE_MPI )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${GSMPI_LIBRARY} ${XXT_LIBRARY})
    IF (NOT MPI_BUILTIN)
        TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${MPI_CXX_LIBRARIES})
        SET_TARGET_PROPERTIES(LibUtilities
            PROPERTIES LINK_FLAGS "${THE_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS}")
        SET_TARGET_PROPERTIES(LibUtilities
            PROPERTIES COMPILE_FLAGS "${THE_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    ENDIF()

    ADD_DEPENDENCIES(LibUtilities gsmpi-1.2.1_2)

    IF (THIRDPARTY_BUILD_MPI)
        ADD_DEPENDENCIES(LibUtilities openmpi-1.10.3)
    ENDIF()
ENDIF( NEKTAR_USE_MPI )

# Lapack and Blas
IF(NOT BLAS_LAPACK_BUILTIN)
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${BLAS_LAPACK})
ENDIF()

IF(THIRDPARTY_BUILD_BLAS_LAPACK)
    ADD_DEPENDENCIES(LibUtilities lapack-3.7.1)
ENDIF()

IF (NEKTAR_ENABLE_DEVICE_SUPPORT)
    # NekBlas
    SET(NEKBLAS_HEADERS
        ${NEKBLAS_HEADERS}
        ./LinearAlgebra/Blas.hpp
        ./LinearAlgebra/NekBlas/NekBlas.hpp
        ./LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp
    )

    SET(NEKBLAS_SOURCES
        ${NEKBLAS_SOURCES}
        ./LinearAlgebra/NekBlas/blas.cpp
    )

    # LIBXSMM specific file
    IF(NEKTAR_ENABLE_SIMD)
        IF(_SYSTEM_PROCESSOR STREQUAL "x86_64")
            SET(NEKBLAS_SOURCES
                ${NEKBLAS_SOURCES}
                ./LinearAlgebra/NekBlas/xsmm.cpp
            )
        ENDIF()
    ENDIF()

    # MAGMA specific file
    IF(NEKTAR_USE_MAGMA)
        SET(NEKBLAS_SOURCES
            ${NEKBLAS_SOURCES}
            ./LinearAlgebra/NekBlas/magma.cpp
        )
    # CUBLAS specific file
    ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        SET(NEKBLAS_SOURCES
            ${NEKBLAS_SOURCES}
            ./LinearAlgebra/NekBlas/cuBlas.cpp
        )
    # HIPBLAS specific file
    ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
        SET(NEKBLAS_SOURCES
            ${NEKBLAS_SOURCES}
            ./LinearAlgebra/NekBlas/hipBlas.cpp
        )
    # MKL/OneMath specific file
    ELSEIF(NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
        SET(NEKBLAS_SOURCES
            ${NEKBLAS_SOURCES}
            ./LinearAlgebra/NekBlas/oneMath.cpp
        )
    ENDIF()

    ADD_NEKTAR_LIBRARY(NekBlas
        SOURCES ${NEKBLAS_SOURCES}
        HEADERS ${NEKBLAS_HEADERS}
        SUMMARY "Nektar++ NekBlas library"
        DESCRIPTION "This library provides classes related to NekBlas for Nektar++.")

    IF(NOT BLAS_LAPACK_BUILTIN)
        TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE ${BLAS_LAPACK})
    ENDIF()

    # NekBlas's SYCL BLAS path references SYCLQueue statics defined in
    # LibUtilities, which is built after it and which NekBlas cannot link
    # without a cycle. A Linux shared library leaves that to load time and
    # says nothing; the macOS linker demands every symbol up front and fails.
    # Defer resolution there so the two platforms behave alike.
    IF(APPLE AND NEKTAR_ENABLE_DEVICE MATCHES "SYCL")
        TARGET_LINK_OPTIONS(NekBlas PRIVATE -Wl,-undefined,dynamic_lookup)
    ENDIF()
    IF(THIRDPARTY_BUILD_BLAS_LAPACK)
        ADD_DEPENDENCIES(NekBlas lapack-3.7.1)
    ENDIF()
    # MAGMA
    IF(NEKTAR_USE_MAGMA)
        # MAGMA require pubic linking
        TARGET_LINK_LIBRARIES(NekBlas LINK_PUBLIC ${MAGMA_LIBRARY})
        # magma.cpp calls cuBLAS/cuSPARSE (resp. hipBLAS/hipSPARSE)
        # directly to build the MAGMA queue, so link them here rather than
        # relying on libmagma's own dependencies.
        IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
            TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE
                ${CUBLAS_LIBRARY} ${CUSPARSE_LIBRARY})
        ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
            TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE
                ${HIPBLAS_LIBRARY} ${HIPSPARSE_LIBRARY})
        ENDIF()
        ADD_DEPENDENCIES(NekBlas magma-v2.10.0)
    # CUBLAS
    ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE ${CUBLAS_LIBRARY})
    # HIPBLAS
    ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
        TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE ${HIPBLAS_LIBRARY})
    # OneMath
    ELSEIF(NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
        TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE ${ONEMATH_LIBRARY})
        IF(NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
            ADD_DEPENDENCIES(NekBlas onemath-v0.9)
        ENDIF()
    ENDIF()

    # LIBXSMM
    IF(NEKTAR_ENABLE_SIMD)
        IF(_SYSTEM_PROCESSOR STREQUAL "x86_64")
            TARGET_LINK_LIBRARIES(NekBlas LINK_PRIVATE ${LIBXSMM_LIBRARY})
            ADD_DEPENDENCIES(NekBlas libxsmm-1.17)
        ENDIF()
    ENDIF()

    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC NekBlas)

    # CUFFT
    IF (NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
        IF(NEKTAR_USE_CUFFT_STATIC)
            TARGET_COMPILE_DEFINITIONS(LibUtilities PRIVATE NEKTAR_USE_CUFFT_STATIC)
            SET_TARGET_PROPERTIES(LibUtilities PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

            # NekDeviceFFT.cu.o path under the CMake Makefile generator
            SET(_nekcufft_obj
                "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LibUtilities.dir/FFT/NekDeviceFFT.cu.o")
            SET(_merged_dlink
                "${CMAKE_CURRENT_BINARY_DIR}/cufft_merged_device_link.o")

            ADD_CUSTOM_COMMAND(
                OUTPUT  "${_merged_dlink}"
                COMMAND "${CMAKE_CUDA_COMPILER}"
                        ${CUFFT_ARCH_FLAGS}
                        --compiler-options=-fPIC
                        -dlink
                        "${_nekcufft_obj}"
                        "$<TARGET_FILE:CUDA::cufft_static>"
                        -o "${_merged_dlink}"
                DEPENDS "${_nekcufft_obj}" "$<TARGET_FILE:CUDA::cufft_static>"
                COMMENT "nvlink: merged device-link NekDeviceFFT.cu + libcufft_static (callback fat-binary)"
                VERBATIM
            )
            ADD_CUSTOM_TARGET(cufft_merged_dlink DEPENDS "${_merged_dlink}")
            ADD_DEPENDENCIES(LibUtilities cufft_merged_dlink)
            TARGET_LINK_LIBRARIES(LibUtilities PRIVATE "${_merged_dlink}")
        ENDIF()

        IF(NEKTAR_USE_CUFFTDX)
            # Private: only DerivZDeviceFFT.cu includes the cuFFTDx headers
            # or tests the option.
            TARGET_INCLUDE_DIRECTORIES(LibUtilities PRIVATE
                    $<BUILD_INTERFACE:${CUFFTDX_INCLUDE_DIR}>)
            TARGET_INCLUDE_DIRECTORIES(LibUtilities PRIVATE
                    $<BUILD_INTERFACE:${CUFFTDX_CUTLASS_INCLUDE_DIR}>)
            TARGET_COMPILE_DEFINITIONS(LibUtilities PRIVATE NEKTAR_USE_CUFFTDX)
            TARGET_COMPILE_DEFINITIONS(LibUtilities PRIVATE
                CUFFTDX_TARGET_SM=${CUFFTDX_TARGET_SM})
            ADD_DEPENDENCIES(LibUtilities cufftdx-25.12)
            MESSAGE(STATUS "cuFFTDx: enabled (${CUFFTDX_INCLUDE_DIR}), SM=${CUFFTDX_TARGET_SM}")
        ENDIF()
        # Private: the device FFT libraries are named only by the .cu/.hip/.cpp
        # translation units under FFT/, and the headers that reach them are not
        # included outside LibUtilities.
        TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${CUFFT_LIBRARY})
    # HIPFFT
    ELSEIF (NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
        TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${HIPFFT_LIBRARY})
    # ONEMATH / ONEMKL DFT
    ELSEIF (NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
        TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${ONEMATH_LIBRARY})
        IF(NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
            ADD_DEPENDENCIES(LibUtilities onemath-v0.9)
        ENDIF()
    ENDIF()

    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${NEKTAR_CUDA_DEPENDS})
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${NEKTAR_HIP_DEPENDS})
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${NEKTAR_SYCL_DEPENDS})
ENDIF()

IF( NEKTAR_USE_PETSC )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PRIVATE ${PETSC_LIBRARIES})
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${CMAKE_DL_LIBS})
    ADD_DEPENDENCIES(LibUtilities petsc-3.19.3)
ENDIF( NEKTAR_USE_PETSC )

IF (NEKTAR_BUILD_PYTHON)
    SUBDIRS(Python)
ENDIF()

# HDF5
IF( NEKTAR_USE_HDF5 )
    TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${HDF5_LIBRARIES})
    ADD_DEPENDENCIES(LibUtilities hdf5-1.12.3)
ENDIF( NEKTAR_USE_HDF5 )

# Likwid
IF (NEKTAR_USE_LIKWID)
   TARGET_LINK_LIBRARIES(LibUtilities LINK_PUBLIC ${LIKWID_LIBRARY})
   TARGET_INCLUDE_DIRECTORIES(LibUtilities PUBLIC ${LIKWID_INCLUDE_DIR})
ENDIF()

# Install
INSTALL(DIRECTORY ./
    DESTINATION ${NEKTAR_INCLUDE_DIR}/LibUtilities
    COMPONENT dev
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
