IF (NEKTAR_ENABLE_CUDA)
    SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cu)
ELSEIF (NEKTAR_ENABLE_HIP)
    SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.hip)
ELSE()
    SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cpp)
ENDIF()

SET(ProfilerElmtOps SOURCES ProfilerElmtOps.cpp)

ADD_SOLVER_EXECUTABLE(ProfilerMathKernels SOURCES ${ProfilerMathKernelsSource})

ADD_SOLVER_EXECUTABLE(ProfilerElmtOps SOURCES ${ProfilerElmtOps})
