CMAKE_DEPENDENT_OPTION(NEKTAR_LIBRARY_PROFILER
    "Build the operator profiler." ON
    "NEKTAR_BUILD_REDESIGN" OFF)

IF (NOT NEKTAR_BUILD_SOLVERS)
    SET(NEKTAR_SOLVER_PROFILER OFF CACHE INTERNAL "")
ENDIF()

IF (NOT NEKTAR_BUILD_REDESIGN)
    SET(NEKTAR_SOLVER_PROFILER OFF CACHE INTERNAL "")
ENDIF()

IF (NEKTAR_ENABLE_CUDA)
    SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cu)
ELSEIF (NEKTAR_ENABLE_HIP)
    SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.hip)
ELSE()
    SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cpp)
ENDIF()

SET(ProfilerElmtOps SOURCES ProfilerElmtOps.cpp)

ADD_SOLVER_EXECUTABLE(ProfilerMathKernels SOURCES
    ${ProfilerMathKernelsSource})

ADD_SOLVER_EXECUTABLE(ProfilerElmtOps SOURCES
    ${ProfilerElmtOps})
