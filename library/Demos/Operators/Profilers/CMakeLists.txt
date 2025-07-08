SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cpp)

SET(ProfilerElmtOps SOURCES ProfilerElmtOps.cpp)

ADD_SOLVER_EXECUTABLE(ProfilerMathKernels SOURCES ${ProfilerMathKernelsSource})

ADD_SOLVER_EXECUTABLE(ProfilerElmtOps SOURCES ${ProfilerElmtOps})
