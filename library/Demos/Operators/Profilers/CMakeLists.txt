SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cpp)

SET(ProfilerElmtOps SOURCES ProfilerElmtOps.cpp)

SET(ProfilerBP SOURCES ProfilerBP.cpp)

ADD_SOLVER_EXECUTABLE(ProfilerMathKernels SOURCES ${ProfilerMathKernelsSource})

ADD_SOLVER_EXECUTABLE(ProfilerElmtOps SOURCES ${ProfilerElmtOps})

ADD_SOLVER_EXECUTABLE(ProfilerBP SOURCES ${ProfilerBP})
