SET(ProfilerMathKernelsSource SOURCES ProfilerMathKernels.cpp)

SET(ProfilerElmtOps SOURCES ProfilerElmtOps.cpp)

SET(BP1 SOURCES BP1.cpp)

SET(BP3 SOURCES BP3.cpp)

ADD_SOLVER_EXECUTABLE(ProfilerMathKernels SOURCES ${ProfilerMathKernelsSource})

ADD_SOLVER_EXECUTABLE(ProfilerElmtOps SOURCES ${ProfilerElmtOps})

ADD_SOLVER_EXECUTABLE(BP1 SOURCES ${BP1})

ADD_SOLVER_EXECUTABLE(BP3 SOURCES ${BP3})
