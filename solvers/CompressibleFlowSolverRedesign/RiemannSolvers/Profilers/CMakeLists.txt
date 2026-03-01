SET(ProfilerRiemannOps SOURCES ProfilerRiemannOps.cpp ${RIEMANN_FACTORY_SOURCES})

ADD_SOLVER_EXECUTABLE(ProfilerRiemannOps SOURCES ${ProfilerRiemannOps})

TARGET_INCLUDE_DIRECTORIES(ProfilerRiemannOps PRIVATE
    "${CMAKE_SOURCE_DIR}/solvers/CompressibleFlowSolverRedesign"
)