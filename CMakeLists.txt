cmake_minimum_required(VERSION 3.0)
project(Redesign CXX)

option(NEKTAR_USE_CUDA "Enable CUDA support" FALSE)

set(SRC main.cpp Operator.cpp)

if (NEKTAR_USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DNEKTAR_USE_CUDA)
    set(SRC ${SRC} Operator.cu)
endif()

add_executable(main ${SRC})
