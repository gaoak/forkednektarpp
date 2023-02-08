cmake_minimum_required(VERSION 3.0)
project(Redesign CXX)

option(NEKTAR_USE_CUDA "Enable CUDA support" FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRC LibUtilities/ErrorUtil.cpp Operators/Operator.cpp Operators/OperatorBwdTrans.cpp Operators/BwdTrans/BwdTransImpl.cpp)

if (NEKTAR_USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DNEKTAR_USE_CUDA)
    set(SRC ${SRC} MemoryRegionCUDA.cu Operators/BwdTrans/BwdTransCUDA.cu)
endif()

add_library(Operators SHARED ${SRC})
target_include_directories(Operators PRIVATE "${CMAKE_SOURCE_DIR}")

set(TEST_SRC main.cpp)
add_executable(main ${TEST_SRC})
target_link_libraries(main Operators)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}")
