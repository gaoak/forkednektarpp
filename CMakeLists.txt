cmake_minimum_required(VERSION 3.0)
project(Redesign CXX)

enable_testing()

option(NEKTAR_USE_CUDA "Enable CUDA support" FALSE)
option(NEKTAR_USE_SIMD "Enable SIMD support, if available" TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Default install location: build/dist
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "" FORCE)
ENDIF()

find_package(Nektar++ REQUIRED)
include_directories(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

set(SRC Field.cpp Operators/Operator.cpp 
    Operators/OperatorBwdTrans.cpp Operators/BwdTrans/BwdTransImpl.cpp 
    Operators/OperatorIProductWRTBase.cpp Operators/IProductWRTBase/IProductWRTBaseImpl.cpp
    Operators/OperatorMass.cpp Operators/Mass/Mass.cpp
    Operators/OperatorIdentity.cpp Operators/Identity/Identity.cpp
    Operators/OperatorConjGrad.cpp Operators/ConjGrad/ConjGrad.cpp
    Operators/OperatorFwdTrans.cpp Operators/FwdTrans/FwdTrans.cpp
    Operators/OperatorAssmbScatr.cpp Operators/AssmbScatr/AssmbScatr.cpp
    Operators/OperatorDirBndCond.cpp Operators/DirBndCond/DirBndCond.cpp
    Operators/OperatorPhysDeriv.cpp Operators/PhysDeriv/PhysDerivImpl.cpp
    Operators/OperatorHelm.cpp Operators/Helm/Helm.cpp
    Operators/OperatorDiagPrecon.cpp Operators/DiagPrecon/DiagPrecon.cpp
    )

if (NEKTAR_USE_CUDA AND NEKTAR_USE_SIMD)
    MESSAGE(FATAL_ERROR "Cannot use both SIMD and CUDA")
endif()

if (NEKTAR_USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DNEKTAR_USE_CUDA)
    set(SRC ${SRC} MemoryRegionCUDA.cu Operators/BwdTrans/BwdTransCUDA.cu Operators/IProductWRTBase/IProductWRTBaseCUDA.cu Operators/PhysDeriv/PhysDerivCUDA.cu)
endif()

if (NEKTAR_USE_SIMD)
    add_compile_definitions(NEKTAR_ENABLE_SIMD_AVX512 NEKTAR_ENABLE_SIMD_AVX2 NEKTAR_ENABLE_SIMD_SSE2)
    add_compile_options("-march=native")
endif()

# add_compile_options("-fsanitize=address,undefined")
# add_link_options("-fsanitize=address,undefined")

add_compile_definitions(NEKTAR_FULLDEBUG) # Test ASSERT/WARNING macros

add_library(Operators SHARED ${SRC})
target_link_libraries(Operators PUBLIC ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES})
target_include_directories(Operators PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(Operators PUBLIC ${NEKTAR++_DEFINITIONS})

set(TEST_SRC main.cpp)
add_executable(main ${TEST_SRC})
target_link_libraries(main Operators)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(main PUBLIC ${NEKTAR++_DEFINITIONS})

add_executable(Helmholtz2D Helmholtz2D.cpp)
target_link_libraries(Helmholtz2D Operators)
target_include_directories(Helmholtz2D PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(Helmholtz2D PUBLIC ${NEKTAR++_DEFINITIONS})

add_executable(MyTest MyTest.cpp)
target_link_libraries(MyTest Operators)
target_include_directories(MyTest PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(MyTest PUBLIC ${NEKTAR++_DEFINITIONS})

add_subdirectory(tests)

find_package(Doxygen)
add_custom_target(doc COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile)
