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
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    # CMake 3.24
    # set(CMAKE_CUDA_ARCHITECTURES native)
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
    Operators/OperatorMass.cpp Operators/Mass/MassImpl.cpp
    Operators/OperatorIdentity.cpp Operators/Identity/IdentityImpl.cpp
    Operators/OperatorConjGrad.cpp Operators/ConjGrad/ConjGradImpl.cpp
    Operators/OperatorFwdTrans.cpp Operators/FwdTrans/FwdTransImpl.cpp
    Operators/OperatorAssmbScatr.cpp Operators/AssmbScatr/AssmbScatrImpl.cpp
    Operators/OperatorDirBndCond.cpp Operators/DirBndCond/DirBndCondImpl.cpp
    Operators/OperatorPhysDeriv.cpp Operators/PhysDeriv/PhysDerivImpl.cpp
    Operators/OperatorHelmSolve.cpp Operators/HelmSolve/HelmSolveImpl.cpp
    Operators/OperatorNullPrecon.cpp Operators/NullPrecon/NullPreconImpl.cpp
    Operators/OperatorDiagPrecon.cpp Operators/DiagPrecon/DiagPreconImpl.cpp
    Operators/OperatorMatrix.cpp Operators/Matrix/MatrixImpl.cpp
    Operators/OperatorNeuBndCond.cpp Operators/NeuBndCond/NeuBndCondImpl.cpp
    Operators/OperatorRobBndCond.cpp Operators/RobBndCond/RobBndCondImpl.cpp
    Operators/OperatorHelmholtz.cpp Operators/Helmholtz/HelmholtzImpl.cpp
    Operators/OperatorIProductWRTDerivBase.cpp Operators/IProductWRTDerivBase/IProductWRTDerivBaseImpl.cpp
    )

if (NEKTAR_USE_CUDA AND NEKTAR_USE_SIMD)
    MESSAGE(FATAL_ERROR "Cannot use both SIMD and CUDA")
endif()

if (NEKTAR_USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DNEKTAR_USE_CUDA)
    set(SRC ${SRC} MemoryRegionCUDA.cu Operators/BwdTrans/BwdTransCUDA.cu 
                   Operators/IProductWRTBase/IProductWRTBaseCUDA.cu 
                   Operators/IProductWRTDerivBase/IProductWRTDerivBaseCUDA.cu 
                   Operators/PhysDeriv/PhysDerivCUDA.cu
                   Operators/Identity/IdentityCUDA.cu
                   Operators/Matrix/MatrixCUDA.cu 
                   Operators/Mass/MassCUDA.cu
                   Operators/AssmbScatr/AssmbScatrCUDA.cu 
                   Operators/Helmholtz/HelmholtzCUDA.cu)
endif()

if (NEKTAR_USE_SIMD)
    add_compile_definitions(NEKTAR_ENABLE_SIMD_AVX512 NEKTAR_ENABLE_SIMD_AVX2 NEKTAR_ENABLE_SIMD_SSE2)
    add_compile_options("-march=native")
endif()

SET(OPERATORS_HEADERS
    Operators/Operators.hpp
)

# # Boilerplate cpp implementation files for registering an operator
# # with the factory. There are individual files for each operator shape
# # so to reduce compile time timeout failures due to the large number
# # of template impementations.

# SET(OPERATORS BwdTrans)
# SET(SHAPES Seg Tri Quad Hex Tet Prism Pyr )
# SET(DIMENSIONS 1 2 2 3 3 3 3 )

# # Note the commented out code is future simpler code for when cmake 3.17 can
# # be used on the tester.

# foreach(OPERATOR IN LISTS OPERATORS)
# #  if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
# #    Remove the code in this part of the conditional once CMake 3.17
# #    can be used on the CI tester.
#      foreach(SHAPE IN LISTS SHAPES)
#         IF( "${SHAPE}" STREQUAL "Seg" )
# 	    SET(DIMENSION 1)
#         ELSEIF( "${SHAPE}" STREQUAL "Tri" OR "${SHAPE}" STREQUAL "Quad" )
# 	    SET(DIMENSION 2)
# 	ELSE()
# 	    SET(DIMENSION 3)
#         ENDIF()
# #  else()
# #     Use this code once CMake 3.17 can be used on the CI tester.
# #     foreach(SHAPE DIMENSION IN ZIP_LISTS SHAPES DIMENSIONS)
# #  endif()

# 	# Uppercase is used for the #defines in the header files.
# 	string(TOUPPER ${SHAPE} TYPE)	

#         # INST sets the macro used in NektarExpMacros.h
#         # The BwdTrans only has regular elements.
#         IF( "${OPERATOR}" STREQUAL "BwdTrans" )
#             SET(INST 1)
#         # All other operators have regular and deformed elements.
#         ELSE()
#             SET(INST 2)
#         ENDIF()

#         # Create the cpp file from the implementation file.
#         configure_file(Operators/OperatorImp.cpp.in ${OPERATOR}${SHAPE}.cpp)

#         # Add the respective headers and cpp file.
#         SET(OPERATORS_HEADERS ${OPERATORS_HEADERS}
#         ${OPERATOR}/${OPERATOR}MatFree.hpp)
#         SET(OPERATORS_HEADERS ${OPERATORS_HEADERS}
#         ${OPERATOR}/${OPERATOR}MatFreeKernels.hpp)
#         SET(SRC ${SRC} ${OPERATORS_SOURCES}
#             ${CMAKE_CURRENT_BINARY_DIR}/${OPERATOR}${SHAPE}.cpp)

#     endforeach()
# endforeach()

# add_compile_options("-fsanitize=address,undefined")
# add_link_options("-fsanitize=address,undefined")

add_compile_definitions(NEKTAR_FULLDEBUG) # Test ASSERT/WARNING macros

add_library(Operators SHARED ${SRC})
target_link_libraries(Operators PUBLIC ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES})
target_include_directories(Operators PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(Operators PUBLIC ${NEKTAR++_DEFINITIONS})

add_subdirectory(tests)

find_package(Doxygen)
add_custom_target(doc COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile)
