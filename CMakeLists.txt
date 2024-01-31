cmake_minimum_required(VERSION 3.24)
project(Redesign CXX)
#SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
#INCLUDE (NektarCommon)

option(NEKTAR_USE_DEBUG " Enable Graphless execution" OFF)
option(NEKTAR_USE_SERIAL "Enable Serial execution" ON)
option(NEKTAR_USE_THREADS "Enable Multithreading execution" ON)
option(NEKTAR_USE_CUDA "Enable CUDA execution" ON)
OPTION(NEKTAR_MPI "Use MPI for parallelisation." ON)
option(NEKTAR_USE_SIMD "Enable SIMD support, if available" TRUE)

if (NEKTAR_USE_CUDA AND NEKTAR_USE_SIMD)
    MESSAGE(FATAL_ERROR "Cannot use both SIMD and CUDA")
endif()

OPTION(NEKTAR_BUILD_DEMOS "Build demonstration codes."     OFF)
OPTION(NEKTAR_BUILD_TESTS "Build tests."     OFF)

OPTION(NEKTAR_FULL_DEBUG "Enable Full Debugging." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

INCLUDE(CMakeDependentOption)
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "" FORCE)
ENDIF()

find_package(Nektar++ REQUIRED)
include_directories(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")
set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(NEKTAR_LIBRARY_TYPE "SHARED")

if (NEKTAR_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DNEKTAR_USE_CUDA)

    if(NEKTAR_FULL_DEBUG)
        add_definitions(-DNEKTAR_CUDA_DEBUG)
    endif()

    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_RUNTIME_LIBRARY "SHARED")
    IF(NOT DEFINED CMAKE_CUDA_STANDARD)
        SET(CMAKE_CUDA_STANDARD 20) 
        SET(CMAKE_CUDA_STANDARD_REQUIRED ON) 
    ENDIF()
    IF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        SET(CMAKE_CUDA_ARCHITECTURES native)
    ENDIF()

endif()

if (NEKTAR_USE_THREADS)
    add_definitions(-DNEKTAR_USE_THREADS)
endif()

if (NEKTAR_USE_SERIAL)
    add_definitions(-DNEKTAR_USE_SERIAL)
endif()

if (NEKTAR_USE_DEBUG)
    add_definitions(-DNEKTAR_USE_DEBUG)
endif()

if (NEKTAR_USE_SIMD)
    add_compile_definitions(NEKTAR_ENABLE_SIMD_AVX512 NEKTAR_ENABLE_SIMD_AVX2 NEKTAR_ENABLE_SIMD_SSE2)
    add_compile_options("-march=native")
endif()

if (NEKTAR_MPI)
    FIND_PACKAGE(MPI REQUIRED)
    INCLUDE_DIRECTORIES(SYSTEM ${MPI_CXX_INCLUDE_PATH} )
    MESSAGE(STATUS "Found MPI: ${MPI_CXX_LIBRARIES}")
    add_definitions(-DNEKTAR_USE_MPI)
endif()

#INCLUDE (ThirdPartyBoost)
SET(Boost_USE_STATIC_LIBS OFF)
IF( WIN32 )
    # The auto-linking feature has problems with USE_STATIC_LIBS off, so we use
    # BOOST_ALL_NO_LIB to turn it off.
    # Several boost libraries headers aren't configured correctly if
    # USE_STATIC_LIBS is off, so we explicitly say they are dynamic with the
    # remaining definitions.
    ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB -DBOOST_PROGRAM_OPTIONS_DYN_LINK -DBOOST_IOSTREAMS_DYN_LINK -DBOOST_THREAD_DYN_LINK)
ENDIF( )

add_compile_definitions(NEKTAR_FULLDEBUG) # Test ASSERT/WARNING macros

INCLUDE_DIRECTORIES(Operators)
ADD_SUBDIRECTORY(Operators)

find_package(Doxygen)
add_custom_target(doc COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile)

IF (NEKTAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_definitions(-DNEKTAR_TEST_ALL)
ENDIF (NEKTAR_BUILD_TESTS)

# Turn-on warnings
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4>>
  $<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror -Wnon-virtual-dtor -Wno-sign-compare -Wno-narrowing -Wno-conversion>>
)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/MP>>)
