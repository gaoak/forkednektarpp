find_package(
  Boost
  REQUIRED
  COMPONENTS unit_test_framework
)

add_executable(test_bwdtrans test_bwdtrans.cpp)
target_link_libraries(test_bwdtrans PRIVATE Operators)
target_link_libraries(test_bwdtrans PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_bwdtrans PRIVATE Boost::unit_test_framework)
target_include_directories(test_bwdtrans PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_bwdtrans PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_bwdtrans PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_bwdtranscuda test_bwdtranscuda.cpp)
    target_link_libraries(test_bwdtranscuda PRIVATE Operators)
    target_link_libraries(test_bwdtranscuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_bwdtranscuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_bwdtranscuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_bwdtranscuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_bwdtranscuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_ipwrtbase test_ipwrtbase.cpp)
target_link_libraries(test_ipwrtbase PRIVATE Operators)
target_link_libraries(test_ipwrtbase PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_ipwrtbase PRIVATE Boost::unit_test_framework)
target_include_directories(test_ipwrtbase PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_ipwrtbase PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_ipwrtbase PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_ipwrtbasecuda test_ipwrtbasecuda.cpp)
    target_link_libraries(test_ipwrtbasecuda PRIVATE Operators)
    target_link_libraries(test_ipwrtbasecuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_ipwrtbasecuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_ipwrtbasecuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_ipwrtbasecuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_ipwrtbasecuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_physderiv test_physderiv.cpp)
target_link_libraries(test_physderiv PRIVATE Operators)
target_link_libraries(test_physderiv PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_physderiv PRIVATE Boost::unit_test_framework)
target_include_directories(test_physderiv PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_physderiv PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_physderiv PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_physderivcuda test_physderivcuda.cpp)
    target_link_libraries(test_physderivcuda PRIVATE Operators)
    target_link_libraries(test_physderivcuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_physderivcuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_physderivcuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_physderivcuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_physderivcuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_test(
  NAME BwdTrans
  COMMAND test_bwdtrans
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME BwdTransCUDA
      COMMAND test_bwdtranscuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME IProductWRTBase
  COMMAND test_ipwrtbase
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME IProductWRTBaseCUDA
      COMMAND test_ipwrtbasecuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME PhysDeriv
  COMMAND test_physderiv
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME PhysDerivCUDA
      COMMAND test_physderivcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()
