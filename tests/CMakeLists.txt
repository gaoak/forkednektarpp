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

add_executable(test_bwdtrans_matfree test_bwdtrans_matfree.cpp)
target_link_libraries(test_bwdtrans_matfree PRIVATE Operators)
target_link_libraries(test_bwdtrans_matfree PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_bwdtrans_matfree PRIVATE Boost::unit_test_framework)
target_include_directories(test_bwdtrans_matfree PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_bwdtrans_matfree PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_bwdtrans_matfree PRIVATE
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

add_executable(test_ipwrtbase_matfree test_ipwrtbase_matfree.cpp)
target_link_libraries(test_ipwrtbase_matfree PRIVATE Operators)
target_link_libraries(test_ipwrtbase_matfree PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_ipwrtbase_matfree PRIVATE Boost::unit_test_framework)
target_include_directories(test_ipwrtbase_matfree PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_ipwrtbase_matfree PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_ipwrtbase_matfree PRIVATE
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

add_executable(test_physderiv_matfree test_physderiv_matfree.cpp)
target_link_libraries(test_physderiv_matfree PRIVATE Operators)
target_link_libraries(test_physderiv_matfree PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_physderiv_matfree PRIVATE Boost::unit_test_framework)
target_include_directories(test_physderiv_matfree PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_physderiv_matfree PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_physderiv_matfree PRIVATE
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

add_executable(test_ipwrtderivbase test_ipwrtderivbase.cpp)
target_link_libraries(test_ipwrtderivbase PRIVATE Operators)
target_link_libraries(test_ipwrtderivbase PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_ipwrtderivbase PRIVATE Boost::unit_test_framework)
target_include_directories(test_ipwrtderivbase PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_ipwrtderivbase PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_ipwrtderivbase PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_ipwrtderivbasecuda test_ipwrtderivbasecuda.cpp)
    target_link_libraries(test_ipwrtderivbasecuda PRIVATE Operators)
    target_link_libraries(test_ipwrtderivbasecuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_ipwrtderivbasecuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_ipwrtderivbasecuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_ipwrtderivbasecuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_ipwrtderivbasecuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_test(
  NAME BwdTrans
  COMMAND test_bwdtrans
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
  NAME BwdTransMatFree
  COMMAND test_bwdtrans_matfree
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

add_test(
  NAME IProductWRTBaseMatFree
  COMMAND test_ipwrtbase_matfree
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

add_test(
  NAME PhysDerivMatFree
  COMMAND test_physderiv_matfree
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME PhysDerivCUDA
      COMMAND test_physderivcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME IProductWRTDerivBase
  COMMAND test_ipwrtderivbase
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME IProductWRTDerivBaseCUDA
      COMMAND test_ipwrtderivbasecuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()
