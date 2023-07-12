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

add_executable(test_ipwrtbase test_ipwrtbase.cpp)
target_link_libraries(test_ipwrtbase PRIVATE Operators)
target_link_libraries(test_ipwrtbase PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_ipwrtbase PRIVATE Boost::unit_test_framework)
target_include_directories(test_ipwrtbase PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_ipwrtbase PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_ipwrtbase PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

add_test(
  NAME BwdTrans
  COMMAND test_bwdtrans
  # Test executable is looking for input session file in the working
  # directory
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
  NAME IProductWRTBase
  COMMAND test_ipwrtbase
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
