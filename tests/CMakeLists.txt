set(test_src test_bwdtrans.cpp)

find_package(
  Boost
  REQUIRED
  COMPONENTS unit_test_framework
)

add_executable(test_bwdtrans ${test_src})
target_link_libraries(test_bwdtrans PRIVATE Operators)
target_link_libraries(test_bwdtrans PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_bwdtrans PRIVATE Boost::unit_test_framework)

target_include_directories(test_bwdtrans PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_bwdtrans PRIVATE ${NEKTAR++_INCLUDE_DIRS})

add_test(
  NAME BwdTrans
  COMMAND test_bwdtrans
  # Test executable is looking for input session file in the working
  # directory
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/input"
)
