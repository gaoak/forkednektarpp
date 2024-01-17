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

add_executable(test_ipwrtbasesumfac test_ipwrtbasesumfac.cpp)
target_link_libraries(test_ipwrtbasesumfac PRIVATE Operators)
target_link_libraries(test_ipwrtbasesumfac PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_ipwrtbasesumfac PRIVATE Boost::unit_test_framework)
target_include_directories(test_ipwrtbasesumfac PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_ipwrtbasesumfac PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_ipwrtbasesumfac PRIVATE
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

add_executable(test_identitycoeff test_identitycoeff.cpp)
target_link_libraries(test_identitycoeff PRIVATE Operators)
target_link_libraries(test_identitycoeff PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_identitycoeff PRIVATE Boost::unit_test_framework)
target_include_directories(test_identitycoeff PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_identitycoeff PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_identitycoeff PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

add_executable(test_identityphys test_identityphys.cpp)
target_link_libraries(test_identityphys PRIVATE Operators)
target_link_libraries(test_identityphys PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_identityphys PRIVATE Boost::unit_test_framework)
target_include_directories(test_identityphys PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_identityphys PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_identityphys PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_identitycoeffcuda test_identitycoeffcuda.cpp)
    target_link_libraries(test_identitycoeffcuda PRIVATE Operators)
    target_link_libraries(test_identitycoeffcuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_identitycoeffcuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_identitycoeffcuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_identitycoeffcuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_identitycoeffcuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")

    add_executable(test_identityphyscuda test_identityphyscuda.cpp)
    target_link_libraries(test_identityphyscuda PRIVATE Operators)
    target_link_libraries(test_identityphyscuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_identityphyscuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_identityphyscuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_identityphyscuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_identityphyscuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_mass test_mass.cpp)
target_link_libraries(test_mass PRIVATE Operators)
target_link_libraries(test_mass PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_mass PRIVATE Boost::unit_test_framework)
target_include_directories(test_mass PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_mass PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_mass PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

add_executable(test_masscuda test_masscuda.cpp)
target_link_libraries(test_masscuda PRIVATE Operators)
target_link_libraries(test_masscuda PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_masscuda PRIVATE Boost::unit_test_framework)
target_include_directories(test_masscuda PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_masscuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_masscuda PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

add_executable(test_assmbscatr test_assmbscatr.cpp)
target_link_libraries(test_assmbscatr PRIVATE Operators)
target_link_libraries(test_assmbscatr PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_assmbscatr PRIVATE Boost::unit_test_framework)
target_include_directories(test_assmbscatr PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_assmbscatr PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_assmbscatr PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_assmbscatrcuda test_assmbscatrcuda.cpp)
    target_link_libraries(test_assmbscatrcuda PRIVATE Operators)
    target_link_libraries(test_assmbscatrcuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_assmbscatrcuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_assmbscatrcuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_assmbscatrcuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_assmbscatrcuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_dirichlet test_dirichlet.cpp)
target_link_libraries(test_dirichlet PRIVATE Operators)
target_link_libraries(test_dirichlet PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_dirichlet PRIVATE Boost::unit_test_framework)
target_include_directories(test_dirichlet PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_dirichlet PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_dirichlet PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_dirichletcuda test_dirichletcuda.cpp)
    target_link_libraries(test_dirichletcuda PRIVATE Operators)
    target_link_libraries(test_dirichletcuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_dirichletcuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_dirichletcuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_dirichletcuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_dirichletcuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_nullprecon test_nullprecon.cpp)
target_link_libraries(test_nullprecon PRIVATE Operators)
target_link_libraries(test_nullprecon PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_nullprecon PRIVATE Boost::unit_test_framework)
target_include_directories(test_nullprecon PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_nullprecon PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_nullprecon PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_nullpreconcuda test_nullpreconcuda.cpp)
    target_link_libraries(test_nullpreconcuda PRIVATE Operators)
    target_link_libraries(test_nullpreconcuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_nullpreconcuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_nullpreconcuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_nullpreconcuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_nullpreconcuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_diagprecon test_diagprecon.cpp)
target_link_libraries(test_diagprecon PRIVATE Operators)
target_link_libraries(test_diagprecon PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_diagprecon PRIVATE Boost::unit_test_framework)
target_include_directories(test_diagprecon PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_diagprecon PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_diagprecon PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

add_executable(test_fwdtrans test_fwdtrans.cpp)
target_link_libraries(test_fwdtrans PRIVATE Operators)
target_link_libraries(test_fwdtrans PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_fwdtrans PRIVATE Boost::unit_test_framework)
target_include_directories(test_fwdtrans PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_fwdtrans PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_fwdtrans PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

add_executable(test_helmholtz test_helmholtz.cpp)
target_link_libraries(test_helmholtz PRIVATE Operators)
target_link_libraries(test_helmholtz PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_helmholtz PRIVATE Boost::unit_test_framework)
target_include_directories(test_helmholtz PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_helmholtz PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_helmholtz PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

IF (NEKTAR_USE_CUDA)
    add_executable(test_helmholtzcuda test_helmholtzcuda.cpp)
    target_link_libraries(test_helmholtzcuda PRIVATE Operators)
    target_link_libraries(test_helmholtzcuda PRIVATE ${NEKTAR++_LIBRARIES})
    target_link_libraries(test_helmholtzcuda PRIVATE Boost::unit_test_framework)
    target_include_directories(test_helmholtzcuda PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(test_helmholtzcuda PRIVATE ${NEKTAR++_INCLUDE_DIRS})
    target_compile_definitions(test_helmholtzcuda PRIVATE
        -DTEST_PATH="${CMAKE_SOURCE_DIR}")
ENDIF()

add_executable(test_helmsolve test_helmsolve.cpp)
target_link_libraries(test_helmsolve PRIVATE Operators)
target_link_libraries(test_helmsolve PRIVATE ${NEKTAR++_LIBRARIES})
target_link_libraries(test_helmsolve PRIVATE Boost::unit_test_framework)
target_include_directories(test_helmsolve PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(test_helmsolve PRIVATE ${NEKTAR++_INCLUDE_DIRS})
target_compile_definitions(test_helmsolve PRIVATE
    -DTEST_PATH="${CMAKE_SOURCE_DIR}")

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
  NAME IProductWRTBaseSumFac
  COMMAND test_ipwrtbasesumfac
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

add_test(
  NAME IdentityCoeff
  COMMAND test_identitycoeff
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
  NAME IdentityPhys
  COMMAND test_identityphys
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME IdentityCoeffCUDA
      COMMAND test_identitycoeffcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )

    add_test(
      NAME IdentityPhysCUDA
      COMMAND test_identityphyscuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME Mass
  COMMAND test_mass
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME MassCUDA
      COMMAND test_masscuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME AssmbScatr
  COMMAND test_assmbscatr
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME AssmbScatrCUDA
      COMMAND test_assmbscatrcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME Dirichlet
  COMMAND test_dirichlet
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME DirichletCUDA
      COMMAND test_dirichletcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME NullPrecon
  COMMAND test_nullprecon
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME NullPreconCUDA
      COMMAND test_nullpreconcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()

add_test(
  NAME DiagPrecon
  COMMAND test_diagprecon
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
  NAME FwdTrans
  COMMAND test_fwdtrans
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
  NAME Helmholtz
  COMMAND test_helmholtz
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
  NAME HelmSolve
  COMMAND test_helmsolve
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

IF (NEKTAR_USE_CUDA)
    add_test(
      NAME HelmholtzCUDA
      COMMAND test_helmholtzcuda
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
ENDIF()
