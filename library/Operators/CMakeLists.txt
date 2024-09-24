CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Common/SwitchLimits.h.in
               SwitchLimits.h
               @ONLY)

# Currently two primary execution spaces.
SET(ExecSpaces Serial)

# Optional AVX execution space.
IF (NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
    SET(ExecSpaces ${ExecSpaces} AVX)
ENDIF()

# Optional CUDA execution space.
IF(NEKTAR_ENABLE_CUDA)
    SET(ExecSpaces ${ExecSpaces} CUDA)
ENDIF()

# Optional SYCL execution space.
IF(NEKTAR_ENABLE_SYCL)
    SET(ExecSpaces ${ExecSpaces} SYCL)
ENDIF()

# Kokkos execution space
IF(NEKTAR_ENABLE_KOKKOS)
    SET(ExecSpaces ${ExecSpaces} Kokkos)
ENDIF()

# Currently two primary implementation types.
SET(Implementations StdMat SumFac)

# Shape types for seperate code compilation
SET(Shapes Seg Tri Quad Hex Pyr Prism Tet)

# Macro for each operator
MACRO(ADD_OPERATOR dir)

    #extract operator name form directory if it exists
    IF(IS_DIRECTORY ${dir})
        get_filename_component(name "${dir}" NAME)
        message("Name identified from dir: " "${name}")
    ELSE()
        continue()
    ENDIF()

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)

    SET(OPERATOR ${name})

    # These operator have an additional argument for the FieldState
    # which requires two delcarations.
    SET(CONFIG_FILE OperatorFactoryDec.cpp.in)

    # Add the main operator header to the list of headers.
    # Does no seem to do abything     SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${OPERTOR_HEADER})

    # These are variales used to determine the operator
    # implementation.
    UNSET(COMMON_HEADER)
    UNSET(BASE_HEADER)
    UNSET(BASE)
    UNSET(HAS_KERNELS)

    # Common header (*Impl.hpp) that can be used without a specific
    # implementation. It may have specific implementation kerenels.
    IF(EXISTS "${dir}/${name}Impl.hpp")
         FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                 ${dir}/${name}Impl.hpp)
         SET(COMMON_HEADER "#include \"${include_file}\"")

        # Check for implementation specific kernels (Note the "?*" in
        # the glob which skips general kernels). If there are kernels
        # then each implementation must have a kernel header file.
        FILE(GLOB HAS_KERNELS RELATIVE ${dir} ${name}?*Kernels.*)

    # A base header (*ImplBase.hpp) defines part but not all of the
    # implementation and as such requires a specific implementation
    # header.
    ELSEIF(EXISTS "${dir}/${name}ImplBase.hpp")
         FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                 ${dir}/${name}ImplBase.hpp)
        SET(BASE_HEADER "#include \"${include_file}\"")
        SET(BASE Base) # Used in the Impl.cpp.in file
    ENDIF()

    # Check to see if shape specific BlockOpSwitch .h files have
    # been generated and if not generate for later potential use
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitchCodeSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitchCode.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName BlockOpSwitchCode${Shape}.h)
            message("Generating block header  file file: " "${ShapeIncName}")
            configure_file(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)

        # Look for file ${name}${ExexSpace}SumFacBlockOp.cpp.in file
        # when using Serial or AVX This loop will then generate a
        # shape specific cpp file containing the shape block method in
        # seperate file which contain the boost_pp generated switch
        # statements 
        IF("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
            IF(EXISTS "${dir}/${name}${ExecSpace}SumFacBlockOp.cpp.in")
                 SET(ShapeConfigName "${dir}/${name}${ExecSpace}SumFacBlockOp.cpp.in")
                 FOREACH (Shape IN LISTS Shapes)
                     SET(ShapeName ${name}${ExecSpace}SumFac${Shape}.cpp)
                     message("Generating shape cpp file: " "${ShapeName}")
                     configure_file(${ShapeConfigName} ${ShapeName})
                     SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                     SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                 ENDFOREACH()
             ENDIF()
        ENDIF()

        # CUDA has cu and cuh extentions
        IF("${ExecSpace}" STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
            SET(HEADER_EXT cuh)
            # This tag is generic - see Spaces.hpp
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_DEVICE_TAG)
        ELSE()
            SET(SOURCE_EXT cpp)
            SET(HEADER_EXT hpp)

            IF("${ExecSpace}" STREQUAL "Kokkos")
                # This tag is generic - see Spaces.hpp
                SET(EXEC_MEM_SPACE_TAG KOKKOS_DEFAULT_DEVICE_TAG)
            ELSEIF("${ExecSpace}" STREQUAL "AVX")
                SET(EXEC_MEM_SPACE_TAG "NektarSpaces::AVX" )
            ELSEIF("${ExecSpace}" STREQUAL "SYCL")
                SET(EXEC_MEM_SPACE_TAG "NektarSpaces::SYCL" )
            ELSE()
                # This tag is generic - see Spaces.hpp
                SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_HOST_TAG)
            ENDIF()
        ENDIF()

        # Look for a generic implementation header
        UNSET(GENERIC_HEADER)

        IF(EXISTS "${dir}/${name}${ExecSpace}Generic.${HEADER_EXT}")
            FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}${ExecSpace}Generic.${HEADER_EXT})
             SET(GENERIC_HEADER "#include \"${include_file}\"")
        ENDIF()

        # Loop through each possible implementation.
        FOREACH (Impl IN LISTS Implementations)

            UNSET(SPECIFIC_HEADER)
            UNSET(KERNEL_HEADER)

            # These combinations do/should not exist.
            IF( ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "CUDA" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "SYCL" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "Kokkos" AND "${Impl}" STREQUAL "StdMat") )
                continue()
            ENDIF()

	    # Set up the name used for the .cpp declaration file.
            SET(FactoryDeclName ${name}${ExecSpace}${Impl}.${SOURCE_EXT})

            # Get the source and header files for this operator,
            # execution space, and implementation.
            SET(CURRENT_HEADERS ${name}_${ExecSpace}_${Impl}_HEADERS)
            SET(CURRENT_SOURCES ${name}_${ExecSpace}_${Impl}_SOURCES)

            # Get the headers matching the name, execution space, and
            # Implementation (specific and kernel).
            FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                 ${dir}/${name}${ExecSpace}${Impl}*.${HEADER_EXT})

            # If there are headers then there is a specific
            # implementation of some sort.
            IF(${CURRENT_HEADERS})
		# Look for a specific header and a kernel header.
                FOREACH (header_file ${${CURRENT_HEADERS}})
                    get_filename_component(file "${header_file}" NAME)
                    get_filename_component(file_dir "${header_file}" DIRECTORY)
                    IF("${file}" STREQUAL "${name}${ExecSpace}${Impl}.${HEADER_EXT}")
                        SET(SPECIFIC_HEADER "#include \"${file_dir}/${name}${ExecSpace}${Impl}.${HEADER_EXT}\"")
                    ELSEIF("${file}" STREQUAL "${name}${ExecSpace}${Impl}Kernels.${HEADER_EXT}")
                        SET(KERNEL_HEADER "#include \"${file_dir}/${name}${ExecSpace}${Impl}Kernels.${HEADER_EXT}\"")
                    ENDIF()
                ENDFOREACH()

                IF(COMMON_HEADER AND KERNEL_HEADER)
                    message("Add Op with a common impl. and kernels: " "${FactoryDeclName}")
                ELSEIF(BASE_HEADER AND SPECIFIC_HEADER AND KERNEL_HEADER)
                    message("Add OP with a base and specific impl. and kernels: " "${FactoryDeclName}")
                    UNSET(KERNEL_HEADER) # The kernel header should be in the specific implementation.
                ELSEIF(GENERIC_HEADER)
                    message("Add OP with a generic impl.: " "${FactoryDeclName}")
                ELSEIF(BASE_HEADER AND SPECIFIC_HEADER)
                    message("Add Op with a base and specific impl.: " "${FactoryDeclName}")
                ELSEIF(SPECIFIC_HEADER AND KERNEL_HEADER)
                    message("Add Op with a specific impl. and kernels: " "${FactoryDeclName}")
                    UNSET(KERNEL_HEADER) # The kernel header should be in the specific implementation.
                ELSEIF(SPECIFIC_HEADER)
                    message("Add Op with a specific impl.: " "${FactoryDeclName}")
                ENDIF()

                # Create the cpp file from the implementation file.
                configure_file(${CONFIG_FILE} ${FactoryDeclName})

                # Add the respective cpp file.
                SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})

            # Found a generic header
            ELSEIF(GENERIC_HEADER)
                message("Adding operator with a generic implementation: " "${FactoryDeclName}")

                # Create the cpp file from the implementation file.
                configure_file(${CONFIG_FILE} ${FactoryDeclName})

                # Add the respective cpp file.
                SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})

            # Found a base header so there must be a specific
            # implementation which was not found above so skip.
            ELSEIF(BASE_HEADER)

            # No headers so totally common but those with kernels
            # would be declared above.
            ELSEIF(COMMON_HEADER)
                # If there are kernels the declaration would be above.
                IF(NOT HAS_KERNELS)
                    message("Add Op using kernel: " "${FactoryDeclName}")

                    # Create the cpp file from the implementation file.
                    configure_file(${CONFIG_FILE} ${FactoryDeclName})

                    # Add the respective cpp file.
                    SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                ENDIF()
            ENDIF()

            # Add this specific implementation to the operator's
            # source and header files.
            SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
            SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})
        ENDFOREACH()
    ENDFOREACH()

    # Add this operator's implemenations to the global operator source
    # and header files.
    SET(OPERATORS_HEADERS
        ${OPERATORS_HEADERS}
        ${${NAME_HEADERS}}
        )
    SET(OPERATORS_SOURCES
        ${OPERATORS_SOURCES}
        ${${NAME_SOURCES}}
        )
ENDMACRO()


SET(OPERATORS_HEADERS
    Field/Field.hpp
    Field/MemoryRegionHost.hpp
    Field/MemoryRegionDevice.hpp
    LoopExecution/LoopExecution.hpp
    Common/Operator.hpp
    Common/OperatorHelper.hpp
    Common/Spaces.hpp
    MathKernels/MathKernels.hpp
    )

SET(OPERATORS_SOURCES
    Field/Field.cpp
    Common/Operator.cpp
    )

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children ./*  ElmtOps/* GlobalLinSysOps/* PreconOps/* BndCondOps/*)
FOREACH(child ${children})
    ADD_OPERATOR(${child})
ENDFOREACH()

SET(OPERATORS_DEPENDS
    MultiRegions)

IF(NEKTAR_ENABLE_CUDA)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      Field/UtilsCUDALaunchers.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      Field/UtilsCUDAKernels.cu
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKLIB_CUDA_LIBRARIES})
ENDIF()

IF(NEKTAR_ENABLE_SYCL)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      SYCLQueue.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      SYCLQueue.cpp
    )
ENDIF()

# Kokkos - if no Kokkos loops are serial
IF(NEKTAR_ENABLE_KOKKOS)
    SET(OPERATORS_DEPENDS
        ${OPERATORS_DEPENDS}
        ${NEKTAR_KOKKOS_DEPENDS})
ENDIF()

SOURCE_GROUP(TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${OPERATORS_HEADERS} ${GROUP_SOURCES})

ADD_DEFINITIONS(-DOPERATORS_EXPORTS -I${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

ADD_NEKTAR_LIBRARY(Operators
    SOURCES ${OPERATORS_SOURCES}
    HEADERS ${OPERATORS_HEADERS}
    DEPENDS ${OPERATORS_DEPENDS}
    SUMMARY "Nektar++ Operators library"
    DESCRIPTION "This library provides classes related to operators for Nektar++.")

INSTALL(DIRECTORY ./
  DESTINATION ${NEKTAR_INCLUDE_DIR}/Operators
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")
