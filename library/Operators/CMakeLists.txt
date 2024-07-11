CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Common/SwitchLimits.h.in
               SwitchLimits.h
               @ONLY)

# Currently two primary execution spaces.
SET(ExecSpaces Serial AVX)

# Optional CUDA execution space.
IF(NEKTAR_ENABLE_CUDA)
    SET(ExecSpaces ${ExecSpaces} CUDA)
ENDIF()

# Kokkos execution space - if no Kokkos loops are serial
SET(ExecSpaces ${ExecSpaces} Kokkos)

# Currently two primary implementation types.
SET(Implementations StdMat SumFac)

# Macro for each operator
MACRO(ADD_OPERATOR name)
    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)

    SET(OPERATOR ${name})

    # These operator have an additional argument for the FieldState
    # which requires two delcarations.
    IF("${OPERATOR}" STREQUAL "Identity" OR
        "${OPERATOR}" STREQUAL "Matrix")
      SET(CONFIG_FILE OperatorFactoryDecCoeffPhys.cpp.in)
    ELSE()
      SET(CONFIG_FILE OperatorFactoryDec.cpp.in)
    ENDIF()

    # Add the main operator header to the list of headers.
    SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${OPERTOR_HEADER})

    # These are variales used to determine the operator
    # implementation.
    UNSET(GENERIC_HEADER)
    UNSET(BASE_HEADER)
    UNSET(BASE)
    UNSET(HAS_KERNELS)

    # Generic header (*Impl.hpp) that can be used without a specific
    # implementation. It may have specific implementation kerenels.
    IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}/${name}Impl.hpp")
        SET(GENERIC_HEADER "#include \"${name}/${name}Impl.hpp\"")

        # Check for implementation specific kernels (Note the "?*" in
        # the glob which skips general kernels). If there are kernels
        # then each implementation must have a kernel header file.
        FILE(GLOB HAS_KERNELS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${name}/${name}?*Kernels.*)

    # A base header (*ImplBase.hpp) defines part but not all of the
    # implementation and as such requires a specific implementation
    # header.
    ELSEIF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}/${name}ImplBase.hpp")
        SET(BASE_HEADER "#include \"${name}/${name}ImplBase.hpp\"")
        SET(BASE Base) # Used in the Impl.cpp.in file
    ENDIF()

    # Loop through each possible execution space.
    foreach (ExecSpace IN LISTS ExecSpaces)

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
            ELSE()
                # This tag is generic - see Spaces.hpp
                SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_HOST_TAG)
            ENDIF()
        ENDIF()

        # Loop through each possible implementation.
        foreach (Impl IN LISTS Implementations)

            UNSET(SPECIFIC_HEADER)
            UNSET(KERNEL_HEADER)

            # These combinations do/should not exist.
            IF( ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "CUDA" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "Kokkos" AND "${Impl}" STREQUAL "SumFac") )
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
                ${name}/${name}${ExecSpace}${Impl}*.${HEADER_EXT})

            # If there are headers then there is a specific
            # implementation of some sort.
            IF(${CURRENT_HEADERS})
		# Look for a specific header and a kernel header.
                foreach (file ${${CURRENT_HEADERS}})
                    IF("${file}" STREQUAL "${name}/${name}${ExecSpace}${Impl}.${HEADER_EXT}")
                        SET(SPECIFIC_HEADER "#include \"${name}/${name}${ExecSpace}${Impl}.${HEADER_EXT}\"")
                    ELSEIF("${file}" STREQUAL "${name}/${name}${ExecSpace}${Impl}Kernels.${HEADER_EXT}")
                        SET(KERNEL_HEADER "#include \"${name}/${name}${ExecSpace}${Impl}Kernels.${HEADER_EXT}\"")
                    ENDIF()
                endforeach()

                IF(GENERIC_HEADER AND KERNEL_HEADER)
                    message("Adding operator with a generic implementation and kernels: " "${FactoryDeclName}")
                ELSEIF(BASE_HEADER AND SPECIFIC_HEADER AND KERNEL_HEADER)
                    message("Adding operator with a base and specific implementation and kernels: " "${FactoryDeclName}")
                    UNSET(KERNEL_HEADER) # The kernel header should be in the specific implementation.
                ELSEIF(BASE_HEADER AND SPECIFIC_HEADER)
                    message("Adding operator with a base and specific implementations: " "${FactoryDeclName}")
                ELSEIF(SPECIFIC_HEADER AND KERNEL_HEADER)
                    message("Adding operator with a specific implementation and kernels: " "${FactoryDeclName}")
                    UNSET(KERNEL_HEADER) # The kernel header should be in the specific implementation.
                ELSEIF(SPECIFIC_HEADER)
                    message("Adding operator with a specific implementation: " "${FactoryDeclName}")
                ENDIF()

                # Replace this GLOB with a declaration based on what
                # was found above.
#               FILE(GLOB ${CURRENT_SOURCES} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#                    ${name}/${name}${ExecSpace}${Impl}*.${SOURCE_EXT})

                # Create the cpp file from the implementation file.
                configure_file(${CONFIG_FILE} ${FactoryDeclName})

                # Add the respective cpp file.
                SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})

            # Found a base header so there must be a specific
            # implementation which was not found above so skip.
            ELSEIF(BASE_HEADER)

            # No headers so totally generic but those with kernels
            # would be declared above.
            ELSEIF(GENERIC_HEADER)
                # Replace this GLOB with a generic declaration (no
                # addtional headers other than the generic header).
#               FILE(GLOB ${CURRENT_SOURCES} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#                    ${name}/${name}${ExecSpace}${Impl}*.${SOURCE_EXT})

#               IF(${CURRENT_SOURCES})
#                   message("-Adding operator with a generic implementation: " "${FactoryDeclName}")
#               ENDIF()

                # If there are kernels the declaration would be above.
                IF(NOT HAS_KERNELS)
                    message("Adding operator with a generic implementation: " "${FactoryDeclName}")

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
        endforeach()
    endforeach()

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
    Field.hpp
    LoopExecution.hpp
    MemoryRegionHost.hpp
    MemoryRegionDevice.hpp
    Operator.hpp
    OperatorHelper.hpp
    Spaces.hpp
    MathKernels.hpp
    )

SET(OPERATORS_SOURCES
    Field.cpp
    LoopExecution.cpp
    Operator.cpp
    )

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories (which are operators) and add
# their source and header files.
FILE(GLOB children ${CMAKE_CURRENT_SOURCE_DIR}/*)
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child})
        get_filename_component(operator "${child}" NAME)
        ADD_OPERATOR(${operator})
    ENDIF()
ENDFOREACH()

SET(OPERATORS_DEPENDS
    MultiRegions)

if(NEKTAR_ENABLE_CUDA)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      CUDAMathKernels.cuh
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKLIB_CUDA_LIBRARIES})
endif()

# Kokkos - if no Kokkos loops are serial
if(NEKTAR_ENABLE_KOKKOS)
    SET(OPERATORS_DEPENDS
        ${OPERATORS_DEPENDS}
        ${NEKTAR_KOKKOS_DEPENDS})
endif()

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
