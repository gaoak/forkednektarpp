CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Common/SwitchLimits.h.in
               SwitchLimits.h
               @ONLY)

# Default execution spaces.
SET(ExecSpaces Serial)

# Optional AVX execution space.
IF(NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
    SET(ExecSpaces ${ExecSpaces} AVX)
ENDIF()

# Optional CUDA execution space.
IF(NEKTAR_ENABLE_CUDA)
    SET(ExecSpaces ${ExecSpaces} CUDA)
ENDIF()

# Optional Kokkos execution space.
IF(NEKTAR_ENABLE_KOKKOS)
    SET(ExecSpaces ${ExecSpaces} Kokkos)
ENDIF()

# Optional SYCL execution space.
IF(NEKTAR_ENABLE_SYCL)
    SET(ExecSpaces ${ExecSpaces} SYCL)
ENDIF()

# Set implementation types.
SET(Implementations StdMat SumFac SumFacQP)

# Shape types for seperate code compilation.
SET(Shapes Seg Tri Quad Hex Pyr Prism Tet)

# Macro for each operator.
MACRO(ADD_OPERATOR dir)

    # Extract operator name form directory if it exists.
    IF(IS_DIRECTORY ${dir})
        GET_FILENAME_COMPONENT(name "${dir}" NAME)
        MESSAGE("Name identified from dir: " "${name}")
    ELSE()
        CONTINUE()
    ENDIF()

    # Skip those files.
    IF( ("${name}" STREQUAL "Common") OR
        ("${name}" STREQUAL "Field") OR
        ("${name}" STREQUAL "LoopExecution") OR
        ("${name}" STREQUAL "MathKernels"))
        CONTINUE()
    ENDIF()

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)
    SET(OPERATOR ${name})

    # These operator have an additional argument for the FieldState
    # which requires two delcarations.
    SET(CONFIG_FILE OperatorFactoryDec.cpp.in)

    # These are variales used to determine the operator.
    # implementation.
    UNSET(COMMON_HEADER)

    # Common header (*Impl.hpp) that can be used without a specific
    # implementation. It may have specific implementation kerenels.
    IF(EXISTS "${dir}/${name}Impl.hpp")
         FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                 ${dir}/${name}Impl.hpp)
         SET(COMMON_HEADER "#include \"${include_file}\"")
    ENDIF()

    # Check to see if shape specific BlockOpSwitch*.h files have
    # been generated and if not generate for later potential use.
    IF(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitchCodeSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitchCode.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName BlockOpSwitchCode${Shape}.h)
            MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    IF(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitch1DCoordsCodeSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitch1DCoordsCode.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName BlockOpSwitch1DCoordsCode${Shape}.h)
            MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        # Look for file ${name}${ExexSpace}SumFacBlockOp.cpp.in file
        # when using Serial or AVX This loop will then generate a
        # shape specific cpp file containing the shape block method in
        # seperate file which contain the boost_pp generated switch
        # statements.
        IF("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
            IF(EXISTS "${dir}/${name}SerialAVXSumFacBlockOp.cpp.in")
                SET(ShapeConfigName "${dir}/${name}SerialAVXSumFacBlockOp.cpp.in")
                FOREACH (Shape IN LISTS Shapes)
                   SET(ShapeName ${name}${ExecSpace}SumFac${Shape}.cpp)
                   MESSAGE("Generating shape cpp file: " "${ShapeName}")
                   CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                   SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                   SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                ENDFOREACH()
            ENDIF()
        ENDIF()

        # Look for a generic implementation header.
        UNSET(GENERIC_HEADER)

        IF(EXISTS "${dir}/${name}${ExecSpace}Generic.${HEADER_EXT}")
            FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}${ExecSpace}Generic.${HEADER_EXT})
            SET(GENERIC_HEADER "#include \"${include_file}\"")
        ENDIF()

        # Set file extension.
        IF("${ExecSpace}" STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
            SET(HEADER_EXT cuh)
        ELSE()
            SET(SOURCE_EXT cpp)
            SET(HEADER_EXT hpp)
        ENDIF()

        # Set execuation space - see Spaces.hpp
        IF("${ExecSpace}" STREQUAL "Serial")
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_HOST_TAG)
        ELSE()
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_DEVICE_TAG)
        ENDIF()

        # Loop through each possible implementation.
        FOREACH (Impl IN LISTS Implementations)

            UNSET(SERIAL_STDMAT_HEADER)
            UNSET(SERIAL_AVX_SUMFAC_HEADER)
            UNSET(DEVICE_SUMFAC_HEADER)

            # These combinations do/should not exist.
            IF( ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "CUDA" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "Kokkos" AND "${Impl}" STREQUAL "StdMat") OR
                ("${ExecSpace}" STREQUAL "SYCL" AND "${Impl}" STREQUAL "StdMat") )
                CONTINUE()
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

            # Found a Serial StdMat header.
            IF(EXISTS "${dir}/${name}SerialStdMat.hpp" AND
                ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "StdMat"))

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}SerialStdMat.hpp)
                SET(SERIAL_STDMAT_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a Serial StdMat implementation: " "${FactoryDeclName}")

            # Found a Serial SumFac header.
            ELSEIF((EXISTS "${dir}/${name}SerialSumFac.hpp") AND
                ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "SumFac"))

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}SerialSumFac.hpp)
                SET(SERIAL_AVX_SUMFAC_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} SumFac implementation: " "${FactoryDeclName}")

            # Found a AVX SumFac header.
            ELSEIF((EXISTS "${dir}/${name}AVXSumFac.hpp") AND
                ("${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "SumFac"))

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}AVXSumFac.hpp)
                SET(SERIAL_AVX_SUMFAC_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} SumFac implementation: " "${FactoryDeclName}")

            # Found a Serial/AVX SumFac header.
            ELSEIF((EXISTS "${dir}/${name}SerialAVXSumFac.hpp") AND
                ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "SumFac"))

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}SerialAVXSumFac.hpp)
                SET(SERIAL_AVX_SUMFAC_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} SumFac implementation: " "${FactoryDeclName}")

            # Found a Device SumFac header.
            ELSEIF((EXISTS "${dir}/${name}DeviceSumFac.hpp") AND
                   (("${ExecSpace}" STREQUAL "Device") OR # Generic device implementation
                    ("${ExecSpace}" STREQUAL "CUDA") OR 
                    ("${ExecSpace}" STREQUAL "Kokkos") OR 
                    ("${ExecSpace}" STREQUAL "SYCL")) AND 
                   (("${Impl}" STREQUAL "SumFac") OR ("${Impl}" STREQUAL "SumFacQP")))

                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}DeviceSumFac.hpp)
                SET(DEVICE_SUMFAC_HEADER "#include \"${include_file}\"")
                IF (("${ExecSpace}" STREQUAL "Device" AND EXISTS "${dir}/${name}DeviceSumFacKernels.hpp") OR
                    ("${ExecSpace}" STREQUAL "CUDA" AND EXISTS "${dir}/${name}CUDASumFacKernels.cuh") OR
                    ("${ExecSpace}" STREQUAL "Kokkos" AND EXISTS "${dir}/${name}KokkosSumFacKernels.hpp") OR
                    ("${ExecSpace}" STREQUAL "SYCL" AND EXISTS "${dir}/${name}SYCLSumFacKernels.hpp"))
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")
                ELSE()
                    CONTINUE() # This avoid creating a *.cpp file
                ENDIF()

            # Found a generic header.
            ELSEIF(GENERIC_HEADER)
                MESSAGE("Adding operator with a generic implementation: " "${FactoryDeclName}")

            # Found a common header.
            ELSEIF(COMMON_HEADER)
                MESSAGE("Adding operator with a common implementation: " "${FactoryDeclName}")

            # No implementation, skip.
            ELSE()
                CONTINUE() # This avoid creating a *.cpp file
            ENDIF()

            # Create the cpp file from the implementation file.
            CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

            # Add the respective cpp file.
            SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})

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

# Common header file.
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

# Common source file.
SET(OPERATORS_SOURCES
    Field/Field.cpp
    Common/Operator.cpp
    Common/OperatorHelper.cpp
    )

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children ./* ElmtOps/* GlobalLinSysOps/* PreconOps/* BndCondOps/*)
FOREACH(child ${children})
    ADD_OPERATOR(${child})
ENDFOREACH()

SET(OPERATORS_DEPENDS
    MultiRegions)

# CUDA specific file
IF(NEKTAR_ENABLE_CUDA)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKLIB_CUDA_LIBRARIES})
ENDIF()

# Kokkos specific file
IF(NEKTAR_ENABLE_KOKKOS)
    SET(OPERATORS_DEPENDS
        ${OPERATORS_DEPENDS}
        ${NEKTAR_KOKKOS_DEPENDS})
ENDIF()

# SYCL specific file
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
