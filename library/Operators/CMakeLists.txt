CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Common/SwitchLimits.h.in
               SwitchLimits.h
               @ONLY)

# Default execution spaces.
SET(ExecSpaces Serial)

# Optional AVX execution space.
IF(NEKTAR_ENABLE_SIMD_AVX2 OR NEKTAR_ENABLE_SIMD_AVX512)
    SET(ExecSpaces ${ExecSpaces} AVX)
ENDIF()

# Optional Device execution space.
IF(NEKTAR_ENABLE_CUDA OR NEKTAR_ENABLE_HIP OR NEKTAR_ENABLE_SYCL OR NEKTAR_ENABLE_DEVICEONHOST)
    SET(ExecSpaces ${ExecSpaces} Device)
ENDIF()

# Set implementation types.
SET(Implementations StdMat SumFac SumFacQP Generic)

# Shape types for seperate code compilation.
SET(Shapes Seg Tri NodalTri Quad Hex Pyr Prism NodalPrism Tet NodalTet)

# set default data types
SET(DataTypes double)
IF(NEKTAR_ENABLE_SINGLE_PRECISION)
    SET(DataTypes ${DataTypes} float)
ENDIF()

# Macro for each operator.
MACRO(ADD_OPERATOR dir)

    # Extract operator name form directory if it exists.
    IF(IS_DIRECTORY ${dir})
        GET_FILENAME_COMPONENT(name "${dir}" NAME)
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

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        # Set file extension.
        SET(HEADER_EXT hpp)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_CUDA)
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_HIP)
            SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

        # Set execuation space - see Spaces.hpp
        IF("${ExecSpace}" STREQUAL "Serial")
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_HOST_TAG)
        ELSE()
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_DEVICE_TAG)
        ENDIF()

        # Loop through each possible data type.
        FOREACH (TData IN LISTS DataTypes)
            # Set up the name used for the .cpp declaration file.
            SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})

            # Get the source and header files for this operator,
            # execution space, and implementation.
            SET(CURRENT_HEADERS ${name}_${ExecSpace}_${Impl}_HEADERS)
            SET(CURRENT_SOURCES ${name}_${ExecSpace}_${Impl}_SOURCES)
                
            # Get the headers matching the name, execution space, and
            # Implementation (specific and kernel).
            FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                ${dir}/${name}${ExecSpace}${Impl}*.hpp)

            # Found a impl header.
            IF(EXISTS "${dir}/${name}Impl.hpp")
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}Impl.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
                # Found a general kernel
                IF(EXISTS "${dir}/${name}ImplKernels.hpp" )
                    MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
                # Found a specific kernel
                ELSEIF(EXISTS "${dir}/${name}${ExecSpace}Kernels.hpp")
                    MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
                ELSEIF(("${ExecSpace}" STREQUAL "Serial" AND EXISTS "${dir}/${name}SerialAVXKernels.hpp") OR
                    ("${ExecSpace}" STREQUAL "AVX" AND EXISTS "${dir}/${name}SerialAVXKernels.hpp"))
                    MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
                # No kernel found
                ELSEIF(NOT EXISTS "${dir}/${name}ImplKernels.hpp" AND
                    NOT EXISTS "${dir}/${name}SerialKernels.hpp" AND
                    NOT EXISTS "${dir}/${name}AVXKernels.hpp" AND
                    NOT EXISTS "${dir}/${name}SerialAVXKernels.hpp" AND
                    NOT EXISTS "${dir}/${name}DeviceKernels.hpp")
                    MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
                ELSE()
                    CONTINUE() # This avoid creating a *.cpp file
                ENDIF()

            # Found a Serial header.
            ELSEIF((EXISTS "${dir}/${name}Serial.hpp") AND ("${ExecSpace}" STREQUAL "Serial"))
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}Serial.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")

            # Found a AVX header.
            ELSEIF((EXISTS "${dir}/${name}AVX.hpp") AND ("${ExecSpace}" STREQUAL "AVX"))
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}AVX.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")

            # Found a Device header.
            ELSEIF(EXISTS "${dir}/${name}DeviceSumFac.hpp" AND ("${ExecSpace}" STREQUAL "Device"))
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}Device.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
                # Found general kernel
                IF(EXISTS "${dir}/${name}DeviceKernels.hpp")
                    MESSAGE("Adding operator with a ${ExecSpace} implementation and a generic device kernel: " "${FactoryDeclName}")
                # No kernel found, generic implementation
                ELSE()
                    MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
                ENDIF()

            # No implementation, skip.
            ELSE()
                CONTINUE() # This avoid creating a *.cpp file
            ENDIF()

            # Create the cpp file from the implementation file.
            CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

            # Add the respective cpp file.
            SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
            SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
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

# Macro for each operator.
MACRO(ADD_ELMT_OPERATOR dir)

    # Extract operator name form directory if it exists.
    IF(IS_DIRECTORY ${dir})
        GET_FILENAME_COMPONENT(name "${dir}" NAME)
    ELSE()
        CONTINUE()
    ENDIF()

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)
    SET(OPERATOR ${name})

    # These operator have an additional argument for the FieldState
    # which requires two delcarations.
    SET(CONFIG_FILE ElmtOps/BlockOperatorFactoryDec.cpp.in)

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

    IF(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitchPhysInterp1DSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitchPhysInterp1D.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName BlockOpSwitchPhysInterp1D${Shape}.h)
            MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        # Set file extension.
        SET(HEADER_EXT hpp)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_CUDA)
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_HIP)
            SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

        # Look for file ${name}${ExexSpace}SumFacBlockOp.cpp.in file
        # when using Serial or AVX This loop will then generate a
        # shape specific cpp file containing the shape block method in
        # seperate file which contain the boost_pp generated switch
        # statements.
        IF("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
            IF(EXISTS "${dir}/${name}SerialAVXSumFacBlockOp.cpp.in")
                FOREACH (TData IN LISTS DataTypes)
                    SET(ShapeConfigName "${dir}/${name}SerialAVXSumFacBlockOp.cpp.in")
                    FOREACH (Shape IN LISTS Shapes)
                        SET(ShapeName ${name}${ExecSpace}SumFac${TData}${Shape}.cpp)
                        MESSAGE("Generating shape cpp file: " "${ShapeName}")
                        CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                        SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                        SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                    ENDFOREACH()
                ENDFOREACH()
            ENDIF()
        ELSEIF("${ExecSpace}" STREQUAL "Device")
            SET(ExecSpaceName ${ExecSpace})
            IF(EXISTS "${dir}/${name}DeviceSumFacBlockOp.cpp.in")
                FOREACH (TData IN LISTS DataTypes)
                    SET(ShapeConfigName "${dir}/${name}DeviceSumFacBlockOp.cpp.in")
                    FOREACH (Shape IN LISTS Shapes)
                        SET(ShapeName ${name}${ExecSpaceName}SumFac${TData}${Shape}.${SOURCE_EXT})
                        MESSAGE("Generating shape cpp file: " "${ShapeName}")
                        CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                        SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                        SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                    ENDFOREACH()
                ENDFOREACH()
            ENDIF()
            IF(EXISTS "${dir}/${name}DeviceSumFacQPBlockOp.cpp.in")
                FOREACH (TData IN LISTS DataTypes)
                    SET(ShapeConfigName "${dir}/${name}DeviceSumFacQPBlockOp.cpp.in")
                    FOREACH (Shape IN LISTS Shapes)
                        SET(ShapeName ${name}${ExecSpaceName}SumFacQP${TData}${Shape}.${SOURCE_EXT})
                        MESSAGE("Generating shape cpp file: " "${ShapeName}")
                        CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                        SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                        SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                    ENDFOREACH()
                ENDFOREACH()
            ENDIF()
        ENDIF()

        # Set execuation space - see Spaces.hpp
        IF("${ExecSpace}" STREQUAL "Serial")
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_HOST_TAG)
        ELSE()
            SET(EXEC_MEM_SPACE_TAG NEKTAR_DEFAULT_DEVICE_TAG)
        ENDIF()

        # Loop through each possible implementation.
        FOREACH (Impl IN LISTS Implementations)

            UNSET(IMPL_HEADER)

            # These combinations do/should not exist.
            IF(("${ExecSpace}" STREQUAL "Serial" AND "${Impl}" STREQUAL "SumFacQP") OR
               ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "StdMat") OR
               ("${ExecSpace}" STREQUAL "Device" AND "${Impl}" STREQUAL "StdMat") )
                CONTINUE()
            ENDIF()

            # Loop through each possible data type.
            FOREACH (TData IN LISTS DataTypes)
                # Set up the name used for the .cpp declaration file.
                SET(FactoryDeclName "Block${name}${ExecSpace}${Impl}${TData}.${SOURCE_EXT}")

                # Get the source and header files for this operator,
                # execution space, and implementation.
                SET(CURRENT_HEADERS ${name}_${ExecSpace}_${Impl}_HEADERS)
                SET(CURRENT_SOURCES ${name}_${ExecSpace}_${Impl}_SOURCES)
                
                # Get the headers matching the name, execution space, and
                # Implementation (specific and kernel).
                FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${dir}/${name}${ExecSpace}${Impl}*.hpp)

                # Found a StdMat header.
                IF((EXISTS "${dir}/${name}StdMat.hpp") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}StdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a SumFac header.
                ELSEIF((EXISTS "${dir}/${name}SumFac.hpp") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Generic header.
                ELSEIF((EXISTS "${dir}/${name}Generic.hpp") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}Generic.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial StdMat header.
                ELSEIF((EXISTS "${dir}/${name}SerialStdMat.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialStdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial SumFac header.
                ELSEIF((EXISTS "${dir}/${name}SerialSumFac.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial Generic header.
                ELSEIF((EXISTS "${dir}/${name}SerialGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a AVX SumFac header.
                ELSEIF((EXISTS "${dir}/${name}AVXSumFac.hpp") AND
                    ("${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}AVXSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a AVX Generic header.
                ELSEIF((EXISTS "${dir}/${name}AVXGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}AVXGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial/AVX SumFac header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXSumFac.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial/AVX generic header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Device SumFac header.
                ELSEIF(EXISTS "${dir}/${name}DeviceSumFac.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND 
                    ("${Impl}" STREQUAL "SumFac" OR "${Impl}" STREQUAL "SumFacQP"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    # Found general kernel
                    IF(EXISTS "${dir}/${name}DeviceSumFacKernels.hpp")
                        MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation and a generic device kernel: " "${FactoryDeclName}")
                    # No kernel found, generic implementation
                    ELSE()
                        MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")
                    ENDIF()

                # Found a generic header.
                ELSEIF(EXISTS "${dir}/${name}DeviceGeneric.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    # Found a general kernel
                    IF(EXISTS "${dir}/${name}DeviceKernels.hpp")
                        MESSAGE("Adding operator with a ${ExecSpace} Generic implementation and a generic device kernel: " "${FactoryDeclName}")
                    # Found specific kernel
                    ELSEIF(EXISTS "${dir}/${name}${ExecSpace}Kernels.hpp")
                        MESSAGE("Adding operator with a ${ExecSpace} Generic implementation and a specific ${ExecSpace} kernel: " "${FactoryDeclName}")
                    # No kernel found, generic implementation
                    ELSE()
                        CONTINUE() # This avoid creating a *.cpp file
                    ENDIF()

                # No implementation, skip.
                ELSE()
                    CONTINUE() # This avoid creating a *.cpp file
                ENDIF()

                # Create the cpp file from the implementation file.
                CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                # Add this specific implementation to the operator's
                # source and header files.
                SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
                SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})
            ENDFOREACH()
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
    Common/Operator.hpp
    ElmtOps/OperatorElmt.hpp
    NekBlas/NekBlas.hpp 
    )

# Common source file.
SET(OPERATORS_SOURCES
    Common/Operator.cpp
    ElmtOps/OperatorElmt.cpp
    NekBlas/blas.cpp
    )

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children ./* ElmtOps/* GlobalLinSysOps/* PreconOps/* BndCondOps/*)
FOREACH(child ${children})
    ADD_OPERATOR(${child})
ENDFOREACH()

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children ElmtOps/*)
FOREACH(child ${children})
    ADD_ELMT_OPERATOR(${child})
ENDFOREACH()

SET(OPERATORS_DEPENDS
    MultiRegions)

# CUDA specific file
IF(NEKTAR_ENABLE_CUDA)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
       NekBlas/cuBlasHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       NekBlas/cuBlasHandle.cpp
       NekBlas/cuBlas.cpp
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_CUDA_DEPENDS})
# HIP specific file
ELSEIF(NEKTAR_ENABLE_HIP)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
       NekBlas/hipBlasHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       NekBlas/hipBlasHandle.cpp
       NekBlas/hipBlas.cpp
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_HIP_DEPENDS})
# SYCL specific file
ELSEIF(NEKTAR_ENABLE_SYCL)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      Utils/SYCLQueue.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      Utils/SYCLQueue.cpp
      NekBlas/syclBlas.cpp
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_SYCL_DEPENDS})
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
