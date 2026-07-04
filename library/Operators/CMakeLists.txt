CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Common/SwitchLimits.h.in
               SwitchLimits.h
               @ONLY)

# Default execution spaces.
SET(ExecSpaces Serial)

# Optional AVX execution space.
IF(NEKTAR_ENABLE_SIMD)
    SET(ExecSpaces ${ExecSpaces} AVX)
ENDIF()

# Optional Device execution space.
IF(NEKTAR_ENABLE_DEVICE)
    SET(ExecSpaces ${ExecSpaces} Device)
ENDIF()

# Set implementation types.
SET(ElmtOpImpl StdMat SumFac SumFacTOP Generic)

# Shape types for seperate code compilation.
SET(Shapes Seg Tri NodalTri Quad Hex Pyr Prism NodalPrism Tet NodalTet)

# Set default data types.
IF(NEKTAR_ENABLE_DOUBLE_PRECISION)
SET(DataTypes double)
ENDIF()
IF(NEKTAR_ENABLE_SINGLE_PRECISION)
    SET(DataTypes ${DataTypes} float)
ENDIF()

# Macro for each operator.
MACRO(ADD_OPERATOR dir)

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
    If("${name}" STREQUAL "IProductWRTDerivBase")
        SET(CONFIG_FILE Common/OpFactoryDecTwoOutStates.cpp.in)
    ELSE()
        SET(CONFIG_FILE Common/OpFactoryDec.cpp.in)
    ENDIF()

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        # Set file extension.
        SET(HEADER_EXT hpp)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
            SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

        # Loop through each possible data type.
        FOREACH (TData IN LISTS DataTypes)
            # Set up the name used for the .cpp declaration file.
            SET(FactoryDeclName
	         src/${name}${ExecSpace}${TData}.${SOURCE_EXT})

            # Get the source and header files for this operator,
            # execution space, and implementation.
            SET(CURRENT_HEADERS ${name}_${ExecSpace}_HEADERS)
            SET(CURRENT_SOURCES ${name}_${ExecSpace}_SOURCES)
                
            # Get the headers matching the name and execution space
            FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                ${dir}/${name}${ExecSpace}*.hpp)

            # Found a impl header.
            IF(EXISTS "${dir}/${name}OpImpl.hpp")
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}OpImpl.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
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
    GET_FILENAME_COMPONENT(name "${dir}" NAME)

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)
    SET(OPERATOR ${name})

    # These operator have an additional argument for the FieldState
    # which requires two delcarations.
    If("${name}" STREQUAL "IProductWRTDerivBase")
        SET(CONFIG_FILE ElmtOps/ElmtBlockOpTwoOutStates.cpp.in)
    ELSE()
        SET(CONFIG_FILE ElmtOps/ElmtBlockOpFactoryDec.cpp.in)
    ENDIF()

    # Check to see if shape specific BlockOpSwitch*.h files have
    # been generated and if not generate for later potential use.
    IF(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitchCodeSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitchCode.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName include/BlockOpSwitchCode${Shape}.h)
            MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    IF(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitch1DCoordsCodeSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitch1DCoordsCode.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName include/BlockOpSwitch1DCoordsCode${Shape}.h)
            MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    IF(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitchPhysInterp1DSeg.h")
    ELSE()
        SET(ShapeIncludeName "Common/BlockOpSwitchPhysInterp1D.h.in")
        FOREACH (Shape IN LISTS Shapes)
            SET(ShapeIncName include/BlockOpSwitchPhysInterp1D${Shape}.h)
            MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDFOREACH()
    ENDIF()

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        # Set file extension.
        SET(HEADER_EXT hpp)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
            SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

        # Loop through each possible implementation.
        FOREACH (Impl IN LISTS ElmtOpImpl)

            UNSET(IMPL_HEADER)

            # These combinations do/should not exist.
            IF(("${ExecSpace}" STREQUAL "Serial" AND "${Impl}" STREQUAL "SumFacTOP") OR
               ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "SumFacTOP"))
                CONTINUE()
            ENDIF()

            # Loop through each possible data type.
            FOREACH (TData IN LISTS DataTypes)
                # Set up the name used for the .cpp declaration file.
                SET(FactoryDeclName
	        "src/Block${name}${ExecSpace}${Impl}${TData}.${SOURCE_EXT}")

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

                # Found a Serial/AVX StdMat header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXStdMat.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXStdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial/AVX SumFac header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXSumFac.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    IF(EXISTS "${dir}/${name}SumFacBlockOp.cpp.in")
                        FOREACH (TData IN LISTS DataTypes)
                            SET(ShapeConfigName "${dir}/${name}SumFacBlockOp.cpp.in")
                            FOREACH (Shape IN LISTS Shapes)
                                SET(ShapeName src/${name}${ExecSpace}SumFac${TData}${Shape}.cpp)
                                MESSAGE("Generating shape cpp file: " "${ShapeName}")
                                CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                                SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                            ENDFOREACH()
                        ENDFOREACH()
                    ENDIF()
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Serial/AVX generic header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Device StdMat header.
                ELSEIF(EXISTS "${dir}/${name}DeviceStdMat.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceStdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a Device SumFac header.
                ELSEIF(EXISTS "${dir}/${name}DeviceSumFac.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND
                    ("${Impl}" STREQUAL "SumFac" OR "${Impl}" STREQUAL "SumFacTOP"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    IF(EXISTS "${dir}/${name}SumFacBlockOp.cpp.in")
                        FOREACH (TData IN LISTS DataTypes)
                            SET(ShapeConfigName "${dir}/${name}SumFacBlockOp.cpp.in")
                            FOREACH (Shape IN LISTS Shapes)
                                SET(ShapeName src/${name}${ExecSpace}${Impl}${TData}${Shape}.${SOURCE_EXT})
                                MESSAGE("Generating shape cpp file: " "${ShapeName}")
                                CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                                SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                            ENDFOREACH()
                        ENDFOREACH()
                    ENDIF()
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # Found a generic header.
                ELSEIF(EXISTS "${dir}/${name}DeviceGeneric.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

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


# Macro for single operator.
MACRO(ADD_SINGLE_OPERATOR dir opname)

  SET(FactoryDeclName "src/${opname}${ExecSpace}${TData}.${SOURCE_EXT}")
  CONFIGURE_FILE(${dir}/${opname}FactoryDec.cpp.in
                 ${FactoryDeclName})
  SET(OPERATORS_SOURCES  ${OPERATORS_SOURCES}  ${FactoryDeclName})
  MESSAGE("Adding operator with a ${ExecSpace} implementation: "
                 "${FactoryDeclName}")
ENDMACRO()


# Macro for each operator.
MACRO(ADD_BLOCK_OPERATOR dir)

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
    SET(CONFIG_FILE_TRACE Common/BlockOperatorFactoryDec.cpp.in)


    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        # Set file extension.
        SET(HEADER_EXT hpp)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
            SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

        # Set execution space.
        IF("${ExecSpace}" STREQUAL "Serial")
            SET(EXEC_MEM_SPACE_TAG NektarSpaces::Serial)
        ELSEIF("${ExecSpace}" STREQUAL "AVX")
            SET(EXEC_MEM_SPACE_TAG NektarSpaces::AVX)
        ELSEIF("${ExecSpace}" STREQUAL "Device")
            SET(EXEC_MEM_SPACE_TAG NektarSpaces::Device)
        ENDIF()
    
        # Loop through each possible data type.
        FOREACH (TData IN LISTS DataTypes)
            # Reset for this (ExecSpace, TData)
            UNSET(IMPL_HEADER)  
            # Set up the name used for the .cpp declaration file.
            SET(FactoryDeclName "src/Block${name}${ExecSpace}${TData}.${SOURCE_EXT}")
            # Found a Serial/AVX header.
            IF((EXISTS "${dir}/${name}SerialAVX.hpp") AND
                ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX"))
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}SerialAVX.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
            # Found a Device header.
            ELSEIF(EXISTS "${dir}/${name}Device.hpp" AND ("${ExecSpace}" STREQUAL "Device"))
                FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                    "${dir}/${name}Device.hpp")
                SET(IMPL_HEADER "#include \"${include_file}\"")
                MESSAGE("Adding operator with a ${ExecSpace} implementation: " "${FactoryDeclName}")
            ENDIF()

            # If no header was found, skip this combination.
            IF(NOT DEFINED IMPL_HEADER)
                CONTINUE()
            ENDIF()

            # Create the cpp file from the implementation file.
            CONFIGURE_FILE(${CONFIG_FILE_TRACE} ${FactoryDeclName})

            # Add this specific implementation to the operator's
            # source and header files.
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
MACRO(ADD_TIME_OPERATOR dir)

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
    ("${name}" STREQUAL "Math"))
        CONTINUE()
    ENDIF()

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)
    SET(OPERATOR ${name})

    # These operator have an additional argument for the FieldState
    # which requires two delcarations.
    IF (${name} STREQUAL "IMEXdirk")
         SET(CONFIG_FILE TimeOps/IMEXdirk/IMEXdirkOpFactoryDec.cpp.in)
    ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
         SET(CONFIG_FILE TimeOps/SDC/SDCOpFactoryDec.cpp.in)
    ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
         SET(CONFIG_FILE TimeOps/GEM/GEMOpFactoryDec.cpp.in)
    ELSE() 
         SET(CONFIG_FILE TimeOps/TimeOperatorFactoryDec.cpp.in)
    ENDIF()

    # Set timeintegration order
    SET(Variants Dummy)
    IF (${name} STREQUAL "RungeKutta")
        LIST(APPEND Variants SSP)
    ELSEIF (${name} STREQUAL "DIRK")
        LIST(APPEND Variants _ES)
    ELSEIF (${name} STREQUAL "CNAB")
        LIST(APPEND Variants Modified)
    ENDIF()

    FOREACH (Variant IN LISTS Variants)
        # Set timeintegration order
        IF (${name} STREQUAL "RungeKutta") 
            IF ("${Variant}" STREQUAL "Dummy")
                SET(TimeIntegrationOrders 1 2 3 4 5)
            ELSEIF ("${Variant}" STREQUAL "SSP")
                SET(TimeIntegrationOrders 1 2 3)
            ENDIF()
        ELSEIF (${name} STREQUAL "DIRK")
            IF ("${Variant}" STREQUAL "Dummy")
                SET(TimeIntegrationOrders 1 2 3)
            ELSEIF ("${Variant}" STREQUAL "_ES")
                SET(TimeIntegrationOrders 2 3 4)
            ENDIF()
        ELSEIF (${name} STREQUAL "IMEXdirk")
            SET(TimeIntegrationImpStages 1 1 1 2 2 2 3 4)
            SET(TimeIntegrationExpStages 1 2 2 2 3 3 4 4)
            SET(TimeIntegrationOrders 1 1 2 2 2 3 3 3)
        ELSEIF (${name} STREQUAL "CNAB")
            SET(TimeIntegrationOrders 2)
        ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
            SET(TimeIntegrationOrders -1)
        ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
            SET(TimeIntegrationOrders -1)
        ELSE()
            SET(TimeIntegrationOrders 1 2 3 4)
        ENDIF()

        # Loop through each possible execution space.
        FOREACH (ExecSpace IN LISTS ExecSpaces)

            # Set file extension.
            SET(HEADER_EXT hpp)
            IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                SET(SOURCE_EXT cu)
            ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                SET(SOURCE_EXT hip)
            ELSE()
                SET(SOURCE_EXT cpp)
            ENDIF()

            IF ("${Variant}" STREQUAL "Dummy")
                SET(Variant "")
            ENDIF()

            # Loop through each possible data type.
            FOREACH (TData IN LISTS DataTypes)
                LIST(LENGTH TimeIntegrationOrders NINDEX)
                MATH(EXPR NINDEX "${NINDEX} - 1")
                FOREACH (INDEX RANGE ${NINDEX})
                    LIST(GET TimeIntegrationOrders ${INDEX} IntOrder)
                    # Set up the name used for the .cpp declaration file.
                    IF (${name} STREQUAL "IMEXdirk")
                        LIST(GET TimeIntegrationImpStages ${INDEX} ImpStage)
                        LIST(GET TimeIntegrationExpStages ${INDEX} ExpStage)
                        SET(FactoryDeclName src/${name}${Variant}${ExecSpace}${TData}${ImpStage}${ExpStage}Order${IntOrder}.${SOURCE_EXT})
                    ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
                        SET(FactoryDeclName src/${name}${ExecSpace}${TData}.${SOURCE_EXT})
                    ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
                        SET(FactoryDeclName src/${name}${ExecSpace}${TData}.${SOURCE_EXT})
                    ELSE()
                        SET(FactoryDeclName src/${name}${Variant}${ExecSpace}${TData}Order${IntOrder}.${SOURCE_EXT})
                    ENDIF()

                    # Get the source and header files for this operator,
                    # execution space, and implementation.
                    SET(CURRENT_HEADERS ${name}_${ExecSpace}_HEADERS)
                    SET(CURRENT_SOURCES ${name}_${ExecSpace}_SOURCES)

                    # Get the headers matching the name and execution space
                    FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                            ${dir}/${name}${ExecSpace}*.hpp)

                    # Found a impl header.
                    IF(EXISTS "${dir}/${name}OpImpl.hpp")
                        FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                                "${dir}/${name}OpImpl.hpp")
                        SET(IMPL_HEADER "#include \"${include_file}\"")
                        MESSAGE("Adding operator with a ${ExecSpace} Generic implementation: " "${FactoryDeclName}")
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
    Common/BlockOperator.hpp
    Common/Operator.hpp
    Common/DeviceProperties.hpp
    Common/Memory/MemoryAlloc.hpp
    Common/DataWarehouse/BasisDataWarehouse.hpp
    Common/DataWarehouse/GeometricDataWarehouse.hpp
    Common/DataWarehouse/LocalToGlobalDataWarehouse.hpp
    Common/DataWarehouse/ModeIndexDataWarehouse.hpp
    Common/DataWarehouse/StdMatDataWarehouse.hpp
    Common/DataWarehouse/TraceDataWarehouse.hpp
    Field/Block.hpp
    Field/Field.hpp
    ElmtOps/ElmtBlockOp.hpp
    Math/Math.hpp
    NekBlas/NekBlas.hpp
    NekBlas/blasHandle.hpp
    TimeOps/TimeOp.hpp
    )

# Common source file.
SET(OPERATORS_SOURCES
    Common/BlockOperator.cpp
    Common/Operator.cpp
    Common/DeviceProperties.cpp
    Common/DataWarehouse/BasisDataWarehouse.cpp
    Common/DataWarehouse/GeometricDataWarehouse.cpp
    Common/DataWarehouse/LocalToGlobalDataWarehouse.cpp
    Common/DataWarehouse/ModeIndexDataWarehouse.cpp
    Common/DataWarehouse/StdMatDataWarehouse.cpp
    Common/DataWarehouse/TraceDataWarehouse.cpp
    Field/Block.cpp
    NekBlas/blas.cpp
    NekBlas/blasHandle.cpp
    TimeOps/TimeOp.cpp
    )

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children AddTraceIntegral BndCondOps/* ElmtOps/*
     GlobalLinSysOps/LinearSystems/* GlobalLinSysOps/LinearSolvers/* 
     PreconOps/* Norm/* GetFwdBwdTracePhys 
     SolverUtilsOps/Advection/* SolverUtilsOps/Diffusion/*)

FOREACH(child ${children})
   MESSAGE("Child : " "${child}")
    ADD_OPERATOR(${child})
ENDFOREACH()

# Loop through all of the directories which contain block operators and add
# their source and header files.
FILE(GLOB children ElmtOps/*)
FOREACH(child ${children})
    ADD_ELMT_OPERATOR(${child})
ENDFOREACH()

# Loop through all of the directories which contain block operators and add
# their source and header files.
FILE(GLOB children GetFwdBwdTracePhys BndCondOps/FwdTransBC Norm/*)
FOREACH(child ${children})
    ADD_BLOCK_OPERATOR(${child})
ENDFOREACH()

# Set compilation for Riemann solvers
SET(SolverTypes Upwind)

SET(name RiemannSolver)
SET(dir ${CMAKE_CURRENT_SOURCE_DIR}/SolverUtilsOps/RiemannSolvers)

SET(NAME_HEADERS ${name}_HEADERS)
SET(NAME_SOURCES ${name}_SOURCES)
SET(OPERATOR ${name})

# These operator have an additional argument for the FieldState
# which requires two delcarations.
SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/Operators/SolverUtilsOps/RiemannSolvers/RiemannOperatorFactoryDec.cpp.in)

FOREACH (SolverType IN LISTS SolverTypes)
    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)

        # Set file extension.
        SET(HEADER_EXT hpp)
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
            SET(SOURCE_EXT hip)
        ELSE()
            SET(SOURCE_EXT cpp)
        ENDIF()

        # Loop through each possible data type.
        FOREACH (TData IN LISTS DataTypes)
            # Set up the name used for the .cpp declaration file.
            SET(FactoryDeclName src/${name}${SolverType}${ExecSpace}${TData}.${SOURCE_EXT})

            # Get the source and header files for this operator,
            # execution space, and implementation.
            SET(CURRENT_HEADERS ${name}${SolverType}_${ExecSpace}_HEADERS)
            SET(CURRENT_SOURCES ${name}${SolverType}_${ExecSpace}_SOURCES)
                
            # Get the headers matching the name and execution space
            FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                ${dir}/${name}${SolverType}${ExecSpace}*.hpp)

            FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                "${dir}/${name}OpImpl.hpp")
            SET(IMPL_HEADER "#include \"${include_file}\"")

            FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                "${dir}/${SolverType}SolverKernels.hpp")
            SET(KERNEL_HEADER "#include \"${include_file}\"")

            # Create the cpp file from the implementation file.
            CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

            # Add the respective cpp file.
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

  #MESSAGE("Sources 0 : " "${OPERATORS_SOURCES}")

# Assmbscater
#Serial double cases
SET(TData double)

SET(ExecSpace Serial)
SET(SOURCE_EXT cpp)

ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatr)
ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatrZeroDir)

#AVX double cases
IF (AVX IN_LIST ExecSpaces)
SET(ExecSpace AVX)
SET(SOURCE_EXT cpp)

ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatr)
ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatrZeroDir)
ENDIF()

#Device double cases
IF (Device IN_LIST ExecSpaces)
SET(ExecSpace Device)
IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
     SET(SOURCE_EXT cu)
ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
     SET(SOURCE_EXT hip)
ELSE()
     SET(SOURCE_EXT cpp)
ENDIF()

ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatr)
ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatrZeroDir)
ENDIF()

# Loop through TimeIntegration directory which contain operators and add
# their source and header files.
FILE(GLOB children TimeOps/*)
FOREACH(child ${children})
    ADD_TIME_OPERATOR(${child})
ENDFOREACH()
FILE(GLOB children TimeOps/SDC/*)
FOREACH(child ${children})
    ADD_TIME_OPERATOR(${child})
ENDFOREACH()
FILE(GLOB children TimeOps/GEM/*)
FOREACH(child ${children})
    ADD_TIME_OPERATOR(${child})
ENDFOREACH()

SET(OPERATORS_DEPENDS
    MultiRegions)
    
# LIBXSMM specific file
IF(NEKTAR_ENABLE_SIMD)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
       NekBlas/xsmmHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      NekBlas/xsmmHandle.cpp
    )

    IF(_SYSTEM_PROCESSOR STREQUAL "x86_64")
        SET(OPERATORS_SOURCES
          ${OPERATORS_SOURCES}
          NekBlas/xsmm.cpp
        )
    ENDIF()
ENDIF()

# MAGMA specific file
IF(NEKTAR_USE_MAGMA)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
       NekBlas/magmaHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       NekBlas/magmaHandle.cpp
       NekBlas/magma.cpp
    )
# CUBLAS specific file
ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      NekBlas/cuBlasHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      NekBlas/cuBlasHandle.cpp
      NekBlas/cuBlas.cpp
    )
# HIPBLAS specific file
ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      NekBlas/hipBlasHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      NekBlas/hipBlasHandle.cpp
      NekBlas/hipBlas.cpp
    )
# MKL/OneMath specific file
ELSEIF(NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      NekBlas/oneMathHandle.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      NekBlas/oneMathHandle.cpp
      NekBlas/syclBlas.cpp
    )
ENDIF()

# AVX specific file
IF(NEKTAR_ENABLE_SIMD)
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      NekBlas/libXSMMDispatchWrapper.hpp
    )
ENDIF()

# CUDA specific file
IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
       Common/Backends/CUDAStream.cpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       Common/Backends/CUDAStream.hpp
       Common/Memory/MemoryAlloc.cu
       Math/Math.cu
       Field/BlockHelper.cu
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_CUDA_DEPENDS})
# HIP specific file
ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
       Common/Backends/HIPStream.cpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       Common/Backends/HIPStream.hpp
       Common/Memory/MemoryAlloc.hip
       Math/Math.hip
       Field/BlockHelper.hip
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_HIP_DEPENDS})
ELSE()
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      Math/Math.cpp
      Field/BlockHelper.cpp
    )
ENDIF()

SET(OPERATORS_HEADERS
  ${OPERATORS_HEADERS}
   LoopExecution/LoopExecution.hpp
   Utils/UtilsDeviceKernelsHelper.hpp
)

SET(OPERATORS_SOURCES
  ${OPERATORS_SOURCES}
   LoopExecution/LoopExecution.cpp
   Utils/UtilsDeviceKernels.cpp
)

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

#ONEMATH
IF(NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-CUDA" OR NEKTAR_ENABLE_DEVICE STREQUAL "SYCL-HIP")
    TARGET_LINK_LIBRARIES(Operators LINK_PUBLIC ${ONEMATH_LIBRARY})
    ADD_DEPENDENCIES(Operators onemath-v0.9)
ENDIF()

#MAGMA
IF(NEKTAR_USE_MAGMA)
    TARGET_LINK_LIBRARIES(Operators LINK_PUBLIC ${MAGMA_LIBRARY})
    ADD_DEPENDENCIES(Operators magma-v2.10.0)
ENDIF()

#LIBXSMM
IF(_SYSTEM_PROCESSOR STREQUAL "x86_64" AND NEKTAR_ENABLE_SIMD)
    TARGET_LINK_LIBRARIES(Operators LINK_PRIVATE ${LIBXSMM_LIBRARY})
    ADD_DEPENDENCIES(Operators libxsmm-1.17)
ENDIF()

INSTALL(DIRECTORY ./
  DESTINATION ${NEKTAR_INCLUDE_DIR}/Operators
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")

IF(NEKTAR_BUILD_UNIT_TESTS)
    ADD_SUBDIRECTORY(SolverUtilsOps/RiemannSolvers/UnitTests)
ENDIF()
