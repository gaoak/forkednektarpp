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
IF(NEKTAR_ENABLE_MAGMA)
    SET(ElmtOpImpl StdMat SumFac SumFacTOP Generic)
ELSE()
    SET(ElmtOpImpl StdMat SumFac SumFacTOP SumFacMat Generic)
ENDIF()

# Shape types for seperate code compilation.
SET(Shapes Seg Tri NodalTri Quad Hex Pyr Prism NodalPrism Tet NodalTet)

# set default data types
SET(DataTypes double)
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
    SET(CONFIG_FILE OperatorFactoryDec.cpp.in)

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
            # Set up the name used for the .cpp declaration file.
            SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})

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
        IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
            SET(SOURCE_EXT cu)
        ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
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
            IF(EXISTS "${dir}/${name}DeviceSumFacTOPBlockOp.cpp.in")
                FOREACH (TData IN LISTS DataTypes)
                    SET(ShapeConfigName "${dir}/${name}DeviceSumFacTOPBlockOp.cpp.in")
                    FOREACH (Shape IN LISTS Shapes)
                        SET(ShapeName ${name}${ExecSpaceName}SumFacTOP${TData}${Shape}.${SOURCE_EXT})
                        MESSAGE("Generating shape cpp file: " "${ShapeName}")
                        CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                        SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                        SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                    ENDFOREACH()
                ENDFOREACH()
            ENDIF()
        ENDIF()

        # Set execution space.
        IF("${ExecSpace}" STREQUAL "Serial")
            SET(EXEC_MEM_SPACE_TAG NektarSpaces::Serial)
        ELSEIF("${ExecSpace}" STREQUAL "AVX")
            SET(EXEC_MEM_SPACE_TAG NektarSpaces::AVX)
        ELSEIF("${ExecSpace}" STREQUAL "Device")
            SET(EXEC_MEM_SPACE_TAG NektarSpaces::Device)
        ENDIF()

        # Loop through each possible implementation.
        FOREACH (Impl IN LISTS ElmtOpImpl)

            UNSET(IMPL_HEADER)

            # These combinations do/should not exist.
            IF(("${ExecSpace}" STREQUAL "Serial" AND "${Impl}" STREQUAL "SumFacTOP") OR
               ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "SumFacTOP") OR
               ("${ExecSpace}" STREQUAL "Serial" AND "${Impl}" STREQUAL "SumFacMat") OR
               ("${ExecSpace}" STREQUAL "AVX" AND "${Impl}" STREQUAL "SumFacMat"))
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
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")

                # ----------------- ADDED FOR CUBLAS IMPLEMENTATION -----------------#
                # Found a CUDA SumFacCUBLAS header.
                ELSEIF((EXISTS "${dir}/${name}CUDASumFacCUBLAS.cuh") AND (NEKTAR_ENABLE_DEVICE STREQUAL "CUDA") AND
                    ("${ExecSpace}" STREQUAL "Device") AND ("${Impl}" STREQUAL "SumFacMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}CUDASumFacCUBLAS.cuh")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding operator with a ${ExecSpace} ${Impl} implementation: " "${FactoryDeclName}")
                # -------------------------------------------------------------------#

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

  SET(FactoryDeclName "${opname}${ExecSpace}${TData}.${SOURCE_EXT}")
  CONFIGURE_FILE(${dir}/${opname}FactoryDec.cpp.in
                 ${FactoryDeclName})
  SET(OPERATORS_SOURCES  ${OPERATORS_SOURCES}  ${FactoryDeclName})
  MESSAGE("Adding operator with a ${ExecSpace} implementation: "
                 "${FactoryDeclName}")
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

            # Set execution space.
            IF("${ExecSpace}" STREQUAL "Serial")
                SET(EXEC_MEM_SPACE_TAG NektarSpaces::Serial)
            ELSEIF("${ExecSpace}" STREQUAL "AVX")
                SET(EXEC_MEM_SPACE_TAG NektarSpaces::AVX)
            ELSEIF("${ExecSpace}" STREQUAL "Device")
                SET(EXEC_MEM_SPACE_TAG NektarSpaces::Device)
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
                        SET(FactoryDeclName ${name}${Variant}${ExecSpace}${TData}${ImpStage}${ExpStage}Order${IntOrder}.${SOURCE_EXT})
                    ELSEIF (${name} STREQUAL "ExplicitSDC" OR ${name} STREQUAL "ImplicitSDC" OR ${name} STREQUAL "IMEXSDC")
                        SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})
                    ELSEIF (${name} STREQUAL "ExplicitGEM" OR ${name} STREQUAL "ImplicitGEM" OR ${name} STREQUAL "IMEXGEM")
                        SET(FactoryDeclName ${name}${ExecSpace}${TData}.${SOURCE_EXT})
                    ELSE()
                        SET(FactoryDeclName ${name}${Variant}${ExecSpace}${TData}Order${IntOrder}.${SOURCE_EXT})
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
    Common/Operator.hpp
    Common/DeviceProperties.hpp
    Common/MemoryAlloc.hpp
    ElmtOps/BlockOperator.hpp
    Math/Math.hpp
    NekBlas/NekBlas.hpp
    NekBlas/blasHandle.hpp
    )

# Common source file.
SET(OPERATORS_SOURCES
    Common/Operator.cpp
    Common/DeviceProperties.cpp
    ElmtOps/BlockOperator.cpp
    NekBlas/blas.cpp
    NekBlas/blasHandle.cpp
    )

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children AddTraceIntegral  BndCondOps/*  ElmtOps/*  GlobalLinSysOps/* PreconOps/*)

FOREACH(child ${children})
   MESSAGE("Child : " "${child}")
    ADD_OPERATOR(${child})
ENDFOREACH()
  MESSAGE("Sources 0 : " "${OPERATORS_SOURCES}")

# add operators manually 

  #Serial double cases
  SET(TData double)

  SET(EXEC_MEM_SPACE_TAG NektarSpaces::Serial)
  SET(ExecSpace Serial)
  SET(SOURCE_EXT cpp)

  ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatr)
  ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatrZeroDir)

  #AVX double cases
  IF (AVX IN_LIST ExecSpaces)
    SET(EXEC_MEM_SPACE_TAG NektarSpaces::AVX)
    SET(ExecSpace AVX)
    SET(SOURCE_EXT cpp)

    ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatr)
    ADD_SINGLE_OPERATOR(AssmbScatr AssmbScatrZeroDir)
  ENDIF()

  #Device double cases
  IF (Device IN_LIST ExecSpaces)
    SET(EXEC_MEM_SPACE_TAG NektarSpaces::Device)
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

  MESSAGE("Sources 1 : " "${OPERATORS_SOURCES}")

# Loop through all of the directories which contain block operators and add
# their source and header files.
FILE(GLOB children ElmtOps/*)
FOREACH(child ${children})
    ADD_ELMT_OPERATOR(${child})
ENDFOREACH()

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
IF(NEKTAR_ENABLE_MAGMA)
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
# MKL specific file
ELSEIF(NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
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
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       Common/MemoryAlloc.cu
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_CUDA_DEPENDS})
# HIP specific file
ELSEIF(NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
       Common/MemoryAlloc.hip
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_HIP_DEPENDS})
# SYCL specific file
ELSEIF(NEKTAR_ENABLE_DEVICE MATCHES "SYCL*")
    SET(OPERATORS_HEADERS
      ${OPERATORS_HEADERS}
      Common/SYCLQueue.hpp
    )

    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      Common/SYCLQueue.cpp
    )

    SET(OPERATORS_DEPENDS
      ${OPERATORS_DEPENDS}
      ${NEKTAR_SYCL_DEPENDS})
ELSE()
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
    )
ENDIF()

# CUDA specific file
IF(NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      Math/Math.cu
    )
ELSE()
    SET(OPERATORS_SOURCES
      ${OPERATORS_SOURCES}
      Math/Math.cpp
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

#MAGMA
IF(NEKTAR_ENABLE_MAGMA)
    TARGET_LINK_LIBRARIES(Operators LINK_PUBLIC ${LIBMAGMA_LIBRARY})
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
