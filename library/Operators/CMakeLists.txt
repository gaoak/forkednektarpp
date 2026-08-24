# Set implementation types.
SET(ElmtOpImpl StdMat SumFac SumFacTOP Generic)

# Shape types for seperate code compilation.
SET(Shapes Seg Tri NodalTri Quad Hex Pyr Prism NodalPrism Tet NodalTet)

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

    SET(SWITCH_DIR ${CMAKE_SOURCE_DIR}/library/LibUtilities/BasicUtils/Switch)

    # Check to see if shape specific BlockOpSwitch*.h files have
    # been generated and if not generate for later potential use.
    SET(ShapeIncludeName "${SWITCH_DIR}/BlockOpSwitchCode.h.in")
    FOREACH (Shape IN LISTS Shapes)
        IF(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitchCodeSeg.h")
            SET(ShapeIncName include/BlockOpSwitchCode${Shape}.h)
            # MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDIF()
    ENDFOREACH()

    SET(ShapeIncludeName "${SWITCH_DIR}/BlockOpSwitch1DCoordsCode.h.in")
    FOREACH (Shape IN LISTS Shapes)
        IF(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/BlockOpSwitch1DCoordsCodeSeg.h")
            SET(ShapeIncName include/BlockOpSwitch1DCoordsCode${Shape}.h)
            # MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDIF()
    ENDFOREACH()

    SET(ShapeIncludeName "${SWITCH_DIR}/BlockOpSwitchPhysInterp1D.h.in")
    FOREACH (Shape IN LISTS Shapes)
        IF(NOT EXISTS "${SWITCH_DIR}/BlockOpSwitchPhysInterp1DSeg.h")
            SET(ShapeIncName include/BlockOpSwitchPhysInterp1D${Shape}.h)
            # MESSAGE("Generating block header  file file: " "${ShapeIncName}")
            CONFIGURE_FILE(${ShapeIncludeName} ${ShapeIncName})
        ENDIF()
    ENDFOREACH()

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
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a SumFac header.
                ELSEIF((EXISTS "${dir}/${name}SumFac.hpp") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Generic header.
                ELSEIF((EXISTS "${dir}/${name}Generic.hpp") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}Generic.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Serial StdMat header.
                ELSEIF((EXISTS "${dir}/${name}SerialStdMat.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialStdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Serial SumFac header.
                ELSEIF((EXISTS "${dir}/${name}SerialSumFac.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Serial Generic header.
                ELSEIF((EXISTS "${dir}/${name}SerialGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a AVX SumFac header.
                ELSEIF((EXISTS "${dir}/${name}AVXSumFac.hpp") AND
                    ("${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "SumFac"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}AVXSumFac.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a AVX Generic header.
                ELSEIF((EXISTS "${dir}/${name}AVXGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}AVXGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Serial/AVX StdMat header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXStdMat.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXStdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

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
                                MESSAGE("Generating shape cpp file: " "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                                CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                                SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                            ENDFOREACH()
                        ENDFOREACH()
                    ENDIF()
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Serial/AVX generic header.
                ELSEIF((EXISTS "${dir}/${name}SerialAVXGeneric.hpp") AND
                    ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}SerialAVXGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a Device StdMat header.
                ELSEIF(EXISTS "${dir}/${name}DeviceStdMat.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceStdMat.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

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
                                MESSAGE("Generating shape cpp file: " "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                                CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                                SET(CPP_SHAPE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${CPP_SHAPE_SOURCE})
                            ENDFOREACH()
                        ENDFOREACH()
                    ENDIF()
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a generic header.
                ELSEIF(EXISTS "${dir}/${name}DeviceGeneric.hpp" AND ("${ExecSpace}" STREQUAL "Device") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_CURRENT_SOURCE_DIR}
                        "${dir}/${name}DeviceGeneric.hpp")
                    SET(IMPL_HEADER "#include \"${include_file}\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

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

SET(OPNAMESPACE "Nektar::Operators::detail")

# Common header file.
SET(OPERATORS_HEADERS
    Common/BlockOperator.hpp
    Common/Operator.hpp
)

# Common source file.
SET(OPERATORS_SOURCES
    Common/BlockOperator.cpp
    Common/Operator.cpp
)

# Only these cpp files go into the group set below.
SET(GROUP_SOURCES ${OPERATORS_SOURCES})

# Loop through all of the directories which contain operators and add
# their source and header files.
FILE(GLOB children AddTraceIntegral AssmbScatr BndCondOps/* ElmtOps/*
     GlobalLinSysOps/LinearSystems/* GlobalLinSysOps/LinearSolvers/*
     PreconOps/* Norm/* GetFwdBwdTracePhys)

FOREACH(child ${children})
    # MESSAGE("Child : " "${child}")
    ADD_OPERATOR(${child} OPNAMESPACE OPERATORS_HEADERS OPERATORS_SOURCES)
ENDFOREACH()

# Loop through all of the directories which contain block operators and add
# their source and header files.
# 1) ElmtOps
FILE(GLOB children ElmtOps/*)
FOREACH(child ${children})
    ADD_ELMT_OPERATOR(${child})
ENDFOREACH()
# 2) Others
FILE(GLOB children GetFwdBwdTracePhys BndCondOps/FwdTransBC Norm/*)
FOREACH(child ${children})
    ADD_BLOCK_OPERATOR(${child} OPNAMESPACE OPERATORS_HEADERS OPERATORS_SOURCES)
ENDFOREACH()

SOURCE_GROUP(TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${OPERATORS_HEADERS} ${GROUP_SOURCES})

ADD_DEFINITIONS(-DOPERATORS_EXPORTS -I${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

ADD_NEKTAR_LIBRARY(Operators
    SOURCES ${OPERATORS_SOURCES}
    HEADERS ${OPERATORS_HEADERS}
    DEPENDS LibUtilities MultiRegions
    SUMMARY "Nektar++ Operators library"
    DESCRIPTION "This library provides classes related to operators for Nektar++.")

INSTALL(DIRECTORY ./
  DESTINATION ${NEKTAR_INCLUDE_DIR}/Operators
  COMPONENT dev
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")
