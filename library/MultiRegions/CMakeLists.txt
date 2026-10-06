# Macro for each element operator.
MACRO(ADD_ELMT_OPERATOR dir)
    # Set implementation types.
    SET(ElmtOpImpl StdMat SumFac SumFacTOP Generic)

    # Shape types for seperate code compilation.
    SET(Shapes Seg Tri NodalTri Quad Hex Pyr Prism NodalPrism Tet NodalTet)

    # Extract operator name form directory if it exists.
    GET_FILENAME_COMPONENT(name "${dir}" NAME)

    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)
    SET(OPERATOR ${name})

    # An operator templated on its output field state has an additional
    # argument for the FieldState, which requires two declarations and two
    # ShapeBlock definitions. That is a property of the operator's block
    # class, so ask the header rather than keeping a list of names here.
    UNSET(TwoOutStates)
    IF(EXISTS "${dir}/${name}BlockOp.hpp")
        FILE(STRINGS "${dir}/${name}BlockOp.hpp" TwoOutStates
            REGEX "FieldState TFieldOut")
    ENDIF()
    IF(TwoOutStates)
        SET(CONFIG_FILE ElmtOps/ElmtBlockOpTwoOutStates.cpp.in)
        SET(SHARED_SHAPE_TEMPLATE BlockOpShapeBlockTwoOutStates.cpp.in)
    ELSE()
        SET(CONFIG_FILE ElmtOps/ElmtBlockOpFactoryDec.cpp.in)
        SET(SHARED_SHAPE_TEMPLATE BlockOpShapeBlock.cpp.in)
    ENDIF()

    # Loop through each possible execution space.
    FOREACH (ExecSpace IN LISTS ExecSpaces)
    
        IF ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
            SET(ExecName "SerialAVX")
        ELSE()
            SET(ExecName "Device")
        ENDIF()

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
            UNSET(IMPL_HEADER_FILE)

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

                # Found a Impl header.
                IF(EXISTS "${dir}/${name}${Impl}.hpp")
                    FILE(RELATIVE_PATH include_file ${CMAKE_SOURCE_DIR}/library
                        "${dir}/${name}${Impl}.hpp")
                    SET(IMPL_HEADER "#include <${include_file}>")
                    SET(IMPL_HEADER_FILE "${dir}/${name}${Impl}.hpp")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a ExecName + StdMat header.
                ELSEIF((EXISTS "${dir}/${name}${ExecName}StdMat.hpp") AND ("${Impl}" STREQUAL "StdMat"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_SOURCE_DIR}/library
                        "${dir}/${name}${ExecName}StdMat.hpp")
                    SET(IMPL_HEADER "#include <${include_file}>")
                    SET(IMPL_HEADER_FILE "${dir}/${name}${ExecName}StdMat.hpp")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a ExecName + SumFac header.
                ELSEIF((EXISTS "${dir}/${name}${ExecName}SumFac.hpp") AND ("${Impl}" STREQUAL "SumFac" OR "${Impl}" STREQUAL "SumFacTOP"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_SOURCE_DIR}/library
                        "${dir}/${name}${ExecName}SumFac.hpp")
                    SET(IMPL_HEADER "#include <${include_file}>")
                    SET(IMPL_HEADER_FILE "${dir}/${name}${ExecName}SumFac.hpp")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # Found a ExecName + generic header.
                ELSEIF((EXISTS "${dir}/${name}${ExecName}Generic.hpp") AND ("${Impl}" STREQUAL "Generic"))
                    FILE(RELATIVE_PATH include_file ${CMAKE_SOURCE_DIR}/library
                        "${dir}/${name}${ExecName}Generic.hpp")
                    SET(IMPL_HEADER "#include <${include_file}>")
                    SET(IMPL_HEADER_FILE "${dir}/${name}${ExecName}Generic.hpp")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                # No implementation, skip.
                ELSE()
                    CONTINUE() # This avoid creating a *.cpp file
                ENDIF()

                # An implementation that splits its shape specific
                # ShapeBlock definitions out of the header is compiled one
                # file per shape. Whether it does so is a property of the
                # implementation header, so ask the header rather than
                # keeping a list here.
                UNSET(ShapeBlockDecl)
                IF(DEFINED IMPL_HEADER_FILE)
                    FILE(STRINGS "${IMPL_HEADER_FILE}" ShapeBlockDecl
                        REGEX "void ShapeBlock")
                ENDIF()

                # Operators share a template: one for the usual single
                # output state, one for operators templated on their output
                # field state. Nothing operator specific is passed to either
                # beyond the substitutions made here; the field states and
                # the switch construction are picked up from the operator's
                # own headers. An operator whose declarations neither shared
                # template can express keeps a local template of its own,
                # named after the operator and the implementation.
                IF(ShapeBlockDecl)
                    IF(EXISTS "${dir}/${name}${Impl}BlockOp.cpp.in")
                        SET(ShapeConfigName
                            "${dir}/${name}${Impl}BlockOp.cpp.in")
                    ELSE()
                        SET(ShapeConfigName
                            "${SWITCH_DIR}/${SHARED_SHAPE_TEMPLATE}")
                    ENDIF()
                    FOREACH (Shape IN LISTS Shapes)
                        SET(ShapeName src/${name}${ExecSpace}${Impl}${TData}${Shape}.${SOURCE_EXT})
                        CONFIGURE_FILE(${ShapeConfigName} ${ShapeName})
                        LIST(APPEND ${NAME_SOURCES}
                            "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                    ENDFOREACH()
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
    SET(MULTI_REGIONS_HEADERS
        ${MULTI_REGIONS_HEADERS}
        ${${NAME_HEADERS}}
    )
    SET(MULTI_REGIONS_SOURCES
        ${MULTI_REGIONS_SOURCES}
        ${${NAME_SOURCES}}
    )
ENDMACRO()

SET(MULTI_REGIONS_SOURCES
    ./AssemblyMap/AssemblyCommDG.cpp
    ./AssemblyMap/AssemblyMap.cpp
    ./AssemblyMap/AssemblyMapCG.cpp
    ./AssemblyMap/AssemblyMapDG.cpp
    ./AssemblyMap/InterfaceMapDG.cpp
    ./AssemblyMap/LocTraceToTraceMap.cpp
    ContField.cpp
    ContField3DHomogeneous1D.cpp
    ContField3DHomogeneous2D.cpp
    DisContField.cpp
    DisContField3DHomogeneous1D.cpp
    DisContField3DHomogeneous2D.cpp
    ExpList.cpp
    ExpListHomogeneous1D.cpp
    ExpListHomogeneous2D.cpp
    ExpList2DHomogeneous1D.cpp
    ExpList2DHomogeneous2D.cpp
    ExpList3DHomogeneous1D.cpp
    ExpList3DHomogeneous2D.cpp
    GJPStabilisation.cpp
    GlobalLinSys.cpp
    GlobalLinSysKey.cpp
    GlobalLinSysDirect.cpp
    GlobalLinSysDirectFull.cpp
    GlobalLinSysDirectStaticCond.cpp
    GlobalLinSysIterative.cpp
    GlobalLinSysIterativeFull.cpp
    GlobalLinSysIterativeStaticCond.cpp
    GlobalLinSysStaticCond.cpp
    GlobalMatrix.cpp
    GlobalMatrixKey.cpp
    Preconditioner.cpp
    PreconditionerDiagonal.cpp
    PreconditionerLowEnergy.cpp
    PreconditionerBlock.cpp
    PreconditionerLOR.cpp
    SubStructuredGraph.cpp
)

SET(MULTI_REGIONS_HEADERS
    ContField.h
    ContField3DHomogeneous1D.h
    ContField3DHomogeneous2D.h
    DisContField.h
    DisContField3DHomogeneous1D.h
    DisContField3DHomogeneous2D.h
    ExpList.h
    ExpListHomogeneous1D.h
    ExpListHomogeneous2D.h
    ExpList2DHomogeneous1D.h
    ExpList2DHomogeneous2D.h
    ExpList3DHomogeneous1D.h
    ExpList3DHomogeneous2D.h
    GJPStabilisation.h
    GlobalLinSys.h
    GlobalLinSysKey.h
    GlobalLinSysDirect.h
    GlobalLinSysDirectFull.h
    GlobalLinSysDirectStaticCond.h
    GlobalLinSysIterative.h
    GlobalLinSysIterativeFull.h
    GlobalLinSysIterativeStaticCond.h
    GlobalLinSysStaticCond.h
    GlobalMatrix.h
    GlobalMatrixKey.h
    MultiRegions.hpp
    MultiRegionsDeclspec.h
    Preconditioner.h
    PreconditionerDiagonal.h
    PreconditionerLowEnergy.h
    PreconditionerBlock.h
    PreconditionerLOR.h
    SubStructuredGraph.h
)

SET(ASSEMBLY_MAP_HEADERS
    ./AssemblyMap/AssemblyComm.h
    ./AssemblyMap/AssemblyCommDG.h
    ./AssemblyMap/AssemblyMap.h
    ./AssemblyMap/AssemblyMapCG.h
    ./AssemblyMap/AssemblyMapDG.h
    ./AssemblyMap/InterfaceMapDG.h
    ./AssemblyMap/LocTraceToTraceMap.h
)

IF(NEKTAR_USE_MPI)
    SET(MULTI_REGIONS_HEADERS ${MULTI_REGIONS_HEADERS}
        GlobalLinSysXxt.h
        GlobalLinSysXxtFull.h
        GlobalLinSysXxtStaticCond.h
        PreconditionerLinear.h
        PreconditionerLinearWithLowEnergy.h
        PreconditionerLinearWithDiag.h
        PreconditionerLinearWithBlock.h
    )
    SET(MULTI_REGIONS_SOURCES ${MULTI_REGIONS_SOURCES}
        GlobalLinSysXxt.cpp
        GlobalLinSysXxtFull.cpp
        GlobalLinSysXxtStaticCond.cpp
        PreconditionerLinear.cpp
        PreconditionerLinearWithLowEnergy.cpp
        PreconditionerLinearWithDiag.cpp
        PreconditionerLinearWithBlock.cpp
    )
ENDIF(NEKTAR_USE_MPI)

IF(NEKTAR_USE_PETSC)
    SET(MULTI_REGIONS_HEADERS ${MULTI_REGIONS_HEADERS}
        GlobalLinSysPETSc.h
        GlobalLinSysPETScFull.h
        GlobalLinSysPETScStaticCond.h
    )
    SET(MULTI_REGIONS_SOURCES ${MULTI_REGIONS_SOURCES}
        GlobalLinSysPETSc.cpp
        GlobalLinSysPETScFull.cpp
        GlobalLinSysPETScStaticCond.cpp
    )
ENDIF(NEKTAR_USE_PETSC)

IF (NEKTAR_ENABLE_DEVICE_SUPPORT)
    SET(MULTI_REGIONS_HEADERS
        ${MULTI_REGIONS_HEADERS}
        ./Common/BlockOperator.hpp
        ./Common/Operator.hpp
        ./DataWarehouse/LocalToGlobalDataWarehouse.hpp
        ./DataWarehouse/TraceDataWarehouse.hpp
    )
    SET(MULTI_REGIONS_SOURCES 
        ${MULTI_REGIONS_SOURCES}
        ./Common/BlockOperator.cpp
        ./Common/Operator.cpp
        ./DataWarehouse/LocalToGlobalDataWarehouse.cpp
        ./DataWarehouse/TraceDataWarehouse.cpp
        ExpListHelper.cpp
    )

    SET(OPNAMESPACE "Nektar::MultiRegions::detail")
    ADD_OPERATOR(${CMAKE_CURRENT_SOURCE_DIR}/AssmbScatr OPNAMESPACE
        MULTI_REGIONS_HEADERS MULTI_REGIONS_SOURCES)

    FILE(GLOB children ${CMAKE_CURRENT_SOURCE_DIR}/BndCondOps/*)
    FOREACH(child ${children})
        ADD_OPERATOR(${child} OPNAMESPACE MULTI_REGIONS_HEADERS
            MULTI_REGIONS_SOURCES)
    ENDFOREACH()
    ADD_BLOCK_OPERATOR(${CMAKE_CURRENT_SOURCE_DIR}/BndCondOps/FwdTransBC
        OPNAMESPACE MULTI_REGIONS_HEADERS MULTI_REGIONS_SOURCES)

    FILE(GLOB children ${CMAKE_CURRENT_SOURCE_DIR}/ElmtOps/*)
    FOREACH(child ${children})
        ADD_OPERATOR(${child} OPNAMESPACE MULTI_REGIONS_HEADERS
            MULTI_REGIONS_SOURCES)
    ENDFOREACH()

    # Shape specific switch code included by the generated block operator
    # sources. This depends only on the shape and on the global switch limits,
    # not on the operator, so it is generated once here rather than once per
    # operator directory.
    SET(SWITCH_DIR ${CMAKE_SOURCE_DIR}/library/LibUtilities/BasicUtils/Switch)
    SET(Shapes Seg Tri NodalTri Quad Hex Pyr Prism NodalPrism Tet NodalTet)
    FOREACH (Shape IN LISTS Shapes)
        CONFIGURE_FILE(${SWITCH_DIR}/BlockOpSwitchCode.h.in
            include/BlockOpSwitchCode${Shape}.h)
        CONFIGURE_FILE(${SWITCH_DIR}/BlockOpSwitch1DCoordsCode.h.in
            include/BlockOpSwitch1DCoordsCode${Shape}.h)
        CONFIGURE_FILE(${SWITCH_DIR}/BlockOpSwitchPhysInterp1D.h.in
            include/BlockOpSwitchPhysInterp1D${Shape}.h)
        CONFIGURE_FILE(${SWITCH_DIR}/BlockOpSwitchPhysTraceExtract.h.in
            include/BlockOpSwitchPhysTraceExtract${Shape}.h)
    ENDFOREACH()

    FOREACH(child ${children})
        ADD_ELMT_OPERATOR(${child})
    ENDFOREACH()

    # GCC 13 crashes with an internal compiler error in gimple_duplicate_bb
    # (gcc/tree-cfg.cc) while completely unrolling the three-dimensional
    # trace kernels: once the templated size parameter is substituted every
    # loop bound in them is a compile-time constant, and the whole call
    # chain is always_inline, so the cunroll pass is handed more to
    # duplicate than it can cope with. Seen on Hex, Pyr, Prism and
    # NodalPrism. Disable that one pass for the affected translation units;
    # every other file, and every other pass in these, is unaffected.
    #
    # Bounded to the 13 series: 14 compiles these files, 12 has not been
    # tried and is assumed good. Widen the bound if the crash turns up on
    # another version, and drop the block entirely once 13 is no longer
    # supported. The lower-impact alternative to the flag is --param
    # max-completely-peeled-insns=<n> below the default of 200, which keeps
    # unrolling the smaller loops.
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
        SET(CUNROLL_ICE_SOURCES)
        FOREACH(src IN LISTS MULTI_REGIONS_SOURCES)
            GET_FILENAME_COMPONENT(src_name "${src}" NAME)
            IF(src_name MATCHES "^(PhysTraceExtract|IProductWRTPhysTrace|IProductWRTPhysNormalDerivTrace)(Serial|AVX)SumFac(double|float)(Hex|Pyr|Prism|NodalPrism|Tet|NodalTet)\\.cpp$")
                LIST(APPEND CUNROLL_ICE_SOURCES "${src}")
            ENDIF()
        ENDFOREACH()
        IF(CUNROLL_ICE_SOURCES)
            MESSAGE(STATUS
                "Disabling complete loop unrolling for the three-dimensional "
                "trace operators: GCC ${CMAKE_CXX_COMPILER_VERSION} crashes in "
                "gimple_duplicate_bb on them")
            SET_SOURCE_FILES_PROPERTIES(${CUNROLL_ICE_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-fdisable-tree-cunroll")
        ENDIF()
    ENDIF()
ENDIF()
ADD_DEFINITIONS(-DMULTI_REGIONS_EXPORTS)

ADD_NEKTAR_LIBRARY(MultiRegions
    SOURCES ${MULTI_REGIONS_SOURCES}
    HEADERS ${MULTI_REGIONS_HEADERS}
    DEPENDS Collections
    SUMMARY "Nektar++ MultiRegions library"
    DESCRIPTION "This library provides global expansions on multi-element domains.")

# MultiRegions uses additional routines from GSMPI
IF( NEKTAR_USE_MPI )
    TARGET_LINK_LIBRARIES(MultiRegions LINK_PRIVATE ${GSMPI_LIBRARY} ${XXT_LIBRARY})

    # If compiling Scotch, we are building a static library so we do
    # not want to propagate the dependency in the link interface
    IF (NEKTAR_USE_SCOTCH AND THIRDPARTY_BUILD_SCOTCH)
        TARGET_LINK_LIBRARIES(MultiRegions LINK_PRIVATE
            ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY})
    ENDIF()
ELSE (NEKTAR_USE_MPI )
    IF (NEKTAR_USE_SCOTCH AND THIRDPARTY_BUILD_SCOTCH)
        TARGET_LINK_LIBRARIES(MultiRegions LINK_PRIVATE
            ${SCOTCH_LIBRARY} ${SCOTCHERR_LIBRARY})
    ENDIF()
ENDIF( NEKTAR_USE_MPI )

IF( NEKTAR_USE_PETSC )
    TARGET_LINK_LIBRARIES(MultiRegions LINK_PRIVATE ${PETSC_LIBRARIES})
    ADD_DEPENDENCIES(MultiRegions petsc-3.19.3)
ENDIF( NEKTAR_USE_PETSC )

IF (NEKTAR_BUILD_PYTHON)
    SUBDIRS(Python)
ENDIF()

INSTALL(DIRECTORY ./ DESTINATION ${NEKTAR_INCLUDE_DIR}/MultiRegions COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")



