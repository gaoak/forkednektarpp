##
## NektarCommon.cmake
##
## Frequently used Nektar++ CMake configuration macros and functions
##

# Executables that use std::thread carry direct references to pthread symbols,
# so the threading library has to appear on their link line. Threads::Threads
# is language-aware, which matters here since executables may also contain CUDA
# or HIP objects that are compiled and linked by a device compiler.
FIND_PACKAGE(Threads REQUIRED)

#
# SET_SHAPE_SWITCH_COMPILE_OPTIONS(file shape ext)
#
# The compile options of a generated per-shape source of an element
# operator, @p file of shape @p shape with extension @p ext, shared by
# ADD_ELMT_OPERATOR and ADD_POLICY_ELMT_BLOCK_OPERATOR. GCC 13 fails with
# an internal compiler error in its loop-splitting pass on the
# collapsed-shape units of the larger operators (Helmholtz, Laplacian)
# once the equal-count cases of the size switch are compiled; the pass
# gains nothing here. With the switch widened past order 6 the same
# assertion fires in complete unrolling on the pyramid units, so those
# lose that pass as well.
#
FUNCTION(SET_SHAPE_SWITCH_COMPILE_OPTIONS file shape ext)
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND "${ext}" STREQUAL "cpp"
       AND "${shape}" MATCHES "(Tri|Prism|Pyr|Tet)$")
        SET(SwitchFlags "-fno-split-loops")
        IF("${shape}" STREQUAL "Pyr" AND NEKTAR_SWITCH_MAX GREATER 6)
            LIST(APPEND SwitchFlags "-fdisable-tree-cunroll")
        ENDIF()
        SET_SOURCE_FILES_PROPERTIES("${file}"
            PROPERTIES COMPILE_OPTIONS "${SwitchFlags}")
    ENDIF()
ENDFUNCTION()

IF(NEKTAR_ENABLE_DEVICE_SUPPORT)
    # Macro for each operator.
    MACRO(ADD_OPERATOR dir OPNAMESPACE OPERATORS_HEADERS OPERATORS_SOURCES)

        IF(IS_DIRECTORY ${dir})
            GET_FILENAME_COMPONENT(name "${dir}" NAME)
            GET_FILENAME_COMPONENT(abs_dir "${dir}" ABSOLUTE)
        ELSE()
            CONTINUE()
        ENDIF()

        SET(NAME_HEADERS ${name}_HEADERS)
        SET(NAME_SOURCES ${name}_SOURCES)
        SET(OPERATOR ${name})

        # Counted so that a directory registering nothing can say so below,
        # rather than surfacing later as a missing operator at run time.
        SET(OP_REGISTERED 0)

        # These operator have an additional argument for the FieldState
        # which requires two delcarations.
        IF("${name}" STREQUAL "IProductWRTDerivBase")
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/MultiRegions/Common/OpFactoryDecTwoOutStates.cpp.in)
        ELSEIF("${name}" STREQUAL "AssmbScatr")
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/MultiRegions/AssmbScatr/AssmbScatrFactoryDec.cpp.in)
        ELSE()
            SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/MultiRegions/Common/OpFactoryDec.cpp.in)
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

            IF ("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
                SET(ExecName "SerialAVX")
            ELSE()
                SET(ExecName "Device")
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
                    
                # Get the headers matching the name and execution space.
                FILE(GLOB ${CURRENT_HEADERS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${abs_dir}/${name}${ExecSpace}*.hpp
                    ${abs_dir}/${name}BlockOp${ExecSpace}*.hpp)

                # Found a impl header.
                IF(EXISTS "${abs_dir}/${name}OpImpl.hpp")
                    SET(IMPL_HEADER "#include \"${abs_dir}/${name}OpImpl.hpp\"")
                    MESSAGE("Adding operator with a ${ExecSpace} execution space: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")
                # Found a ExecName header.
                ELSEIF(EXISTS "${abs_dir}/${name}${ExecName}.hpp")
                    SET(IMPL_HEADER "#include \"${abs_dir}/${name}${ExecName}.hpp\"")
                    MESSAGE("Adding operator with a ${ExecSpace} execution space: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")
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
                MATH(EXPR OP_REGISTERED "${OP_REGISTERED}+1")

            ENDFOREACH()
        ENDFOREACH()

        # As above: a globbed directory that produces no operator is almost
        # always a mistake, and says nothing until the factory is asked for it.
        # Only for a directory that declares an operator: a glob such as
        # BndCondOps/* also picks up shared directories which hold no operator
        # and should stay quiet.
        IF(OP_REGISTERED EQUAL 0 AND EXISTS "${abs_dir}/${name}Op.hpp")
            MESSAGE(WARNING "ADD_OPERATOR: '${name}' declares ${name}Op.hpp but "
                "registered no operator for any execution space. Expected an "
                "implementation header named ${name}OpImpl.hpp in ${dir}.")
        ENDIF()

        # Add this operator's implemenations to the global operator source
        # and header files.
        SET(${OPERATORS_HEADERS}
           "${${OPERATORS_HEADERS}}"
            ${${NAME_HEADERS}}
        )
        SET(${OPERATORS_SOURCES}
           "${${OPERATORS_SOURCES}}"
            ${${NAME_SOURCES}}
        )
    ENDMACRO()

    # Macro for each operator.
    MACRO(ADD_BLOCK_OPERATOR dir OPNAMESPACE OPERATORS_HEADERS OPERATORS_SOURCES)

        # Extract operator name form directory if it exists.
        IF(IS_DIRECTORY ${dir})
            GET_FILENAME_COMPONENT(name "${dir}" NAME)
            GET_FILENAME_COMPONENT(abs_dir "${dir}" ABSOLUTE)
        ELSE()
            CONTINUE()
        ENDIF()

        SET(NAME_HEADERS ${name}_HEADERS)
        SET(NAME_SOURCES ${name}_SOURCES)
        SET(OPERATOR ${name})

        # Counted so that a directory registering nothing can say so below,
        # rather than surfacing later as a missing operator at run time.
        SET(BLOCK_OP_REGISTERED 0)

        # These operator have an additional argument for the FieldState
        # which requires two delcarations.
        SET(CONFIG_FILE ${CMAKE_SOURCE_DIR}/library/MultiRegions/Common/BlockOperatorFactoryDec.cpp.in)

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
        
            IF("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
                SET(ExecName "SerialAVX")
            ELSE()
                SET(ExecName "${ExecSpace}")
            ENDIF()

            # Loop through each possible data type.
            FOREACH (TData IN LISTS DataTypes)
                # Reset for this (ExecSpace, TData)
                UNSET(IMPL_HEADER)  
                # Set up the name used for the .cpp declaration file.
                SET(FactoryDeclName "src/Block${name}${ExecSpace}${TData}.${SOURCE_EXT}")
                # Found a impl header.
		IF(EXISTS "${dir}/${name}BlockOpImpl.hpp")
                    SET(IMPL_HEADER "#include \"${abs_dir}/${name}BlockOpImpl.hpp\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")
                # Found a ExecName header.
		ELSEIF(EXISTS "${dir}/${name}BlockOp${ExecName}.hpp")
                    SET(IMPL_HEADER "#include \"${abs_dir}/${name}BlockOp${ExecName}.hpp\"")
                    MESSAGE("Adding block operator with a ${ExecSpace} implementation: " "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")
                ENDIF()

                # If no header was found, skip this combination.
                IF(NOT DEFINED IMPL_HEADER)
                    CONTINUE()
                ENDIF()

                # Create the cpp file from the implementation file.
                CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})

                # Add this specific implementation to the operator's
                # source and header files.
                SET(${NAME_HEADERS} ${${NAME_HEADERS}} ${${CURRENT_HEADERS}})
                SET(${CURRENT_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName})
                SET(${NAME_SOURCES} ${${NAME_SOURCES}} ${${CURRENT_SOURCES}})
                MATH(EXPR BLOCK_OP_REGISTERED "${BLOCK_OP_REGISTERED}+1")
            ENDFOREACH()
        ENDFOREACH()

        # A globbed directory that produces no block operator is almost always
        # a mistake, and its only symptom is "No such operator" from the
        # factory at run time, a long way from the cause.
        # Only for a directory that declares a block operator, so that shared
        # directories picked up by a glob stay quiet.
        IF(BLOCK_OP_REGISTERED EQUAL 0 AND EXISTS "${abs_dir}/${name}BlockOp.hpp")
            MESSAGE(WARNING "ADD_BLOCK_OPERATOR: '${name}' declares "
                "${name}BlockOp.hpp but registered no block operator for any "
                "execution space. Expected an implementation header named "
                "${name}BlockOp<SerialAVX|Device>.hpp in ${dir}.")
        ENDIF()

        # Add this operator's implemenations to the global operator source
        # and header files.
        SET(${OPERATORS_HEADERS}
           "${${OPERATORS_HEADERS}}"
            ${${NAME_HEADERS}}
        )
        SET(${OPERATORS_SOURCES}
           "${${OPERATORS_SOURCES}}"
            ${${NAME_SOURCES}}
        )
    ENDMACRO()

    #
    # ADD_POLICY_ELMT_BLOCK_OPERATOR(dir policyName policyType policyHeader
    #                                ndim sources)
    #
    # Register, for one policy, the block operators of a SolverCore element
    # operator whose implementations are templated on a policy as well as on
    # the execution space, implementation and data type. The policy is the
    # physics the operator evaluates at each quadrature point: for the fused
    # element-local operators (SolverCore/Advection/AdvectionFusedElmt,
    # SolverCore/AdvDiffusion/AdvDiffFusedElmt), a trivially copyable type
    # supplying the advective flux AdvFlux() and, for the advection-diffusion
    # operator, the viscous flux ViscFlux(), such as a compressible Euler or
    # Navier-Stokes model. It belongs to the code that uses the operator, so
    # the operator's own library cannot generate the registrations; this is
    # called where the policy is defined, and the generated sources are
    # compiled there.
    #
    # Implementations are found by file name, as ADD_ELMT_OPERATOR finds
    # them and in its order: `<name>BlockOp<Impl>.hpp`, then
    # `<name>BlockOp<SerialAVX|Device><StdMat|SumFac|Generic>.hpp`, with
    # SumFac and SumFacTOP sharing the SumFac header and SumFacTOP skipped on
    # Serial and AVX. A `<name>BlockOpDeviceSumFacTOP.hpp` header, where there
    # is one, serves SumFacTOP on its own, ahead of the shared SumFac header,
    # for an operator whose Device SumFac and SumFacTOP are separate
    # implementations or which has only the latter. The registrations and per-shape ShapeBlock definitions are
    # generated from SolverCore/ElmtBlockOpWithPolicyFactoryDec.cpp.in and
    # SolverCore/BlockOpWithPolicyShapeBlock.cpp.in, under the factory key
    # `<name>BlockOp<TData>::name + policyType::name + <ExecSpace><Impl>`,
    # for the shapes of dimension @p ndim.
    #
    # Arguments:
    #   - `dir`: the operator's directory under library/, named after the
    #     operator, e.g.
    #     ${CMAKE_SOURCE_DIR}/library/SolverCore/AdvDiffusion/AdvDiffFusedElmt.
    #   - `policyName`: the policy's name in the generated file names, e.g.
    #     NavierStokesModel3D.
    #   - `policyType`: the policy's fully qualified C++ type, e.g.
    #     "Nektar::UnitTests::NavierStokesModel<3>".
    #   - `policyHeader`: the header declaring the policy, as it is included,
    #     e.g. UnitTests/SolverCore/TestFusedElmtPolicies.hpp.
    #   - `ndim`: the policy's dimension, 1, 2 or 3.
    #   - `sources`: the list the generated sources are appended to.
    #
    FUNCTION(ADD_POLICY_ELMT_BLOCK_OPERATOR dir policyName policyType
             policyHeader ndim sources)
        # Set implementation types.
        SET(ElmtOpImpl StdMat SumFac SumFacTOP Generic)

        # Shape types for separate code compilation: those of the policy's
        # dimension. The fused operators take no nodal shape types.
        IF("${ndim}" STREQUAL "1")
            SET(Shapes Seg)
        ELSEIF("${ndim}" STREQUAL "2")
            SET(Shapes Tri Quad)
        ELSEIF("${ndim}" STREQUAL "3")
            SET(Shapes Hex Pyr Prism Tet)
        ELSE()
            MESSAGE(FATAL_ERROR "ADD_POLICY_ELMT_BLOCK_OPERATOR: ${policyName} "
                "has dimension '${ndim}'; expected 1, 2 or 3.")
        ENDIF()

        # Extract the operator name from the directory.
        GET_FILENAME_COMPONENT(abs_dir "${dir}" ABSOLUTE)
        GET_FILENAME_COMPONENT(name "${abs_dir}" NAME)
        SET(OPERATOR ${name})
        SET(POLICY_NAME ${policyName})
        SET(POLICY_TYPE ${policyType})

        # The policy counterparts of the MultiRegions registration and
        # ShapeBlock templates.
        SET(CONFIG_FILE
            ${CMAKE_SOURCE_DIR}/library/SolverCore/ElmtBlockOpWithPolicyFactoryDec.cpp.in)
        SET(SHAPE_CONFIG_FILE
            ${CMAKE_SOURCE_DIR}/library/SolverCore/BlockOpWithPolicyShapeBlock.cpp.in)
        SET(generated "")

        # Loop through each possible execution space.
        FOREACH(ExecSpace IN LISTS ExecSpaces)

            IF("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
                SET(ExecName "SerialAVX")
            ELSE()
                SET(ExecName "Device")
            ENDIF()

            # Set file extension.
            IF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "CUDA")
                SET(SOURCE_EXT cu)
            ELSEIF("${ExecSpace}" STREQUAL "Device" AND NEKTAR_ENABLE_DEVICE STREQUAL "HIP")
                SET(SOURCE_EXT hip)
            ELSE()
                SET(SOURCE_EXT cpp)
            ENDIF()

            # Loop through each possible implementation.
            FOREACH(Impl IN LISTS ElmtOpImpl)

                UNSET(implFile)

                # These combinations do/should not exist.
                IF(("${ExecSpace}" STREQUAL "Serial" OR "${ExecSpace}" STREQUAL "AVX")
                   AND "${Impl}" STREQUAL "SumFacTOP")
                    CONTINUE()
                ENDIF()

                # The headers are looked for in ADD_ELMT_OPERATOR's order
                # (library/MultiRegions/CMakeLists.txt), with one addition:
                # a Device SumFacTOP header of its own, for an operator whose
                # Device SumFac and SumFacTOP are separate implementations or
                # which has only the latter.

                # Found an Impl header.
                IF(EXISTS "${abs_dir}/${name}BlockOp${Impl}.hpp")
                    SET(implFile "${abs_dir}/${name}BlockOp${Impl}.hpp")

                # Found a Device SumFacTOP header.
                ELSEIF((EXISTS "${abs_dir}/${name}BlockOp${ExecName}SumFacTOP.hpp") AND ("${Impl}" STREQUAL "SumFacTOP"))
                    SET(implFile "${abs_dir}/${name}BlockOp${ExecName}SumFacTOP.hpp")

                # Found a ExecName + StdMat header.
                ELSEIF((EXISTS "${abs_dir}/${name}BlockOp${ExecName}StdMat.hpp") AND ("${Impl}" STREQUAL "StdMat"))
                    SET(implFile "${abs_dir}/${name}BlockOp${ExecName}StdMat.hpp")

                # Found a ExecName + SumFac header.
                ELSEIF((EXISTS "${abs_dir}/${name}BlockOp${ExecName}SumFac.hpp") AND ("${Impl}" STREQUAL "SumFac" OR "${Impl}" STREQUAL "SumFacTOP"))
                    SET(implFile "${abs_dir}/${name}BlockOp${ExecName}SumFac.hpp")

                # Found a ExecName + generic header.
                ELSEIF((EXISTS "${abs_dir}/${name}BlockOp${ExecName}Generic.hpp") AND ("${Impl}" STREQUAL "Generic"))
                    SET(implFile "${abs_dir}/${name}BlockOp${ExecName}Generic.hpp")

                # No implementation, skip.
                ELSE()
                    CONTINUE() # This avoid creating a *.cpp file
                ENDIF()

                # The generated sources include the implementation and the
                # header declaring the policy.
                FILE(RELATIVE_PATH include_file ${CMAKE_SOURCE_DIR}/library
                    "${implFile}")
                SET(IMPL_HEADER
                    "#include <${include_file}>\n#include <${policyHeader}>")

                # Loop through each possible data type.
                FOREACH(TData IN LISTS DataTypes)
                    # Set up the name used for the .cpp declaration file and
                    # create it from the registration template.
                    SET(FactoryDeclName
                        "src/Block${name}${policyName}${ExecSpace}${Impl}${TData}.${SOURCE_EXT}")
                    CONFIGURE_FILE(${CONFIG_FILE} ${FactoryDeclName})
                    LIST(APPEND generated
                        "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")
                    MESSAGE("Adding block operator with a ${ExecSpace} ${Impl} implementation for ${policyName}: "
                        "${CMAKE_CURRENT_BINARY_DIR}/${FactoryDeclName}")

                    # The shape-specific ShapeBlock definitions, compiled one
                    # file per shape.
                    FOREACH(Shape IN LISTS Shapes)
                        SET(ShapeName
                            "src/${name}${policyName}${ExecSpace}${Impl}${TData}${Shape}.${SOURCE_EXT}")
                        CONFIGURE_FILE(${SHAPE_CONFIG_FILE} ${ShapeName})
                        LIST(APPEND generated
                            "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}")
                        SET_SHAPE_SWITCH_COMPILE_OPTIONS(
                            "${CMAKE_CURRENT_BINARY_DIR}/${ShapeName}"
                            "${Shape}" "${SOURCE_EXT}")
                    ENDFOREACH()
                ENDFOREACH()
            ENDFOREACH()
        ENDFOREACH()

        # A policy that produces no block operator is almost always a
        # mistake, and its only symptom is "No such operator" from the
        # factory at run time, a long way from the cause.
        IF(NOT generated)
            MESSAGE(WARNING "ADD_POLICY_ELMT_BLOCK_OPERATOR: '${name}' "
                "registered no block operator for ${policyName}. Expected an "
                "implementation header named ${name}BlockOp<SerialAVX|Device>"
                "<StdMat|SumFac|SumFacTOP|Generic>.hpp in ${dir}.")
        ENDIF()
        SET(${sources} ${${sources}} ${generated} PARENT_SCOPE)
    ENDFUNCTION()
ENDIF()

#
# THIRDPARTY_LIBRARY(varname DESCRIPTION <description> [STATIC|SHARED] lib1 [lib2]...)
#
# Updates a variable containing the name of a third-party shared or static
# library to point to an absolute path defining its location instead of adding
# `-llibname` to the linker flags. This avoids the issue of e.g. linking against
# an outdated system zlib installation.
#
# On Windows, it is necessary to link using .lib files which will have been
# generated based on whether the target library was built as STATIC or SHARED.
# Where the library has been built as a SHARED library the associated DLLs
# are used at runtime. The setting of LIBTYPE determines the file names to
# link against so on Windows setting this to SHARED results in trying to link
# against .dll files. LIBTYPE is therefore forced to be STATIC on WIN32
# platforms to ensure linking against the .lib files.
#
# Arguments:
#   - `varname`: variable name containing the third-party library name. On
#     output will be altered to update the correct path.
#   - `DESCRIPTION`: a brief description of the variable (used in the SET
#     command).
#   - `SHARED`: if the library will be built as a shared library
#   - `STATIC`: if the library will be built as a static library
#
MACRO(THIRDPARTY_LIBRARY varname)
    CMAKE_PARSE_ARGUMENTS(TPLIB "" "DESCRIPTION" "STATIC;SHARED" ${ARGN})

    IF(TPLIB_SHARED)
        IF(WIN32)
            # Ensure linking against .lib files on Windows
            SET(LIBTYPE "STATIC")
        ELSE()
            SET(LIBTYPE "SHARED")
        ENDIF()
        SET(TPLIBS ${TPLIB_SHARED})
    ELSEIF(TPLIB_STATIC)
        SET(LIBTYPE "STATIC")
        SET(TPLIBS ${TPLIB_STATIC})
    ENDIF()

    FOREACH (lib ${TPLIBS})
        LIST(APPEND tmplist "${TPDIST}/lib/${CMAKE_${LIBTYPE}_LIBRARY_PREFIX}${lib}${CMAKE_${LIBTYPE}_LIBRARY_SUFFIX}")
    ENDFOREACH()

    SET(${varname} ${tmplist} CACHE FILEPATH ${TPLIB_DESCRIPTION} FORCE)
    UNSET(tmplist)
    UNSET(LIBTYPE)
    UNSET(TPLIBS)
    UNSET(TPLIB_SHARED)
    UNSET(TPLIB_STATIC)
    UNSET(lib)
ENDMACRO()

#
# SET_COMMON_PROPERTIES(target)
#
# Sets properties that are common to either library or executable targets. This
# includes:
#
# - Name suffixes: -g for debug, -ms for minsize, -rg for release w/debug.
# - Disable some MSVC compiler warnings
# - Add -pg flag if NEKTAR_ENABLE_PROFILE is switched on and we're using gcc
# - Add compiler definitions and appropriate warning levels to gcc-like
#   compilers (e.g. clang)
# - Define versions for the target
# - Make sure that -fPIC is enabled for library code if building shared
#   libraries.
#
# Arguments:
#   - `target`: target name
#
MACRO(SET_COMMON_PROPERTIES name)
    SET_TARGET_PROPERTIES(${name} PROPERTIES DEBUG_POSTFIX -g)
    SET_TARGET_PROPERTIES(${name} PROPERTIES MINSIZEREL_POSTFIX -ms)
    SET_TARGET_PROPERTIES(${name} PROPERTIES RELWITHDEBINFO_POSTFIX -rg)

    IF (MSVC)
        # Enable production-level warnings
        TARGET_COMPILE_OPTIONS(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /W4>)
        # Temporarily disable signed/unsigned comparison warning
        TARGET_COMPILE_OPTIONS(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /wd4018>)
        # Temporarily disable narrowing warnings
        TARGET_COMPILE_OPTIONS(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /wd4244 /wd4267>)
        # Enable source-level parallel builds
        TARGET_COMPILE_OPTIONS(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /MP>)
        # Specify minimum Windows version (501=WinXP, 601=Windows 7)
        TARGET_COMPILE_DEFINITIONS(${name} PRIVATE _WIN32_WINNT=0x0601)
    ELSE ()
        # Enable all warnings
        TARGET_COMPILE_OPTIONS(${name} PRIVATE -Wall -Wextra)
        IF (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # For GNU compilers add pedantic warnings
            TARGET_COMPILE_OPTIONS(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wpedantic>)
            TARGET_COMPILE_OPTIONS(${name} PRIVATE -Wnon-virtual-dtor)
        ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            TARGET_COMPILE_OPTIONS(${name} PRIVATE -Wno-tautological-constant-compare)
        ENDIF()
        # Temporarily disable warnings about comparing signed and unsigned
        TARGET_COMPILE_OPTIONS(${name} PRIVATE -Wno-sign-compare)
        # Temporarily disable warnings about narrowing of data types
        TARGET_COMPILE_OPTIONS(${name} PRIVATE -Wno-narrowing -Wno-conversion)
        # Disable warnings about unknown pragama
        TARGET_COMPILE_OPTIONS(${name} PRIVATE -Wno-unknown-pragmas)

        # Disable dignostic about partially overloaded virtual functions
        IF (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            TARGET_COMPILE_OPTIONS(${name} PRIVATE -diag-disable 654)
        ENDIF()

        IF (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            TARGET_COMPILE_OPTIONS(${name} PRIVATE -fno-fast-math)
        ENDIF()

        IF ( NEKTAR_ERROR_ON_WARNINGS )
            TARGET_COMPILE_OPTIONS(${name} PRIVATE -Werror)
        ENDIF()
    ENDIF()

    IF (${CMAKE_COMPILER_IS_GNUCXX})
        IF(NEKTAR_ENABLE_PROFILE)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
            SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
            SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pg")
            SET(LINK_FLAGS "${LINK_FLAGS} -pg")
        ENDIF(NEKTAR_ENABLE_PROFILE)
    ENDIF()

    # Prevent including these common flags multiple times.
    IF (NOT ${CMAKE_CXX_FLAGS_DEBUG} MATCHES ".*DNEKTAR_DEBUG.*")
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DNEKTAR_DEBUG")

        IF ( NEKTAR_FULL_DEBUG )
            SET(CMAKE_CXX_FLAGS_DEBUG
                    "${CMAKE_CXX_FLAGS_DEBUG} -DNEKTAR_FULLDEBUG")
        ENDIF( NEKTAR_FULL_DEBUG)

        # Define version
        SET_PROPERTY(TARGET ${name}
            APPEND PROPERTY COMPILE_DEFINITIONS
            NEKTAR_VERSION=\"${NEKTAR_VERSION}\")

        SET(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_CXX_FLAGS_RELEASE} -DNEKTAR_RELEASE")
    ENDIF(NOT ${CMAKE_CXX_FLAGS_DEBUG} MATCHES ".*DNEKTAR_DEBUG.*")

    IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        # The static libraries must be compiled with position independent
        # code on 64 bit Linux.
        SET_PROPERTY(TARGET ${name} APPEND PROPERTY COMPILE_FLAGS "-fPIC")
    ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
ENDMACRO(SET_COMMON_PROPERTIES name)

#
# ADD_NEKTAR_EXECUTABLE(name COMPONENT <component> [DEPENDS dep1 ...] [SOURCES src1 ...])
#
# Adds a new executable to a component with the supplied component dependencies
# and sources files.
#
# Arguments:
#   - `name`: target name to construct
#   - `COMPONENT`: component name in which this target will live (e.g. demos)
#   - `DEPENDS`: a list of components on which this target depends on
#   - `SOURCES`: a list of source files for this target
#
MACRO(ADD_NEKTAR_EXECUTABLE name)
    CMAKE_PARSE_ARGUMENTS(NEKEXE "" "COMPONENT" "DEPENDS;SOURCES" ${ARGN})
    ADD_EXECUTABLE(${name} ${NEKEXE_SOURCES})
    SET_COMMON_PROPERTIES(${name})

    STRING(TOLOWER ${NEKEXE_COMPONENT} NEKEXE_COMPONENT)
    STRING(TOUPPER ${NEKEXE_COMPONENT} NEKEXE_COMPVAR)

    SET_PROPERTY(TARGET ${name} PROPERTY FOLDER ${NEKEXE_COMPONENT})
    INSTALL(TARGETS ${name}
        RUNTIME DESTINATION ${NEKTAR_BIN_DIR} COMPONENT ${NEKEXE_COMPONENT} OPTIONAL
        ARCHIVE DESTINATION ${NEKTAR_LIB_DIR} COMPONENT ${NEKEXE_COMPONENT} OPTIONAL
        LIBRARY DESTINATION ${NEKTAR_LIB_DIR} COMPONENT ${NEKEXE_COMPONENT} OPTIONAL)

    # Add dependencies for executable.
    TARGET_LINK_LIBRARIES(${name} LINK_PUBLIC ${NEKEXE_DEPENDS})

    # Appended as a property rather than through TARGET_LINK_LIBRARIES: several
    # callers add their own libraries to these targets with the plain signature,
    # and CMake refuses to mix that with the keyword signature used above.
    SET_PROPERTY(TARGET ${name} APPEND PROPERTY LINK_LIBRARIES Threads::Threads)
ENDMACRO()

#
# ADD_NEKTAR_LIBRARY(name
#                    DESCRIPTION <description>
#                    SUMMARY <summary>
#                    DEPENDS dep1 dep2 ...
#                    SOURCES src1 src2 ...
#                    HEADERS head1 head2 ...)
#
# Adds a new library to a component with the supplied component dependencies and
# sources files. A new component will be set up automatically with a lower-case
# name: e.g. if the supplied library name is `LibUtilities` the corresponding
# component is `libutilities`.
#
# Arguments:
#   - `name`: target name to construct
#   - `DESCRIPTION`: a more detailed description of the library
#   - `LINK_MODE`: mode in which to link dependencies: accepts public (default) interface or private; case insensitive
#   - `SUMMARY`: a brief summary of the library
#   - `TARGET_SUFFIX`: Optional suffix to attach to the target name. Allows different targets to share the same (output) `name`
#   - `DEPENDS`: a list of components on which this target depends on
#   - `SOURCES`: a list of source files for this target
#   - `HEADERS`: a list of header files for this target. These will be
#     automatically put into a `dev` package.
#
MACRO(ADD_NEKTAR_LIBRARY name)
    CMAKE_PARSE_ARGUMENTS(NEKLIB "" "DESCRIPTION;LINK_MODE;SUMMARY;TARGET_SUFFIX" "DEPENDS;SOURCES;HEADERS" ${ARGN})

    SET (NEKLIB_TARGET_NAME ${name}${NEKLIB_TARGET_SUFFIX})
    ADD_LIBRARY(${NEKLIB_TARGET_NAME} ${NEKTAR_LIBRARY_TYPE} ${NEKLIB_SOURCES} ${NEKLIB_HEADERS})

    # SimSYCL provides sycl/sycl.hpp, and when it is built from ThirdParty
    # that header does not exist until the external project has installed it.
    # Every Nektar library is a potential includer - directly, or through a
    # LibUtilities header - and the ones that do not link LibUtilities (NekBlas
    # above all) have nothing else ordering them after the external project, so
    # a parallel build compiles them against a directory that is not there yet.
    # Order every library after it rather than naming them one at a time.
    IF (NEKTAR_USE_SIMSYCL)
        ADD_DEPENDENCIES(${NEKLIB_TARGET_NAME} simsycl-${SIMSYCL_VERSION})
    ENDIF()

    # Infer component name from lower-case library name, variables should use
    # upper-case.
    STRING(TOLOWER ${NEKLIB_TARGET_NAME} NEKLIB_COMPONENT)
    STRING(TOUPPER ${NEKLIB_TARGET_NAME} NEKLIB_COMPVAR)

    # Add name to a list so that we know for constructing dependencies.
    SET(NEKTAR++_LIBRARIES ${NEKTAR++_LIBRARIES} ${NEKLIB_TARGET_NAME} CACHE INTERNAL "")

    SET_PROPERTY(TARGET ${NEKLIB_TARGET_NAME} PROPERTY FOLDER ${NEKLIB_COMPONENT})
    SET_PROPERTY(TARGET ${NEKLIB_TARGET_NAME} PROPERTY VERSION ${NEKTAR_VERSION})
    # Output name is always ${name}, even if a suffix was used for the target name
    SET_PROPERTY(TARGET ${NEKLIB_TARGET_NAME} PROPERTY OUTPUT_NAME ${name})

    SET_COMMON_PROPERTIES(${NEKLIB_TARGET_NAME})

    INSTALL(TARGETS ${NEKLIB_TARGET_NAME}
        EXPORT Nektar++Libraries
        RUNTIME DESTINATION ${NEKTAR_BIN_DIR} COMPONENT ${NEKLIB_COMPONENT} OPTIONAL
        ARCHIVE DESTINATION ${NEKTAR_LIB_DIR} COMPONENT ${NEKLIB_COMPONENT} OPTIONAL
        LIBRARY DESTINATION ${NEKTAR_LIB_DIR} COMPONENT ${NEKLIB_COMPONENT} OPTIONAL)

    # Headers always installed in <include_dir>/${name} rather than <include_dir>/${NEKLIB_TARGET_NAME}
    FOREACH(HEADER ${NEKLIB_HEADERS})
        STRING(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
        INSTALL(FILES ${HEADER}
            DESTINATION ${NEKTAR_INCLUDE_DIR}/${name}/${DIR}
            COMPONENT dev)
    ENDFOREACH()

    # If we have dependencies then link against them.
    IF(NEKLIB_DEPENDS)
        IF (NEKLIB_LINK_MODE)
            string(TOLOWER "${NEKLIB_LINK_MODE}" link_mode)
        ELSE()
            set(link_mode "public")
        ENDIF()

        IF(link_mode STREQUAL "private")
            TARGET_LINK_LIBRARIES(${NEKLIB_TARGET_NAME} PRIVATE ${NEKLIB_DEPENDS})
        ELSEIF(link_mode STREQUAL "interface")
            TARGET_LINK_LIBRARIES(${NEKLIB_TARGET_NAME} INTERFACE ${NEKLIB_DEPENDS})
        ELSEIF(link_mode STREQUAL "public")
            TARGET_LINK_LIBRARIES(${NEKLIB_TARGET_NAME} LINK_PUBLIC ${NEKLIB_DEPENDS})
        ELSE()
            message(FATAL_ERROR "ADD_NEKTAR_LIBRARY: Unknown link mode [${NEKLIB_LINK_MODE}] for [${NEKLIB_TARGET_NAME}]")
        ENDIF()
    ENDIF()
ENDMACRO()

#
# ADD_NEKTAR_TEST(name [LENGTHY])
#
# Adds a test with a given name.  The Test Definition File should be in a
# subdirectory called Tests relative to the CMakeLists.txt file calling this
# macros. The test file should be called NAME.tst, where NAME is given as a
# parameter to this macro. If the LENGTHY flag is given, the test will only be
# run if `NEKTAR_TEST_ALL` is enabled.
#
# Arguments:
#   - `name`: name of the test file
#   - `LENGTHY`: denotes a test that requires extended runtime.
#
MACRO(ADD_NEKTAR_TEST name)
    CMAKE_PARSE_ARGUMENTS(NEKTEST "LENGTHY" "" "" ${ARGN})

    IF ((NEKTAR_BUILD_TESTS) AND (NOT NEKTEST_LENGTHY OR NEKTAR_TEST_ALL))
        GET_FILENAME_COMPONENT(dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)

        SET(test_file ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${name}.tst)

        ADD_TEST(NAME ${dir}_${name}
            COMMAND Tester ${test_file})

        # Read the test file and extract <processes> value
        IF(EXISTS ${test_file})
            FILE(READ ${test_file} test_contents)

            SET(num_procs 1)

            # A .tst file may hold several <test> elements inside a <tests>
            # root, each with its own <processes>. They run one after another
            # within the single CTest test, so the cores this test needs is the
            # largest of them, not the first one we happen to match.
            STRING(REGEX MATCHALL "<processes>[ \t\r\n]*([0-9]+)[ \t\r\n]*</processes>"
                   matches ${test_contents})

            FOREACH(match ${matches})
                STRING(REGEX REPLACE "<processes>[ \t\r\n]*([0-9]+)[ \t\r\n]*</processes>"
                       "\\1" this_procs ${match})
                IF(this_procs GREATER num_procs)
                    SET(num_procs ${this_procs})
                ENDIF()
            ENDFOREACH()
        ELSE()
            SET(num_procs 1)
        ENDIF()

        # Set the number of processor cores required for this test
        SET_PROPERTY(TEST ${dir}_${name} PROPERTY PROCESSORS "${num_procs}")

        # Prioritise tests by parallelism so parallel tests get executed earlier
        SET_PROPERTY(TEST ${dir}_${name} PROPERTY COST "${num_procs}")
    ENDIF()
ENDMACRO(ADD_NEKTAR_TEST)

#
# ADD_NEKTAR_PERFORMANCE_TEST
#
# Adds a performance test with a given name.  The Test Definition File should be in a
# subdirectory called Tests relative to the CMakeLists.txt file calling this
# macros. The test file should be called NAME.tst, where NAME is given as a
# parameter to this macro.
#
# Arguments:
#   - `name`: name of the test file
#
MACRO(ADD_NEKTAR_PERFORMANCE_TEST name)
    IF (NEKTAR_BUILD_PERFORMANCE_TESTS)
        GET_FILENAME_COMPONENT(dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
        ADD_TEST(NAME ${dir}_${name}
            COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${name}.tst)
    ENDIF()
ENDMACRO(ADD_NEKTAR_PERFORMANCE_TEST)

#
# ADD_NEKPY_LIBRARY(name SOURCES src1 src2 ...)
#
# Adds a new NekPy library with the given sources.
#
MACRO(ADD_NEKPY_LIBRARY name)
    CMAKE_PARSE_ARGUMENTS(NEKPY "" "DEPENDS;LIBDEPENDS" "SOURCES" ${ARGN})

    # Create library.
    ADD_LIBRARY(_${name} SHARED ${NEKPY_SOURCES})

    # Python requires a .so extension, even on OS X.
    SET_TARGET_PROPERTIES(_${name} PROPERTIES PREFIX "")
    SET_TARGET_PROPERTIES(_${name} PROPERTIES SUFFIX ".so")
    SET_TARGET_PROPERTIES(_${name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${NEKPY_BASE_DIR}/NekPy/${name})

    # Add target link libraries.
    TARGET_LINK_LIBRARIES(_${name} ${Python3_LIBRARIES})

    # Make sure we build/install pybind11
    ADD_DEPENDENCIES(_${name} pybind11)

    IF (NEKPY_LIBDEPENDS)
        TARGET_LINK_LIBRARIES(_${name} ${NEKPY_LIBDEPENDS})
    ELSE()
        TARGET_LINK_LIBRARIES(_${name} ${name})
    ENDIF()

    # Install __init__.py files.
    SET(TMPOUT "")
    IF (NEKPY_DEPENDS)
        SET(TMPOUT "from ..${NEKPY_DEPENDS} import _${NEKPY_DEPENDS}\n")
    ENDIF()
    SET(TMPOUT "${TMPOUT}from ._${name} import *")

    FILE(WRITE ${NEKPY_BASE_DIR}/NekPy/${name}/__init__.py ${TMPOUT})
ENDMACRO()

#
# ADD_NEKPY_EXECUTABLE(name source)
#
# Adds a NekPy Python-based executable. No compilation is obviously required
# for Python scripts, so this macro simply copies the given file into the
# appropriate install directory.
#
MACRO(ADD_NEKPY_EXECUTABLE name source)
    # Copy the files into binary directory.
    INSTALL(FILES ${source} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    FILE(COPY ${source} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    ADD_CUSTOM_TARGET(${name} SOURCES ${source})
ENDMACRO()

#
# ADD_NEKPY_TEST(name)
#
# Adds a NekPy test with a given name. The Test Definition File should be in a
# subdirectory called Tests relative to the CMakeLists.txt file calling this
# macros. The test file should be called NAME.tst, where NAME is given as a
# parameter to this macro.
#
# Arguments:
#   - `name`: name of the test file
#
MACRO(ADD_NEKPY_TEST name)
    GET_FILENAME_COMPONENT(dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    ADD_TEST(NAME NekPy_${dir}_${name}
             COMMAND Tester ${CMAKE_CURRENT_SOURCE_DIR}/Tests/${name}.tst)
ENDMACRO(ADD_NEKPY_TEST)

# Device Support
MACRO(SET_UNIT_TEST testname name execspace implementation mpi_flag)
    IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
        SET(name-ext "${name}-g")
    ELSEIF (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        SET(name-ext "${name}-ms")
    ELSEIF (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        SET(name-ext "${name}-rg")
    ELSE()
        SET(name-ext "${name}")
    ENDIF()

    IF (${mpi_flag} AND NEKTAR_USE_MPI)
        # A parallel run partitions its session file into a run/<session>_xml
        # directory beside the mesh. Tests sharing a working directory then
        # write the same partition files at once under a parallel ctest, and a
        # rank reading one half-written stops with "Unable to find 'NEKTAR' XML
        # node" while its peers wait in the trace exchange. Give each parallel
        # test a directory, and its own copy of the meshes to partition.
        SET(unit_test_dir ${CMAKE_CURRENT_BINARY_DIR}/mpi-${testname})
        FILE(COPY ${CMAKE_SOURCE_DIR}/library/UnitTests/run
             DESTINATION ${unit_test_dir})

        # mpiexec needs the binary's path, and only the generator knows it:
        # the -g/-ms/-rg suffix is a per-configuration postfix, Windows adds
        # .exe, and a multi-configuration generator puts the file in a
        # configuration subdirectory. $<TARGET_FILE> is all of that resolved.
        ADD_TEST(
          NAME Operators${testname}_MPI
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:${name}>
                  --detect_memory_leaks=0 -- ${execspace} ${implementation}
          WORKING_DIRECTORY ${unit_test_dir}
        )
    ELSE()
        ADD_TEST(
          NAME Operators${testname} 
          COMMAND ${name-ext} --detect_memory_leaks=0 -- ${execspace} ${implementation})
    ENDIF()
ENDMACRO()

