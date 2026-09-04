########################################################################
#
# ThirdParty configuration for Nektar++
#
# SimSYCL
#
# SimSYCL (https://github.com/celerity/SimSYCL) is a SYCL *simulator* that
# runs on an ordinary host compiler. It is used as the SYCL-CPU
# implementation so that the device code paths can be developed and tested
# on a machine with no GPU.
#
# This module must be included AFTER ThirdPartyBoost: SimSYCL requires
# Boost.context, and finds Boost in CONFIG mode.
#
########################################################################

IF (NEKTAR_USE_SIMSYCL)
    # SimSYCL has no releases track; use specific commit instead.
    SET(SIMSYCL_VERSION "7053164")
    SET(SIMSYCL_URL_MD5 "a216768ba5152511063a97699eb87973")
    # The snapshot is not on the distribution node yet, so the default names the
    # host it is served from meanwhile. Point this at ${TPURL} once it is, or at
    # a local file:// copy. The md5 is checked whatever the source, so a wrong or
    # truncated download fails loudly rather than half-building.
    SET(SIMSYCL_URL "${TPURL}/simsycl-${SIMSYCL_VERSION}.tar.gz"
        CACHE STRING "URL of the SimSYCL source snapshot.")

    # SimSYCL is a C++20 library in earnest: it uses <source_location> and
    # std::bit_cast throughout its headers, and its own README rules out Clang
    # before 17 over incomplete CTAD, testing GCC 13 on macOS. The libc++ that
    # ships with Xcode 14 has neither header, and the build then fails deep
    # inside SimSYCL's own sources with a page of missing-include errors that
    # never mention the toolchain. Check the standard library up front instead.
    # CMAKE_CXX_STANDARD is already 20 here, and CMP0067 carries it into the
    # check.
    INCLUDE(CheckCXXSourceCompiles)
    CHECK_CXX_SOURCE_COMPILES("
        #include <bit>
        #include <source_location>
        int main()
        {
            (void)std::source_location::current();
            return std::bit_cast<int>(1.0f) == 0;
        }" NEKTAR_CXX20_LIBRARY_FOR_SIMSYCL)

    IF (NOT NEKTAR_CXX20_LIBRARY_FOR_SIMSYCL)
        MESSAGE(FATAL_ERROR
            "SimSYCL needs a C++20 standard library providing "
            "<source_location> and std::bit_cast, and "
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not "
            "provide them. On macOS that means Xcode 15 or newer, or a "
            "MacPorts/Homebrew toolchain: SimSYCL's own CI tests GCC 13 and "
            "Clang 17, and states that Clang before 17 does not work.")
    ENDIF()

    # A Debug build will not LINK without this. SimSYCL wraps every device
    # operation in several layers of templates, and at -O0 none of them
    # inline: each layer emits a standalone out-of-line function, across all
    # ~630 device kernel objects in Operators. The result is more code and
    # more symbols than ld64 survives, and it dies linking
    # libOperators-g.dylib with a bare
    #     c++: error: unable to execute command: Segmentation fault: 11
    # that says nothing about the cause.
    #
    # It is the optimisation level, not the debug info: the link fails
    # identically at -g, at -gline-tables-only and at -g0, where the objects
    # are still 5.2 GB with no DWARF in them at all. -Og restores inlining,
    # collapses the template layers, and links with full -g kept.
    IF (CMAKE_BUILD_TYPE STREQUAL "Debug" AND
        NOT CMAKE_CXX_FLAGS_DEBUG MATCHES "-O")
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og"
            CACHE STRING "Flags used by the CXX compiler during Debug builds."
            FORCE)
        MESSAGE(STATUS
            "SimSYCL: added -Og to Debug flags; -O0 does not link")
    ENDIF()

    # A system or user-supplied install takes precedence, as elsewhere in
    # the ThirdParty modules. SimSYCL exports SimSYCL::simsycl.
    FIND_PACKAGE(SimSYCL CONFIG QUIET)

    IF (SimSYCL_FOUND)
        SET(BUILD_SIMSYCL OFF)
    ELSE()
        SET(BUILD_SIMSYCL ON)
    ENDIF()

    OPTION(THIRDPARTY_BUILD_SIMSYCL
        "Build SimSYCL library from ThirdParty." ${BUILD_SIMSYCL})

    IF (THIRDPARTY_BUILD_SIMSYCL)
        INCLUDE(ExternalProject)

        UNSET(PATCH CACHE)
        FIND_PROGRAM(PATCH patch)
        IF(NOT PATCH)
            MESSAGE(FATAL_ERROR
                "'patch' tool for modifying files not found. Cannot build SimSYCL.")
        ENDIF()
        MARK_AS_ADVANCED(PATCH)

        # The snapshot is packed on macOS, so every entry in it is shadowed
        # by an AppleDouble "._" companion - the top-level directory
        # included. That leaves the archive with two top-level entries, and
        # ExternalProject strips the leading directory only when there is
        # exactly one, so the tree arrives one level below SOURCE_DIR and
        # the patch below has nothing to patch. Flatten it before anything
        # else looks at it. The script is a no-op on a tree that extracted
        # correctly, so it costs nothing if the snapshot is ever repacked.
        SET(SIMSYCL_FLATTEN_COMMAND ${CMAKE_COMMAND}
            -DSRC:PATH=${TPSRC}/simsycl-${SIMSYCL_VERSION}
            -P ${PROJECT_SOURCE_DIR}/cmake/scripts/flatten-source-dir.cmake)

        # Without this patch nothing built against SimSYCL will RUN on macOS
        # - every binary aborts during dynamic initialisation. It is not
        # optional on any platform using libc++.
        SET(SIMSYCL_PATCH_COMMAND ${PATCH} -p1 <
            ${PROJECT_SOURCE_DIR}/cmake/thirdparty-patches/simsycl-macos-static-init-order.patch)

        # SimSYCL asks for Boost in CONFIG mode, which needs BoostConfig.cmake
        # to be installed. Nektar sets Boost_NO_BOOST_CMAKE and asks in MODULE
        # mode, which does not - so a Boost that satisfies Nektar, such as the
        # MacPorts one on the macOS CI runners, can leave the sub-build failing
        # with "Could not find a package configuration file provided by Boost".
        # Drop CONFIG so both halves agree on how Boost is found; FindBoost
        # defines the same Boost::context target that SimSYCL links.
        SET(SIMSYCL_BOOST_PATCH_COMMAND ${PATCH} -p1 <
            ${PROJECT_SOURCE_DIR}/cmake/thirdparty-patches/simsycl-boost-module-mode.patch)

        # SimSYCL pulls its dependencies with FetchContent, which would need
        # network access at build time. The Nektar snapshot bundles them
        # under deps/, and FETCHCONTENT_SOURCE_DIR_<NAME> redirects
        # FetchContent at those copies instead of the network. The names must
        # be upper-cased, as FetchContent requires.
        #
        # All four are needed. Catch2 is pulled by test/CMakeLists.txt, which
        # SimSYCL adds unconditionally - there is no option to disable its
        # tests, so the configure needs Catch2 even though only the simsycl
        # target is built. fmt is pulled by libenvpp from its own bundled
        # submodule; a local path is still a download as far as FetchContent
        # is concerned, so FULLY_DISCONNECTED blocks it unless the source dir
        # is given explicitly.
        SET(SIMSYCL_DEPS ${TPSRC}/simsycl-${SIMSYCL_VERSION}/deps)
        SET(SIMSYCL_FETCHCONTENT_ARGS
            -DFETCHCONTENT_SOURCE_DIR_LIBENVPP:PATH=${SIMSYCL_DEPS}/libenvpp
            -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON:PATH=${SIMSYCL_DEPS}/nlohmann_json
            -DFETCHCONTENT_SOURCE_DIR_CATCH2:PATH=${SIMSYCL_DEPS}/Catch2
            -DFETCHCONTENT_SOURCE_DIR_FMT:PATH=${SIMSYCL_DEPS}/libenvpp/external/fmt)

        # Hand SimSYCL the prefix of the Boost that Nektar is actually using,
        # and make it resolve Boost the same way Nektar does.
        #
        # Which prefix that is cannot be read off one variable. When Nektar
        # builds its own Boost it is ${TPDIST}, and Boost_INCLUDE_DIR is
        # misleading: FindBoost sets it from whatever headers it found - the
        # system ones - even when the run failed because a component was
        # missing, which is precisely why the ThirdParty build was triggered.
        # When Boost is taken from the system, Boost_INCLUDE_DIR is the only
        # thing that records where, since ThirdPartyBoost defaults BOOST_ROOT
        # to ${TPDIST} whether or not it builds anything.
        #
        # Boost_NO_BOOST_CMAKE mirrors what ThirdPartyBoost sets for Nektar,
        # and is not cosmetic: without it FindBoost forwards to BoostConfig,
        # which demands component packages matching its own version exactly.
        # A machine with system Boost 1.83 headers and a ThirdParty Boost 1.82
        # - the CMAKE_INSTALL_PREFIX below puts the latter on the search path -
        # then fails with
        #     Could not find a configuration file for package "boost_context"
        #     that exactly matches requested version "1.83.0"
        SET(SIMSYCL_BOOST_ARGS -DBoost_NO_BOOST_CMAKE:BOOL=ON)
        IF (THIRDPARTY_BUILD_BOOST)
            SET(SIMSYCL_BOOST_PREFIX ${TPDIST})
            LIST(APPEND SIMSYCL_BOOST_ARGS -DBoost_NO_SYSTEM_PATHS:BOOL=ON)
        ELSEIF (Boost_INCLUDE_DIR)
            GET_FILENAME_COMPONENT(SIMSYCL_BOOST_PREFIX
                ${Boost_INCLUDE_DIR} DIRECTORY)
        ELSE()
            SET(SIMSYCL_BOOST_PREFIX ${BOOST_ROOT})
        ENDIF()
        IF (SIMSYCL_BOOST_PREFIX)
            LIST(APPEND SIMSYCL_BOOST_ARGS
                -DBOOST_ROOT:PATH=${SIMSYCL_BOOST_PREFIX}
                -DCMAKE_PREFIX_PATH:PATH=${SIMSYCL_BOOST_PREFIX})
        ENDIF()

        # Defines SIMSYCL_LIBRARY, which BUILD_BYPRODUCTS below relies on
        # being set already.
        THIRDPARTY_LIBRARY(SIMSYCL_LIBRARY SHARED simsycl
            DESCRIPTION "SimSYCL library")

        EXTERNALPROJECT_ADD(
            simsycl-${SIMSYCL_VERSION}
            PREFIX ${TPSRC}
            URL ${SIMSYCL_URL}
            URL_MD5 ${SIMSYCL_URL_MD5}
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            SOURCE_DIR ${TPSRC}/simsycl-${SIMSYCL_VERSION}
            BINARY_DIR ${TPBUILD}/simsycl-${SIMSYCL_VERSION}
            TMP_DIR ${TPBUILD}/simsycl-${SIMSYCL_VERSION}-tmp
            INSTALL_DIR ${TPDIST}
            BUILD_BYPRODUCTS ${SIMSYCL_LIBRARY}
            PATCH_COMMAND ${SIMSYCL_FLATTEN_COMMAND}
            COMMAND ${SIMSYCL_PATCH_COMMAND}
            COMMAND ${SIMSYCL_BOOST_PATCH_COMMAND}
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                ${NEKTAR_EXTERNAL_PROJECT_CMAKE_GENERATOR_ARGS}
                -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
                -DCMAKE_CXX_STANDARD:STRING=20
                # SimSYCL keeps its runtime state, including the USM allocation
                # registry, in namespace-scope globals. Nektar links it into
                # every one of its shared libraries, so a STATIC SimSYCL gives
                # each library a private registry and memory allocated in one is
                # unknown to another when freed. Do not change this to static.
                -DBUILD_SHARED_LIBS:BOOL=ON
                -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                -DCMAKE_BUILD_TYPE:STRING=Release
                ${SIMSYCL_BOOST_ARGS}
                ${SIMSYCL_FETCHCONTENT_ARGS}
                -DCMAKE_INSTALL_PREFIX:PATH=${TPDIST}
                # The loader resolves a shared library's own dependencies through
                # that library's RUNPATH, not through the RUNPATH of the executable
                # - which is the only place Nektar records the ThirdParty lib
                # directory. Without this, libsimsycl.so cannot find the libenvpp
                # sitting right beside it and every binary dies at startup with
                #     error while loading shared libraries: libenvpp.so.1.5.3
                -DCMAKE_INSTALL_RPATH:PATH=${TPDIST}/lib
                ${TPSRC}/simsycl-${SIMSYCL_VERSION}
                -Wno-dev 
            # SimSYCL's test and example targets do not build on macOS, and
            # are of no use to Nektar in any case. Build only the library.
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target simsycl
            INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
                --target install
            COMMAND ${CMAKE_COMMAND} -DBINARY_DIR=<BINARY_DIR>
                -DDEST=${TPDIST}/lib
                -P ${PROJECT_SOURCE_DIR}/cmake/scripts/install-simsycl-deps.cmake
        )

        # The configure step runs find_package(Boost), so it cannot start until
        # a ThirdParty-built Boost has installed. ThirdPartyBoost declares an
        # empty boost target when it builds none, so this is unconditional.
        ADD_DEPENDENCIES(simsycl-${SIMSYCL_VERSION} boost)

        SET(SIMSYCL_INCLUDE_DIRS ${TPDIST}/include CACHE FILEPATH
            "SimSYCL include" FORCE)
        MESSAGE(STATUS "Build SimSYCL: ${SIMSYCL_LIBRARY}")
        SET(SIMSYCL_CONFIG_INCLUDE_DIR ${TPINC})
    ELSE()
        ADD_CUSTOM_TARGET(simsycl-${SIMSYCL_VERSION} ALL)
        GET_TARGET_PROPERTY(SIMSYCL_INCLUDE_DIRS SimSYCL::simsycl
                            INTERFACE_INCLUDE_DIRECTORIES)
        MESSAGE(STATUS "Found SimSYCL: ${SimSYCL_DIR}")
        SET(SIMSYCL_CONFIG_INCLUDE_DIR ${SIMSYCL_INCLUDE_DIRS})
    ENDIF()

    # SimSYCL is header-only *to the compiler* but not to the linker, and
    # unlike AdaptiveCpp or DPC++ it is used with the ordinary compiler
    # rather than a SYCL wrapper. Nothing therefore puts sycl/sycl.hpp on the
    # default include path, and only LibUtilities links NEKTAR_SYCL_DEPENDS -
    # which leaves every other target including SYCLQueue.hpp, NekBlas first
    # among them, unable to compile or link. Put both in scope globally.
    IF (SIMSYCL_INCLUDE_DIRS)
        INCLUDE_DIRECTORIES(${SIMSYCL_INCLUDE_DIRS})
    ENDIF()

    IF (THIRDPARTY_BUILD_SIMSYCL)
        LINK_LIBRARIES(${SIMSYCL_LIBRARY})
        SET(NEKTAR_SYCL_DEPENDS ${SIMSYCL_LIBRARY})
    ELSE()
        LINK_LIBRARIES(SimSYCL::simsycl)
        SET(NEKTAR_SYCL_DEPENDS SimSYCL::simsycl)
    ENDIF()

    ADD_DEPENDENCIES(thirdparty simsycl-${SIMSYCL_VERSION})

    MARK_AS_ADVANCED(SIMSYCL_INCLUDE_DIRS)
    MARK_AS_ADVANCED(SIMSYCL_LIBRARY)
    MARK_AS_ADVANCED(SIMSYCL_URL)
    MARK_AS_ADVANCED(SimSYCL_DIR)
ENDIF()
