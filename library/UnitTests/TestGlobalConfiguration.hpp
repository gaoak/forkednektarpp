///////////////////////////////////////////////////////////////////////////////
//
// File: TestGlobalConfiguration.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: The Boost.Test global fixture shared by every Nektar unit
// test. It starts and stops MPI and MAGMA, and captures the execution space
// and implementation names the test binary was invoked with, so that the
// fixtures can read them back as GlobalConfiguration::ExecStr() and
// GlobalConfiguration::ImplStr().
//
// Include this after TestBoostSetup.hpp and before TestBoostTeardown.hpp:
// it needs the Boost.Test headers the former pulls in, and it tests
// BOOST_TEST_NO_MAIN, which the latter undefines. Then invoke
// NEKTAR_TEST_GLOBAL_CONFIGURATION once, naming the arguments the fixture's
// tests require.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <UnitTests/TestBoostSetup.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#if defined(NEKTAR_USE_MPI)
#include <mpi.h>
#endif

#if defined(NEKTAR_USE_MAGMA)
#include "magma_v2.h"
#endif

namespace Nektar::UnitTests
{

/// The arguments a fixture's tests require, passed to the binary after "--".
enum class TestArgs
{
    None,       ///< exe
    Exec,       ///< exe -- ExecName
    ExecAndImpl ///< exe -- ExecName ImplName
};

/**
 * @struct GlobalConfiguration
 *
 * The Boost.Test global fixture. The constructor runs once before the first
 * test case and the destructor once after the last one, so this is where the
 * runtimes the tests share - MPI and MAGMA - are started and stopped.
 *
 * @tparam Args The arguments the including fixture's tests require. A binary
 * invoked with fewer prints the usage message and exits.
 */
template <TestArgs Args> struct GlobalConfiguration
{
    /// The number of arguments, the executable name included, that are needed.
    static constexpr int RequiredArgc = (Args == TestArgs::ExecAndImpl) ? 3
                                        : (Args == TestArgs::Exec)      ? 2
                                                                        : 1;

    /// The execution space name the binary was invoked with.
    static std::string &ExecStr()
    {
        static std::string s;
        return s;
    }

    /// The implementation name the binary was invoked with.
    static std::string &ImplStr()
    {
        static std::string s;
        return s;
    }

    GlobalConfiguration()
    {
        int argc    = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        // Copy the names out of argv up front. MPI_Init below is handed the
        // live argv and may rewrite it, and reading argv[1]/argv[2] per test
        // without a bounds check is undefined when the arguments are missing
        // - which happens more easily than it sounds, e.g. zsh passes an
        // unquoted $var holding "Serial SumFac" as a single argument. The
        // bounds check below turns that into a usage message instead.
        std::string execname(argv[0]);

        if (argc > 1)
        {
            ExecStr() = argv[1];
        }

        if (argc > 2)
        {
            ImplStr() = argv[2];
        }

#if defined(NEKTAR_USE_MAGMA)
        magma_init();
#endif
#if defined(NEKTAR_USE_MPI)
        MPI_Init(&argc, &argv);
#endif

        if (argc < RequiredArgc)
        {
            PrintUsage(execname);

#if defined(NEKTAR_USE_MPI)
            MPI_Finalize();
#endif
            exit(1);
        }
    }

    ~GlobalConfiguration()
    {
#if defined(NEKTAR_USE_MPI)
        MPI_Finalize();
#endif
#if defined(NEKTAR_USE_MAGMA)
        magma_finalize();
#endif
    }

    /// Writes the accepted arguments to stderr, on the root rank only.
    static void PrintUsage(const std::string &execname)
    {
        int rank = 0;

#if defined(NEKTAR_USE_MPI)
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Comm_rank(comm, &rank);
#endif
        if (rank != 0)
        {
            return;
        }

        std::cerr << "Usage: " << execname << " -- ExecName";

        if constexpr (Args == TestArgs::ExecAndImpl)
        {
            std::cerr << " ImplName";
        }

        std::cerr << std::endl;

#if defined(NEKTAR_ENABLE_DEVICE) && defined(NEKTAR_ENABLE_SIMD)
        std::cerr << "\t ExecName = Serial, AVX, Device" << std::endl;
#elif defined(NEKTAR_ENABLE_DEVICE)
        std::cerr << "\t ExecName = Serial, Device" << std::endl;
#elif defined(NEKTAR_ENABLE_SIMD)
        std::cerr << "\t ExecName = Serial, AVX" << std::endl;
#else
        std::cerr << "\t ExecName = Serial" << std::endl;
#endif

        if constexpr (Args == TestArgs::ExecAndImpl)
        {
#if defined(NEKTAR_ENABLE_DEVICE)
            std::cerr << "\t ImplName = StdMat, SumFac, SumFacTOP" << std::endl;
#else
            std::cerr << "\t ImplName = StdMat, SumFac" << std::endl;
#endif
        }
    }
};

} // namespace Nektar::UnitTests

/**
 * Registers the global fixture with Boost.Test, and names it
 * GlobalConfiguration for the enclosing fixture header. Invoke it once, at
 * file scope, passing the Nektar::UnitTests::TestArgs the tests require.
 *
 * Under BOOST_TEST_NO_MAIN the fixture cannot be registered, so the entry
 * point is defined here instead and constructs it directly.
 */
#if defined(BOOST_TEST_NO_MAIN)

#define NEKTAR_TEST_GLOBAL_CONFIGURATION(args)                                 \
    using GlobalConfiguration = Nektar::UnitTests::GlobalConfiguration<args>;  \
                                                                               \
    inline bool init_function()                                                \
    {                                                                          \
        return true;                                                           \
    }                                                                          \
                                                                               \
    int main(int argc, char *argv[])                                           \
    {                                                                          \
        GlobalConfiguration gc;                                                \
                                                                               \
        return boost::unit_test::unit_test_main(&init_function, argc, argv);   \
    }

#else

#define NEKTAR_TEST_GLOBAL_CONFIGURATION(args)                                 \
    using GlobalConfiguration = Nektar::UnitTests::GlobalConfiguration<args>;  \
    BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration)

#endif
