///////////////////////////////////////////////////////////////////////////////
//
// File: TestMaxStdVelocity.hpp
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
// Description: Fixture for the MaxStdVelocity operator. Self contained: it
// owns its session, expansion list and input field, and takes the expected
// value from the legacy per-element formula as an independent oracle.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/CFL/MaxStdVelocity/MaxStdVelocityOp.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <UnitTests/TestBoostSetup.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;
using namespace Nektar::SolverCore;

struct GlobalConfiguration
{
    // The execution space string, captured once up front: MPI_Init below is
    // handed the live argv and may rewrite it, and reading argv[1] per test
    // without a bounds check is undefined when the argument is missing. The
    // bounds check in the constructor turns that into a usage message
    // instead.
    static std::string &ExecStr()
    {
        static std::string s;
        return s;
    }

    GlobalConfiguration()
    {
        int argc    = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        if (argc > 1)
        {
            ExecStr() = argv[1];
        }

#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif

        // The execution space must be present: exe -- ExecName. Anything
        // less gets the usage message.
        if (argc < 2)
        {
            int rank = 0;

#ifdef NEKTAR_USE_MPI
            MPI_Comm comm = MPI_COMM_WORLD;
            MPI_Comm_rank(comm, &rank);
#endif
            if (rank == 0)
            {
                std::string execname(argv[0]);

                std::cerr << "Usage: " << execname << " -- ExecName"
                          << std::endl;
#if defined(NEKTAR_ENABLE_DEVICE) && defined(NEKTAR_ENABLE_SIMD)
                std::cerr << "\t ExecName = Serial, AVX, Device" << std::endl;
#elif defined(NEKTAR_ENABLE_DEVICE)
                std::cerr << "\t ExecName = Serial, Device" << std::endl;
#elif defined(NEKTAR_ENABLE_SIMD)
                std::cerr << "\t ExecName = Serial, AVX" << std::endl;
#else
                std::cerr << "\t ExecName = Serial" << std::endl;
#endif
            }
#ifdef NEKTAR_USE_MPI
            MPI_Finalize();
#endif
            exit(1);
        }
    }

    ~GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
    }
};

#if defined(BOOST_TEST_NO_MAIN)

bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    GlobalConfiguration gc;

    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else
BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);
#endif

/**
 * @class TestMaxStdVelocity
 *
 * @brief Compares MaxStdVelocityOp against the legacy formula.
 *
 * The expected value is computed here from the legacy expansion interface -
 * GetGeomFactors()->GetDerivFactors() element by element - rather than from
 * the packed factors the operator reads out of the data warehouse. The two
 * reach the same numbers by different routes, so the comparison checks the
 * packing and its indexing, the deformed and regular layouts, the exclusion
 * of padded elements and the per block order weighting, on every mesh the
 * fixture offers.
 *
 * The field carries one velocity component per coordinate direction followed
 * by a wave speed, and the operator is asked for both weightings: zero, the
 * incompressible and scalar advection case, and one, where the wave speed
 * contributes as a sound speed does.
 *
 * The fixture constructor (destructor) is called before (after) each call to
 * the BOOST_FIXTURE_TEST_CASE(<test name>, <fixture>) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */
template <typename TData> class TestMaxStdVelocity
{
public:
    TestMaxStdVelocity()  = default;
    ~TestMaxStdVelocity() = default;

    /**
     * @brief Read the session, build the expansion list and allocate the
     * field the test cases work on.
     */
    void Configure()
    {
        SetSession();
        SetExpList();
        m_expList->SetDataWarehouse();
        SetFixture();
    }

    /**
     * @brief Fill the velocity components and the wave speed, and record the
     * value the operator is expected to return.
     */
    void SetTestCase()
    {
        const auto nphys   = m_expList->GetTotPoints();
        const auto coordim = m_expList->GetCoordim(0);

        Array<OneD, double> x(nphys);
        Array<OneD, double> y(nphys, 0.0);
        Array<OneD, double> z(nphys, 0.0);
        m_expList->GetCoords(x, y, z);

        // A velocity that varies over the mesh, so the maximum is not
        // attained everywhere and a mistake in which point or element is
        // examined shows up. The wave speed is kept strictly positive, as a
        // physical sound speed is.
        for (unsigned int blk = 0; blk < m_in.GetBlocks().size(); ++blk)
        {
            auto &inblock = m_in.GetBlocks()[blk];
            auto inptr =
                inblock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int n = 0; n < m_in.GetNumComponents(); ++n)
            {
                for (size_t el = 0, cnt = 0; el < inblock.GetNumElements();
                     ++el)
                {
                    for (unsigned int phys = 0; phys < inblock.GetNumData();
                         ++phys, ++cnt)
                    {
                        const auto xc = static_cast<TData>(x[cnt]);
                        const auto yc = static_cast<TData>(y[cnt]);
                        const auto zc = static_cast<TData>(z[cnt]);

                        if (n < coordim)
                        {
                            inptr[cnt] = static_cast<TData>(0.7 + 0.3 * n) +
                                         static_cast<TData>(0.4) * xc -
                                         static_cast<TData>(0.2) * yc +
                                         static_cast<TData>(0.1) * zc * zc;
                        }
                        else
                        {
                            inptr[cnt] = static_cast<TData>(1.3) +
                                         static_cast<TData>(0.2) * xc * xc +
                                         static_cast<TData>(0.05) * yc;
                        }
                    }
                }
                inptr += inblock.CompSize();
            }
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    /**
     * @brief Apply the operator with and without the wave speed, once on the
     * non-interleaved storage and once on the interleaved storage.
     */
    void RunTestCase()
    {
        auto op = MaxStdVelocityOp<TData>::Create(m_expList,
                                                  m_session->GetVariables());

        // Non-interleaved case.
        op->SetSoundSpeedFactor(TData(0));
        m_testNoSound = op->Apply(m_in);

        op->SetSoundSpeedFactor(TData(1));
        m_testWithSound = op->Apply(m_in);

        // Interleaved case.
        std::string execStr = Operator<TData>::GetOpExecSpace(m_session);
        m_in.ReshapeStorage(NektarSpaces::GetVectorWidth<TData>(execStr),
                            execStr);

        op->SetSoundSpeedFactor(TData(0));
        m_testNoSound2 = op->Apply(m_in);

        op->SetSoundSpeedFactor(TData(1));
        m_testWithSound2 = op->Apply(m_in);
    }

    /**
     * @brief Compare both weightings and both storage layouts against the
     * expected values, with relative tolerance tol.
     *
     * @return bool
     */
    bool Compare(const TData tol)
    {
        bool match = true;

        const auto report = [&match, tol](const std::string &desc,
                                          const TData actual,
                                          const TData expected) {
            const auto diff = std::abs(actual - expected);
            const auto rel =
                diff / (std::max)(std::abs(expected), static_cast<TData>(1));
            if (rel > tol)
            {
                std::cout << desc << " mismatch: actual=" << actual
                          << ", expected=" << expected << ", rel diff=" << rel
                          << std::endl;
                match = false;
            }
        };

        report("Non-interleaved case without wave speed", m_testNoSound,
               m_expectedNoSound);
        report("Non-interleaved case with wave speed", m_testWithSound,
               m_expectedWithSound);
        report("Interleaved case without wave speed", m_testNoSound2,
               m_expectedNoSound);
        report("Interleaved case with wave speed", m_testWithSound2,
               m_expectedWithSound);

        // The wave speed can only add to the transport rate, and the test
        // velocity is not uniform, so the two must actually differ - a
        // factor that was silently ignored would otherwise pass.
        if (!(m_testWithSound > m_testNoSound))
        {
            std::cout << "Wave speed made no difference: with="
                      << m_testWithSound << ", without=" << m_testNoSound
                      << std::endl;
            match = false;
        }

        return match;
    }

protected:
    std::string m_meshName = "";
    SessionReaderSharedPtr m_session;
    ExpListSharedPtr m_expList;
    Field<TData, FieldState::Phys> m_in;

    TData m_testNoSound       = 0;
    TData m_testWithSound     = 0;
    TData m_testNoSound2      = 0;
    TData m_testWithSound2    = 0;
    TData m_expectedNoSound   = 0;
    TData m_expectedWithSound = 0;

    void SetSession()
    {
        std::string execStr(GlobalConfiguration::ExecStr());

        BOOST_TEST_MESSAGE("Creating input field");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 3;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = strdup(m_meshName.data());
        argv[2]     = strdup(("--opExecSpace=" + execStr).c_str());

        m_session = SessionReader::CreateInstance(argc, argv);

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;
    }

    void SetExpList()
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(m_session);
        m_expList  = MemoryManager<ExpList>::AllocateSharedPtr(
            m_session, graph, true, "u", Collections::eNoCollection);
    }

    void SetFixture()
    {
        // One component per coordinate direction, plus the wave speed. This
        // is the operator's input contract, and is unrelated to the number of
        // variables the session happens to declare.
        const unsigned int ncomp = m_expList->GetCoordim(0) + 1;
        auto blockAttr = GetBlockAttributes<TData, FieldState::Phys>(m_expList);

        m_in = Field<TData, FieldState::Phys>("f_in", blockAttr, ncomp, 1);
    }

    /**
     * @brief The oracle: the legacy computation, element by element on the
     * host.
     *
     * Mirrors CompressibleFlowSystem::v_GetMaxStdVelocity, including its
     * indexing of the derivative factors and its treatment of the wave speed
     * as acting equally in every direction, then applies the order weight the
     * operator folds in per block.
     */
    void ExpectedSolution()
    {
        const auto coordim = m_expList->GetCoordim(0);
        const auto expdim  = m_expList->GetExp(0)->GetShapeDimension();
        const auto nElmts  = m_expList->GetExpSize();

        Array<OneD, TData> in = m_in.template ToArray<TData>();
        const auto nphys      = m_expList->GetTotPoints();

        m_expectedNoSound   = 0;
        m_expectedWithSound = 0;

        for (size_t el = 0; el < nElmts; ++el)
        {
            auto exp          = m_expList->GetExp(el);
            const auto offset = m_expList->GetPhys_Offset(el);
            const auto nqTot  = exp->GetTotPoints();
            const auto &gmat  = exp->GetGeomFactors()->GetDerivFactors();
            const bool deformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;

            const TData order =
                (std::max)(static_cast<TData>(exp->EvalBasisNumModesMax()) -
                               static_cast<TData>(1),
                           static_cast<TData>(1));
            const TData weight = order * order;

            for (unsigned int q = 0; q < nqTot; ++q)
            {
                TData sqNoSound   = 0;
                TData sqWithSound = 0;

                for (unsigned int i = 0; i < expdim; ++i)
                {
                    TData stdVel   = 0;
                    TData stdSound = 0;
                    for (unsigned int j = 0; j < coordim; ++j)
                    {
                        const TData g = static_cast<TData>(
                            gmat[expdim * j + i][deformed ? q : 0]);
                        stdVel += g * in[j * nphys + offset + q];
                        stdSound += g;
                    }

                    const TData c = in[coordim * nphys + offset + q];

                    const TData velNoSound = std::abs(stdVel);
                    const TData velSound =
                        std::abs(stdVel) + std::abs(stdSound * c);

                    sqNoSound += velNoSound * velNoSound;
                    sqWithSound += velSound * velSound;
                }

                m_expectedNoSound   = (std::max)(m_expectedNoSound,
                                               std::sqrt(sqNoSound) * weight);
                m_expectedWithSound = (std::max)(
                    m_expectedWithSound, std::sqrt(sqWithSound) * weight);
            }
        }
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestMaxStdVelocity<float>                       \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            m_meshName = filename;                                             \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public TestMaxStdVelocity<double>                             \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            m_meshName = filename;                                             \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(Seg, "run/segment.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(Tri, "run/tri.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(Prism, "run/prism.xml")

TEST(Tet, "run/tet.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")

#include <UnitTests/TestBoostTeardown.hpp>
