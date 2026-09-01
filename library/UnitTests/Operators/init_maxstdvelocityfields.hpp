///////////////////////////////////////////////////////////////////////////////
//
// File: init_maxstdvelocityfields.hpp
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
// Description: Fixture for the MaxStdVelocity operator.
//
///////////////////////////////////////////////////////////////////////////////

#include "init_fields.hpp"

#include "Operators/CFL/MaxStdVelocity/MaxStdVelocityOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

/**
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
 */
template <typename TData>
class MaxStdVelocityField
    : public InitFields<TData, FieldState::Phys, FieldState::Phys>
{
public:
    MaxStdVelocityField()
        : InitFields<TData, FieldState::Phys, FieldState::Phys>()
    {
    }

    void SetFixture(const unsigned int nhomo) override
    {
        // One component per coordinate direction, plus the wave speed.
        const unsigned nin = this->fixt_explist->GetCoordim(0) + 1;

        auto blockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", blockAttr, nin, nhomo);
        auto f_out =
            Field<TData, FieldState::Phys>("f_out", blockAttr, nin, nhomo);
        auto f_expected =
            Field<TData, FieldState::Phys>("f_expected", blockAttr, nin, nhomo);

        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));
    }

    void SetTestCase()
    {
        const auto nphys   = this->fixt_explist->GetTotPoints();
        const auto coordim = this->fixt_explist->GetCoordim(0);

        Array<OneD, double> x(nphys);
        Array<OneD, double> y(nphys, 0.0);
        Array<OneD, double> z(nphys, 0.0);
        this->fixt_explist->GetCoords(x, y, z);

        // A velocity that varies over the mesh, so the maximum is not
        // attained everywhere and a mistake in which point or element is
        // examined shows up. The wave speed is kept strictly positive, as a
        // physical sound speed is.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &inblock = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                inblock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int n = 0; n < this->fixt_in->GetNumComponents(); ++n)
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

    void RunTestCase()
    {
        auto op = MaxStdVelocityOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        // Non-interleaved case.
        op->SetSoundSpeedFactor(TData(0));
        m_testNoSound = op->Apply(*this->fixt_in);

        op->SetSoundSpeedFactor(TData(1));
        m_testWithSound = op->Apply(*this->fixt_in);

        // Interleaved case.
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);
        this->fixt_in->ReshapeStorage(
            NektarSpaces::GetVectorWidth<TData>(execStr), execStr);

        op->SetSoundSpeedFactor(TData(0));
        m_testNoSound2 = op->Apply(*this->fixt_in);

        op->SetSoundSpeedFactor(TData(1));
        m_testWithSound2 = op->Apply(*this->fixt_in);
    }

    bool Compare(const TData tol) override
    {
        bool match = true;

        const auto report = [&match, tol](const std::string &desc,
                                          const TData actual,
                                          const TData expected) {
            const auto diff = std::abs(actual - expected);
            const auto rel =
                diff / std::max(std::abs(expected), static_cast<TData>(1));
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
    TData m_testNoSound       = 0;
    TData m_testWithSound     = 0;
    TData m_testNoSound2      = 0;
    TData m_testWithSound2    = 0;
    TData m_expectedNoSound   = 0;
    TData m_expectedWithSound = 0;

    /**
     * @brief The legacy computation, element by element on the host.
     *
     * Mirrors CompressibleFlowSystem::v_GetMaxStdVelocity, including its
     * indexing of the derivative factors and its treatment of the wave speed
     * as acting equally in every direction, then applies the order weight the
     * operator folds in per block.
     */
    void ExpectedSolution()
    {
        const auto coordim = this->fixt_explist->GetCoordim(0);
        const auto expdim  = this->fixt_explist->GetExp(0)->GetShapeDimension();
        const auto nElmts  = this->fixt_explist->GetExpSize();

        Array<OneD, TData> in = this->fixt_in->template ToArray<TData>();
        const auto nphys      = this->fixt_explist->GetTotPoints();

        m_expectedNoSound   = 0;
        m_expectedWithSound = 0;

        for (size_t el = 0; el < nElmts; ++el)
        {
            auto exp          = this->fixt_explist->GetExp(el);
            const auto offset = this->fixt_explist->GetPhys_Offset(el);
            const auto nqTot  = exp->GetTotPoints();
            const auto &gmat  = exp->GetGeomFactors()->GetDerivFactors();
            const bool deformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;

            const TData order =
                std::max(static_cast<TData>(exp->EvalBasisNumModesMax()) -
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

                m_expectedNoSound =
                    std::max(m_expectedNoSound, std::sqrt(sqNoSound) * weight);
                m_expectedWithSound = std::max(m_expectedWithSound,
                                               std::sqrt(sqWithSound) * weight);
            }
        }
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public MaxStdVelocityField<float>                      \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public MaxStdVelocityField<double>                            \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
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
