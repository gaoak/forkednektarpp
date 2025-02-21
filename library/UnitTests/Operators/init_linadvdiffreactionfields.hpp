///////////////////////////////////////////////////////////////////////////////
//
// File: init_linadvdiffreactionfields.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include "init_fields.hpp"

#include "Operators/ElmtOps/OperatorLinAdvDiffReaction.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class LinAdvDiffReactionField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff>
{
public:
    LinAdvDiffReactionField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int nc = 0; nc < fixt_in->GetNumComponents(); ++nc)
            {
                for (unsigned int el = 0, cnt = 0; el < block.GetNumElements();
                     ++el)
                {
                    for (unsigned int coeff = 0; coeff < block.GetNumData();
                         ++coeff, ++cnt)
                    {
                        inptr[cnt] = coeff;
                    }
                }
                inptr += block.size();
            }
        }

        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto LinADR = OperatorLinAdvDiffReaction<double>::Create(fixt_explist);

        // seem to have the negative definitio of lambda implemented currently
        LinADR->SetLambda(-1.0 * m_lambda);
        LinADR->SetAdvVel(m_dim, m_vel);
        LinADR->apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        int compSize = fixt_in->GetNumComponents();
        int ncoeffs  = fixt_explist->GetNcoeffs();
        int nphys    = fixt_explist->GetTotPoints();
        Array<OneD, double> tmp;

        // set advection velocity
        m_dim = fixt_explist->GetCoordim(0);
        m_vel = Array<OneD, double>(nphys * m_dim, 1.0);
        for (int d = 1; d < m_dim; ++d)
        {
            Vmath::Fill(nphys, d + 1.0, tmp = m_vel + d * nphys, 1);
            // Vmath::Fill(nphys, 0.0, tmp = m_vel + d * nphys, 1);
        }

        m_lambda = 1.0;
        StdRegions::FactorMap factors;
        factors[StdRegions::eFactorLambda] = m_lambda;
        Array<OneD, double> incoeffs       = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(compSize * ncoeffs);
        std::vector<StdRegions::VarCoeffType> velCoeffType = {
            StdRegions::eVarCoeffVelX, StdRegions::eVarCoeffVelY,
            StdRegions::eVarCoeffVelZ};

        for (int i = 0; i < compSize; ++i)
        {
            unsigned int e          = 0;
            unsigned int offset     = i * ncoeffs;
            unsigned int physoffset = 0;

            for (const auto &block : fixt_expected->GetBlocks())
            {
                auto nmTot    = fixt_explist->GetExp(e)->GetNcoeffs();
                auto nphysloc = fixt_explist->GetExp(e)->GetTotPoints();

                for (unsigned int el = 0; el < block.GetNumElements(); ++el)
                {
                    // Restrict varcoeffs to size of element
                    StdRegions::VarCoeffMap varcoeffs;

                    for (int d = 0; d < m_dim; ++d)
                    {
                        varcoeffs[velCoeffType[d]] =
                            m_vel + d * nphys + physoffset;
                    }

                    StdRegions::StdMatrixKey mkey(
                        StdRegions::eLinearAdvectionDiffusionReaction,
                        fixt_explist->GetExp(e)->DetShapeType(),
                        *(fixt_explist->GetExp(e)), factors, varcoeffs);

                    fixt_explist->GetExp(e)->GeneralMatrixOp(
                        incoeffs + offset, tmp = outcoeffs + offset, mkey);
                    e++;
                    offset += nmTot;
                    physoffset += nphysloc;
                }
            }
        }
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }

private:
    int m_dim;
    double m_lambda;
    Array<OneD, double> m_vel;
};

#define TEST(type, filename)                                                   \
    class type : public LinAdvDiffReactionField                                \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
