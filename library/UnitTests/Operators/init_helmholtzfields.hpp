///////////////////////////////////////////////////////////////////////////////
//
// File: init_helmholtzfields.hpp
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

#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class HelmholtzField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff>
{
public:
    HelmholtzField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0;
                 n < fixt_in->GetNumComponents() * fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
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

        // Get lambda from session or default to 10.0
        m_lambda = session->DefinesParameter("Lambda")
                       ? session->GetParameter("Lambda")
                       : 10.0;

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = HelmholtzOp<double>::Create(fixt_explist[0]);
        op->SetLambda(m_lambda);
        op->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int compSize =
            fixt_in->GetNumComponents() * fixt_in->GetNumHomoModes();
        const size_t ncoeffs =
            fixt_explist[0]->GetNcoeffs() / fixt_in->GetNumHomoModes();

        StdRegions::FactorMap factors;
        factors[StdRegions::eFactorLambda] = m_lambda;

        Array<OneD, double> incoeffs = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(compSize * ncoeffs);
        Array<OneD, double> tmp;

        for (unsigned int i = 0; i < compSize; ++i)
        {
            size_t e      = 0;
            size_t offset = i * ncoeffs;
            for (const auto &block : fixt_expected->GetBlocks())
            {
                auto nmTot = fixt_explist[0]->GetExp(e)->GetNcoeffs();
                for (size_t el = 0; el < block.GetNumElements(); ++el)
                {
                    StdRegions::StdMatrixKey mkey(
                        StdRegions::eHelmholtz,
                        fixt_explist[0]->GetExp(e)->DetShapeType(),
                        *(fixt_explist[0]->GetExp(e)), factors);
                    fixt_explist[0]->GetExp(e)->GeneralMatrixOp(
                        incoeffs + offset, tmp = outcoeffs + offset, mkey);
                    e++;
                    offset += nmTot;
                }
            }
        }
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }

private:
    double m_lambda;
};

#define TEST(type, filename)                                                   \
    class type : public HelmholtzField                                         \
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

TEST(TriNodal, "run/tri_nodal.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(PrismNodal, "run/prism_nodal.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(TetNodal, "run/tet_nodal.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
