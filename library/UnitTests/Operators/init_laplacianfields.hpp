///////////////////////////////////////////////////////////////////////////////
//
// File: init_laplacianfields.hpp
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

#include "Operators/ElmtOps/Laplacian/LaplacianOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TData>
class LaplacianField
    : public InitFields<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    LaplacianField() : InitFields<TData, FieldState::Coeff, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < this->fixt_in->GetNumComponents() *
                                             this->fixt_in->GetNumHomoModes();
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
                inptr += block.CompSize();
            }
        }

        // Set up diffusion coefficient.
        const auto coordDim      = this->fixt_explist->GetCoordim(0);
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        // Set up (isotropic) diffusion coefficient.
        if (coordDim == 1)
        {
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (coordDim == 2)
        {
            m_diffCoeff[0] = 2.0; // D00
            m_diffCoeff[2] = 3.0; // D11
        }
        else
        {
            m_diffCoeff[0] = 2.0; // D00
            m_diffCoeff[2] = 3.0; // D11
            m_diffCoeff[5] = 4.0; // D22
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = LaplacianOp<TData>::Create(this->fixt_explist,
                                             this->session->GetVariables());
        op->SetDiffCoeff(m_diffCoeff);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int numComp = this->fixt_in->GetNumComponents() *
                                     this->fixt_in->GetNumHomoModes();
        const size_t ncoeffs =
            this->fixt_explist->GetNcoeffs() / this->fixt_in->GetNumHomoModes();

        // Set up diffusion coefficient.
        StdRegions::FactorMap factors;
        if (this->fixt_explist->GetCoordim(0) == 1)
        {
            factors[StdRegions::eFactorLambda] = 0.0;
        }
        else if (this->fixt_explist->GetCoordim(0) == 2)
        {
            factors[StdRegions::eFactorLambda]   = 0.0;
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
        }
        else
        {
            factors[StdRegions::eFactorLambda]   = 0.0;
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
            factors[StdRegions::eFactorCoeffD02] = m_diffCoeff[3];
            factors[StdRegions::eFactorCoeffD12] = m_diffCoeff[4];
            factors[StdRegions::eFactorCoeffD22] = m_diffCoeff[5];
        }

        Array<OneD, TData> incoeffs = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(numComp * ncoeffs);
        Array<OneD, TData> tmp;

        for (unsigned int i = 0; i < numComp; ++i)
        {
            size_t e      = 0;
            size_t offset = i * ncoeffs;
            for (const auto &block : this->fixt_expected->GetBlocks())
            {
                auto nmTot = this->fixt_explist->GetExp(e)->GetNcoeffs();
                for (size_t el = 0; el < block.GetNumElements(); ++el)
                {
                    StdRegions::StdMatrixKey mkey(
                        StdRegions::eLaplacian,
                        this->fixt_explist->GetExp(e)->DetShapeType(),
                        *(this->fixt_explist->GetExp(e)), factors);
                    this->fixt_explist->GetExp(e)->GeneralMatrixOp(
                        incoeffs + offset, tmp = outcoeffs + offset, mkey);
                    e++;
                    offset += nmTot;
                }
            }
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

private:
    std::vector<TData> m_diffCoeff;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public LaplacianField<float>                           \
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
    class type : public LaplacianField<double>                                 \
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
