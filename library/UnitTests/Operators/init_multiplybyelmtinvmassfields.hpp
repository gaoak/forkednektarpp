///////////////////////////////////////////////////////////////////////////////
//
// File: init_multiplybyelmtinvmassfields.hpp
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

#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TData>
class MultiplyByElmtInvMassField
    : public InitFields<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    MultiplyByElmtInvMassField()
        : InitFields<TData, FieldState::Coeff, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
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
                        inptr[cnt] = coeff + n;
                    }
                }
                inptr += block.CompSize();
            }
        }
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = MultiplyByElmtInvMassOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        unsigned int numComp = this->fixt_in->GetNumComponents();
        size_t ncoeffs       = this->fixt_explist->GetNcoeffs();

        Array<OneD, TData> incoeffs = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(ncoeffs * numComp), tmp;

        for (unsigned int i = 0; i < numComp; ++i)
        {
            this->fixt_explist->MultiplyByElmtInvMass(
                incoeffs + i * ncoeffs, tmp = outcoeffs + i * ncoeffs);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public MultiplyByElmtInvMassField<float>               \
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
    class type : public MultiplyByElmtInvMassField<double>                     \
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
