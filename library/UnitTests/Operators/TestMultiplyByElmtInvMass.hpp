///////////////////////////////////////////////////////////////////////////////
//
// File: TestMultiplyByElmtInvMass.hpp
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

#include <MultiRegions/ExpListHomogeneous1D.h>

#include "TestOp.hpp"

#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

template <typename TData>
class TestMultiplyByElmtInvMass
    : public TestOp<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    TestMultiplyByElmtInvMass() = default;

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

        // MultiplyByElmtInvMass is not virtual and builds its block matrix
        // from the expansion list's own elements, so a homogeneous list is
        // driven one plane at a time. The planes of a variable are
        // consecutive within that variable's slice.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist);
        const unsigned int nhomo = this->fixt_in->GetNumHomoModes();

        if (homoExpList && nhomo > 1)
        {
            const size_t planeNcoeffs = ncoeffs / nhomo;
            for (unsigned int i = 0; i < numComp; ++i)
            {
                for (unsigned int pl = 0; pl < nhomo; ++pl)
                {
                    const size_t offset = i * ncoeffs + pl * planeNcoeffs;
                    homoExpList->GetPlane(pl)->MultiplyByElmtInvMass(
                        incoeffs + offset, tmp = outcoeffs + offset);
                }
            }
        }
        else
        {
            for (unsigned int i = 0; i < numComp; ++i)
            {
                this->fixt_explist->MultiplyByElmtInvMass(
                    incoeffs + i * ncoeffs, tmp = outcoeffs + i * ncoeffs);
            }
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestMultiplyByElmtInvMass<float>                \
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
    class type : public TestMultiplyByElmtInvMass<double>                      \
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

template <typename TData>
class TestMultiplyByElmtInvMassFFT : public TestMultiplyByElmtInvMass<TData>
{
public:
    TestMultiplyByElmtInvMassFFT() = default;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TEST_FFTFLOAT(type, filename)                                          \
    class type##float : public TestMultiplyByElmtInvMassFFT<float>                              \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TEST_FFTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TEST_FFTDOUBLE(type, filename)                                         \
    class type : public TestMultiplyByElmtInvMassFFT<double>                                    \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TEST_FFTDOUBLE(type, filename)
#endif
#define TEST_FFT(type, filename)                                               \
    TEST_FFTFLOAT(type, filename)                                              \
    TEST_FFTDOUBLE(type, filename)
// clang-format on

TEST_FFT(QuadFFT, "run/square.xml")
TEST_FFT(TriFFT, "run/tri.xml")
TEST_FFT(SquareAllElementsFFT, "run/square_all_elements.xml")
