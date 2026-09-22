///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvection.hpp
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

#include "TestOp.hpp"

#include "Operators/ElmtOps/Advection/AdvectionOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

template <typename TData>
class TestAdvection : public TestOp<TData, FieldState::Phys, FieldState::Phys>
{
public:
    TestAdvection() = default;

    void SetFixture(const unsigned int nhomo) override
    {
        auto nin  = this->session->GetVariables().size();
        auto nout = this->session->GetVariables().size();
        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", inblockAttr, nin, nhomo);
        auto f_out =
            Field<TData, FieldState::Phys>("f_out", outblockAttr, nout, nhomo);
        auto f_expected = Field<TData, FieldState::Phys>(
            "f_expected", outblockAttr, nout, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));
    }

    void RunTestCase()
    {
        auto advelblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto vel =
            Field<TData, FieldState::Phys>("vel", advelblockAttr, m_coordDim,
                                           this->fixt_in->GetNumHomoModes());
        vel.template CopyArray<NektarSpaces::HostSpace>(m_vel);

        auto op = AdvectionOp<TData>::Create(this->fixt_explist,
                                             this->session->GetVariables());
        op->SetAdvVel(vel);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void SetTestCase()
    {
        // Set initial conditions.
        auto coordim   = this->fixt_explist->GetCoordim(0);
        auto totpoints = this->fixt_explist->GetTotPoints();
        Array<OneD, TData> x(totpoints);
        Array<OneD, TData> y(totpoints);
        Array<OneD, TData> z(totpoints);
        this->fixt_explist->GetCoords(x, y, z);
        if (coordim == 1)
        {
            Vmath::Fill(totpoints, 1.0, y, 1);
            Vmath::Fill(totpoints, 1.0, z, 1);
        }
        else if (coordim == 2)
        {
            Vmath::Fill(totpoints, 1.0, z, 1);
        }

        size_t el = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int n = 0; n < this->fixt_in->GetNumComponents(); ++n)
            {
                size_t pts = 0;
                for (unsigned int m = 0; m < this->fixt_in->GetNumHomoModes();
                     ++m)
                {
                    for (size_t e = 0, cnt = 0; e < block.GetNumElements(); ++e)
                    {
                        // set M[3] to the point in the zero direction
                        // otherwise to the points in the basis direction if
                        // that basis exists
                        unsigned int M[3];
                        M[0] = M[1] = M[2] =
                            this->fixt_explist->GetExp(el)->GetNumPoints(0);
                        for (unsigned int i = 1;
                             i < this->fixt_explist->GetExp(el)->GetNumBases();
                             ++i)
                        {
                            M[i] =
                                this->fixt_explist->GetExp(el)->GetNumPoints(i);
                        }

                        for (unsigned int phys = 0; phys < block.GetNumData();
                             ++phys, ++pts, ++cnt)
                        {
                            TData tmp = 0.0;
                            for (unsigned int i = 0; i < M[0] / 2; i++)
                            {
                                for (unsigned int j = 0; j < M[1] / 2; j++)
                                {
                                    for (unsigned int k = 0; k < M[2] / 2; ++k)
                                    {
                                        tmp += std::pow(x[pts], i) *
                                               std::pow(y[pts], j) *
                                               std::pow(z[pts], k);
                                    }
                                }
                            }
                            inptr[cnt] = tmp;
                        }
                    }
                    inptr += block.CompSize();
                }
            }
            el += block.GetNumElements();
        }

        // Set advection velocity. It carries the same planes as the input,
        // and on a 3DH1 expansion a third component along the homogeneous
        // direction, which the planes themselves do not count.
        const bool is3DH1 =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist) != nullptr;
        size_t nphys       = this->fixt_explist->GetTotPoints();
        m_coordDim         = (is3DH1 && this->fixt_in->GetNumHomoModes() > 1)
                                 ? 3
                                 : this->fixt_explist->GetCoordim(0);
        m_vel              = Array<OneD, TData>(nphys * m_coordDim, 1.0);
        unsigned int count = 0;
        for (int i = 0; i < m_coordDim; i++)
        {
            for (int j = 0; j < nphys; j++)
            {
                m_vel[count] = i + 1;
                count += 1;
            }
        }
        // Compute expected solution.
        ExpectedSolution();
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        const unsigned int numComp = this->fixt_in->GetNumComponents();
        const size_t nphys         = this->fixt_explist->GetTotPoints();
        Array<OneD, TData> inphys  = this->fixt_in->ToArray();
        Array<OneD, TData> outphys = Array<OneD, TData>(nphys * numComp, 0.0);
        Array<OneD, TData> grad0   = Array<OneD, TData>(nphys);
        Array<OneD, TData> grad1   = Array<OneD, TData>(nphys);
        Array<OneD, TData> grad2   = Array<OneD, TData>(nphys);

        const bool is3DH1 =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist) != nullptr;
        const bool wave = is3DH1 && this->fixt_in->GetNumHomoModes() > 1;

        // The input is in physical space, while SetExpList3DH1 leaves
        // WaveSpace true, so it is turned off for the derivatives.
        if (wave)
        {
            this->fixt_explist->SetWaveSpace(false);
        }

        unsigned int count = 0;
        for (unsigned int i = 0; i < numComp; i++)
        {
            Vmath::Zero(nphys, grad0, 1);
            Vmath::Zero(nphys, grad1, 1);
            Vmath::Zero(nphys, grad2, 1);
            // Get PhysDeriv or GradU for the ith compoment
            this->fixt_explist->PhysDeriv(inphys + i * nphys, grad0, grad1,
                                          grad2);

            // Dot Product by advection velocity to Grad(U)
            for (int j = 0; j < nphys; j++)
            {
                outphys[count] = grad0[j] * m_vel[j];
                if (m_coordDim >= 2)
                {
                    outphys[count] += grad1[j] * m_vel[j + nphys];
                }
                if (m_coordDim == 3)
                {
                    outphys[count] += grad2[j] * m_vel[j + 2 * nphys];
                }
                count += 1;
            }
        }

        if (wave)
        {
            this->fixt_explist->SetWaveSpace(true);
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outphys);
    }

protected:
    unsigned int m_coordDim;
    Array<OneD, TData> m_vel;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestAdvection<float>                            \
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
    class type : public TestAdvection<double>                                  \
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

TEST(Seg3D, "run/segment_3D.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(Tri3D, "run/tri_3D.xml")

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

template <typename TData> class TestAdvectionFFT : public TestAdvection<TData>
{
public:
    TestAdvectionFFT() = default;
};

#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TEST_FFTFLOAT(type, filename)                                          \
    class type##float : public TestAdvectionFFT<float>{                        \
        public : type##float(){meshName = filename;                            \
    }                                                                          \
    }                                                                          \
    ;
#else
#define TEST_FFTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TEST_FFTDOUBLE(type, filename)                                         \
    class type : public TestAdvectionFFT<double>                               \
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

TEST_FFT(QuadFFT, "run/square.xml")
TEST_FFT(TriFFT, "run/tri.xml")
TEST_FFT(SquareAllElementsFFT, "run/square_all_elements.xml")
