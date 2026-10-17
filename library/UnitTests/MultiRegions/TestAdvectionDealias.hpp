///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvectionDealias.hpp
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
// Description: Test fixture for AdvectionDealiasOp.
//
///////////////////////////////////////////////////////////////////////////////

#include <cmath>

#include <MultiRegions/ExpListHomogeneous1D.h>

#include "TestOp.hpp"

#include <MultiRegions/ElmtOps/AdvectionDealias/AdvectionDealiasOp.hpp>

#include <LibUtilities/Foundations/PhysGalerkinProject.h>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

template <typename TData>
class TestAdvectionDealias
    : public TestOp<TData, FieldState::Phys, FieldState::Phys>
{
public:
    TestAdvectionDealias() = default;

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

    void RunTestCase(bool append)
    {
        auto advelblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto vel =
            Field<TData, FieldState::Phys>("vel", advelblockAttr, m_nVelComp,
                                           this->fixt_in->GetNumHomoModes());
        vel.template CopyArray<NektarSpaces::HostSpace>(m_vel);

        if (append)
        {
            auto outSize = this->fixt_out->GetNumComponents() *
                           this->fixt_explist->GetTotPoints();
            Array<OneD, TData> offsetArr(outSize, m_appendOffset);
            this->fixt_out->template CopyArray<NektarSpaces::HostSpace>(
                offsetArr);
        }

        auto op = AdvectionDealiasOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        op->SetAdvVel(vel);
        op->SetAppend(append);
        op->Apply(*this->fixt_in, *this->fixt_out);
        ExpectedSolution(append);
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
        // A 3DH1 expansion has a real third coordinate although its planes
        // are two-dimensional, and the initial condition has to vary along it
        // for the z terms to be exercised at all.
        const bool isHomogeneous = this->fixt_in->GetNumHomoModes() > 1;
        if (coordim == 1)
        {
            Vmath::Fill(totpoints, 1.0, y, 1);
            Vmath::Fill(totpoints, 1.0, z, 1);
        }
        else if (coordim == 2 && !isHomogeneous)
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
                                        // The homogeneous direction carries
                                        // a Fourier basis over a unit
                                        // period, so it is driven by a mode
                                        // it can represent. A monomial in z
                                        // is not periodic: its Fourier
                                        // derivative rings, which would make
                                        // this a test of how alike the two
                                        // sides' Gibbs noise is rather than
                                        // of the operator.
                                        const TData zfac =
                                            isHomogeneous
                                                ? std::cos(2.0 * M_PI * k *
                                                           z[pts])
                                                : std::pow(z[pts], k);
                                        tmp += std::pow(x[pts], i) *
                                               std::pow(y[pts], j) * zfac;
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

        // Set advection velocity. The operator reads one velocity per
        // coordinate direction on every plane, so a component spans them all.
        // A 3DH1 expansion wants a third component along the homogeneous
        // direction, which its two-dimensional planes do not count.
        const unsigned int nhomo = this->fixt_in->GetNumHomoModes();
        m_coordDim               = this->fixt_explist->GetCoordim(0);
        m_nVelComp               = (nhomo == 1) ? m_coordDim : 3u;
        m_velCompStride          = this->fixt_explist->GetTotPoints();

        m_vel = Array<OneD, TData>(m_velCompStride * m_nVelComp, 1.0);
        unsigned int count = 0;
        for (unsigned int i = 0; i < m_nVelComp; i++)
        {
            for (size_t j = 0; j < m_velCompStride; j++)
            {
                m_vel[count] = i + 1;
                count += 1;
            }
        }
    }

    // ExpList only dispatches this operation for 2D and 3D expansions.
    void PhysGalerkinProjection1DScaledDimAware(const TData scale,
                                                Array<OneD, TData> &in,
                                                Array<OneD, TData> &out)
    {
        if (this->fixt_explist->GetExp(0)->GetShapeDimension() == 1)
        {
            int cnt = 0, cnt1 = 0;
            for (int e = 0; e < this->fixt_explist->GetExpSize(); ++e)
            {
                auto exp = this->fixt_explist->GetExp(e);
                int pt0  = exp->GetNumPoints(0);
                int npt0 = (int)(pt0 * scale);

                LibUtilities::PointsKey newPointsKey0(npt0,
                                                      exp->GetPointsType(0));

                LibUtilities::PhysGalerkinProject1D(
                    newPointsKey0, &in[cnt], exp->GetBasis(0)->GetPointsKey(),
                    &out[cnt1]);

                cnt += npt0;
                cnt1 += pt0;
            }
        }
        else
        {
            this->fixt_explist->PhysGalerkinProjection1DScaled(scale, in, out);
        }
    }

    void ExpectedSolution(bool append)
    {
        // Calculate the expected result using ExpList primitives.
        const unsigned int numComp = this->fixt_in->GetNumComponents();
        const size_t nphys         = this->fixt_explist->GetTotPoints();

        // On 3DH1 the dealiasing is in the plane only, so the fine grid is
        // one plane's scaled points on each plane. Get1DScaledTotPoints()
        // counts the expansions of a single plane, which is what
        // ExpListHomogeneous1D's own interpolation strides by.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist);
        const unsigned int nhomo = this->fixt_in->GetNumHomoModes();
        const bool is3DH1        = homoExpList && nhomo > 1;

        const size_t nphys1D =
            is3DH1 ? nhomo * homoExpList->GetPlane(0)->Get1DScaledTotPoints(1.5)
                   : this->fixt_explist->Get1DScaledTotPoints(1.5);

        // The gradient of a 3DH1 field has three directions although its
        // planes carry two, and the velocity has a component for each.
        const unsigned int ndir = is3DH1 ? 3u : m_coordDim;

        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outphys(numComp * nphys, 0.0);

        Array<OneD, TData> grad0(nphys), grad1(nphys), grad2(nphys);
        Array<OneD, TData> grad0Fine(nphys1D), grad1Fine(nphys1D),
            grad2Fine(nphys1D);
        Array<OneD, TData> velFine(ndir * nphys1D);
        Array<OneD, TData> combinedFine(nphys1D);
        Array<OneD, TData> combinedNative(nphys), tmp;

        for (unsigned int c = 0; c < ndir; ++c)
        {
            this->fixt_explist->PhysInterp1DScaled(
                1.5, m_vel + c * m_velCompStride, tmp = velFine + c * nphys1D);
        }

        for (unsigned int i = 0; i < numComp; i++)
        {
            Vmath::Zero(nphys, grad0, 1);
            Vmath::Zero(nphys, grad1, 1);
            Vmath::Zero(nphys, grad2, 1);
            // Get PhysDeriv or GradU for the ith compoment. On 3DH1 the
            // input is in physical space while SetExpList3DH1 leaves
            // WaveSpace=true, so it is turned off around the call: otherwise
            // PhysDeriv takes the planes for Fourier coefficients and skips
            // the transforms its z-derivative needs.
            if (is3DH1)
            {
                this->fixt_explist->SetWaveSpace(false);
            }

            this->fixt_explist->PhysDeriv(inphys + i * nphys, grad0, grad1,
                                          grad2);

            if (is3DH1)
            {
                this->fixt_explist->SetWaveSpace(true);
            }

            this->fixt_explist->PhysInterp1DScaled(1.5, grad0, tmp = grad0Fine);
            if (ndir >= 2)
            {
                this->fixt_explist->PhysInterp1DScaled(1.5, grad1,
                                                       tmp = grad1Fine);
            }
            if (ndir == 3)
            {
                this->fixt_explist->PhysInterp1DScaled(1.5, grad2,
                                                       tmp = grad2Fine);
            }

            for (size_t j = 0; j < nphys1D; ++j)
            {
                TData val = grad0Fine[j] * velFine[j];
                if (ndir >= 2)
                {
                    val += grad1Fine[j] * velFine[j + nphys1D];
                }
                if (ndir == 3)
                {
                    val += grad2Fine[j] * velFine[j + 2 * nphys1D];
                }
                combinedFine[j] = val;
            }

            PhysGalerkinProjection1DScaledDimAware(1.5, combinedFine,
                                                   tmp = combinedNative);

            Vmath::Vcopy(nphys, combinedNative, 1, tmp = outphys + i * nphys,
                         1);
        }

        if (append)
        {
            Vmath::Sadd(numComp * nphys, m_appendOffset, outphys, 1, outphys,
                        1);
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outphys);
    }

protected:
    unsigned int m_coordDim;
    /// Velocity components: one per coordinate direction, or three on a 3DH1
    /// expansion, whose planes do not count the homogeneous direction.
    unsigned int m_nVelComp = 0;
    /// Step from one velocity component to the next: every plane of one.
    size_t m_velCompStride = 0;
    Array<OneD, TData> m_vel;
    static constexpr TData m_appendOffset = TData(3.0);
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestAdvectionDealias<float>                     \
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
    class type : public TestAdvectionDealias<double>                           \
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

// Exercise point-varying derivative factors.
TEST(QuadDeformed, "run/Helmholtz2D_Quad.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(TriNodal, "run/tri_nodal.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(HexVarP, "run/hex_varp.xml")

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

template <typename TData>
class TestAdvectionDealiasFFT : public TestAdvectionDealias<TData>
{
public:
    TestAdvectionDealiasFFT() = default;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TEST_FFTFLOAT(type, filename)                                          \
    class type##float : public TestAdvectionDealiasFFT<float>                  \
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
    class type : public TestAdvectionDealiasFFT<double>                        \
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
