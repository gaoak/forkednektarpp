///////////////////////////////////////////////////////////////////////////////
//
// File: init_advectiondealias.hpp
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

#include "init_fields.hpp"

#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasOp.hpp"

#include <LibUtilities/Foundations/PhysGalerkinProject.h>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;
using namespace Nektar::MultiRegions;

template <typename TData>
class AdvectionDealiasField
    : public InitFields<TData, FieldState::Phys, FieldState::Phys>
{
public:
    AdvectionDealiasField()
        : InitFields<TData, FieldState::Phys, FieldState::Phys>()
    {
    }

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
        auto vel = Field<TData, FieldState::Phys>("vel", advelblockAttr,
                                                  m_coordDim, 1);
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

        // Set advection velocity
        size_t nphys = this->fixt_explist->GetTotPoints() /
                       this->fixt_in->GetNumHomoModes();
        m_coordDim         = this->fixt_explist->GetCoordim(0);
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
        const size_t nphys1D = this->fixt_explist->Get1DScaledTotPoints(1.5);

        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outphys(numComp * nphys, 0.0);

        Array<OneD, TData> grad0(nphys), grad1(nphys), grad2(nphys);
        Array<OneD, TData> grad0Fine(nphys1D), grad1Fine(nphys1D),
            grad2Fine(nphys1D);
        Array<OneD, TData> velFine(m_coordDim * nphys1D);
        Array<OneD, TData> combinedFine(nphys1D);
        Array<OneD, TData> combinedNative(nphys), tmp;

        for (int c = 0; c < m_coordDim; ++c)
        {
            this->fixt_explist->PhysInterp1DScaled(1.5, m_vel + c * nphys,
                                                   tmp = velFine + c * nphys1D);
        }

        for (unsigned int i = 0; i < numComp; i++)
        {
            Vmath::Zero(nphys, grad0, 1);
            Vmath::Zero(nphys, grad1, 1);
            Vmath::Zero(nphys, grad2, 1);
            // Get PhysDeriv or GradU for the ith compoment
            this->fixt_explist->PhysDeriv(inphys + i * nphys, grad0, grad1,
                                          grad2);

            this->fixt_explist->PhysInterp1DScaled(1.5, grad0, tmp = grad0Fine);
            if (m_coordDim >= 2)
            {
                this->fixt_explist->PhysInterp1DScaled(1.5, grad1,
                                                       tmp = grad1Fine);
            }
            if (m_coordDim == 3)
            {
                this->fixt_explist->PhysInterp1DScaled(1.5, grad2,
                                                       tmp = grad2Fine);
            }

            for (size_t j = 0; j < nphys1D; ++j)
            {
                TData val = grad0Fine[j] * velFine[j];
                if (m_coordDim >= 2)
                {
                    val += grad1Fine[j] * velFine[j + nphys1D];
                }
                if (m_coordDim == 3)
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
    Array<OneD, TData> m_vel;
    static constexpr TData m_appendOffset = TData(3.0);
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public AdvectionDealiasField<float>                    \
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
    class type : public AdvectionDealiasField<double>                         \
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
