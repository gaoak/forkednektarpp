///////////////////////////////////////////////////////////////////////////////
//
// File: init_physgalerkinproject1dscaled.hpp
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
// Description: fixture is the mirror image of
// init_physinterp1dscaled.hpp's PhysInterp1DScaledField: input lives on
// the scaled (finer) grid, output on the native grid.
//
///////////////////////////////////////////////////////////////////////////////

#include "init_fields.hpp"

#include "Operators/ElmtOps/PhysGalerkinProject1DScaled/PhysGalerkinProject1DScaledOp.hpp"

#include <LibUtilities/Foundations/PhysGalerkinProject.h>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;
using namespace Nektar::MultiRegions;

template <typename TData>
class PhysGalerkinProject1DScaledField
    : public InitFields<TData, FieldState::Phys, FieldState::Phys>
{
public:
    PhysGalerkinProject1DScaledField()
        : InitFields<TData, FieldState::Phys, FieldState::Phys>()
    {
    }

    void Configure(const TData scale)
    {
        this->scale = scale;
        this->SetSession();
        this->SetExpList();
        this->fixt_explist->SetDataWarehouse();
        this->SetFixture();
    }

    void SetFixture(const unsigned int nhomo = 1) override
    {
        auto nin  = this->session->GetVariables().size();
        auto nout = this->session->GetVariables().size();

        // Native-sized blocks are the *output* of this operator.
        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        std::vector<BlockAttributes<FieldState::Phys>> inblockAttr;

        size_t eid = 0;
        for (unsigned int blk = 0; blk < outblockAttr.size(); ++blk)
        {
            auto expPtr = this->fixt_explist->GetExp(eid);

            unsigned int npts0 = expPtr->GetNumPoints(0);
            unsigned int ndata = 1;
            for (unsigned int d = 0; d < expPtr->GetNumBases(); ++d)
            {
                unsigned int npts = expPtr->GetNumPoints(d);
                ndata *= (npts0 - npts == 1) ? (int)(npts0 * this->scale - 1)
                                             : (int)(npts * this->scale);
            }

            BlockAttributes<FieldState::Phys> new_block(
                outblockAttr[blk].GetNumElements(),
                outblockAttr[blk].GetNumElementsWithPadding(), ndata,
                outblockAttr[blk].GetInterleaveWidth());

            inblockAttr.push_back(new_block);

            eid += outblockAttr[blk].GetNumElements();
        }

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

    void SetTestCase(void)
    {
        // Set initial conditions on the (fine) input grid.
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
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys + n;
                    }
                }
                inptr += block.CompSize();
            }
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = PhysGalerkinProject1DScaledOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        op->SetScaleFactor(this->scale);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        const unsigned int numComp = this->fixt_in->GetNumComponents();
        const size_t nphys         = this->fixt_explist->GetTotPoints();
        const size_t nphys1D =
            this->fixt_explist->Get1DScaledTotPoints(this->scale);

        // Calculate expected result from Nektar++
        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outphys(numComp * nphys), tmp;

        // ExpList::PhysGalerkinProjection1DScaled only implements its 2D
        // and 3D switch cases (see ExpList.cpp:6748), so for 1D/Seg
        // meshes drive the same underlying, dimension-general primitive,
        // LibUtilities::PhysGalerkinProject1D, directly per element -
        // mirroring exactly the per-element loop
        // ExpList::v_PhysGalerkinProjection1DScaled itself uses for its
        // 2D (PhysGalerkinProject2D) and 3D (PhysGalerkinProject3D)
        // cases. This is still an independent legacy reference: it uses
        // the same PointsManager()-cached Galerkin projection matrix
        // machinery that both this new operator and the legacy 2D/3D
        // path (already validated) are built on.
        if (this->fixt_explist->GetExp(0)->GetShapeDimension() == 1)
        {
            for (unsigned int i = 0; i < numComp; ++i)
            {
                int cnt = 0, cnt1 = 0;
                for (int e = 0; e < this->fixt_explist->GetExpSize(); ++e)
                {
                    auto exp = this->fixt_explist->GetExp(e);
                    int pt0  = exp->GetNumPoints(0);
                    int npt0 = (int)(pt0 * this->scale);

                    LibUtilities::PointsKey newPointsKey0(
                        npt0, exp->GetPointsType(0));

                    LibUtilities::PhysGalerkinProject1D(
                        newPointsKey0, &inphys[i * nphys1D + cnt],
                        exp->GetBasis(0)->GetPointsKey(),
                        &outphys[i * nphys + cnt1]);

                    cnt += npt0;
                    cnt1 += pt0;
                }
            }
        }
        else
        {
            // Calculate expected result using the legacy Nektar++
            // Galerkin projection utility
            // (LibUtilities::PhysGalerkinProject2D/3D via
            // ExpList::PhysGalerkinProjection1DScaled), independent of
            // this new operator's implementation.
            for (unsigned int i = 0; i < numComp; ++i)
            {
                this->fixt_explist->PhysGalerkinProjection1DScaled(
                    this->scale, inphys + i * nphys1D,
                    tmp = outphys + i * nphys);
            }
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outphys);
    }

private:
    TData scale;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public PhysGalerkinProject1DScaledField<float>         \
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
    class type : public PhysGalerkinProject1DScaledField<double>               \
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

// NOTE: the homogeneous (3DH1/3DH2) extension is not implemented for
// PhysGalerkinProject1DScaled, so unlike init_physinterp1dscaled.hpp this
// fixture set has no Configure3DH1/Configure3DH2 methods and no
// corresponding *_3dh1/_3dh2 test cases below.

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Seg3D, "run/segment_3D.xml")

TEST(Quad, "run/square.xml")

TEST(Quad3D, "run/square_3D.xml")

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
