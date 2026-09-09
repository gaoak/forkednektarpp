///////////////////////////////////////////////////////////////////////////////
//
// File: init_ipwrtphysnormderivtrace.hpp
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
// Description: Fixture for the IProductWRTPhysNormalDerivTrace operator.
//
///////////////////////////////////////////////////////////////////////////////

#include "init_fields.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"
#include "Operators/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceOp.hpp"
#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

/**
 * The operator under test evaluates, for every expansion mode phi of every
 * element,
 *
 *     < dphi/dn , g >_dOmega
 *
 * where g is a scalar carried on the element local traces.
 *
 * Taking g as the restriction to dOmega of a smooth volume function G, the
 * divergence theorem applied to the vector field G grad(phi) gives
 *
 *     < dphi/dn , G >_dOmega  =  int_Omega div( G grad(phi) )
 *                             =  < grad(phi), grad(G) >_Omega
 *                              + < laplacian(phi), G >_Omega
 *
 * so the expected value is built entirely from volume quantities. The first
 * term is the existing IProductWRTDerivBase operator applied to the vector
 * field grad(G). The second term needs a second derivative of the basis,
 * which no operator provides, so it is evaluated directly on the legacy
 * expansion mode by mode.
 *
 * Neither term touches a trace normal, a trace Jacobian or a collapsed
 * coordinate deriv factor, so this reference is genuinely independent of the
 * geometric factors the operator consumes.
 *
 * G is taken to be linear, G = sum_d (d+1) x_d, so that G grad(phi) is a
 * polynomial the element quadrature integrates exactly and the identity
 * therefore holds discretely as well as continuously.
 */
template <typename TData>
class IproductWRTPhysNormDerivTraceField
    : public InitFields<TData, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::DisContField>
{
private:
    Field<TData, FieldState::Phys> *phys = nullptr;

    // Coefficients of the linear function G = sum_d m_gcoeff[d] * x_d.
    static NekDouble GCoeff(unsigned d)
    {
        return static_cast<NekDouble>(d + 1);
    }

public:
    IproductWRTPhysNormDerivTraceField()
        : InitFields<TData, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::DisContField>()
    {
    }

    void Configure(bool divtest = false)
    {
        this->SetSession();
        this->SetExpList();
        this->fixt_explist->SetDataWarehouse();
        if (divtest)
        {
            SetFixtureImpl(1, 1, 1);
        }
        else
        {
            SetFixture(1);
        }
    }

    void SetFixture(const unsigned int nhomo) override
    {
        SetFixtureImpl(nhomo);
    }

    // The divergence test drives a single scalar field in and out, whereas
    // the standard case takes one component per session variable. Zero for
    // either count means "take it from the session variables".
    void SetFixtureImpl(const unsigned int nhomo, const unsigned p_nin = 0,
                        const unsigned p_nout = 0)
    {
        unsigned nin, nout;
        nin  = (p_nin == 0) ? this->session->GetVariables().size() : p_nin;
        nout = (p_nout == 0) ? this->session->GetVariables().size() : p_nout;
        auto inblockAttr = GetLocTraceBlockAttributes<TData, FieldState::Phys>(
            this->fixt_explist);
        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(this->fixt_explist);

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", inblockAttr, nin, nhomo);
        auto f_out =
            Field<TData, FieldState::Coeff>("f_out", outblockAttr, nout, nhomo);

        auto f_expected = Field<TData, FieldState::Coeff>(
            "f_expected", outblockAttr, nout, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Coeff>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Coeff>(std::move(f_expected));

        // set up physical space intermediate soln.
        auto physblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto phys_tmp = Field<TData, FieldState::Phys>(
            "phys_tmp", physblockAttr, nout, nhomo);
        phys = new Field<TData, FieldState::Phys>(std::move(phys_tmp));
    }

    /**
     * Fill a volume Phys field with G at the element quadrature points.
     */
    template <typename TAttr>
    void FillVolumeG(Field<TData, FieldState::Phys> &fld,
                     const TAttr &blockAttr)
    {
        const auto dim = this->fixt_explist->GetCoordim(0);

        size_t eid = 0;
        for (unsigned blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto numElmts = blockAttr[blk].GetNumElements();
            auto numData  = blockAttr[blk].GetNumData();

            auto ptr =
                fld.GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned e = 0; e < numElmts; ++e)
            {
                auto expPtr = this->fixt_explist->GetExp(eid + e);
                auto coords = expPtr->GetCoords();

                for (unsigned q = 0; q < numData; ++q)
                {
                    TData val = 0.0;
                    for (unsigned d = 0; d < dim; ++d)
                    {
                        val += static_cast<TData>(GCoeff(d) * coords[d][q]);
                    }
                    ptr[q] = val;
                }
                ptr += numData;
            }
            eid += numElmts;
        }
    }

    void SetTestCaseDivTest()
    {
        // Build g on the element local traces by extracting the trace values
        // of the volume function G with the PhysTraceExtract operator. Going
        // through the operator framework rather than filling the loc trace
        // buffer by hand keeps the input in exactly the layout and the
        // element local trace orientation that the operator expects, which
        // for a quad means edges 2 and 3 run along decreasing xi rather than
        // in their own edge direction.
        auto physblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);

        auto Gvol = Field<TData, FieldState::Phys>("Gvol", physblockAttr, 1, 1);
        Gvol.template Initialize<NektarSpaces::HostSpace>(TData(0.0));

        FillVolumeG(Gvol, physblockAttr);

        auto extractOp = PhysTraceExtractOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        extractOp->Apply(Gvol, *this->fixt_in);

        // Compute expected solution.
        ExpectedSolutionDivTest();
    }

    void RunTestCase(bool nonCollocated = false)
    {
        auto op = IProductWRTPhysNormalDerivTraceOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        if (nonCollocated)
        {
            op->SetIsCollocated(false);
        }

        auto iProdOp = IProductWRTBaseOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        iProdOp->SetIntegration(false); // just want B^T not B^T W

        // < dphi/dn, g > evaluated into the nodal phys representation
        op->Apply(*this->fixt_in, *this->phys);

        // transform to Coeff space using B^T
        iProdOp->Apply(*this->phys, *this->fixt_out);
    }

    /**
     * The vector-input lift against the scalar one: with g_k = n_k g the
     * per-direction lift sum_k <dphi/dx_k, n_k g> must equal <dphi/dn, g>
     * exactly - same factors, same quadrature - which exercises the
     * uncontracted per-direction factor sets and the ApplyVector staging on
     * every shape, deformed and reoriented cases included. The normals are
     * indexed straight against the loc-trace point order, which is the same
     * assumption the scalar factor creator makes when it contracts.
     */
    void RunVectorIdentityCase(const TData tol)
    {
        const unsigned dim = this->fixt_explist->GetExp(0)->GetNumBases();

        // Scalar reference into fixt_expected.
        auto op = IProductWRTPhysNormalDerivTraceOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        auto iProdOp = IProductWRTBaseOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        iProdOp->SetIntegration(false);

        op->Apply(*this->fixt_in, *this->phys);
        iProdOp->Apply(*this->phys, *this->fixt_expected);

        // Vector input g_k = n_k g in the loc-trace layout.
        auto attrs = GetLocTraceBlockAttributes<TData, FieldState::Phys>(
            this->fixt_explist);
        Field<TData, FieldState::Phys> vecIn("vecIn", attrs, dim, 1);
        vecIn.template Initialize<NektarSpaces::HostSpace>(TData(0.0));

        size_t eid = 0;
        for (unsigned blk = 0; blk < attrs.size(); ++blk)
        {
            const auto numElmts = attrs[blk].GetNumElements();
            const auto numData  = attrs[blk].GetNumData();
            const auto compSize = vecIn.GetBlocks()[blk].CompSize();

            auto gptr =
                this->fixt_in->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto vptr =
                vecIn.GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned e = 0; e < numElmts; ++e)
            {
                auto expPtr = this->fixt_explist->GetExp(eid + e);
                auto shape  = expPtr->DetShapeType();

                unsigned offset = 0;
                for (unsigned dir = 0; dir < dim; ++dir)
                {
                    // Per-trace point count, computed exactly as
                    // GetLocTraceBlockAttributes lays the storage out.
                    unsigned ndata = 1;
                    for (unsigned d1 = 0, cnt = 0; d1 < dim; ++d1)
                    {
                        if (d1 == dir)
                        {
                            continue;
                        }
                        ndata *= expPtr->GetTraceBasisKey(dim - 1 - dir, cnt++)
                                     .GetNumPoints();
                    }

                    const auto ntr =
                        LibUtilities::ShapeTypeNumTraceInDir[shape][dir];
                    for (unsigned n = 0; n < ntr; ++n)
                    {
                        const auto traceId =
                            LibUtilities::ShapeTypeTraceIDInDir[shape][dir][n];
                        const auto &norm = expPtr->GetTraceNormal(traceId);

                        for (unsigned k = 0; k < dim; ++k)
                        {
                            const auto &nk     = norm[k];
                            const size_t nlast = nk.size() - 1;
                            for (unsigned pt = 0; pt < ndata; ++pt)
                            {
                                vptr[k * compSize + e * numData + offset + pt] =
                                    static_cast<TData>(
                                        nk[std::min<size_t>(pt, nlast)]) *
                                    gptr[e * numData + offset + pt];
                            }
                        }
                        offset += ndata;
                    }
                }
            }
            eid += numElmts;
        }

        // The vector apply, through the same B^T, into fixt_out.
        op->ApplyVector(vecIn, *this->phys);
        iProdOp->Apply(*this->phys, *this->fixt_out);

        BOOST_TEST(this->Compare(tol));
    }

    void ExpectedSolutionDivTest()
    {
        const auto dim = this->fixt_explist->GetCoordim(0);

        auto blockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);

        // grad(G) is constant, one component per Cartesian direction.
        auto GradG = Field<TData, FieldState::Phys>("GradG", blockAttr, dim, 1);
        GradG.template Initialize<NektarSpaces::HostSpace>(TData(0.0));

        for (unsigned blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto numElmts = blockAttr[blk].GetNumElements();
            auto numData  = blockAttr[blk].GetNumData();
            auto compSize = blockAttr[blk].CompSize();

            auto ptr =
                GradG.GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // Leave the padding elements zero, as the other fixtures do.
            for (unsigned e = 0; e < numElmts; ++e)
            {
                for (unsigned d = 0; d < dim; ++d)
                {
                    Vmath::Fill(numData, TData(GCoeff(d)), ptr + d * compSize,
                                1);
                }
                ptr += numData;
            }
        }

        // < grad(phi), grad(G) >
        std::vector<std::string> Xstr(dim);
        auto iProdDerivOp =
            IProductWRTDerivBaseOp<FieldState::Coeff, TData>::Create(
                this->fixt_explist, Xstr);
        iProdDerivOp->SetAppend(false);
        iProdDerivOp->Apply(GradG, *this->fixt_expected);

        // + < laplacian(phi), G >
        AddLaplacianOfBasisTerm();
    }

    /**
     * Add < laplacian(phi_p), G >_Omega to the expected coefficients, mode by
     * mode, straight off the legacy expansion.
     */
    void AddLaplacianOfBasisTerm()
    {
        const auto dim = this->fixt_explist->GetCoordim(0);

        size_t eid = 0;
        for (unsigned blk = 0; blk < this->fixt_expected->GetBlocks().size();
             ++blk)
        {
            auto &block      = this->fixt_expected->GetBlocks()[blk];
            auto numElmts    = block.GetNumElements();
            auto numElmtsPad = block.GetNumElementsWithPadding();
            auto numDataOut  = block.GetNumData();

            auto outptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

            for (unsigned e = 0; e < numElmts; ++e)
            {
                auto exp = this->fixt_explist->GetExp(eid + e);

                const unsigned nq = exp->GetTotPoints();
                const unsigned nc = exp->GetNcoeffs();

                ASSERTL1(nc <= numDataOut,
                         "Element has more coefficients than the block holds");

                // G at the element quadrature points
                auto coords = exp->GetCoords();
                Array<OneD, NekDouble> G(nq, 0.0);
                for (unsigned d = 0; d < dim; ++d)
                {
                    Vmath::Svtvp(nq, GCoeff(d), coords[d], 1, G, 1, G, 1);
                }

                Array<OneD, NekDouble> phi(nq), lap(nq), tmp(nq);
                Array<OneD, NekDouble> d0(nq), d1(nq), d2(nq);

                for (unsigned p = 0; p < nc; ++p)
                {
                    exp->FillMode(p, phi);

                    Vmath::Zero(nq, lap, 1);

                    if (dim == 1)
                    {
                        exp->PhysDeriv(phi, d0);
                        exp->PhysDeriv(0, d0, tmp);
                        Vmath::Vadd(nq, tmp, 1, lap, 1, lap, 1);
                    }
                    else if (dim == 2)
                    {
                        exp->PhysDeriv(phi, d0, d1);
                        exp->PhysDeriv(0, d0, tmp);
                        Vmath::Vadd(nq, tmp, 1, lap, 1, lap, 1);
                        exp->PhysDeriv(1, d1, tmp);
                        Vmath::Vadd(nq, tmp, 1, lap, 1, lap, 1);
                    }
                    else
                    {
                        exp->PhysDeriv(phi, d0, d1, d2);
                        exp->PhysDeriv(0, d0, tmp);
                        Vmath::Vadd(nq, tmp, 1, lap, 1, lap, 1);
                        exp->PhysDeriv(1, d1, tmp);
                        Vmath::Vadd(nq, tmp, 1, lap, 1, lap, 1);
                        exp->PhysDeriv(2, d2, tmp);
                        Vmath::Vadd(nq, tmp, 1, lap, 1, lap, 1);
                    }

                    Vmath::Vmul(nq, G, 1, lap, 1, tmp, 1);

                    outptr[p] += static_cast<TData>(exp->Integral(tmp));
                }

                outptr += numDataOut;
            }

            eid += numElmts;
            outptr += numDataOut * (numElmtsPad - numElmts);
        }
    }

private:
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public IproductWRTPhysNormDerivTraceField<float>       \
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
    class type : public IproductWRTPhysNormDerivTraceField<double>             \
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

TEST(QuadOrtho, "run/square_ortho.xml")

TEST(QuadOrthoGauss, "run/square_ortho_gauss_p10.xml")

TEST(TriOrtho, "run/tri.xml")

TEST(QuadAffine, "run/square_affine.xml")

TEST(QuadAffineCurved, "run/square_affine_curved.xml")

TEST(QuadAffineGauss, "run/square_affine_gauss.xml")

TEST(QuadAffineCurvedGauss, "run/square_affine_curved_gauss.xml")

TEST(HexOrientVarQ, "run/hex_orient_varq.xml")

TEST(HexAffine, "run/hex_affine.xml")

TEST(HexAffineGauss, "run/hex_affine_gauss.xml")

TEST(Hex, "run/hex_fixedp.xml")

TEST(Prism, "run/prism.xml")

TEST(Pyr, "run/pyr.xml")

// Disabled with ipwrtNormDerivTrace_tet_p1; see
// test_ipwrtphysnormderivtrace.cpp.
// TEST(TetP1, "run/tet_p1.xml")

TEST(Seg, "run/segment.xml")

TEST(SegCurvedAffine, "run/segment_curved_affine.xml")

TEST(SegCurved, "run/segment_curved.xml")

TEST(TetP2, "run/tet_p2.xml")

TEST(TetP2Gauss, "run/tet_p2_gauss.xml")

TEST(TetOrtho, "run/tet_ortho.xml")

TEST(TetOrthoGauss, "run/tet_ortho_gauss.xml")
