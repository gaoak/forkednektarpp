///////////////////////////////////////////////////////////////////////////////
//
// File: TestIProductWRTPhysTrace.hpp
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
// Description: Take the inner product of the trace expansion and put
// its action onto the element
//
///////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "TestOp.hpp"

#include <MultiRegions/ElmtOps/Divergence/DivergenceOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceOp.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

template <typename TData>
class TestIProductWRTPhysTrace
    : public TestOp<TData, FieldState::Phys, FieldState::Coeff,
                    MultiRegions::DisContField>
{
public:
    TestIProductWRTPhysTrace() = default;

private:
    Field<TData, FieldState::Phys> *phys = nullptr;

public:
    /**
     * @brief Build the fixture.
     *
     * @param divtest  Configure for a divergence test: the check of
     *                 SetTestCaseDivTest(), which lifts a linear field over
     *                 the element traces and compares the result with the
     *                 volume integral the divergence theorem gives for it.
     *                 That check needs one component and no homogeneous
     *                 directions, so the fixture is built accordingly.
     */
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

    Array<OneD, double> GetXNOnTrace(unsigned tid,
                                     LocalRegions::ExpansionSharedPtr &exp)
    {
        // Every length here must come from the *local* trace. The normals are
        // local, and the result is written into the local trace buffer, so
        // sizing anything from exp->GetTraceExp() -- the shared trace, which
        // under variable p takes the higher of the two adjacent orders --
        // overruns the normals.
        //
        // Take the coordinates from the aligned trace expansion, which carries
        // this element's own trace basis keys and so is always of local size,
        // then scatter them into the local frame.
        auto stdTrace = exp->GetStdTraceExp(tid);
        auto alignExp = exp->GetAlignedTraceExp(tid);

        auto tnorm   = exp->GetTraceNormal(tid);
        auto tcoords = alignExp->GetCoords();

        const auto dim    = alignExp->GetCoordim();
        const auto ntrace = stdTrace->GetTotPoints();

        ASSERTL1(tnorm[0].size() >= ntrace,
                 "Trace normals are shorter than the local trace.");

        Array<OneD, double> output(ntrace, 0.0);

        const auto orient = exp->GetTraceOrient(tid);
        if (exp->GetShapeDimension() == 3 &&
            orient != StdRegions::eDir1FwdDir1_Dir2FwdDir2)
        {
            Array<OneD, double> tmp(ntrace);
            for (unsigned d = 0; d < dim; ++d)
            {
                exp->ReOrientTracePhysVals(orient, tcoords[d], tmp,
                                           stdTrace->GetNumPoints(0),
                                           stdTrace->GetNumPoints(1), false);
                Vmath::Vcopy(ntrace, tmp, 1, tcoords[d], 1);
            }
        }

        // calculate n.x (with correct sign)
        for (unsigned d = 0; d < dim; ++d)
        {
            Vmath::Vvtvp(ntrace, tcoords[d], 1, tnorm[d], 1, output, 1, output,
                         1);
        }
        return output;
    }

    void SetTestCaseDivTest()
    {
        //  Set initial conditions.
        for (unsigned blk = 0, eid = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block      = this->fixt_in->GetBlocks()[blk];
            auto numElmts    = block.GetNumElements();
            auto numElmtsPad = block.GetNumElementsWithPadding();
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned n = 0; n < this->fixt_in->GetNumComponents() *
                                         this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                unsigned ntracePts = 0;
                for (size_t el = 0; el < numElmts; ++el)
                {
                    // get expansion
                    auto exp = this->fixt_explist->GetExp(eid + el);
                    unsigned npts;
                    auto Shape = exp->DetShapeType();

                    // set up local trace expansions
                    auto ntrace = exp->GetNtraces();
                    std::vector<LocalRegions::ExpansionSharedPtr> TraceExp;
                    TraceExp.resize(ntrace);
                    if (Shape != LibUtilities::Seg) // ignore Segs since method
                                                    // does not exist
                    {
                        for (unsigned t = 0; t < ntrace; ++t)
                        {
                            TraceExp[t] = exp->GetLocTraceExp(t);
                        }
                    }

                    switch (Shape)
                    {
                        case LibUtilities::Seg:
                        {
                            auto XN  = GetXNOnTrace(0, exp);
                            inptr[0] = XN[0];

                            XN       = GetXNOnTrace(1, exp);
                            inptr[1] = XN[0];

                            inptr += 2;
                            ntracePts = 2;
                        }
                        break;
                        case LibUtilities::Quad:
                        {
                            npts = TraceExp[3]->GetTotPoints();

                            // edge 3
                            auto XN = GetXNOnTrace(3, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts = npts;

                            // edge 1
                            XN = GetXNOnTrace(1, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            npts = TraceExp[0]->GetTotPoints();
                            // edge 0
                            XN = GetXNOnTrace(0, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            // edge 2
                            XN = GetXNOnTrace(2, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;
                        }
                        break;
                        case LibUtilities::NodalTri:
                        case LibUtilities::Tri:
                        {
                            npts = TraceExp[2]->GetTotPoints();

                            // edge 2
                            auto XN = GetXNOnTrace(2, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts = npts;

                            // edge 1
                            XN = GetXNOnTrace(1, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            npts = TraceExp[0]->GetTotPoints();
                            // edge 0
                            XN = GetXNOnTrace(0, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;
                        }
                        break;
                        case LibUtilities::Hex:
                        case LibUtilities::NodalPrism:
                        case LibUtilities::Prism:
                        case LibUtilities::Pyr:
                        {
                            npts = TraceExp[4]->GetTotPoints();

                            // face 4
                            auto XN = GetXNOnTrace(4, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts = npts;

                            // face 2
                            XN = GetXNOnTrace(2, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            npts = TraceExp[1]->GetTotPoints();
                            // face 1
                            XN = GetXNOnTrace(1, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            // face 3
                            XN = GetXNOnTrace(3, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            npts = TraceExp[0]->GetTotPoints();
                            // face 0
                            XN = GetXNOnTrace(0, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            if (Shape == LibUtilities::Hex)
                            {
                                // face 5
                                XN = GetXNOnTrace(5, exp);
                                Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                                inptr += npts;
                                ntracePts += npts;
                            }
                        }
                        break;
                        case LibUtilities::NodalTet:
                        case LibUtilities::Tet:
                        {
                            npts = TraceExp[3]->GetTotPoints();

                            // face 3
                            auto XN = GetXNOnTrace(3, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts = npts;

                            // face 2
                            XN = GetXNOnTrace(2, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            npts = TraceExp[1]->GetTotPoints();
                            // face 1
                            XN = GetXNOnTrace(1, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;

                            npts = TraceExp[0]->GetTotPoints();
                            // face 0
                            XN = GetXNOnTrace(0, exp);
                            Vmath::Vcopy(npts, XN.data(), 1, inptr, 1);

                            inptr += npts;
                            ntracePts += npts;
                        }
                        break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "Need to define shape details");
                            break;
                    }
                }
                // offset for padded case
                inptr += ntracePts * (numElmtsPad - numElmts);
            }
            eid += block.GetNumElements();
        }
        // Compute expected solution.
        ExpectedSolutionDivTest();
    }

    void SetTestCase()
    {
        // Set initial conditions.
        for (unsigned blk = 0; blk < this->fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned n = 0; n < this->fixt_in->GetNumComponents() *
                                         this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned phys = 0; phys < block.GetNumData();
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

    void RunTestCase(bool nonCollocated = false, bool append = false)
    {
        auto op = IProductWRTPhysTraceOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        if (nonCollocated)
        {
            op->SetIsCollocated(false);
        }

        auto iProdOp = IProductWRTBaseOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        iProdOp->SetIntegration(false); // just want B^T not B^T W

        // Under append: pre-fill the volume field, lift onto it with
        // the append flag set, then subtract the pre-fill again -- what
        // remains must be the plain lift, so the standard expected
        // solution applies.
        if (append)
        {
            this->phys->template Initialize<NektarSpaces::HostSpace>(
                TData(0.0));
            AddPreFillPhys(TData(1.0));
            op->SetAppend(true);
        }

        // do iproduct WRT Phys trace to phys space
        op->Apply(*this->fixt_in, *this->phys);

        if (append)
        {
            AddPreFillPhys(TData(-1.0));
        }

        // transform to Coeff space using B^T
        iProdOp->Apply(*this->phys, *this->fixt_out);
    }

    // Non-constant pattern used to pre-fill the volume field for the
    // append test; period 17 so no interleave reshape maps it onto
    // itself, magnitude O(1) so the comparison tolerance is unaffected.
    TData PreFillValue(size_t cnt, unsigned n)
    {
        return TData(1.0) + TData(0.01) * TData((cnt + n) % 17);
    }

    // Add sign * pattern to every entry (padding included) of the
    // volume field, on the host.
    void AddPreFillPhys(TData sign)
    {
        for (unsigned blk = 0; blk < this->phys->GetBlocks().size(); ++blk)
        {
            auto &block = this->phys->GetBlocks()[blk];
            auto ptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

            for (unsigned n = 0; n < this->phys->GetNumComponents() *
                                         this->phys->GetNumHomoModes();
                 ++n)
            {
                for (size_t cnt = 0; cnt < block.CompSize(); ++cnt)
                {
                    ptr[cnt] += sign * PreFillValue(cnt, n);
                }
                ptr += block.CompSize();
            }
        }
    }

    // Drive the per-trace entry point over the whole element boundary,
    // one call per trace in packed order.
    //
    // With @p append clear the first call zeroes the volume field and
    // the rest accumulate, which is the protocol a caller assembling
    // the boundary uses; the field is pre-filled first so that the
    // clearing is actually exercised rather than landing on zeros.
    // With @p append set every call accumulates onto a pre-filled
    // field, which is then subtracted again, as the bulk append leg
    // does. Either way what remains is the plain lift, so the standard
    // expected solution applies to both.
    void RunTestCaseSingle(bool nonCollocated = false, bool append = false)
    {
        this->phys->template Initialize<NektarSpaces::HostSpace>(TData(0.0));
        AddPreFillPhys(TData(1.0));

        auto op = IProductWRTPhysTraceOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        if (nonCollocated)
        {
            op->SetIsCollocated(false);
        }

        auto iProdOp = IProductWRTBaseOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        iProdOp->SetIntegration(false); // just want B^T not B^T W

        // This test case will only run on problems of fixed p and same
        // shape
        auto exp = this->fixt_explist->GetExp(0);

        // Traces in packed order: by normal direction, then by position
        // within the direction.
        std::vector<unsigned int> order;
        switch (exp->DetShapeType())
        {
            case LibUtilities::NodalTri:
            case LibUtilities::Tri:
                order = {2, 1, 0};
                break;
            case LibUtilities::Quad:
                order = {3, 1, 0, 2};
                break;
            case LibUtilities::NodalTet:
            case LibUtilities::Tet:
                order = {3, 2, 1, 0};
                break;
            case LibUtilities::NodalPrism:
            case LibUtilities::Prism:
            case LibUtilities::Pyr:
                order = {4, 2, 1, 3, 0};
                break;
            case LibUtilities::Hex:
                order = {4, 2, 1, 3, 0, 5};
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Shape not set up ");
                break;
        }

        // Quadrature points of one trace of the element.
        auto tracePts = [&exp](unsigned int tr) -> unsigned int {
            return (exp->GetShapeDimension() == 2)
                       ? exp->GetTraceBasisKey(tr).GetNumPoints()
                       : exp->GetTraceBasisKey(tr, 0).GetNumPoints() *
                             exp->GetTraceBasisKey(tr, 1).GetNumPoints();
        };

        unsigned int inOffset = 0;
        for (size_t k = 0; k < order.size(); ++k)
        {
            // Clear on the first trace unless the caller is appending;
            // accumulate on the rest either way.
            op->SetAppend(append || k > 0);
            op->IProductWRTPhysTrace(order[k], *this->fixt_in, inOffset,
                                     *this->phys);
            inOffset += tracePts(order[k]);
        }

        if (append)
        {
            AddPreFillPhys(TData(-1.0));
        }

        // transform to Coeff space using B^T
        iProdOp->Apply(*this->phys, *this->fixt_out);
    }

    void ExpectedSolutionDivTest()
    {
        size_t eid = 0;
        auto dim   = this->fixt_explist->GetCoordim(0);

        auto blockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto Xvals =
            Field<TData, FieldState::Phys>("Coordinates", blockAttr, dim, 1);
        Xvals.template Initialize<NektarSpaces::HostSpace>(TData(0.0));

        // fill x-values
        for (unsigned blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto numElmts    = blockAttr[blk].GetNumElements();
            auto numElmtsPad = blockAttr[blk].GetNumElementsWithPadding();
            auto numData     = blockAttr[blk].GetNumData();
            auto compSize    = blockAttr[blk].CompSize();

            auto Xptr =
                Xvals.GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned e = 0; e < numElmts; ++e)
            {
                auto expPtr = this->fixt_explist->GetExp(eid + e);

                auto coords = expPtr->GetCoords();

                for (unsigned d = 0; d < dim; ++d)
                {
                    Vmath::Vcopy(numData, coords[d].data(), 1,
                                 Xptr + d * compSize, 1);
                }

                Xptr += numData;
            }
            eid += numElmts;
            Xptr += numData * (numElmtsPad - numElmts);
        }

        // operators
        std::vector<std::string> Xstr(dim), Ostr(1);
        auto divOp = DivergenceOp<TData>::Create(this->fixt_explist, Xstr);
        auto iProdOp =
            IProductWRTBaseOp<TData>::Create(this->fixt_explist, Ostr);
        auto iProdDerivOpAppend =
            IProductWRTDerivBaseOp<FieldState::Coeff, TData>::Create(
                this->fixt_explist, Xstr);
        iProdDerivOpAppend->SetAppend(true);

        auto phystmp =
            Field<TData, FieldState::Phys>("Phys tmps", blockAttr, 1, 1);

        // Div.X
        divOp->Apply(Xvals, phystmp);
        // (phy,Div.X)
        iProdOp->Apply(phystmp, *this->fixt_expected);
        // (phy,Div.X) + (Dphi,X);
        iProdDerivOpAppend->Apply(Xvals, *this->fixt_expected);
    }

    void ApplyLocalTraceJac(LocalRegions::ExpansionSharedPtr &expPtr,
                            unsigned t,
                            StdRegions::StdExpansionSharedPtr &traceExp,
                            Array<OneD, TData> &phys)
    {
        auto jacAligned =
            expPtr->GetAlignedTraceExp(t)->GetGeomFactors()->GetJac();

        const unsigned npts = traceExp->GetTotPoints();

        if (jacAligned.size() < npts)
        {
            // Regular trace: the Jacobian is constant.
            for (unsigned i = 0; i < npts; ++i)
            {
                phys[i] *= jacAligned[0];
            }
            return;
        }

        const auto orient = expPtr->GetTraceOrient(t);
        Array<OneD, double> jacLocal(npts);

        if (expPtr->GetShapeDimension() == 3 &&
            orient != StdRegions::eDir1FwdDir1_Dir2FwdDir2)
        {
            expPtr->ReOrientTracePhysVals(orient, jacAligned, jacLocal,
                                          traceExp->GetNumPoints(0),
                                          traceExp->GetNumPoints(1), false);
        }
        else
        {
            Vmath::Vcopy(npts, jacAligned, 1, jacLocal, 1);
        }

        for (unsigned i = 0; i < npts; ++i)
        {
            phys[i] *= jacLocal[i];
        }
    }

    void ExpectedSolution()
    {
        const unsigned numComp = this->fixt_in->GetNumComponents();

        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(this->fixt_explist);

        size_t eid = 0;
        for (unsigned blk = 0; blk < outblockAttr.size(); ++blk)
        {
            auto numElmts    = outblockAttr[blk].GetNumElements();
            auto numElmtsPad = outblockAttr[blk].GetNumElementsWithPadding();
            auto numDataOut  = outblockAttr[blk].GetNumData();

            auto inptr =
                this->fixt_in->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            auto outptr =
                this->fixt_expected->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            Array<OneD, TData> phys(
                this->fixt_in->GetBlocks()[blk].GetNumData() / 2);
            Array<OneD, TData> coeff(
                this->fixt_in->GetBlocks()[blk].GetNumData() / 2);

            Array<OneD, unsigned> maparray;

            for (unsigned nc = 0; nc < numComp; ++nc)
            {
                unsigned ntracePts = 0;
                for (unsigned e = 0; e < numElmts; ++e)
                {
                    auto expPtr = this->fixt_explist->GetExp(eid + e);
                    auto Shape  = expPtr->DetShapeType();

                    // set up local trace expansions
                    auto ntrace = expPtr->GetNtraces();
                    std::vector<StdRegions::StdExpansionSharedPtr> TraceExp;
                    TraceExp.resize(ntrace);
                    if (Shape != LibUtilities::Seg) // ignore Segs since
                                                    // method does not exist
                    {
                        for (unsigned t = 0; t < ntrace; ++t)
                        {
                            TraceExp[t] = expPtr->GetStdTraceExp(t);
                        }
                    }
                    switch (Shape)
                    {
                        case LibUtilities::Seg:
                        {
                            expPtr->GetTraceCoeffMap(0, maparray);
                            outptr[maparray[0]] = inptr[0];

                            expPtr->GetTraceCoeffMap(1, maparray);
                            outptr[maparray[0]] = inptr[1];

                            inptr += 2;
                            ntracePts = 2;
                            outptr += numDataOut;
                        }
                        break;
                        case LibUtilities::Quad:
                        {
                            // zero output array for this element block
                            Vmath::Zero(numDataOut, outptr, 1);

                            // edge 3
                            auto npts = TraceExp[3]->GetTotPoints();
                            auto nm   = TraceExp[3]->GetNcoeffs();

                            // extract phys points
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 3, TraceExp[3], phys);
                            TraceExp[3]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(3, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] = coeff[m];
                            }

                            inptr += npts;
                            ntracePts = npts;

                            // edge 1
                            // extract phys points
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);

                            ApplyLocalTraceJac(expPtr, 1, TraceExp[1], phys);
                            TraceExp[1]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(1, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // edge 0
                            npts = TraceExp[0]->GetTotPoints();
                            nm   = TraceExp[0]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);

                            ApplyLocalTraceJac(expPtr, 0, TraceExp[0], phys);
                            TraceExp[0]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(0, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // edge 2

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);

                            ApplyLocalTraceJac(expPtr, 2, TraceExp[2], phys);
                            TraceExp[2]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(2, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            outptr += numDataOut;
                        }
                        break;
                        case LibUtilities::NodalTri:
                        case LibUtilities::Tri:
                        {
                            // zero output array for this element block
                            Vmath::Zero(numDataOut, outptr, 1);

                            // edge 2
                            auto npts = TraceExp[2]->GetTotPoints();
                            auto nm   = TraceExp[2]->GetNcoeffs();

                            // extract phys points
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);

                            ApplyLocalTraceJac(expPtr, 2, TraceExp[2], phys);
                            TraceExp[2]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(2, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] = coeff[m];
                            }

                            inptr += npts;
                            ntracePts = npts;

                            // edge 1
                            // extract phys points
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);

                            ApplyLocalTraceJac(expPtr, 1, TraceExp[1], phys);
                            TraceExp[1]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(1, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // edge 0
                            npts = TraceExp[0]->GetTotPoints();
                            nm   = TraceExp[0]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);

                            ApplyLocalTraceJac(expPtr, 0, TraceExp[0], phys);
                            TraceExp[0]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(0, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            outptr += numDataOut;
                        }
                        break;
                        case LibUtilities::Hex:
                        case LibUtilities::NodalPrism:
                        case LibUtilities::Prism:
                        case LibUtilities::Pyr:
                        {
                            // zero output array for this element block
                            Vmath::Zero(numDataOut, outptr, 1);

                            // face 4
                            auto npts = TraceExp[4]->GetTotPoints();
                            auto nm   = TraceExp[4]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 4, TraceExp[4], phys);
                            TraceExp[4]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(4, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] = coeff[m];
                            }

                            inptr += npts;
                            ntracePts = npts;

                            // face 2
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 2, TraceExp[2], phys);
                            TraceExp[2]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(2, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // face 1
                            npts = TraceExp[1]->GetTotPoints();
                            nm   = TraceExp[1]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 1, TraceExp[1], phys);
                            TraceExp[1]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(1, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // face 3
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 3, TraceExp[3], phys);
                            TraceExp[3]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(3, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // face 0
                            npts = TraceExp[0]->GetTotPoints();
                            nm   = TraceExp[0]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 0, TraceExp[0], phys);
                            TraceExp[0]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(0, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            if (Shape == LibUtilities::Hex)
                            {
                                // face 5
                                auto npts = TraceExp[5]->GetTotPoints();
                                auto nm   = TraceExp[5]->GetNcoeffs();

                                Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                                ApplyLocalTraceJac(expPtr, 5, TraceExp[5],
                                                   phys);
                                TraceExp[5]->IProductWRTBase(phys, coeff);

                                expPtr->GetTraceCoeffMap(5, maparray);
                                for (unsigned m = 0; m < nm; ++m)
                                {
                                    outptr[maparray[m]] += coeff[m];
                                }

                                inptr += npts;
                                ntracePts += npts;
                            }
                            outptr += numDataOut;
                        }
                        break;
                        case LibUtilities::NodalTet:
                        case LibUtilities::Tet:
                        {
                            // zero output array for this element block
                            Vmath::Zero(numDataOut, outptr, 1);

                            // face 3
                            auto npts = TraceExp[3]->GetTotPoints();
                            auto nm   = TraceExp[3]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 3, TraceExp[3], phys);
                            TraceExp[3]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(3, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] = coeff[m];
                            }

                            inptr += npts;
                            ntracePts = npts;

                            // face 2
                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 2, TraceExp[2], phys);
                            TraceExp[2]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(2, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // face 1
                            npts = TraceExp[1]->GetTotPoints();
                            nm   = TraceExp[1]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 1, TraceExp[1], phys);
                            TraceExp[1]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(1, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            // face 0
                            npts = TraceExp[0]->GetTotPoints();
                            nm   = TraceExp[0]->GetNcoeffs();

                            Vmath::Vcopy(npts, inptr, 1, phys.data(), 1);
                            ApplyLocalTraceJac(expPtr, 0, TraceExp[0], phys);
                            TraceExp[0]->IProductWRTBase(phys, coeff);

                            expPtr->GetTraceCoeffMap(0, maparray);
                            for (unsigned m = 0; m < nm; ++m)
                            {
                                outptr[maparray[m]] += coeff[m];
                            }

                            inptr += npts;
                            ntracePts += npts;

                            outptr += numDataOut;
                        }
                        break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "Need to define shape details");
                            break;
                    }
                }
                // offset for padded case
                outptr += numDataOut * (numElmtsPad - numElmts);
                inptr += ntracePts * (numElmtsPad - numElmts);
            }
            eid += numElmts;
        }
    }

private:
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestIProductWRTPhysTrace<float>                 \
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
    class type : public TestIProductWRTPhysTrace<double>                       \
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

TEST(SegGaussPts, "run/segment_gausspts.xml")

TEST(Quad, "run/square.xml")

TEST(QuadOrtho, "run/square_ortho.xml")

TEST(QuadOrthoGauss, "run/square_ortho_gauss.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadGaussPts, "run/square_gausspts.xml")

TEST(Tri, "run/tri.xml")

TEST(TriOrtho, "run/tri.xml")

TEST(TriOrthoGauss, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(TriNodal, "run/tri_nodal.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexFixedP, "run/hex_fixedp.xml")

TEST(HexAffineGauss, "run/hex_affine_gauss.xml")

TEST(Prism, "run/prism.xml")
TEST(PrismTransposedFace, "run/prism_transposed_face.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(PrismFixedP, "run/prism_fixedp.xml")

TEST(Pyr, "run/pyr.xml")
TEST(PyrOrtho, "run/pyr_ortho.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(TetNodal, "run/tet_nodal.xml")

TEST(TetOrtho, "run/tet_ortho.xml")

TEST(TetOrthoGauss, "run/tet_ortho_gauss.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
