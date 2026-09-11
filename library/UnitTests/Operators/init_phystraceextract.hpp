///////////////////////////////////////////////////////////////////////////////
//
// File: init_phystraceextract.hpp
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

#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TData>
class PhystraceextractField
    : public InitFields<TData, FieldState::Phys, FieldState::Phys>
{
public:
    PhystraceextractField()
        : InitFields<TData, FieldState::Phys, FieldState::Phys>()
    {
    }

    void Configure()
    {
        this->SetSession();
        this->SetExpList();
        this->fixt_explist->SetDataWarehouse();
        SetFixture(1);
    }

    void SetFixture(const unsigned int nhomo) override
    {
        auto nin  = this->session->GetVariables().size();
        auto nout = this->session->GetVariables().size();
        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto outblockAttr = GetLocTraceBlockAttributes<TData, FieldState::Phys>(
            this->fixt_explist);

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

    void RunTestCase(bool nonCollocated = false)
    {
        auto op = PhysTraceExtractOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        if (nonCollocated)
        {
            op->SetIsCollocated(false);
        }

        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void RunTestCaseExtractTrace(bool nonCollocated = false)
    {
        auto op = PhysTraceExtractOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());

        if (nonCollocated)
        {
            op->SetIsCollocated(false);
        }

        // This test case will only run on problems of fixed p and same shape
        auto exp = this->fixt_explist->GetExp(0);
        switch (exp->DetShapeType())
        {
            case LibUtilities::NodalTri:
            case LibUtilities::Tri:
            {
                op->ExtractTrace(2, *this->fixt_in, *this->fixt_out, 0);
                auto pts = exp->GetTraceBasisKey(2).GetNumPoints();
                op->ExtractTrace(1, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(1).GetNumPoints();
                op->ExtractTrace(0, *this->fixt_in, *this->fixt_out, pts);
            }
            break;
            case LibUtilities::Quad:
            {
                op->ExtractTrace(3, *this->fixt_in, *this->fixt_out, 0);
                auto pts = exp->GetTraceBasisKey(3).GetNumPoints();
                op->ExtractTrace(1, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(1).GetNumPoints();
                op->ExtractTrace(0, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(0).GetNumPoints();
                op->ExtractTrace(2, *this->fixt_in, *this->fixt_out, pts);
            }
            break;
            case LibUtilities::NodalTet:
            case LibUtilities::Tet:
            {
                op->ExtractTrace(3, *this->fixt_in, *this->fixt_out, 0);
                auto pts = exp->GetTraceBasisKey(3, 0).GetNumPoints() *
                           exp->GetTraceBasisKey(3, 1).GetNumPoints();
                op->ExtractTrace(2, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(2, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(2, 1).GetNumPoints();
                op->ExtractTrace(1, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(1, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(1, 1).GetNumPoints();
                op->ExtractTrace(0, *this->fixt_in, *this->fixt_out, pts);
            }
            break;
            case LibUtilities::NodalPrism:
            case LibUtilities::Prism:
            case LibUtilities::Pyr:
            {
                op->ExtractTrace(4, *this->fixt_in, *this->fixt_out, 0);
                auto pts = exp->GetTraceBasisKey(4, 0).GetNumPoints() *
                           exp->GetTraceBasisKey(4, 1).GetNumPoints();
                op->ExtractTrace(2, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(2, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(2, 1).GetNumPoints();
                op->ExtractTrace(1, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(1, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(1, 1).GetNumPoints();
                op->ExtractTrace(3, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(3, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(3, 1).GetNumPoints();
                op->ExtractTrace(0, *this->fixt_in, *this->fixt_out, pts);
            }
            break;
            case LibUtilities::Hex:
            {
                op->ExtractTrace(4, *this->fixt_in, *this->fixt_out, 0);
                auto pts = exp->GetTraceBasisKey(4, 0).GetNumPoints() *
                           exp->GetTraceBasisKey(4, 1).GetNumPoints();
                op->ExtractTrace(2, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(2, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(2, 1).GetNumPoints();
                op->ExtractTrace(1, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(1, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(1, 1).GetNumPoints();
                op->ExtractTrace(3, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(3, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(3, 1).GetNumPoints();
                op->ExtractTrace(0, *this->fixt_in, *this->fixt_out, pts);
                pts += exp->GetTraceBasisKey(0, 0).GetNumPoints() *
                       exp->GetTraceBasisKey(0, 1).GetNumPoints();
                op->ExtractTrace(5, *this->fixt_in, *this->fixt_out, pts);
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "Shape not set up ");
                break;
        }
    }

    void ExpectedSolution()
    {
        const unsigned numComp = this->fixt_in->GetNumComponents();

        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);

        size_t eid = 0;
        for (unsigned blk = 0; blk < inblockAttr.size(); ++blk)
        {
            auto numElmts    = inblockAttr[blk].GetNumElements();
            auto numElmtsPad = inblockAttr[blk].GetNumElementsWithPadding();
            auto numData     = inblockAttr[blk].GetNumData();

            auto inptr =
                this->fixt_in->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            auto outptr =
                this->fixt_expected->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            Array<OneD, TData> outphys(
                this->fixt_expected->GetBlocks()[blk].GetNumData() / 2);

            for (unsigned nc = 0; nc < numComp; ++nc)
            {
                unsigned physoffset = 0;
                unsigned ntrace     = 0;
                for (unsigned e = 0; e < numElmts; ++e)
                {
                    auto expPtr = this->fixt_explist->GetExp(eid + e);
                    Array<OneD, double> indata(numData, inptr + physoffset);

                    auto Shape = expPtr->DetShapeType();

                    // set up local trace expansions
                    auto nface = expPtr->GetNtraces();
                    std::vector<LocalRegions::ExpansionSharedPtr> TraceExp;
                    TraceExp.resize(nface);
                    if (Shape != LibUtilities::Seg) // ignore Segs since
                                                    // method does not exist
                    {
                        for (unsigned t = 0; t < nface; ++t)
                        {
                            TraceExp[t] = expPtr->GetLocTraceExp(t);
                        }
                    }
                    ntrace = 0;

                    switch (Shape)
                    {
                        case LibUtilities::Seg:
                        {
                            expPtr->GetTracePhysVals(0, expPtr, indata, outphys,
                                                     StdRegions::eForwards);
                            outptr[0] = outphys[0];
                            expPtr->GetTracePhysVals(1, expPtr, indata, outphys,
                                                     StdRegions::eForwards);
                            outptr[1] = outphys[0];
                            outptr += 2;
                            ntrace = 2;
                        }
                        break;
                        case LibUtilities::Quad:
                        {
                            // edge 3
                            auto npts = TraceExp[3]->GetTotPoints();
                            expPtr->GetTracePhysVals(3, TraceExp[3], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);

                            // edge 1
                            expPtr->GetTracePhysVals(1, TraceExp[1], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr + npts);

                            outptr += 2 * npts;
                            ntrace = 2 * npts;

                            // edge 0
                            npts = TraceExp[0]->GetTotPoints();
                            expPtr->GetTracePhysVals(0, TraceExp[0], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // edge 2
                            expPtr->GetTracePhysVals(2, TraceExp[2], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;
                        }
                        break;
                        case LibUtilities::NodalTri:
                        case LibUtilities::Tri:
                        {
                            // edge 2
                            auto npts = TraceExp[2]->GetTotPoints();
                            expPtr->GetTracePhysVals(2, TraceExp[2], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);

                            // edge 1
                            expPtr->GetTracePhysVals(1, TraceExp[1], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr + npts);

                            outptr += 2 * npts;
                            ntrace = 2 * npts;

                            // edge 0
                            npts = TraceExp[0]->GetTotPoints();
                            expPtr->GetTracePhysVals(0, TraceExp[0], indata,
                                                     outphys,
                                                     StdRegions::eForwards);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;
                        }
                        break;
                        case LibUtilities::Hex:
                        case LibUtilities::NodalPrism:
                        case LibUtilities::Prism:
                        case LibUtilities::Pyr:
                        {
                            auto npts = TraceExp[2]->GetTotPoints();
                            // face 4
                            expPtr->GetTracePhysVals(
                                4, TraceExp[4], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 2
                            expPtr->GetTracePhysVals(
                                2, TraceExp[2], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 1
                            npts = TraceExp[1]->GetTotPoints();
                            expPtr->GetTracePhysVals(
                                1, TraceExp[1], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 3
                            expPtr->GetTracePhysVals(
                                3, TraceExp[3], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 0
                            npts = TraceExp[0]->GetTotPoints();
                            expPtr->GetTracePhysVals(
                                0, TraceExp[0], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            if (Shape == LibUtilities::Hex)
                            {
                                // face 5
                                expPtr->GetTracePhysVals(
                                    5, TraceExp[5], indata, outphys,
                                    StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                                std::copy(outphys.data(), outphys.data() + npts,
                                          outptr);
                                outptr += npts;
                                ntrace += npts;
                            }
                        }
                        break;
                        case LibUtilities::NodalTet:
                        case LibUtilities::Tet:
                        {
                            auto npts = TraceExp[2]->GetTotPoints();
                            // face 3
                            expPtr->GetTracePhysVals(
                                3, TraceExp[3], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 2
                            expPtr->GetTracePhysVals(
                                2, TraceExp[2], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 1
                            npts = TraceExp[1]->GetTotPoints();
                            expPtr->GetTracePhysVals(
                                1, TraceExp[1], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;

                            // face 0
                            npts = TraceExp[0]->GetTotPoints();
                            expPtr->GetTracePhysVals(
                                0, TraceExp[0], indata, outphys,
                                StdRegions::eDir1FwdDir1_Dir2FwdDir2);
                            std::copy(outphys.data(), outphys.data() + npts,
                                      outptr);
                            outptr += npts;
                            ntrace += npts;
                        }
                        break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "Need to define shape details");
                            break;
                    }
                    physoffset += numData;
                }
                inptr += numData * numElmtsPad;
                outptr +=
                    ntrace * (numElmtsPad - numElmts); // offset for padded case
            }
            eid += numElmts;
        }
    }

private:
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public PhystraceextractField<float>                  \
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
    class type : public PhystraceextractField<double>                        \
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

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadGaussPts, "run/square_gausspts.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(TriNodal, "run/tri_nodal.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexFixedP, "run/hex_fixedp.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismFixedP, "run/prism_fixedp.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(TetNodal, "run/tet_nodal.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
