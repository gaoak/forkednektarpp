///////////////////////////////////////////////////////////////////////////////
//
// File: TestPhysTraceExtractBndCond.hpp
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
// Description: Fixture for the tests of detail::PhysTraceExtractBndCond: a
// linear polynomial, exact at every order, is extracted from the volume onto
// a boundary region and compared with its values at the region's points.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TestOp.hpp"

#include <MultiRegions/BndCondOps/Common/PhysTraceExtractBndCond.hpp>

#include <SpatialDomains/Conditions.h>

#include <algorithm>

using namespace Nektar::MultiRegions;
using namespace Nektar::LibUtilities;
using namespace Nektar;

// A field exactly representable at any polynomial order >= 1: u = x + 2y + 3z.
template <typename TData>
static inline TData PhysTraceExtractBndCondTestFn(TData x, TData y, TData z)
{
    return x + TData(2.0) * y + TData(3.0) * z;
}

template <typename TData>
class TestPhysTraceExtractBndCond
    : public TestOp<TData, FieldState::Phys, FieldState::Phys>
{
public:
    TestPhysTraceExtractBndCond()
        : TestOp<TData, FieldState::Phys, FieldState::Phys>()
    {
    }

    void Configure(unsigned int boundaryRegionId)
    {
        this->SetSession();
        this->SetExpList();
        this->fixt_explist->SetDataWarehouse();

        auto graph = this->fixt_explist->GetGraph();
        SpatialDomains::BoundaryConditions bcs(this->session, graph);
        const auto &bregions  = bcs.GetBoundaryRegions();
        const auto regionIter = bregions.find(boundaryRegionId);
        ASSERTL0(regionIter != bregions.end(),
                 "Boundary region id not found in this mesh.");

        const auto &vars = this->session->GetVariables();
        m_bcExpList = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
            this->session, *(regionIter->second), graph, true, vars[0], false,
            this->session->GetComm(), Collections::eNoImpType);
        m_bcExpList->SetDataWarehouse();

        SetFixture(1);
    }

    void SetFixture(const unsigned int nhomo) override
    {
        auto nvars = this->session->GetVariables().size();
        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(m_bcExpList);

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", inblockAttr, nvars, nhomo);
        auto f_out =
            Field<TData, FieldState::Phys>("f_out", outblockAttr, nvars, nhomo);
        auto f_expected = Field<TData, FieldState::Phys>(
            "f_expected", outblockAttr, nvars, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));
    }

    void SetTestCase()
    {
        // Seed the volume field with u = x + 2y + 3z at every quadrature
        // point, for every component.
        for (unsigned blk = 0; blk < this->fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto collection =
                MultiRegions::GetCollection(this->fixt_explist, blk);
            const auto &expansions = collection.GetExpVector();

            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned n = 0; n < this->fixt_in->GetNumComponents() *
                                         this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                size_t cnt = 0;
                for (size_t el = 0; el < expansions.size(); ++el)
                {
                    const auto npts = expansions[el]->GetTotPoints();
                    // GetCoords() leaves z unset on a two-dimensional mesh.
                    Array<OneD, NekDouble> x(npts), y(npts), z(npts, 0.0);
                    expansions[el]->GetCoords(x, y, z);
                    for (int q = 0; q < npts; ++q, ++cnt)
                    {
                        inptr[cnt] = PhysTraceExtractBndCondTestFn<TData>(
                            static_cast<TData>(x[q]), static_cast<TData>(y[q]),
                            static_cast<TData>(z[q]));
                    }
                }
                inptr += block.CompSize();
            }
        }

        // Expected: evaluate the same polynomial directly at the boundary
        // region's own coordinates.
        for (unsigned blk = 0; blk < this->fixt_expected->GetBlocks().size();
             ++blk)
        {
            auto &block     = this->fixt_expected->GetBlocks()[blk];
            auto collection = MultiRegions::GetCollection(m_bcExpList, blk);
            const auto &expansions = collection.GetExpVector();

            auto expptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned n = 0; n < this->fixt_expected->GetNumComponents() *
                                         this->fixt_expected->GetNumHomoModes();
                 ++n)
            {
                size_t cnt = 0;
                for (size_t el = 0; el < expansions.size(); ++el)
                {
                    const auto npts = expansions[el]->GetTotPoints();
                    // GetCoords() leaves z unset on a two-dimensional mesh.
                    Array<OneD, NekDouble> x(npts), y(npts), z(npts, 0.0);
                    expansions[el]->GetCoords(x, y, z);
                    for (int q = 0; q < npts; ++q, ++cnt)
                    {
                        expptr[cnt] = PhysTraceExtractBndCondTestFn<TData>(
                            static_cast<TData>(x[q]), static_cast<TData>(y[q]),
                            static_cast<TData>(z[q]));
                    }
                }
                expptr += block.CompSize();
            }
        }
    }

    // The helper is header-only, and its Device kernels compile only in a
    // translation unit built for the device, which this test is not. It is
    // therefore tested here on Serial and AVX.
    void RunTestCase()
    {
        const std::string execStr(GlobalConfiguration::ExecStr());
        if (execStr == NektarSpaces::Serial::name)
        {
            RunTestCase<NektarSpaces::Serial>();
        }
        else if (execStr == NektarSpaces::AVX::name)
        {
            RunTestCase<NektarSpaces::AVX>();
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "TestPhysTraceExtractBndCond runs on Serial and AVX, "
                     "not on " +
                         execStr);
        }
    }

    template <typename ExecSpace> void RunTestCase()
    {
        MultiRegions::detail::PhysTraceExtractBndCond<ExecSpace, TData> op(
            this->fixt_explist, this->session->GetVariables(), m_bcExpList,
            ExecSpace::name);

        if (m_inputInterleaveWidth == 1)
        {
            op.Apply(*this->fixt_in, *this->fixt_out);
            return;
        }

        // Extract at the wider interleave width, then again at width 1 into
        // a cleared output, so that the gather index is built for both.
        for (auto &block : this->fixt_in->GetBlocks())
        {
            block.ReshapeStorage(m_inputInterleaveWidth, ExecSpace::name);
        }
        op.Apply(*this->fixt_in, *this->fixt_out);
        BOOST_TEST(this->Compare(1.0E-10));

        for (auto &block : this->fixt_in->GetBlocks())
        {
            block.ReshapeStorage(1, ExecSpace::name);
        }
        for (auto &block : this->fixt_out->GetBlocks())
        {
            auto outptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            std::fill(outptr,
                      outptr + block.CompSize() * block.GetNumComponents() *
                                   block.GetNumHomoModes(),
                      TData(0));
        }
        op.Apply(*this->fixt_in, *this->fixt_out);
    }

protected:
    MultiRegions::ExpListSharedPtr m_bcExpList;
    /// Interleave width the volume field is given to Apply() at.
    unsigned int m_inputInterleaveWidth = 1;
};
