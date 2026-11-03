///////////////////////////////////////////////////////////////////////////////
//
// File: TestBndCondPhys.hpp
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

#include <LibUtilities/Memory/MemoryAlloc.hpp>
#include <MultiRegions/BndCondOps/BndCondPhys/BndCondPhysOp.hpp>
#include <SpatialDomains/Conditions.h>

#include <algorithm>
#include <cmath>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

template <typename TData>
class TestBndCondPhys : public TestOp<TData, FieldState::Phys, FieldState::Phys,
                                      MultiRegions::DisContField>
{
public:
    TestBndCondPhys() = default;

    void RunTestCase(const TData time)
    {
        m_op = BndCondPhysOp<TData>::Create(this->fixt_explist,
                                            this->session->GetVariables());

        // Refill twice, so that the values read are those of the last update.
        m_time = time;
        m_op->UpdateBndPhys(0.5 * time);
        m_op->UpdateBndPhys(time);

        const std::string execStr(GlobalConfiguration::ExecStr());
        m_values.resize(m_op->GetNumBlocks());
        for (unsigned int blk = 0; blk < m_op->GetNumBlocks(); ++blk)
        {
            const size_t n =
                m_op->GetNumComponents() * m_op->GetBlockCompStride(blk);
            m_values[blk].resize(n);

            const TData *ptr = m_op->GetBlockPtr(blk);
            if (execStr == NektarSpaces::Device::name)
            {
                deviceMemcpy<DeviceToHost>(m_values[blk].data(), ptr,
                                           n * sizeof(TData), 0);
            }
            else
            {
                std::copy(ptr, ptr + n, m_values[blk].begin());
            }
        }
    }

    // Compare every boundary point against the session's expressions,
    // evaluated by the host interpreter at the time of the last update, or at
    // zero for a region that does not depend on time.
    bool Compare(TData tol) override
    {
        const auto &vars = this->session->GetVariables();
        const SpatialDomains::BoundaryConditions bcs(
            this->session, this->fixt_explist->GetGraph());
        const auto &bndExps = this->fixt_explist->GetBndCondExpansions();

        bool pass            = true;
        unsigned int ordinal = 0;
        for (auto &region : bcs.GetBoundaryRegions())
        {
            const auto &bndExp  = bndExps[ordinal++];
            const auto &condMap = bcs.GetBoundaryConditions().at(region.first);

            bool timeDependent = false;
            for (auto &var : vars)
            {
                timeDependent =
                    timeDependent || condMap->at(var)->IsTimeDependent();
            }
            const NekDouble time = timeDependent ? m_time : 0.0;

            for (size_t e = 0; e < bndExp->GetExpSize(); ++e)
            {
                const auto exp = bndExp->GetExp(e);
                const auto tid = exp->GetGeom()->GetGlobalID();
                if (!m_op->HasTrace(tid))
                {
                    continue;
                }

                unsigned int blk;
                size_t offset, compOffset;
                m_op->GetTraceLocation(tid, blk, offset, compOffset);

                const size_t nq = exp->GetTotPoints();
                Array<OneD, NekDouble> x(nq, 0.0), y(nq, 0.0), z(nq, 0.0);
                exp->GetCoords(x, y, z);

                for (unsigned int c = 0; c < vars.size(); ++c)
                {
                    const auto &eqn = condMap->at(vars[c])->GetEquation();
                    for (size_t q = 0; q < nq; ++q)
                    {
                        const TData expected =
                            eqn->Evaluate(x[q], y[q], z[q], time);
                        const TData value =
                            m_values[blk][offset + c * compOffset + q];
                        if (std::abs(value - expected) > tol)
                        {
                            std::cout << "region " << region.first << " trace "
                                      << tid << " point " << q << ": " << value
                                      << " != " << expected << std::endl;
                            pass = false;
                        }
                    }
                }
            }
        }
        return pass;
    }

protected:
    std::shared_ptr<BndCondPhysOp<TData>> m_op;
    std::vector<std::vector<TData>> m_values;
    TData m_time = 0.0;
};

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
class Helmholtz2D_BndCondPhys : public TestBndCondPhys<double>
{
public:
    Helmholtz2D_BndCondPhys()
    {
        meshName = "run/Helmholtz2D_P7_BndCondPhys.xml";
    }
};
#endif
