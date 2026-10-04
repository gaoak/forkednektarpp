///////////////////////////////////////////////////////////////////////////////
//
// File: GatherScatter.cpp
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
// Description: Gather-scatter handle selectable between gslib and
// SharedIdPlan.
//
///////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <LibUtilities/Communication/GatherScatter.h>
#include <LibUtilities/Communication/GsLib.hpp>
#include <LibUtilities/Communication/SharedIdPlan.hpp>

namespace Nektar::LibUtilities
{

namespace
{
const std::string cmdMethod = SessionReader::RegisterCmdLineArgument(
    "gather-scatter", "",
    "Gather-scatter used by assembly maps: gslib (default), or pairwise, "
    "neighbour or crystal for SharedIdPlan with that exchange backend.");

const std::string cmdDiscovery = SessionReader::RegisterCmdLineArgument(
    "gather-scatter-discovery", "",
    "Transport for SharedIdPlan's one-off discovery: crystal (default) or "
    "a2a. Ignored with --gather-scatter gslib.");
} // namespace

GatherScatterConfig GatherScatterConfig::FromSession(
    const SessionReaderSharedPtr &session)
{
    GatherScatterConfig config;
    if (!session)
    {
        return config;
    }

    if (session->DefinesCmdLineArgument(cmdMethod))
    {
        const std::string m =
            session->GetCmdLineArgument<std::string>(cmdMethod);
        if (m == "gslib")
        {
            config.method = GatherScatterMethod::eGslib;
        }
        else if (m == "pairwise")
        {
            config.method = GatherScatterMethod::ePairwise;
        }
        else if (m == "neighbour")
        {
            config.method = GatherScatterMethod::eNeighbour;
        }
        else if (m == "crystal")
        {
            config.method = GatherScatterMethod::eCrystal;
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "--gather-scatter must be one of gslib, pairwise, "
                     "neighbour or crystal; got '" +
                         m + "'.");
        }
    }

    if (session->DefinesCmdLineArgument(cmdDiscovery))
    {
        const std::string d =
            session->GetCmdLineArgument<std::string>(cmdDiscovery);
        if (d == "crystal")
        {
            config.discovery = GatherScatterDiscovery::eCrystalRouter;
        }
        else if (d == "a2a")
        {
            config.discovery = GatherScatterDiscovery::eAlltoallv;
        }
        else
        {
            NEKERROR(
                ErrorUtil::efatal,
                "--gather-scatter-discovery must be crystal or a2a; got '" + d +
                    "'.");
        }
    }

    return config;
}

std::string GatherScatterConfig::Describe() const
{
    if (method == GatherScatterMethod::eGslib)
    {
        return "gslib";
    }

    const std::string backend =
        method == GatherScatterMethod::ePairwise    ? "pairwise"
        : method == GatherScatterMethod::eNeighbour ? "neighbour-collective"
                                                    : "crystal-router";
    const std::string transport =
        discovery == GatherScatterDiscovery::eCrystalRouter ? "crystal-router"
                                                            : "alltoallv";
    return "SharedIdPlan (" + backend + " exchange, " + transport +
           " discovery)";
}

namespace
{

/**
 * @brief The do-nothing implementation, for a serial communicator and for
 * slots no gather-scatter was ever set up over.
 */
class NoOpGatherScatter final : public GatherScatter
{
public:
    void Gather([[maybe_unused]] Array<OneD, NekDouble> u,
                [[maybe_unused]] GatherScatterOp op) override
    {
    }

    void Unique([[maybe_unused]] Array<OneD, long> &ids) override
    {
    }
};

/// gslib's gs_setup / gs_gather / gs_unique.
class GsLibGatherScatter final : public GatherScatter
{
public:
    GsLibGatherScatter(const Array<OneD, long> &ids, CommSharedPtr comm,
                       bool verbose)
        : m_comm(std::move(comm)), m_gsh(Gs::Init(ids, m_comm, verbose))
    {
    }

    ~GsLibGatherScatter() override
    {
        // Null-safe, and skipped once MPI has been finalised.
        Gs::Finalise(m_gsh);
    }

    void Gather(Array<OneD, NekDouble> u, GatherScatterOp op) override
    {
        Gs::Gather(u, ToGsOp(op), m_gsh);
    }

    void Unique(Array<OneD, long> &ids) override
    {
        Gs::Unique(ids, m_comm);
    }

private:
    /// The one place gslib's operation names are needed.
    static Gs::gs_op ToGsOp(GatherScatterOp op)
    {
        switch (op)
        {
            case GatherScatterOp::eAdd:
                return Gs::gs_add;
            case GatherScatterOp::eMul:
                return Gs::gs_mul;
            case GatherScatterOp::eMin:
                return Gs::gs_min;
            case GatherScatterOp::eMax:
                return Gs::gs_max;
            case GatherScatterOp::eAbsMax:
                return Gs::gs_amax;
        }
        NEKERROR(ErrorUtil::efatal, "Unknown GatherScatterOp.");
        return Gs::gs_add;
    }

    CommSharedPtr m_comm;
    Gs::gs_data *m_gsh = nullptr;
};

/// A #SharedIdPlan over one of its exchange backends.
class PlanGatherScatter final : public GatherScatter
{
public:
    PlanGatherScatter(const Array<OneD, long> &ids, const CommSharedPtr &comm,
                      const GatherScatterConfig &config, bool verbose)
        : m_nSlots(ids.size())
    {
        // Only positive ids take part. Id 0 is gslib's "skip"; a negative id
        // is gslib's "flagged" entry, which does not contribute to its
        // group's fold but does receive the result for |id|. Nektar's only
        // source of negative ids is AssemblyMapDG's -1 initialiser, left on
        // trace modes above a lower-order element's own order under variable
        // P -- which gslib therefore aliases onto universal id 1. The plan
        // leaves those padding slots untouched instead.
        std::vector<int64_t> slotIds(m_nSlots);
        for (size_t i = 0; i < m_nSlots; ++i)
        {
            slotIds[i] = ids[i] > 0 ? static_cast<int64_t>(ids[i])
                                    : SharedIdPlan::kIgnore;
        }

        std::shared_ptr<ITransport> transport;
        if (config.discovery == GatherScatterDiscovery::eAlltoallv)
        {
            transport = std::make_shared<AlltoallvTransport>(comm);
        }
        else
        {
            transport = std::make_shared<CrystalRouterTransport>(comm);
        }

        const ExchangeBackend backend =
            config.method == GatherScatterMethod::ePairwise
                ? ExchangeBackend::ePairwise
            : config.method == GatherScatterMethod::eNeighbour
                ? ExchangeBackend::eNeighbourCollective
                : ExchangeBackend::eCrystalRouter;

        const auto t0 = std::chrono::steady_clock::now();
        m_plan = std::make_unique<SharedIdPlan>(transport, slotIds, backend);

        if (verbose && comm->GetRank() == 0)
        {
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - t0)
                                       .count();
            std::cout << "GatherScatter: " << config.Describe() << ", "
                      << m_nSlots << " slots and "
                      << m_plan->Neighbours().size()
                      << " neighbours on rank 0, set up in " << seconds << " s"
                      << std::endl;
        }

        // Say so rather than leave the caller believing they got what they
        // asked for: the neighbourhood collective needs MPI-3, and the plan
        // falls back to the pairwise exchange without it.
        if (backend != m_plan->Backend() && comm->GetRank() == 0)
        {
            NEKERROR(ErrorUtil::ewarning,
                     "--gather-scatter neighbour needs MPI-3; using the "
                     "pairwise exchange instead. Results are unaffected.");
        }
    }

    void Gather(Array<OneD, NekDouble> u, GatherScatterOp op) override
    {
        // Like gslib, the plan only touches slots whose id is shared or
        // repeated locally, and AssemblyMap relies on that: a multi-level
        // static condensation level gathers through its parent level's
        // handle with an array that stops short of the parent's
        // condensed-away slots.
        ASSERTL1(u.size() >= m_plan->ActiveSlotBound(),
                 "GatherScatter::Gather: array too short for the slots the "
                 "handle exchanges");
        NekDouble *slots = u.data();
        switch (op)
        {
            case GatherScatterOp::eAdd:
                m_plan->Gather(slots,
                               [](NekDouble a, NekDouble b) { return a + b; });
                break;
            case GatherScatterOp::eMul:
                m_plan->Gather(slots,
                               [](NekDouble a, NekDouble b) { return a * b; });
                break;
            case GatherScatterOp::eMin:
                m_plan->Gather(slots, [](NekDouble a, NekDouble b) {
                    return std::min(a, b);
                });
                break;
            case GatherScatterOp::eMax:
                m_plan->Gather(slots, [](NekDouble a, NekDouble b) {
                    return std::max(a, b);
                });
                break;
            case GatherScatterOp::eAbsMax:
                // gslib's GS_DO_amax: keep the signed value of largest
                // magnitude.
                m_plan->Gather(slots, [](NekDouble a, NekDouble b) {
                    return std::abs(b) > std::abs(a) ? b : a;
                });
                break;
        }
    }

    void Unique(Array<OneD, long> &ids) override
    {
        ASSERTL0(ids.size() == m_nSlots,
                 "GatherScatter::Unique: ids must be the array this handle "
                 "was built from");
        const std::vector<int> mask = m_plan->UniqueMask();
        for (size_t i = 0; i < m_nSlots; ++i)
        {
            // Zero and negative ids are left as they are, exactly as gslib
            // leaves them: callers read "id >= 0" as owned, so a zero stays
            // owned and a negative stays not-owned.
            if (ids[i] > 0 && !mask[i])
            {
                ids[i] = -ids[i];
            }
        }
    }

private:
    size_t m_nSlots;
    std::unique_ptr<SharedIdPlan> m_plan;
};

} // namespace

GatherScatterSharedPtr GatherScatter::Create(const Array<OneD, long> &ids,
                                             const CommSharedPtr &comm,
                                             const GatherScatterConfig &config,
                                             bool verbose)
{
    if (config.method == GatherScatterMethod::eGslib)
    {
        return std::make_shared<GsLibGatherScatter>(ids, comm, verbose);
    }

    // gslib has nothing to exchange on one rank and hands back a null handle;
    // do the same, so both methods behave identically in serial.
    if (comm->IsSerial())
    {
        return CreateNoOp();
    }

    return std::make_shared<PlanGatherScatter>(ids, comm, config, verbose);
}

GatherScatterSharedPtr GatherScatter::CreateNoOp()
{
    return std::make_shared<NoOpGatherScatter>();
}

} // namespace Nektar::LibUtilities
