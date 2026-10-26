///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
// Copyright (c) 2026 Nektar++ contributors.
// See the LICENSE file in the root directory for details.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/Communication/CommCwipi.h>
#include <SolverUtils/Core/CouplingCoefficients.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>

namespace Nektar::SolverUtils
{

std::string CouplingCoefficients::className =
    GetCouplingFactory().RegisterCreatorFunction(
        "Coefficients", CouplingCoefficients::create,
        "Paired MHD element-coefficient exchange on unified partitions");

CouplingCoefficients::CouplingCoefficients(MultiRegions::ExpListSharedPtr field)
    : Coupling(field)
{
    m_config["ROLE"]       = "";
    m_config["FLUIDRANKS"] = "0";
    m_config["REMOTENAME"] = "";
}

std::vector<NekDouble> CouplingCoefficients::BuildLayout()
{
    std::vector<int> elements(m_evalField->GetExpSize());
    std::iota(elements.begin(), elements.end(), 0);
    std::sort(elements.begin(), elements.end(), [&](int a, int b) {
        return m_evalField->GetExp(a)->GetGeom()->GetGlobalID() <
               m_evalField->GetExp(b)->GetGeom()->GetGlobalID();
    });
    Array<OneD, NekDouble> x(m_evalField->GetTotPoints()), y(x.size()),
        z(x.size());
    m_evalField->GetCoords(x, y, z);
    std::vector<NekDouble> layout{
        static_cast<NekDouble>(elements.size()),
        static_cast<NekDouble>(m_evalField->GetNcoeffs()),
        static_cast<NekDouble>(m_dimension)};
    for (int e : elements)
    {
        auto exp  = m_evalField->GetExp(e);
        auto geom = exp->GetGeom();
        layout.insert(layout.end(),
                      {static_cast<NekDouble>(geom->GetGlobalID()),
                       static_cast<NekDouble>(exp->DetShapeType()),
                       static_cast<NekDouble>(exp->GetNcoeffs()),
                       static_cast<NekDouble>(exp->GetNumBases())});
        for (int d = 0; d < exp->GetNumBases(); ++d)
        {
            auto basis = exp->GetBasis(d);
            layout.insert(layout.end(),
                          {static_cast<NekDouble>(basis->GetBasisType()),
                           static_cast<NekDouble>(basis->GetNumModes()),
                           static_cast<NekDouble>(basis->GetNumPoints()),
                           static_cast<NekDouble>(basis->GetPointsType())});
        }
        layout.push_back(geom->GetNumVerts());
        for (int v = 0; v < geom->GetNumVerts(); ++v)
        {
            NekDouble vx, vy, vz;
            geom->GetVertex(v)->GetCoords(vx, vy, vz);
            layout.insert(layout.end(), {static_cast<NekDouble>(
                                             geom->GetVertex(v)->GetGlobalID()),
                                         vx, vy, vz});
        }
        layout.push_back(exp->GetTotPoints());
        int offset = m_evalField->GetPhys_Offset(e);
        for (int q = 0; q < exp->GetTotPoints(); ++q)
        {
            layout.insert(layout.end(),
                          {x[offset + q], y[offset + q], z[offset + q]});
        }
        for (int c = 0; c < exp->GetNcoeffs(); ++c)
        {
            m_indices.push_back(m_evalField->GetCoeff_Offset(e) + c);
        }
    }
    return layout;
}

void CouplingCoefficients::v_Init()
{
    Coupling::v_Init();
    auto comm = std::dynamic_pointer_cast<LibUtilities::CommCwipi>(
        m_evalField->GetComm());
    ASSERTL0(
        comm,
        "Coefficient exchange requires distinct --cwipi application names.");
    m_local = comm->GetComm();
    MPI_Comm_rank(m_local, &m_rank);
    int size, worldSize;
    MPI_Comm_size(m_local, &size);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    ASSERTL0(StringUtils::IEquals(m_config["ROLE"], "Fluid") ||
                 StringUtils::IEquals(m_config["ROLE"], "Electric"),
             "Coefficient ROLE must be Fluid or Electric.");
    m_fluid        = StringUtils::IEquals(m_config["ROLE"], "Fluid");
    int fluidRanks = std::stoi(m_config["FLUIDRANKS"]);
    ASSERTL0(fluidRanks > 0 &&
                 (m_fluid ? size == fluidRanks : size > fluidRanks),
             "Unified partition sizes must be M fluid ranks and M+N electric "
             "ranks, N>0.");
    auto *geometry = m_evalField->GetSession()->GetElement("Nektar/Geometry");
    int part       = -1;
    geometry->QueryIntAttribute("PARTITION", &part);
    ASSERTL0(
        part == m_rank,
        "Coefficient exchange requires rank-matched pre-partitioned meshes.");
    ASSERTL0(m_sendSteps == 1 && m_recvSteps == 1,
             "Coefficient exchange requires SendSteps=ReceiveSteps=1.");
    m_dimension = m_evalField->GetGraph()->GetMeshDimension();
    std::vector<std::string> velocity{"u", "v"};
    if (m_dimension == 3)
        velocity.push_back("w");
    velocity.push_back("mhd_time");
    velocity.push_back("mhd_stop");
    std::vector<std::string> potential{"phi"};
    ASSERTL0(m_sendFieldNames == (m_fluid ? velocity : potential) &&
                 m_recvFieldNames == (m_fluid ? potential : velocity),
             "Invalid coefficient MHD field lists.");

    // Discover leaders, rather than depending on the ordering of MPMD apps.
    int info[4] = {m_fluid ? 0 : 1, m_rank, size, fluidRanks};
    std::vector<int> all(4 * worldSize);
    MPI_Allgather(info, 4, MPI_INT, all.data(), 4, MPI_INT, MPI_COMM_WORLD);
    int remoteLeader = -1, fluidCount = 0, electricCount = 0;
    for (int r = 0; r < worldSize; ++r)
    {
        ASSERTL0(all[4 * r + 3] == fluidRanks,
                 "FluidRanks differs between applications.");
        (all[4 * r] == 0 ? fluidCount : electricCount)++;
        if (all[4 * r] != info[0] && all[4 * r + 1] == 0)
            remoteLeader = r;
    }
    ASSERTL0(remoteLeader >= 0 && fluidCount == fluidRanks &&
                 electricCount > fluidRanks &&
                 size == (m_fluid ? fluidCount : electricCount),
             "Coefficient mode requires exactly one Fluid and one Electric "
             "application.");
    MPI_Intercomm_create(m_local, 0, MPI_COMM_WORLD, remoteLeader, 27183,
                         &m_bridge);
    m_paired  = m_rank < fluidRanks;
    int valid = 1;
    if (m_paired)
    {
        auto layout = BuildLayout();
        ASSERTL0(layout.size() <= INT_MAX,
                 "Coefficient layout is too large for MPI.");
        int count = layout.size(), otherCount;
        MPI_Sendrecv(&count, 1, MPI_INT, m_rank, 0, &otherCount, 1, MPI_INT,
                     m_rank, 0, m_bridge, MPI_STATUS_IGNORE);
        ASSERTL0(otherCount >= 0, "Invalid coefficient layout size.");
        std::vector<NekDouble> other(otherCount);
        MPI_Sendrecv(layout.data(), count, MPI_DOUBLE, m_rank, 1, other.data(),
                     otherCount, MPI_DOUBLE, m_rank, 1, m_bridge,
                     MPI_STATUS_IGNORE);
        valid = count == otherCount;
        if (valid)
        {
            for (int i = 0; i < count; ++i)
                valid &= std::abs(layout[i] - other[i]) <=
                         1e-12 * std::max({1.0, std::abs(layout[i]),
                                           std::abs(other[i])});
        }
        m_coeffs = Array<OneD, NekDouble>(m_evalField->GetNcoeffs(), 0.0);
    }
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, m_local);
    ASSERTL0(
        valid,
        "Paired fluid layouts differ: check element IDs, geometry orientation, "
        "basis types, orders and quadrature. Regenerate unified partitions.");
    if (m_rank == 0)
        std::cout << "Coefficient coupling: " << fluidRanks
                  << " paired fluid partitions, " << electricCount - fluidRanks
                  << " solid partitions; layout validated once." << std::endl;
}

void CouplingCoefficients::SendPacket(int step, NekDouble time, int stop,
                                      const std::vector<NekDouble> &values)
{
    if (!m_paired)
        return;
    ASSERTL0(values.size() < INT_MAX,
             "Coefficient message is too large for MPI.");
    int header[3] = {step, stop, static_cast<int>(values.size())};
    MPI_Send(header, 3, MPI_INT, m_rank, 2, m_bridge);
    MPI_Send(&time, 1, MPI_DOUBLE, m_rank, 3, m_bridge);
    MPI_Send(values.data(), values.size(), MPI_DOUBLE, m_rank, 4, m_bridge);
}

void CouplingCoefficients::ReceivePacket(int step, int nvalues)
{
    int header[3];
    MPI_Recv(header, 3, MPI_INT, m_rank, 2, m_bridge, MPI_STATUS_IGNORE);
    ASSERTL0(
        header[0] == step && header[2] == nvalues &&
            (header[1] == -1 || header[1] == 1),
        "Coefficient exchange sequence, size or shutdown marker mismatch.");
    m_stop = header[1];
    MPI_Recv(&m_time, 1, MPI_DOUBLE, m_rank, 3, m_bridge, MPI_STATUS_IGNORE);
    ASSERTL0(std::isfinite(m_time), "Non-finite coefficient request time.");
    m_packet.resize(nvalues);
    MPI_Recv(m_packet.data(), nvalues, MPI_DOUBLE, m_rank, 4, m_bridge,
             MPI_STATUS_IGNORE);
}

void CouplingCoefficients::v_Send(
    int step, NekDouble time,
    const Array<OneD, const Array<OneD, NekDouble>> &fields,
    std::vector<std::string> &names)
{
    if (!m_paired)
        return;
    auto map = GenerateVariableMapping(names, m_sendFieldNames);
    if (!m_fluid)
    {
        m_evalField->FwdTransLocalElmt(fields[map[0]], m_coeffs);
        SendCoefficients(step, time, m_coeffs);
        return;
    }
    const int nc = m_indices.size();
    std::vector<NekDouble> values(static_cast<size_t>(m_dimension) * nc);
    for (int d = 0; d < m_dimension; ++d)
    {
        // RHS stage velocities are physical arrays; coefficients stored in
        // the flow fields may belong to a different time-integration stage.
        m_evalField->FwdTransLocalElmt(fields[map[d]], m_coeffs);
        for (int c = 0; c < nc; ++c)
            values[d * nc + c] = m_coeffs[m_indices[c]];
    }
    m_stop = fields[map[m_dimension + 1]][0] > 0.0 ? 1 : -1;
    SendPacket(step, time, m_stop, values);
}

void CouplingCoefficients::v_Receive(
    int step, NekDouble time, Array<OneD, Array<OneD, NekDouble>> &fields,
    std::vector<std::string> &names)
{
    auto map = GenerateVariableMapping(names, m_recvFieldNames);
    for (int i : map)
        Vmath::Zero(fields[i].size(), fields[i], 1);
    if (!m_paired)
        return;
    if (m_fluid)
    {
        ReceiveCoefficients(step, time, m_coeffs);
        m_evalField->BwdTrans(m_coeffs, fields[map[0]]);
        return;
    }
    const int nc = m_indices.size();
    ReceivePacket(step, m_dimension * nc);
    for (int d = 0; d < m_dimension; ++d)
    {
        for (int c = 0; c < nc; ++c)
            m_coeffs[m_indices[c]] = m_packet[d * nc + c];
        m_evalField->BwdTrans(m_coeffs, fields[map[d]]);
    }
    Vmath::Fill(fields[map[m_dimension]].size(), m_time,
                fields[map[m_dimension]], 1);
    Vmath::Fill(fields[map[m_dimension + 1]].size(),
                static_cast<NekDouble>(m_stop), fields[map[m_dimension + 1]],
                1);
}

void CouplingCoefficients::SendCoefficients(
    int step, NekDouble time, const Array<OneD, const NekDouble> &coeffs)
{
    ASSERTL0(!m_fluid, "Only Electric sends potential coefficients.");
    if (!m_paired)
        return;
    ASSERTL0(coeffs.size() == m_evalField->GetNcoeffs(),
             "Potential coefficient size mismatch.");
    std::vector<NekDouble> values(m_indices.size());
    for (int c = 0; c < m_indices.size(); ++c)
        values[c] = coeffs[m_indices[c]];
    SendPacket(step, time, m_stop, values);
}

void CouplingCoefficients::ReceiveCoefficients(int step, NekDouble time,
                                               Array<OneD, NekDouble> &coeffs)
{
    ASSERTL0(m_fluid && m_paired,
             "Only Fluid receives potential coefficients.");
    ASSERTL0(coeffs.size() == m_evalField->GetNcoeffs(),
             "Potential coefficient size mismatch.");
    ReceivePacket(step, m_indices.size());
    ASSERTL0(std::abs(time - m_time) < 1e-10 * std::max(1.0, std::abs(time)),
             "Potential reply time differs from the velocity request.");
    for (int c = 0; c < m_indices.size(); ++c)
        coeffs[m_indices[c]] = m_packet[c];
}

void CouplingCoefficients::v_Finalize()
{
    if (m_bridge != MPI_COMM_NULL)
        MPI_Comm_free(&m_bridge);
}

} // namespace Nektar::SolverUtils
