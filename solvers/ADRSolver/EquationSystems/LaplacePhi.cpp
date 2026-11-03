///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacePhi.cpp
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
// Description: Compute the velocity potentials around a rigid body
//
// 2D, phi0, phi1 [translation]; phi2 [rotation]; phi3 [bulge]
// 3D, phi0, phi1, phi2 [translation]; phi3, phi4, phi5 [rotation]; phi6 [bulge]
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/EquationSystems/LaplacePhi.h>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
namespace Nektar
{
std::string LaplacePhi::className =
    GetEquationSystemFactory().RegisterCreatorFunction("LaplacePhi",
                                                       LaplacePhi::create);

LaplacePhi::LaplacePhi(const LibUtilities::SessionReaderSharedPtr &pSession,
                       const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_factors()
{
    m_factors[StdRegions::eFactorLambda] = 0.0;
    m_factors[StdRegions::eFactorTau]    = 1.0;
}

void LaplacePhi::v_InitObject(bool DeclareFields)
{
    EquationSystem::v_InitObject(DeclareFields);
    int ndim = m_fields[0]->GetCoordim(0);
    // number of potential fields
    int bndsize = (ndim == 2) ? 4 : 7;
    // check variable name phi[0-bndsize]
    for (const auto &it : m_session->GetVariables())
    {
        if (it.size() != 4 || it[0] != 'p' || it[1] != 'h' || it[2] != 'i' ||
            it[3] < '0' || it[3] >= bndsize + '0')
        {
            NEKERROR(ErrorUtil::efatal,
                     "Error: incorrect variable name '" + it +
                         "', should be phi[0-3] or phi[0-6].");
        }
    }
    loadMovingWall();
    // number of physics points for each field
    int numpts = 0;
    Array<OneD, const SpatialDomains::BoundaryConditionShPtr> m_BndConds =
        m_fields[0]->GetBndConditions();
    Array<OneD, MultiRegions::ExpListSharedPtr> m_BndExp =
        m_fields[0]->GetBndCondExpansions();
    for (int i = 0; i < m_BndConds.size(); ++i)
    {
        if (StringUtils::IEquals(m_BndConds[i]->GetUserDefined(), "MOVE"))
        {
            numpts += m_BndExp[i]->GetTotPoints();
        }
    }
    // load pivot point
    m_pivot = Array<OneD, NekDouble>(3, 0.);
    if (m_session->DefinesParameter("Pivotx"))
    {
        m_pivot[0] = m_session->GetParameter("Pivotx");
    }
    if (m_session->DefinesParameter("Pivoty"))
    {
        m_pivot[1] = m_session->GetParameter("Pivoty");
    }
    if (m_session->DefinesParameter("Pivotz"))
    {
        m_pivot[2] = m_session->GetParameter("Pivotz");
    }
    // allocate storage
    m_boundValues = Array<OneD, Array<OneD, NekDouble>>(bndsize);
    for (int i = 0; i < bndsize; ++i)
    {
        m_boundValues[i] = Array<OneD, NekDouble>(numpts, 0.);
    }
    // calculate values
    int offset = 0;
    for (int i = 0; i < m_BndConds.size(); ++i)
    {
        if (StringUtils::IEquals(m_BndConds[i]->GetUserDefined(), "MOVE"))
        {
            int npts = m_BndExp[i]->GetTotPoints();
            Array<OneD, Array<OneD, NekDouble>> n(3);
            Array<OneD, Array<OneD, NekDouble>> x(3);
            for (int j = 0; j < 3; ++j)
            {
                n[j] = Array<OneD, NekDouble>(npts, 0.);
                x[j] = Array<OneD, NekDouble>(npts, 0.);
            }
            m_BndExp[i]->GetNormals(n);
            m_BndExp[i]->GetCoords(x[0], x[1], x[2]);
            for (int j = 0; j < 3; ++j)
            {
                Vmath::Sadd(npts, -m_pivot[j], x[j], 1, x[j], 1);
            }
            Array<OneD, NekDouble> atmp;
            for (int j = 0; j < ndim; ++j)
            {
                Vmath::Smul(npts, -1., n[j], 1,
                            atmp = m_boundValues[j] + offset, 1);
            }
            atmp = m_boundValues[bndsize - 2] + offset;
            Vmath::Vvtvvtm(npts, &x[1][0], 1, &n[0][0], 1, &x[0][0], 1,
                           &n[1][0], 1, &atmp[0], 1);
            if (ndim == 3)
            {
                atmp = m_boundValues[3] + offset;
                Vmath::Vvtvvtm(npts, &x[2][0], 1, &n[1][0], 1, &x[1][0], 1,
                               &n[2][0], 1, &atmp[0], 1);
                atmp = m_boundValues[4] + offset;
                Vmath::Vvtvvtm(npts, &x[0][0], 1, &n[2][0], 1, &x[2][0], 1,
                               &n[0][0], 1, &atmp[0], 1);
            }
            atmp = m_boundValues[bndsize - 1] + offset;
            for (int j = 0; j < 2; ++j)
            {
                Vmath::Vvtvp(npts, x[j], 1, n[j], 1, atmp, 1, atmp, 1);
            }
            Vmath::Neg(npts, atmp, 1);
            offset += npts;
        }
    }
}

void LaplacePhi::v_GenerateSummary(SolverUtils::SummaryList &s)
{
    EquationSystem::SessionSummary(s);
    SolverUtils::AddSummaryItem(s, "Lambda",
                                m_factors[StdRegions::eFactorLambda]);
}

void LaplacePhi::v_DoInitialise([[maybe_unused]] bool dumpInitialConditions)
{
    // Set initial conditions from session file
    SetInitialConditions(0.0, false, 0);
}

void LaplacePhi::v_DoSolve()
{
    for (int i = 0; i < m_fields.size(); ++i)
    {
        setUserDefinedBC(i);
        Array<OneD, NekDouble> forcing(m_fields[i]->GetTotPoints(), 0.);
        m_fields[i]->HelmSolve(forcing, m_fields[i]->UpdateCoeffs(), m_factors);
        m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(),
                              m_fields[i]->UpdatePhys());
        m_fields[i]->SetPhysState(true);
    }
    if (!m_hasMovingWall)
    {
        calculateAddedMass();
    }
}

Array<OneD, bool> LaplacePhi::v_GetSystemSingularChecks()
{
    return Array<OneD, bool>(m_session->GetVariables().size(), true);
}

void LaplacePhi::setUserDefinedBC(int n)
{
    int m = m_session->GetVariable(n)[3] - '0';
    Array<OneD, const SpatialDomains::BoundaryConditionShPtr> m_BndConds =
        m_fields[n]->GetBndConditions();
    Array<OneD, MultiRegions::ExpListSharedPtr> m_BndExp =
        m_fields[n]->GetBndCondExpansions();
    int offset = 0;
    for (int i = 0; i < m_BndConds.size(); ++i)
    {
        if (StringUtils::IEquals(m_BndConds[i]->GetUserDefined(), "MOVE"))
        {
            m_BndExp[i]->IProductWRTBase(m_boundValues[m] + offset,
                                         m_BndExp[i]->UpdateCoeffs());
            offset += m_BndExp[i]->GetTotPoints();
        }
        else if (StringUtils::IEquals(m_BndConds[i]->GetUserDefined(),
                                      "Movingwall"))
        {
            setMovingWallBC(m_BndExp[i]);
        }
    }
}

// Read endpoint velocities once. Coordinates are matched independently of file
// ordering, using an absolute tolerance of 1e-8 in mesh coordinate units.
void LaplacePhi::loadMovingWall()
{
    int hasMovingWall = 0;
    for (const auto &field : m_fields)
    {
        for (const auto &bc : field->GetBndConditions())
        {
            if (StringUtils::IEquals(bc->GetUserDefined(), "Movingwall"))
            {
                ASSERTL0(
                    bc->GetBoundaryConditionType() == SpatialDomains::eNeumann,
                    "Movingwall must be a Neumann (N) boundary condition.");
                hasMovingWall = 1;
            }
        }
    }
    m_session->GetComm()->AllReduce(hasMovingWall, LibUtilities::ReduceMax);
    m_hasMovingWall = hasMovingWall != 0;
    if (!m_hasMovingWall)
    {
        return;
    }
    ASSERTL0(m_graph->GetMeshDimension() == 2 && m_spacedim == 2,
             "Movingwall currently supports two-dimensional meshes with edge "
             "boundaries.");
    std::string filename;
    m_session->LoadSolverInfo("MovingWallFile", filename, "Movingwall.dat");
    std::ifstream input(filename);
    ASSERTL0(input.is_open(),
             "Cannot open Movingwall velocity file: " + filename);
    std::string line;
    int lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof())
        {
            continue;
        }
        std::array<NekDouble, 6> point;
        for (auto &value : point)
        {
            ASSERTL0(static_cast<bool>(row >> value) && std::isfinite(value),
                     filename + ": invalid x y z u v w at line " +
                         std::to_string(lineNumber));
        }
        std::string extra;
        ASSERTL0(!(row >> extra),
                 filename + ": expected exactly six columns at line " +
                     std::to_string(lineNumber));
        m_wallPoints.push_back(point);
    }
    ASSERTL0(!m_wallPoints.empty(),
             "Movingwall velocity file is empty: " + filename);
}

void LaplacePhi::setMovingWallBC(const MultiRegions::ExpListSharedPtr &boundary)
{
    int npts = boundary->GetTotPoints();
    Array<OneD, Array<OneD, NekDouble>> normals(3);
    for (int d = 0; d < 3; ++d)
    {
        normals[d] = Array<OneD, NekDouble>(npts, 0.0);
    }
    boundary->GetNormals(normals);
    Array<OneD, NekDouble> flux(npts, 0.0);
    for (int e = 0; e < boundary->GetExpSize(); ++e)
    {
        auto edge = boundary->GetExp(e);
        std::array<NekDouble, 3> velocity[2];
        for (int v = 0; v < 2; ++v)
        {
            NekDouble x, y, z;
            edge->GetGeom()->GetVertex(v)->GetCoords(x, y, z);
            int match = -1;
            for (int p = 0; p < static_cast<int>(m_wallPoints.size()); ++p)
            {
                const auto &point = m_wallPoints[p];
                if (std::abs(point[0] - x) <= 1e-8 &&
                    std::abs(point[1] - y) <= 1e-8 &&
                    std::abs(point[2] - z) <= 1e-8)
                {
                    ASSERTL0(match == -1, "Movingwall: ambiguous duplicate "
                                          "endpoint coordinates.");
                    match = p;
                }
            }
            ASSERTL0(match >= 0,
                     "Movingwall: missing velocity for edge " +
                         std::to_string(edge->GetGeom()->GetGlobalID()) +
                         ", endpoint " + std::to_string(v) + " (" +
                         std::to_string(x) + ", " + std::to_string(y) + ", " +
                         std::to_string(z) + ").");
            for (int d = 0; d < 3; ++d)
            {
                velocity[v][d] = m_wallPoints[match][d + 3];
            }
        }
        // Reference coordinate interpolation also applies to curved edges.
        const auto &xi = edge->GetBasis(0)->GetZ();
        int offset     = boundary->GetPhys_Offset(e);
        for (int q = 0; q < edge->GetTotPoints(); ++q)
        {
            NekDouble t = 0.5 * (1.0 + xi[q]);
            for (int d = 0; d < m_spacedim; ++d)
            {
                flux[offset + q] +=
                    normals[d][offset + q] *
                    ((1.0 - t) * velocity[0][d] + t * velocity[1][d]);
            }
        }
    }
    boundary->IProductWRTBase(flux, boundary->UpdateCoeffs());
}

void LaplacePhi::calculateAddedMass()
{
    Array<OneD, const SpatialDomains::BoundaryConditionShPtr> m_BndConds =
        m_fields[0]->GetBndConditions();
    Array<OneD, MultiRegions::ExpListSharedPtr> m_BndExp =
        m_fields[0]->GetBndCondExpansions();
    int nfld = m_fields.size();
    Array<OneD, NekDouble> value(nfld * nfld,
                                 std::numeric_limits<NekDouble>::lowest());
    int offset = 0;
    for (int i = 0; i < m_BndConds.size(); ++i)
    {
        if (StringUtils::IEquals(m_BndConds[i]->GetUserDefined(), "MOVE"))
        {
            MultiRegions::ExpListSharedPtr edgeExplist = m_BndExp[i];
            Array<OneD, Array<OneD, NekDouble>> phi(nfld);
            MultiRegions::ExpListSharedPtr BndElmtExp;
            m_fields[0]->GetBndElmtExpansion(i, BndElmtExp, false);
            Array<OneD, NekDouble> phiElm(BndElmtExp->GetTotPoints(), 0.);
            for (int j = 0; j < nfld; ++j)
            {
                m_fields[j]->ExtractPhysToBndElmt(i, m_fields[j]->GetPhys(),
                                                  phiElm);
                m_fields[j]->ExtractElmtToBndPhys(i, phiElm, phi[j]);
            }
            ////phi_0 n_0, phi_0 n_1, phi_1 n_0, phi_1 n_1
            Array<OneD, NekDouble> mularray(edgeExplist->GetTotPoints(), 0.);
            for (int j = 0; j < nfld; ++j)
            {
                for (int k = 0; k < m_boundValues.size(); ++k)
                {
                    Vmath::Vmul(edgeExplist->GetTotPoints(), phi[j], 1,
                                m_boundValues[k] + offset, 1, mularray, 1);
                    value[k + j * nfld] = edgeExplist->Integral(mularray);
                }
            }
            offset += edgeExplist->GetTotPoints();
        }
    }

    m_session->GetComm()->AllReduce(value, LibUtilities::ReduceMax);
    if (m_session->GetComm()->GetRank() == 0)
    {
        for (int j = 0; j < nfld; ++j)
        {
            for (int k = 0; k < nfld; ++k)
            {
                std::cout << "value[" << m_session->GetVariable(j)[3] << ", "
                          << m_session->GetVariable(k)[3]
                          << "] = " << std::scientific << std::setprecision(7)
                          << value[k + j * nfld] << std::endl;
            }
        }
    }
}
} // namespace Nektar
