///////////////////////////////////////////////////////////////////////////////
//
// File: CoupledElectricPotential.cpp
//
// The MIT License
// Copyright (c) 2026 Nektar++ contributors.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/EquationSystems/CoupledElectricPotential.h>
#include <LibUtilities/BasicUtils/SessionNodeXml.h>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <limits>

namespace Nektar
{

std::string CoupledElectricPotential::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "CoupledElectricPotential", CoupledElectricPotential::create);

CoupledElectricPotential::CoupledElectricPotential(
    const LibUtilities::SessionReaderSharedPtr &session,
    const SpatialDomains::MeshGraphSharedPtr &graph)
    : EquationSystem(session, graph)
{
}

void CoupledElectricPotential::v_InitObject(bool declareFields)
{
    EquationSystem::v_InitObject(declareFields);
    ASSERTL0(m_HomogeneousType == eNotHomogeneous &&
                 m_projectionType == MultiRegions::eGalerkin,
             "CoupledElectricPotential requires a full 2D/3D Continuous mesh.");
    ASSERTL0(m_fields.size() == 1 && m_session->GetVariable(0) == "phi",
             "CoupledElectricPotential requires exactly one variable: phi.");
    m_dimension = m_graph->GetMeshDimension();
    ASSERTL0(m_dimension == 2 || m_dimension == 3,
             "CoupledElectricPotential requires a 2D or 3D mesh.");
    ASSERTL0(
        m_graph->GetSpaceDimension() == m_dimension &&
            !m_session->DefinesElement("Nektar/Movement"),
        "CoupledElectricPotential requires a static full-dimensional mesh.");
    const int nq = m_fields[0]->GetTotPoints();
    m_sigma      = Array<OneD, NekDouble>(nq, 0.0);
    m_fluid      = Array<OneD, int>(nq, 0);

    // Material regions refer to volume composites, independently of the
    // electrical boundary regions. Interfaces remain internal CG faces.
    auto *regions = m_session->GetElement("Nektar/ElectricPotential");
    std::map<int, std::pair<NekDouble, bool>> materials;
    for (auto *region = regions->FirstChildElement(); region;
         region       = region->NextSiblingElement())
    {
        std::string type = region->Value();
        ASSERTL0(type == "FLUID" || type == "SOLID",
                 "ELECTRICPOTENTIAL children must be FLUID or SOLID.");
        const char *composite    = region->Attribute("COMPOSITE");
        const char *conductivity = region->Attribute("CONDUCTIVITY");
        ASSERTL0(composite && conductivity,
                 "Electrical regions require COMPOSITE and CONDUCTIVITY.");
        std::string ids(composite);
        ASSERTL0(ids.size() > 3 && ids.substr(0, 2) == "C[" &&
                     ids.back() == ']',
                 "Electrical COMPOSITE must have the form C[...].");
        LibUtilities::Equation expression(m_session->GetInterpreter(),
                                          conductivity);
        NekDouble sigma = expression.Evaluate();
        ASSERTL0(std::isfinite(sigma) && sigma > 0.0,
                 "Electrical conductivity must be finite and positive.");
        SpatialDomains::CompositeMap composites;
        m_graph->GetCompositeList(ids.substr(2, ids.size() - 3), composites);
        for (auto &entry : composites)
        {
            for (auto &geometry : entry.second->m_geomVec)
            {
                ASSERTL0(geometry->GetShapeDim() == m_dimension,
                         "Electrical material composites must contain volume "
                         "elements.");
                ASSERTL0(materials
                             .emplace(geometry->GetGlobalID(),
                                      std::make_pair(sigma, type == "FLUID"))
                             .second,
                         "Electrical material regions overlap.");
            }
        }
    }
    int fluidPoints = 0;
    for (int e = 0; e < m_fields[0]->GetExpSize(); ++e)
    {
        auto exp      = m_fields[0]->GetExp(e);
        auto material = materials.find(exp->GetGeom()->GetGlobalID());
        ASSERTL0(
            material != materials.end(),
            "Every electrical element must belong to one material region.");
        int offset = m_fields[0]->GetPhys_Offset(e);
        for (int q = 0; q < exp->GetTotPoints(); ++q)
        {
            m_sigma[offset + q] = material->second.first;
            m_fluid[offset + q] = material->second.second ? 1 : 0;
            fluidPoints += m_fluid[offset + q];
        }
    }
    m_comm->AllReduce(fluidPoints, LibUtilities::ReduceSum);
    ASSERTL0(fluidPoints > 0, "The electrical mesh has no fluid elements.");
    m_varcoeff[StdRegions::eVarCoeffD00] = m_sigma;
    m_varcoeff[StdRegions::eVarCoeffD11] = m_sigma;
    if (m_dimension == 3)
    {
        m_varcoeff[StdRegions::eVarCoeffD22] = m_sigma;
    }

    m_electric             = Array<OneD, NekDouble>(3, 0.0);
    m_magnetic             = Array<OneD, NekDouble>(3, 0.0);
    const std::string axes = "xyz";
    for (int d = 0; d < 3; ++d)
    {
        for (char kind : {'E', 'B'})
        {
            std::string var = std::string(1, kind) + axes[d];
            if (m_session->DefinesFunction("ElectricMagneticFields", var))
            {
                NekDouble value =
                    m_session->GetFunction("ElectricMagneticFields", var)
                        ->Evaluate(0.0, 0.0, 0.0, 0.0);
                (kind == 'E' ? m_electric : m_magnetic)[d] = value;
            }
        }
    }

    auto config = LibUtilities::SessionNodeFromXml(
        m_session->GetElement("Nektar/Coupling"), "COUPLING");
    ASSERTL0(StringUtils::IEquals(config.GetAttribute("TYPE"), "Cwipi"),
             "CoupledElectricPotential requires TYPE=\"Cwipi\".");
    m_coupling =
        SolverUtils::GetCouplingFactory().CreateInstance("Cwipi", m_fields[0]);
    m_recvNames = {"u", "v"};
    if (m_dimension == 3)
    {
        m_recvNames.push_back("w");
    }
    m_recvNames.push_back("mhd_time");
    m_recvNames.push_back("mhd_stop");
    m_sendNames = {"phi"};
    ASSERTL0(m_coupling->GetRecvFieldNames() == m_recvNames &&
                 m_coupling->GetSendFieldNames() == m_sendNames,
             "Electrical coupling must receive u,v[,w],mhd_time,mhd_stop and "
             "send phi.");
    auto options = m_coupling->GetConfig();
    ASSERTL0(
        options.at("SENDSTEPS") == "1" && options.at("RECEIVESTEPS") == "1" &&
            std::stod(options.at("FILTERWIDTH")) <= 0.0 &&
            StringUtils::IEquals(options.at("RECEIVEPROJECTION"), "Local") &&
            StringUtils::IEquals(options.at("NOTLOCMETHOD"), "Keep"),
        "Electrical coupling requires SendSteps=ReceiveSteps=1, "
        "FilterWidth<=0, ReceiveProjection=Local and NotLocMethod=Keep.");
    m_received = Array<OneD, Array<OneD, NekDouble>>(m_recvNames.size());
    for (auto &field : m_received)
    {
        field = Array<OneD, NekDouble>(nq, 0.0);
    }
}

void CoupledElectricPotential::v_DoInitialise(bool)
{
    Vmath::Zero(m_fields[0]->GetNcoeffs(), m_fields[0]->UpdateCoeffs(), 1);
    Vmath::Zero(m_fields[0]->GetTotPoints(), m_fields[0]->UpdatePhys(), 1);
    m_fields[0]->SetPhysState(true);
}

void CoupledElectricPotential::v_DoSolve()
{
    int exchange = 0;
    Array<OneD, Array<OneD, NekDouble>> reply(1);
    while (true)
    {
        // Not-located solid points retain zero. The material mask below is
        // authoritative even if the location tolerance includes nearby solids.
        for (auto &field : m_received)
        {
            Vmath::Zero(field.size(), field, 1);
        }
        m_coupling->Receive(exchange, m_time, m_received, m_recvNames);
        NekDouble stop    = -std::numeric_limits<NekDouble>::max();
        NekDouble minStop = std::numeric_limits<NekDouble>::max();
        NekDouble time    = -std::numeric_limits<NekDouble>::max();
        NekDouble minTime = std::numeric_limits<NekDouble>::max();
        for (int q = 0; q < m_fluid.size(); ++q)
        {
            if (m_fluid[q])
            {
                stop    = std::max(stop, m_received[m_dimension + 1][q]);
                minStop = std::min(minStop, m_received[m_dimension + 1][q]);
                time    = std::max(time, m_received[m_dimension][q]);
                minTime = std::min(minTime, m_received[m_dimension][q]);
            }
        }
        m_comm->AllReduce(stop, LibUtilities::ReduceMax);
        m_comm->AllReduce(minStop, LibUtilities::ReduceMin);
        m_comm->AllReduce(time, LibUtilities::ReduceMax);
        m_comm->AllReduce(minTime, LibUtilities::ReduceMin);
        ASSERTL0(
            std::abs(stop - minStop) < 1e-8 &&
                std::abs(std::abs(stop) - 1.0) < 1e-8,
            "Electrical fluid points did not receive a complete MHD request. "
            "Check CWIPI overlap; solid points alone may be unlocated.");
        ASSERTL0(time - minTime < 1e-8 * std::max(1.0, std::abs(time)),
                 "Electrical fluid points did not receive a consistent request "
                 "time. "
                 "Check CWIPI overlap and quadrature resolution.");
        m_time = time;
        if (stop < 0.5)
        {
            SetBoundaryConditions(m_time);
            SolvePotential();
        }
        reply[0] = m_fields[0]->UpdatePhys();
        m_coupling->Send(exchange++, m_time, reply, m_sendNames);
        if (stop >= 0.5)
        {
            break;
        }
    }
    m_coupling->Finalize();
    m_coupling.reset();
    if (m_comm->GetRank() == 0)
    {
        std::cout << "Electric potential solves: " << exchange - 1 << std::endl;
    }
}

void CoupledElectricPotential::SolvePotential()
{
    const int nq = m_sigma.size();
    const int nc = m_fields[0]->GetNcoeffs();
    Array<OneD, NekDouble> rhs(nc, 0.0), product(nc), flux(nq);
    for (int d = 0; d < m_dimension; ++d)
    {
        int j = (d + 1) % 3, k = (d + 2) % 3;
        for (int q = 0; q < nq; ++q)
        {
            NekDouble motional = 0.0;
            if (m_fluid[q])
            {
                NekDouble uj = j < m_dimension ? m_received[j][q] : 0.0;
                NekDouble uk = k < m_dimension ? m_received[k][q] : 0.0;
                motional     = uj * m_magnetic[k] - uk * m_magnetic[j];
            }
            flux[q] = m_sigma[q] * (m_electric[d] + motional);
        }
        m_fields[0]->IProductWRTDerivBase(d, flux, product);
        // HelmSolve negates a coefficient-space RHS. Its positive stiffness
        // must receive + integral(sigma (E0 + u x B) . grad(test)).
        Vmath::Vsub(nc, rhs, 1, product, 1, rhs, 1);
    }
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 0.0;
    m_fields[0]->HelmSolve(rhs, m_fields[0]->UpdateCoeffs(), factors,
                           m_varcoeff, StdRegions::NullVarFactorsMap,
                           NullNekDouble1DArray, false);
    m_fields[0]->BwdTrans(m_fields[0]->GetCoeffs(), m_fields[0]->UpdatePhys());
    m_fields[0]->SetPhysState(true);
}

Array<OneD, bool> CoupledElectricPotential::v_GetSystemSingularChecks()
{
    return Array<OneD, bool>(m_session->GetVariables().size(), true);
}

void CoupledElectricPotential::v_ExtraFldOutput(
    std::vector<Array<OneD, NekDouble>> &fieldcoeffs,
    std::vector<std::string> &variables)
{
    // Current is element-local: conductivity and tangential current may jump
    // at material interfaces. Do not apply a continuous mass projection.
    const int nq = m_sigma.size();
    Array<OneD, NekDouble> derivative(nq), current(nq);
    for (int d = 0; d < 3; ++d)
    {
        Vmath::Zero(nq, derivative, 1);
        if (d < m_dimension)
        {
            m_fields[0]->PhysDeriv(MultiRegions::DirCartesianMap[d],
                                   m_fields[0]->GetPhys(), derivative);
        }
        int j = (d + 1) % 3, k = (d + 2) % 3;
        for (int q = 0; q < nq; ++q)
        {
            NekDouble uj =
                m_fluid[q] && j < m_dimension ? m_received[j][q] : 0.0;
            NekDouble uk =
                m_fluid[q] && k < m_dimension ? m_received[k][q] : 0.0;
            current[q] = m_sigma[q] * (m_electric[d] + uj * m_magnetic[k] -
                                       uk * m_magnetic[j] - derivative[q]);
        }
        Array<OneD, NekDouble> coeffs(m_fields[0]->GetNcoeffs());
        m_fields[0]->FwdTransLocalElmt(current, coeffs);
        fieldcoeffs.push_back(coeffs);
        variables.push_back(std::string("J") + "xyz"[d]);
    }
    Array<OneD, NekDouble> conductivity(m_fields[0]->GetNcoeffs());
    m_fields[0]->FwdTransLocalElmt(m_sigma, conductivity);
    fieldcoeffs.push_back(conductivity);
    variables.push_back("sigma");
}

void CoupledElectricPotential::v_GenerateSummary(
    SolverUtils::SummaryList &summary)
{
    EquationSystem::SessionSummary(summary);
    SolverUtils::AddSummaryItem(summary, "Electric potential",
                                "CWIPI fluid/solid");
}

} // namespace Nektar
