////////////////////////////////////////////////////////////////////////////////
//
// File: EquationSystem.cpp
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

#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <SolverCore/EquationSystems/EquationSystem.h>

#include <Operators/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp>
#include <Operators/Norm/NormL2/NormL2Op.hpp>
#include <Operators/Norm/NormLinf/NormLinfOp.hpp>
#include <vector>

#include "Operators/Field/Block.hpp"

namespace Nektar::SolverCore
{

EquationSystemFactory &GetEquationSystemFactory()
{
    static EquationSystemFactory instance;
    return instance;
}

EquationSystem::EquationSystem(
    const LibUtilities::SessionReaderSharedPtr &session,
    const SpatialDomains::MeshGraphSharedPtr &graph)
    : m_comm(session->GetComm()), m_session(session), m_graph(graph)
{
    const auto &filenames = m_session->GetFilenames();
    for (size_t i = 0; i < filenames.size(); ++i)
    {
        m_fieldMetaDataMap["SessionName" + std::to_string(i)] = filenames[i];
    }

    // Get variables and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();

    // Get coordinate system (space) dimensions
    m_coordim = m_graph->GetSpaceDimension();
    m_expdim  = m_graph->GetMeshDimension();

    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    // Check and set definitions for homogeneous/Fourier dimensions
    CheckHomogeneousDimensions();

    // Hard set the GlobalSysSoln to the only supported type: IterativeFull
    m_session->SetSolverInfo("GlobalSysSoln", "IterativeFull");

    // Get projection CG, DG or mixed CG/DG
    const std::string projection = m_session->GetSolverInfo("PROJECTION");
    m_projectionType             = GetProjectionType(projection);
}

void EquationSystem::v_InitObject(bool declareExpansionLists)
{
    m_sessionName = m_session->GetSessionName();
    m_fieldIo     = LibUtilities::FieldIO::CreateDefault(m_session);

    // Check boundary conditions, if one field uses only Neumann/Periodic BCs
    m_checkIfSystemSingular = v_GetSystemSingularChecks();

    // Define ExpLists and Fields, ie allocate memory for entire mesh
    if (declareExpansionLists)
    {
        DeclareExpansionLists();
    }
}

void EquationSystem::v_DoInitialise(bool)
{
}

void EquationSystem::v_DoSolve()
{
}

void EquationSystem::v_Output()
{
    WriteFld(m_sessionName + ".fld");
}

/*
 * Create m_field and m_field_coeff for holding the solution, comparing against
 * an ExactSolution and writing to file. Note any solver implementation can
 * choose to not call EquationSystem::v_InitialiseFields, but still has to
 * initialise m_fields and m_fields_coeff.
 * @param expansionList Allows specialising the expansionList for the Field.
 * @param variables Allows specialising the solution field to contain a subset
 * of all fields.
 */
void EquationSystem::v_InitialiseFields()
{
    // Create solution fields in physical space (at quadrature points)
    auto bAtr_phys =
        GetBlockAttributes<double, FieldState::Phys>(m_expansionLists[0]);
    m_fields = Field<double, FieldState::Phys>("solution", bAtr_phys,
                                               m_nVariables, m_npointsZ);

    // Create solution fields in coefficient space (coefficients of polynomial
    // basis function)
    auto bAtr_coeff =
        GetBlockAttributes<double, FieldState::Coeff>(m_expansionLists[0]);
    m_fields_coeff = Field<double, FieldState::Coeff>(
        "solution coeff", bAtr_coeff, m_nVariables, m_npointsZ);

    // Zero both fields
    m_math.zero(m_fields);
    m_math.zero(m_fields_coeff);
}

void EquationSystem::v_GenerateSummary(SummaryList &summary)
{
    if (m_session->DefinesSolverInfo("EQTYPE"))
    {
        AddSummaryItem(summary, "EquationType",
                       m_session->GetSolverInfo("EQTYPE"));
    }

    AddSummaryItem(summary, "Session Name", m_sessionName);
    AddSummaryItem(summary, "Coordinate Dim.", m_coordim);
    AddSummaryItem(summary, "Expansion Dim.", m_expdim);

    if (m_expansionLists.size() > 0)
    {
        AddSummaryItem(summary, "Max SEM Exp. Order",
                       m_expansionLists[0]->EvalBasisNumModesMax());
    }

    // TODO can we add number of GPUs/Devices?
    if (m_comm->GetSize() > 1)
    {
        AddSummaryItem(summary, "Num. Processes", m_comm->GetSize());
    }

    AddSummaryItem(summary, "Projection Type",
                   GetProjectionString(m_projectionType));
}

void EquationSystem::v_PrintNorms(std::ostream &out)
{
    // Create workspace Field
    auto wsp_phys = Field<double, FieldState::Phys>(
        "exact solution",
        GetBlockAttributes<double, FieldState::Phys>(m_expansionLists[0]),
        m_nVariables, m_npointsZ);
    m_math.zero(wsp_phys);

    // Evaluate exact solution and compute diff to discrete solution
    if (m_session->DefinesFunction("ExactSolution"))
    {
        SessionFunction exactSolution(m_session, m_expansionLists[0],
                                      "ExactSolution");
        exactSolution.EvaluateExpression(m_variables, wsp_phys, m_time);
    }
    m_math.sub(m_fields, wsp_phys, wsp_phys);

    // Compute L2 norm
    auto l2NormOp = NormL2Op<double>::Create(m_expansionLists[0], m_variables);
    l2NormOp->Apply(wsp_phys);
    auto l2Errors = l2NormOp->GetNorms();

    // Compute Linf norm
    auto linfNormOp =
        NormLinfOp<double>::Create(m_expansionLists[0], m_variables);
    linfNormOp->Apply(wsp_phys);
    auto linfErrors = linfNormOp->GetNorms();

    // Print norms
    for (unsigned int n = 0; n < m_nVariables; ++n)
    {
        if (m_comm->GetRank() == 0)
        {
            out << "L 2 error (variable " << m_variables[n]
                << ") : " << l2Errors[n] << std::endl;
            out << "L inf error (variable " << m_variables[n]
                << ") : " << linfErrors[n] << std::endl;
        }
    }
}

void EquationSystem::v_WriteFld(const std::string &outname)
{
    // Ugly and temporary workaround: In-place conversion to coeffs
    // Assume DG has result already in m_fields_coeff
    if (m_projectionType == MultiRegions::eGalerkin)
    {
        m_fwdTransOp->UpdateBndCoeffs(m_time);
        m_math.zero(m_fields_coeff);
        m_fwdTransOp->Apply(m_fields, m_fields_coeff);
    }

    std::vector<std::vector<double>> fieldCoeffs;
    std::vector<std::string> variables(m_expansionLists.size());

    // Get vector with all component data and sort into vectors per component
    std::vector<double> allCoeffs = m_fields_coeff.ToVector<double>();
    auto ncomp                    = m_fields_coeff.GetNumComponents();
    auto ncoeffsTotal = allCoeffs.size(); // use allCoeffs to not have padding
    auto ncoeffsComp  = ncoeffsTotal / ncomp;
    fieldCoeffs.reserve(ncomp);
    for (unsigned int nc = 0; nc < ncomp; ++nc)
    {
        auto first = allCoeffs.begin() + nc * ncoeffsComp;
        auto last  = allCoeffs.begin() + (nc + 1) * ncoeffsComp;

        fieldCoeffs.emplace_back(std::make_move_iterator(first),
                                 std::make_move_iterator(last));
    }

    // Note: FieldDef holds nummodes, basis, shapetype, variable strings, ..
    auto fieldDef = m_expansionLists[0]->GetFieldDefinitions();
    std::vector<std::vector<double>> fieldData(fieldDef.size());

    for (unsigned int nc = 0; nc < ncomp; ++nc)
    {
        for (size_t i = 0; i < fieldDef.size(); ++i)
        {
            // Note Legacy transforms NodalToModal, if necessary, this code does
            // not
            fieldDef[i]->m_fields.push_back(m_variables[nc]);
            m_expansionLists[0]->AppendFieldData(fieldDef[i], fieldData[i],
                                                 fieldCoeffs[nc]);
        }
    }

    m_fieldMetaDataMap["Time"] = std::to_string(m_time);

    m_fieldIo->Write(outname, fieldDef, fieldData, m_fieldMetaDataMap,
                     m_session->GetBackups());
}

std::vector<bool> EquationSystem::v_GetSystemSingularChecks()
{
    return std::vector<bool>(m_session->GetVariables().size(), false);
}

void EquationSystem::DeclareExpansionLists()
{
    m_expansionLists =
        std::vector<MultiRegions::ExpListSharedPtr>(m_nVariables);

    ASSERTL0(m_nVariables > 0, "Session must define at least one variable.");

    if (m_projectionType == MultiRegions::eGalerkin ||
        m_projectionType == MultiRegions::eMixed_CG_Discontinuous)
    {
        auto firstField =
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                m_session, m_graph, m_variables[0], false,
                m_checkIfSystemSingular[0]);
        m_expansionLists[0] = firstField;

        for (int i = 1; i < m_nVariables; ++i)
        {
            if (m_graph->SameExpansionInfo(m_variables[0], m_variables[i]))
            {
                m_expansionLists[i] =
                    MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                        *firstField, m_graph, m_variables[i], false,
                        m_checkIfSystemSingular[i]);
            }
            else
            {
                m_expansionLists[i] =
                    MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                        m_session, m_graph, m_variables[i], false,
                        m_checkIfSystemSingular[i]);
            }
        }
    }
    else
    {
        bool setupJustDG = true;
        auto firstField =
            MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
                m_session, m_graph, m_variables[0], setupJustDG, false);
        m_expansionLists[0] = firstField;

        for (int i = 1; i < m_nVariables; ++i)
        {
            if (m_graph->SameExpansionInfo(m_variables[0], m_variables[i]))
            {
                m_expansionLists[i] =
                    MemoryManager<MultiRegions::DisContField>::
                        AllocateSharedPtr(*firstField, m_graph, m_variables[i],
                                          setupJustDG, false);
            }
            else
            {
                m_expansionLists[i] =
                    MemoryManager<MultiRegions::DisContField>::
                        AllocateSharedPtr(m_session, m_graph, m_variables[i],
                                          setupJustDG, false);
            }
        }
    }

    for (int i = 0; i < m_nVariables; ++i)
    {
        // Initialise DataWarehouse for Operators
        m_expansionLists[i]->SetDataWarehouse();
        m_expansionLists[i]->GetTrace()->SetDataWarehouse();
    }
}

void EquationSystem::CheckHomogeneousDimensions()
{
    if (m_session->DefinesSolverInfo("HOMOGENEOUS"))
    {
        std::string HomoStr = m_session->GetSolverInfo("HOMOGENEOUS");

        if ((HomoStr == "HOMOGENEOUS1D") || (HomoStr == "Homogeneous1D") ||
            (HomoStr == "1D") || (HomoStr == "Homo1D"))
        {
            m_HomogeneousType = eHomogeneous1D;
            m_session->LoadParameter("LZ", m_LhomZ);
            m_HomoDirec = 1;

            if (m_session->DefinesSolverInfo("ModeType"))
            {
                m_session->MatchSolverInfo("ModeType", "SingleMode",
                                           m_singleMode, false);
                m_session->MatchSolverInfo("ModeType", "HalfMode", m_halfMode,
                                           false);
                m_session->MatchSolverInfo("ModeType", "MultipleModes",
                                           m_multipleModes, false);
            }

            // Stability Analysis flags
            if (m_session->DefinesSolverInfo("ModeType"))
            {
                if (m_singleMode)
                {
                    m_npointsZ = 2;
                }
                else if (m_halfMode)
                {
                    m_npointsZ = 1;
                }
                else if (m_multipleModes)
                {
                    m_npointsZ = m_session->GetParameter("HomModesZ");
                }
                else
                {
                    ASSERTL0(false, "SolverInfo ModeType not valid");
                }
            }
            else
            {
                m_npointsZ = m_session->GetParameter("HomModesZ");
            }
        }

        if ((HomoStr == "HOMOGENEOUS2D") || (HomoStr == "Homogeneous2D") ||
            (HomoStr == "2D") || (HomoStr == "Homo2D"))
        {
            m_HomogeneousType = eHomogeneous2D;
            m_session->LoadParameter("HomModesY", m_npointsY);
            m_session->LoadParameter("LY", m_LhomY);
            m_session->LoadParameter("HomModesZ", m_npointsZ);
            m_session->LoadParameter("LZ", m_LhomZ);
            m_HomoDirec = 2;
        }

        if ((HomoStr == "HOMOGENEOUS3D") || (HomoStr == "Homogeneous3D") ||
            (HomoStr == "3D") || (HomoStr == "Homo3D"))
        {
            m_HomogeneousType = eHomogeneous3D;
            m_session->LoadParameter("HomModesY", m_npointsY);
            m_session->LoadParameter("LY", m_LhomY);
            m_session->LoadParameter("HomModesZ", m_npointsZ);
            m_session->LoadParameter("LZ", m_LhomZ);
            m_HomoDirec = 2;
        }

        m_session->MatchSolverInfo("USEFFT", "FFTW", m_useFFT, false);

        m_session->MatchSolverInfo("DEALIASING", "True", m_homogen_dealiasing,
                                   false);
    }
    else
    {
        // set to default value so can use to identify 2d or 3D
        // (homogeneous) expansions
        m_npointsZ = 1;
    }
}

/**
 * @brief Compute the projection for the unsteady ADR problem.
 *
 * @param in    Given fields.
 * @param out   CG-projected fields.
 * @param time  Time.
 */
void EquationSystem::v_DoProjection(Field<double, FieldState::Phys> &in,
                                    Field<double, FieldState::Phys> &out,
                                    const double time)
{
    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Discontinuous projection
            if (&in != &out)
            {
                m_math.copy(in, out);
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Continuous projection
            // Note we could use the cheaper operators: AvgAssemble or
            // GlobalToLocal
            m_fwdTransOp->UpdateBndCoeffs(time);
            m_fwdTransOp->Apply(in, m_fields_coeff);
            m_bwdTransOp->Apply(m_fields_coeff, out);
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }
}

/*
 *  Create and initialise all operators for this solver
 */
void EquationSystem::v_InitialiseOperators()
{
    // Set DataWarehouse for operators
    m_expansionLists[0]->SetDataWarehouse();

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Discontinuous projection requires trace datawarehouse
            m_expansionLists[0]->GetTrace()->SetDataWarehouse();
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Continuous projection
            m_bwdTransOp =
                BwdTransOp<double>::Create(m_expansionLists[0], m_variables);

            // Projection operators
            m_fwdTransOp =
                FwdTransOp<double>::Create(m_expansionLists[0], m_variables);
            auto preconOp =
                PreconOp<double>::Create(m_expansionLists[0], m_variables);
            // The projection solve is a mass-matrix problem, so use
            // ConjGrad regardless of the ADR system solver configured in
            // the session.
            auto linsolverOp = LinearSolverOp<double>::Create(
                m_expansionLists[0], m_variables, "ConjGrad");
            m_fwdTransOp->SetLinearSolver(linsolverOp);
            m_fwdTransOp->SetPrecon(preconOp);
            m_fwdTransOp->UpdatePrecon();
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }
}
} // namespace Nektar::SolverCore
