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

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <SolverCore/EquationSystems/EquationSystem.h>

#include <SolverCore/NormOps/NormL2/NormL2Op.hpp>
#include <SolverCore/NormOps/NormLinf/NormLinfOp.hpp>
#include <algorithm>
#include <iterator>
#include <vector>

#include <LibUtilities/BasicUtils/Field/Block.hpp>

namespace Nektar::SolverCore
{

namespace
{
// True when two field definitions describe the same elements with the same
// expansion, so that their variables can share one block of a field file.
bool SameFieldLayout(const LibUtilities::FieldDefinitionsSharedPtr &a,
                     const LibUtilities::FieldDefinitionsSharedPtr &b)
{
    return a->m_shapeType == b->m_shapeType &&
           a->m_elementIDs == b->m_elementIDs && a->m_basis == b->m_basis &&
           a->m_numHomogeneousDir == b->m_numHomogeneousDir &&
           a->m_homogeneousLengths == b->m_homogeneousLengths &&
           a->m_homoStrips == b->m_homoStrips &&
           a->m_homogeneousSIDs == b->m_homogeneousSIDs &&
           a->m_homogeneousZIDs == b->m_homogeneousZIDs &&
           a->m_homogeneousYIDs == b->m_homogeneousYIDs &&
           a->m_points == b->m_points && a->m_pointsDef == b->m_pointsDef &&
           a->m_uniOrder == b->m_uniOrder && a->m_numModes == b->m_numModes &&
           a->m_numPoints == b->m_numPoints &&
           a->m_numPointsDef == b->m_numPointsDef;
}
} // namespace

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
    std::string execName =
        MultiRegions::Operator<double>::GetOpExecSpace(m_session);
    m_math = Math::MathHelper(execName);

    // Check and set definitions for homogeneous/Fourier dimensions
    CheckHomogeneousDimensions();

    // Hard set the GlobalSysSoln to the only supported type: IterativeFull
    m_session->SetSolverInfo("GlobalSysSoln", "IterativeFull");

    // Without a .opt file the expansion lists would time every
    // implementation of the legacy collection operators to write one, at a
    // cost in time and memory. The operators of this library do not use
    // them, and the few legacy operations still called are served by the
    // default implementations, so they are timed only on request
    // (--write-opt-file).
    // The grouping of the elements into collections, which sets the blocks
    // of the fields, does not depend on the timing.
    if (!m_session->DefinesCmdLineArgument("write-opt-file"))
    {
        m_session->SetUpdateOptFile(false);
    }

    // Get projection CG, DG or mixed CG/DG
    const std::string projection = m_session->GetSolverInfo("PROJECTION");
    m_projectionType             = GetProjectionType(projection);
}

/**
 * @brief Publish #m_fields as the solution, as the one physical group, and
 * the output coefficients of v_GetOutputFieldViews() as the coefficient
 * groups.
 */
FieldRegistry EquationSystem::v_GetFieldRegistry()
{
    ASSERTL0(!m_expansionLists.empty(),
             "EquationSystem has no physical fields to publish.");
    PhysicalFieldView solution{m_fields, m_expansionLists[0], m_coordim,
                               MakeFieldComponents(m_variables)};
    FieldRegistry registry;
    registry.SetRole(std::string(FieldRoles::Solution), solution);
    registry.SetPhysicalGroups({solution});
    registry.SetCoefficientProvider(
        [this]() { return v_GetOutputFieldViews(); });
    return registry;
}

void EquationSystem::v_InitObject(bool declareExpansionLists)
{
    m_sessionName = m_session->GetSessionName();
    m_fieldIo     = LibUtilities::FieldIO::CreateDefault(m_session);

    // How the initial condition is brought into the polynomial space; see
    // UnsteadySystem::v_SetInitialConditions(). A continuous run must be
    // brought into the continuous space, by Average or Galerkin, and
    // defaults to Average. A discontinuous run holds its state at the
    // quadrature points, where it is not confined to the polynomial space.
    // Galerkin confines it, which makes a field file a faithful record of
    // the state and so makes a restart exact - at the price of changing
    // every discontinuous answer, which is why None is its default. True and
    // False are accepted for Galerkin and None.
    const bool continuous        = m_projectionType == MultiRegions::eGalerkin;
    m_initialConditionProjection = continuous
                                       ? InitialConditionProjection::Average
                                       : InitialConditionProjection::None;
    if (m_session->DefinesSolverInfo("ProjectInitialConditions"))
    {
        const std::string value =
            m_session->GetSolverInfo("ProjectInitialConditions");
        if (StringUtils::IEquals(value, "Average"))
        {
            m_initialConditionProjection = InitialConditionProjection::Average;
        }
        else if (StringUtils::IEquals(value, "Galerkin") ||
                 StringUtils::IEquals(value, "True"))
        {
            m_initialConditionProjection = InitialConditionProjection::Galerkin;
        }
        else if (StringUtils::IEquals(value, "None") ||
                 StringUtils::IEquals(value, "False"))
        {
            m_initialConditionProjection = InitialConditionProjection::None;
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "ProjectInitialConditions must be Average, Galerkin or "
                     "None, not " +
                         value + ".");
        }

        const InitialConditionProjection unavailable =
            continuous ? InitialConditionProjection::None
                       : InitialConditionProjection::Average;
        if (m_initialConditionProjection == unavailable)
        {
            NEKERROR(ErrorUtil::efatal,
                     "ProjectInitialConditions = " + value +
                         " is not available for the " +
                         GetProjectionString(m_projectionType) +
                         " projection; use " +
                         (continuous ? "Average or Galerkin."
                                     : "Galerkin or None."));
        }
    }

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
    // Get interleave width
    auto interleaveWidth =
        MultiRegions::Operator<double>::GetDefaultInterleaveWidth(m_session);

    // Create solution fields in physical space (at quadrature points)
    auto bAtr_phys = MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
        m_expansionLists[0], interleaveWidth);
    m_fields = LibUtilities::Field<double, FieldState::Phys>(
        "solution", bAtr_phys, m_nVariables, m_npointsZ);

    // Create solution fields in coefficient space (coefficients of polynomial
    // basis function)
    auto bAtr_coeff =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_expansionLists[0], interleaveWidth);
    m_fields_coeff = LibUtilities::Field<double, FieldState::Coeff>(
        "solution coeff", bAtr_coeff, m_nVariables, m_npointsZ);

    // Scratch for the discontinuous phys to coeff projection, which needs an
    // intermediate between the inner product and the inverse mass multiply.
    // Only ProjectPhysToCoeffs() reads it, and that is discontinuous only, so
    // a continuous run would carry a whole coefficient field for nothing.
    if (m_projectionType == MultiRegions::eDiscontinuous)
    {
        m_fields_coeff_tmp = LibUtilities::Field<double, FieldState::Coeff>(
            "solution coeff tmp", bAtr_coeff, m_nVariables, m_npointsZ);
        m_math.zero(m_fields_coeff_tmp);
    }

    // Zero the solution fields
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
    // Create workspace Field, matching the interleave width of m_fields so
    // that the subtraction below combines consistently laid-out data. This
    // is recomputed from the session rather than read off m_fields' blocks
    // since GetBlocks() can be empty on a rank with no local elements.
    const auto interleaveWidth =
        MultiRegions::Operator<double>::GetDefaultInterleaveWidth(m_session);
    auto wsp_phys = LibUtilities::Field<double, FieldState::Phys>(
        "exact solution",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            m_expansionLists[0], interleaveWidth),
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

/**
 * @brief Project the discontinuous physical field into #m_fields_coeff.
 *
 * The element local L2 projection \f$M^{-1}B^{T}W\f$, assembled from the
 * inner product and the inverse mass multiply. m_fwdTransOp cannot serve:
 * it is a global mass matrix solve carrying boundary conditions, and is
 * only built for eGalerkin. Only valid for eDiscontinuous.
 */
void EquationSystem::ProjectPhysToCoeffs(
    LibUtilities::Field<double, FieldState::Phys> &phys)
{
    ASSERTL1(m_projectionType == MultiRegions::eDiscontinuous,
             "The element local projection is only built for a discontinuous "
             "projection; a continuous one goes through m_fwdTransOp.");

    m_iProductWRTBaseOp->Apply(phys, m_fields_coeff_tmp);
    m_multiplyByElmtInvMassOp->Apply(m_fields_coeff_tmp, m_fields_coeff);
}

/**
 * @brief Build the continuous projection, a mass matrix solve, for the fields
 * of an expansion list.
 *
 * The projection solve is a mass matrix problem, so it uses ConjGrad
 * regardless of the system solver configured in the session. The
 * preconditioner is set up here.
 *
 * @param   expList     Continuous expansion list of the fields.
 * @param   variables   Names of the field components.
 *
 * @return  The projection operator.
 */
std::shared_ptr<FwdTransOp<double>> EquationSystem::CreateFwdTransOp(
    const MultiRegions::ExpListSharedPtr &expList,
    const std::vector<std::string> &variables)
{
    auto fwdTransOp = FwdTransOp<double>::Create(expList, variables);
    auto preconOp   = PreconOp<double>::Create(expList, variables);
    auto linsolverOp =
        LinearSolverOp<double>::Create(expList, variables, "ConjGrad");
    fwdTransOp->SetLinearSolver(linsolverOp);
    fwdTransOp->SetPrecon(preconOp);
    fwdTransOp->UpdatePrecon();
    return fwdTransOp;
}

LibUtilities::FunctionType EquationSystem::GetInitialConditionType(
    const std::vector<std::string> &variables)
{
    LibUtilities::FunctionType vType = LibUtilities::eFunctionTypeNone;
    if (m_session->DefinesFunction("InitialConditions"))
    {
        vType = m_session->GetFunctionType("InitialConditions", variables[0]);

        // Check all variables use same FunctionType
        for (unsigned int i = 1; i < variables.size(); ++i)
        {
            LibUtilities::FunctionType vType2 =
                m_session->GetFunctionType("InitialConditions", variables[i]);
            ASSERTL0(vType == vType2,
                     "Currently support either all InitialConditions from file "
                     "<F> or all from expression <E>.");
        }
    }

    return vType;
}

void EquationSystem::v_WriteFld(const std::string &outname)
{
    m_fieldMetaDataMap["Time"] = std::to_string(m_time);
    WriteOutputFields(outname, v_GetOutputFieldViews(), m_fieldMetaDataMap);
}

CoefficientFieldViews EquationSystem::v_GetOutputFieldViews()
{
    // The solution is advanced in physical space, so project back to
    // coefficients before exposing it to output filters. CG uses its global
    // transform with boundary coefficients; DG uses the element-local L2
    // projection required by the restart path.
    if (m_projectionType == MultiRegions::eGalerkin)
    {
        m_fwdTransOp->UpdateBndCoeffs(m_time);
        m_math.zero(m_fields_coeff);
        m_fwdTransOp->Apply(m_fields, m_fields_coeff);
    }
    else if (m_projectionType == MultiRegions::eDiscontinuous)
    {
        ProjectPhysToCoeffs(m_fields);
    }

    return {{m_fields_coeff, m_expansionLists[0], m_coordim,
             MakeFieldComponents(m_variables)}};
}

/**
 * @brief Write coefficient fields to a field file.
 *
 * A view whose expansion embeds losslessly in that of the first view
 * (ExpList::EmbedsLosslessly()), such as a pressure one order below the
 * velocity, is written through the first view's expansion, its missing
 * modes zero. Every variable then shares one set of definitions, which every
 * Nektar++ reader accepts. Any other view is written with the field
 * definitions of its own expansion, so that fields of different orders can
 * share a file. A view on the same element layout as the previous one adds
 * its variables to the previous definitions instead of starting a new block.
 *
 * @param   outname   Path of the file.
 * @param   fields    Coefficient views to write, in order.
 * @param   metadata  Metadata of the file.
 * @param   scale     Factor applied to every coefficient as it is written.
 */
void EquationSystem::WriteOutputFields(
    const std::string &outname, const CoefficientFieldViews &fields,
    const LibUtilities::FieldMetaDataMap &metadata, const double scale)
{
    ASSERTL0(!fields.empty(),
             "EquationSystem::WriteOutputFields requires at least one field.");

    std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDef;
    std::vector<std::vector<double>> fieldData;

    // First definition and number of definitions of the previous view.
    size_t blockStart = 0;
    size_t blockSize  = 0;

    for (const auto &view : fields)
    {
        auto &field          = view.field.get();
        const auto variables = ComponentNames(view.components);
        const auto ncomp     = field.GetNumComponents();

        ASSERTL0(ncomp == variables.size(),
                 "Output field component and variable count mismatch.");

        // ToVector performs the only device-to-host transfer in this output
        // path, removes block padding and returns the coefficients element by
        // element whatever the interleave width. Apply the average scale while
        // separating components so the device accumulator remains unchanged.
        std::vector<double> allCoeffs = field.ToVector<double>();
        ASSERTL0(allCoeffs.size() % ncomp == 0,
                 "Output field has inconsistent component storage.");
        const auto ncoeffsComp = allCoeffs.size() / ncomp;

        const auto &firstExpansion = fields.front().expansion;
        const bool embed           = view.expansion != firstExpansion &&
                           view.expansion->EmbedsLosslessly(firstExpansion);
        const auto &expansion = embed ? firstExpansion : view.expansion;

        auto viewFieldDef = expansion->GetFieldDefinitions();
        std::vector<std::vector<double>> viewFieldData(viewFieldDef.size());

        for (unsigned int nc = 0; nc < ncomp; ++nc)
        {
            const auto first = allCoeffs.begin() + nc * ncoeffsComp;
            std::vector<double> component(ncoeffsComp);
            std::transform(
                first, first + ncoeffsComp, component.begin(),
                [scale](const double value) { return scale * value; });

            if (embed)
            {
                // TODO replace with a redesign operator, together with the
                // rest of the field-file input and output, which goes
                // through the legacy ExpList interface. The Arrays only wrap
                // the vectors for it.
                std::vector<double> embedded(expansion->GetNcoeffs(), 0.0);
                Array<OneD, double> componentArray(
                    component.size(), component.data(), eArrayWrapper);
                Array<OneD, double> embeddedArray(
                    embedded.size(), embedded.data(), eArrayWrapper);
                expansion->ExtractCoeffsToCoeffs(view.expansion, componentArray,
                                                 embeddedArray);
                component = std::move(embedded);
            }

            for (size_t i = 0; i < viewFieldDef.size(); ++i)
            {
                viewFieldDef[i]->m_fields.push_back(variables[nc]);
                expansion->AppendFieldData(viewFieldDef[i], viewFieldData[i],
                                           component);
            }
        }

        // A view on the same element layout as the previous one adds its
        // variables to the previous view's definitions, so that every block
        // of the file lists all of these variables, as FieldConvert expects.
        bool sameLayout = blockSize == viewFieldDef.size();
        for (size_t i = 0; sameLayout && i < viewFieldDef.size(); ++i)
        {
            sameLayout =
                SameFieldLayout(fieldDef[blockStart + i], viewFieldDef[i]);
        }

        if (sameLayout)
        {
            for (size_t i = 0; i < viewFieldDef.size(); ++i)
            {
                auto &names = fieldDef[blockStart + i]->m_fields;
                names.insert(names.end(), viewFieldDef[i]->m_fields.begin(),
                             viewFieldDef[i]->m_fields.end());
                auto &data = fieldData[blockStart + i];
                data.insert(data.end(), viewFieldData[i].begin(),
                            viewFieldData[i].end());
            }
            continue;
        }

        blockStart = fieldDef.size();
        blockSize  = viewFieldDef.size();
        fieldDef.insert(fieldDef.end(), viewFieldDef.begin(),
                        viewFieldDef.end());
        fieldData.insert(fieldData.end(),
                         std::make_move_iterator(viewFieldData.begin()),
                         std::make_move_iterator(viewFieldData.end()));
    }

    m_fieldIo->Write(outname, fieldDef, fieldData, metadata,
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

        // Set up the trace before the other fields are copied from this one,
        // so that those with the same types of boundary conditions share it
        // rather than each setting up its own below.
        firstField->GetTrace();

        for (int i = 1; i < m_nVariables; ++i)
        {
            if (m_graph->SameExpansionInfo(m_variables[0], m_variables[i]))
            {
                // The copy adds its map to those the first field shares
                // with it, where the operators set up on the first field
                // look up the maps of the other variables.
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
void EquationSystem::v_DoProjection(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double time)
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

            // Coefficient space is used for file IO even though the solve
            // runs in physical space, so both directions are needed here.
            // BwdTrans is element local and so serves either projection;
            // the forward direction is not m_fwdTransOp, which is a global
            // mass matrix solve carrying boundary conditions, but the element
            // local M^-1 B^T W that a discontinuous field calls for.
            m_bwdTransOp = MultiRegions::BwdTransOp<double>::Create(
                m_expansionLists[0], m_variables);
            m_iProductWRTBaseOp =
                MultiRegions::IProductWRTBaseOp<double>::Create(
                    m_expansionLists[0], m_variables);
            m_multiplyByElmtInvMassOp =
                MultiRegions::MultiplyByElmtInvMassOp<double>::Create(
                    m_expansionLists[0], m_variables);
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Continuous projection
            m_bwdTransOp = MultiRegions::BwdTransOp<double>::Create(
                m_expansionLists[0], m_variables);

            // Projection operator
            m_fwdTransOp = CreateFwdTransOp(m_expansionLists[0], m_variables);
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
