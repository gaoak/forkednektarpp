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
#include <SolverCore/MeanRemoval/MeanRemovalOp.hpp>
#include <SolverCore/NormOps/NormL2/NormL2Op.hpp>
#include <SolverCore/NormOps/NormLinf/NormLinfOp.hpp>
#include <algorithm>
#include <iterator>
#include <utility>
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

// True when two variables have the same basis on every element, whether or
// not the session defines their expansions together.
bool SameExpansion(const SpatialDomains::MeshGraphSharedPtr &graph,
                   const std::string &a, const std::string &b)
{
    const auto &expA = graph->GetExpansionInfo(a);
    const auto &expB = graph->GetExpansionInfo(b);
    return &expA == &expB ||
           std::equal(expA.begin(), expA.end(), expB.begin(), expB.end(),
                      [](const auto &x, const auto &y) {
                          return x.first == y.first &&
                                 x.second->m_geomPtr == y.second->m_geomPtr &&
                                 x.second->m_basisKeyVector ==
                                     y.second->m_basisKeyVector;
                      });
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
    for (unsigned int i = 0; i < filenames.size(); ++i)
    {
        m_fieldMetaDataMap["SessionName" + std::to_string(i)] = filenames[i];
    }

    // Get coordinate system (space) dimensions.
    m_coordim = m_graph->GetSpaceDimension();
    m_expdim  = m_graph->GetMeshDimension();

    // Initialise Math.
    std::string execName =
        MultiRegions::Operator<double>::GetOpExecSpace(m_session);
    m_math = Math::MathHelper(execName);

    // Check and set definitions for homogeneous/Fourier dimensions.
    CheckHomogeneousDimensions();

    // Hard set the GlobalSysSoln to the only supported type: IterativeFull.
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

    // Get projection CG, DG or mixed CG/DG.
    const std::string projection = m_session->GetSolverInfo("PROJECTION");
    m_projectionType             = GetProjectionType(projection);
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

    // Group the variables by expansion list and declare the lists, which
    // also checks whether each system is singular, i.e. has only
    // Neumann/Periodic BCs.
    if (declareExpansionLists)
    {
        DeclareExpansionLists();
    }
    else
    {
        // A solver declaring its own lists still gets its variables grouped
        // and the singular checks.
        v_SetExpansionVariables();
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
        int maxExpOrder = 0;
        for (const auto &explist : m_expansionLists)
        {
            maxExpOrder =
                std::max(maxExpOrder, explist->EvalBasisNumModesMax());
        }
        AddSummaryItem(summary, "Max SEM Exp. Order", maxExpOrder);
    }

    // TODO can we add number of GPUs/Devices?
    if (m_comm->GetSize() > 1)
    {
        AddSummaryItem(summary, "Num. Processes", m_comm->GetSize());
    }

    AddSummaryItem(summary, "Projection Type",
                   GetProjectionString(m_projectionType));
}

/*
 * Create m_phys and m_coeff, one Field per expansion list holding the
 * variables of its m_variables group, for holding the solution, comparing
 * against an ExactSolution and writing to file, and the workspaces
 * m_wsp_phys and m_wsp_coeff of the same layout. Note any solver
 * implementation can choose to not call EquationSystem::v_InitialiseFields,
 * but still has to initialise m_phys and m_coeff.
 */
void EquationSystem::v_InitialiseFields()
{
    for (unsigned int i = 0; i < m_expansionLists.size(); i++)
    {
        // Create solution fields in physical space (at quadrature points)
        auto bAtr_phys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[i]);
        m_phys.push_back(LibUtilities::Field<double, FieldState::Phys>(
            "phys", bAtr_phys, m_variables[i], m_npointsZ));
        m_wsp_phys.push_back(LibUtilities::Field<double, FieldState::Phys>(
            "wsp (phys)", bAtr_phys, m_variables[i], m_npointsZ));

        // Create solution fields in coefficient space (coefficients of
        // polynomial basis function)
        auto bAtr_coeff =
            MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
                m_expansionLists[i]);
        m_coeff.push_back(LibUtilities::Field<double, FieldState::Coeff>(
            "coeff", bAtr_coeff, m_variables[i], m_npointsZ));
        m_wsp_coeff.push_back(LibUtilities::Field<double, FieldState::Coeff>(
            "wsp (coeff)", bAtr_coeff, m_variables[i], m_npointsZ));

        // Zero the solution fields
        m_math.zero(m_phys[i]);
        m_math.zero(m_coeff[i]);
    }
}

/*
 *  Hook for the solver to create and initialise its operators.
 */
void EquationSystem::v_InitialiseOperators()
{
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

void EquationSystem::v_WriteFld(const std::string &outname)
{
    m_fieldMetaDataMap["Time"] = std::to_string(m_time);
    WriteOutputFields(outname, v_GetOutputFieldViews(), m_fieldMetaDataMap);
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

        // Whether the view embeds is decided on the local elements, so the
        // ranks agree on it before writing: otherwise one file would hold the
        // view in the first view's definitions on some ranks and in its own
        // on others.
        const auto &firstExpansion = fields.front().expansion;
        int embedOnAllRanks        = view.expansion != firstExpansion &&
                              view.expansion->EmbedsLosslessly(firstExpansion);
        m_comm->AllReduce(embedOnAllRanks, LibUtilities::ReduceMin);
        const bool embed      = embedOnAllRanks == 1;
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

/**
 * @brief All the session variables share one expansion list.
 */
void EquationSystem::v_SetExpansionVariables()
{
    m_variables.push_back(m_session->GetVariables());
    m_checkIfSystemSingular.push_back(v_GetSystemSingularChecks());
}

/**
 * @brief Declare one expansion list per group of m_variables, whose
 * variables share one expansion, from the expansion and boundary conditions
 * of the group's first variable. A group with the same expansion as an
 * earlier one gets a copy of that list, which shares its maps and trace; the
 * maps of the group's other variables are built on demand by
 * ContField::GetLocalToGlobalMap().
 */
void EquationSystem::DeclareExpansionLists()
{
    using MultiRegions::ContField;
    using MultiRegions::DisContField;

    v_SetExpansionVariables();

    ASSERTL0(!m_variables.empty(),
             "Session must define at least one variable.");
    for (const auto &variables : m_variables)
    {
        ASSERTL0(!variables.empty(),
                 "Every expansion list must hold at least one variable.");
        for (const auto &variable : variables)
        {
            ASSERTL0(SameExpansion(m_graph, variables[0], variable),
                     "Variable " + variable + " must share the expansion of " +
                         variables[0] + ", the first of its group.");
        }
    }

    const bool continuous =
        m_projectionType == MultiRegions::eGalerkin ||
        m_projectionType == MultiRegions::eMixed_CG_Discontinuous;
    const bool setupJustDG = true;

    for (unsigned int i = 0; i < m_variables.size(); ++i)
    {
        const std::string &variable = m_variables[i][0];

        // The first earlier list with the same expansion, if any.
        MultiRegions::ExpListSharedPtr sharedExpList;
        for (unsigned int j = 0; j < i && !sharedExpList; ++j)
        {
            if (m_graph->SameExpansionInfo(m_variables[j][0], variable))
            {
                sharedExpList = m_expansionLists[j];
            }
        }

        if (continuous)
        {
            MultiRegions::ContFieldSharedPtr field;
            if (sharedExpList)
            {
                // A copy, which adds its map to those sharedExpList shares.
                field = MemoryManager<ContField>::AllocateSharedPtr(
                    *std::dynamic_pointer_cast<ContField>(sharedExpList),
                    m_graph, variable, false, m_checkIfSystemSingular[i][0]);
            }
            else
            {
                field = MemoryManager<ContField>::AllocateSharedPtr(
                    m_session, m_graph, variable, false,
                    m_checkIfSystemSingular[i][0]);

                // Set up the trace before any later list is copied from this
                // one, so that those with the same types of boundary
                // conditions share it rather than each setting up its own.
                field->GetTrace();
            }
            m_expansionLists.push_back(field);
        }
        else
        {
            MultiRegions::DisContFieldSharedPtr field;
            if (sharedExpList)
            {
                // A copy of sharedExpList.
                field = MemoryManager<DisContField>::AllocateSharedPtr(
                    *std::dynamic_pointer_cast<DisContField>(sharedExpList),
                    m_graph, variable, setupJustDG, false);
            }
            else
            {
                field = MemoryManager<DisContField>::AllocateSharedPtr(
                    m_session, m_graph, variable, setupJustDG, false);
            }
            m_expansionLists.push_back(field);
        }
    }

    // Initialise DataWarehouse for Operators.
    for (unsigned int i = 0; i < m_expansionLists.size(); ++i)
    {
        m_expansionLists[i]->SetDataWarehouse();
        m_expansionLists[i]->GetTrace()->SetDataWarehouse();
    }
}

void EquationSystem::PrintNorms(std::ostream &out)
{
    for (unsigned int i = 0; i < m_expansionLists.size(); i++)
    {
        auto &expansionList = m_expansionLists[i];
        auto &variables     = m_variables[i];

        m_math.zero(m_wsp_phys[i]);

        // Evaluate exact solution and compute diff to discrete solution
        if (m_session->DefinesFunction("ExactSolution"))
        {
            SessionFunction exactSolution(m_session, expansionList, variables,
                                          "ExactSolution");
            exactSolution.Evaluate(m_wsp_phys[i], m_time);
        }
        m_math.sub(m_phys[i], m_wsp_phys[i], m_wsp_phys[i]);

        // A singular system (Neumann/periodic boundaries only) is determined
        // only up to a constant, so the mean of its error is removed before
        // the norms. A non-singular one keeps its error, which the boundary
        // conditions already fix.
        if (m_checkIfSystemSingular[i][0])
        {
            if (m_meanRemovalOp.size() <= i)
            {
                m_meanRemovalOp.resize(m_expansionLists.size());
            }
            if (!m_meanRemovalOp[i])
            {
                m_meanRemovalOp[i] = SolverCore::MeanRemovalOp<double>::Create(
                    m_expansionLists[i], m_variables[i]);
            }

            // The solve leaves its Neumann balance set on the shared
            // operator; the error is reduced to zero mean.
            m_meanRemovalOp[i]->SetNeumannBalance(nullptr);
            m_meanRemovalOp[i]->Apply(m_wsp_phys[i]);
        }

        // Compute L2 norm
        auto l2NormOp = NormL2Op<double>::Create(expansionList, variables);
        l2NormOp->Apply(m_wsp_phys[i]);
        auto l2Errors = l2NormOp->GetNorms();

        // Compute Linf norm
        auto linfNormOp = NormLinfOp<double>::Create(expansionList, variables);
        linfNormOp->Apply(m_wsp_phys[i]);
        auto linfErrors = linfNormOp->GetNorms();

        // Print norms
        for (unsigned int n = 0; n < variables.size(); ++n)
        {
            if (m_comm->GetRank() == 0)
            {
                out << "L 2 error (variable " << variables[n]
                    << ") : " << l2Errors[n] << std::endl;
                out << "L inf error (variable " << variables[n]
                    << ") : " << linfErrors[n] << std::endl;
            }
        }
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
 * @brief Publish #m_phys, one physical group per expansion list, with that
 * of the first as the solution, and the output coefficients of
 * v_GetOutputFieldViews() as the coefficient groups.
 */
FieldRegistry EquationSystem::v_GetFieldRegistry()
{
    ASSERTL0(!m_expansionLists.empty(),
             "EquationSystem has no physical fields to publish.");
    PhysicalFieldViews groups;
    for (unsigned int i = 0; i < m_expansionLists.size(); ++i)
    {
        groups.push_back({m_phys[i], m_expansionLists[i], m_coordim,
                          MakeFieldComponents(m_variables[i])});
    }
    FieldRegistry registry;
    registry.SetRole(std::string(FieldRoles::Solution), groups[0]);
    registry.SetPhysicalGroups(std::move(groups));
    registry.SetCoefficientProvider(
        [this]() { return v_GetOutputFieldViews(); });
    return registry;
}

/**
 * @brief Return views of the coefficients #m_coeff, one per expansion list.
 */
CoefficientFieldViews EquationSystem::v_GetOutputFieldViews()
{
    CoefficientFieldViews views;
    for (unsigned int i = 0; i < m_expansionLists.size(); ++i)
    {
        views.push_back({m_coeff[i], m_expansionLists[i], m_coordim,
                         MakeFieldComponents(m_variables[i])});
    }
    return views;
}

} // namespace Nektar::SolverCore
