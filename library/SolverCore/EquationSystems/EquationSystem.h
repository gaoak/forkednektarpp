///////////////////////////////////////////////////////////////////////////////
//
// File: EquationSystem.h
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
// Description: Base class for device support solvers.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Math/MathHelper.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Communication/Comm.h>
#include <MultiRegions/ExpList.h>
#include <SolverCore/Core/Misc.h>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/Filters/FilterContext.h>
#include <SpatialDomains/Conditions.h>
#include <iosfwd>
#include <vector>

namespace Nektar::SolverCore
{

template <typename TData> class FwdTransOp;
template <typename TData> class MeanRemovalOp;

class EquationSystem;

using EquationSystemSharedPtr = std::shared_ptr<EquationSystem>;
using EquationSystemFactory =
    LibUtilities::NekFactory<std::string, EquationSystem,
                             const LibUtilities::SessionReaderSharedPtr &,
                             const SpatialDomains::MeshGraphSharedPtr &>;

SOLVER_CORE_EXPORT EquationSystemFactory &GetEquationSystemFactory();

class EquationSystem : public std::enable_shared_from_this<EquationSystem>
{
public:
    void InitObject(bool declareExpansionLists = true)
    {
        v_InitObject(declareExpansionLists);
    }

    void DoInitialise(bool dumpInitialConditions = true)
    {
        v_DoInitialise(dumpInitialConditions);
    }

    void DoSolve()
    {
        v_DoSolve();
    }

    void Output()
    {
        v_Output();
    }

    void WriteFld(const std::string &outname)
    {
        v_WriteFld(outname);
    }

    LibUtilities::SessionReaderSharedPtr GetSession() const
    {
        return m_session;
    }

    const std::string &GetSessionName() const
    {
        return m_sessionName;
    }

    const LibUtilities::FieldMetaDataMap &GetFieldMetaDataMap() const
    {
        return m_fieldMetaDataMap;
    }

    bool HasFieldMetaData(const std::string &key) const
    {
        return m_fieldMetaDataMap.count(key) != 0;
    }

    const std::string &GetFieldMetaData(const std::string &key) const
    {
        const auto it = m_fieldMetaDataMap.find(key);
        ASSERTL0(it != m_fieldMetaDataMap.end(),
                 "Missing field metadata: " + key);
        return it->second;
    }

    void SetFieldMetaData(const std::string &key, const std::string &value)
    {
        m_fieldMetaDataMap[key] = value;
    }

    void PrintSummary(std::ostream &out)
    {
        if (m_session->GetComm()->GetRank() != 0)
        {
            return;
        }

        SummaryList summary;
        v_GenerateSummary(summary);

        out << std::string(70, '=') << std::endl;
        for (const auto &item : summary)
        {
            out << "\t";
            out.width(20);
            out << item.first << ": " << item.second << std::endl;
        }
        out << std::string(70, '=') << std::endl;
    }

    /// Print the L2 and Linf errors of each variable. The regression tests
    /// parse this output, so its format must not change.
    void PrintNorms(std::ostream &out);

protected:
    /// How the initial condition is brought into the polynomial space
    enum class InitialConditionProjection
    {
        None,     ///< Discontinuous: keep the quadrature point samples.
        Average,  ///< Continuous: average element coefficients, Dirichlet.
        Galerkin, ///< Continuous: mass matrix solve with the Dirichlet
                  ///< values. Discontinuous: element-local L2 projection.
    };

    /// Parameter for homogeneous expansions
    enum HomogeneousType
    {
        eHomogeneous1D,
        eHomogeneous2D,
        eHomogeneous3D,
        eNotHomogeneous
    };

    LibUtilities::CommSharedPtr m_comm;
    LibUtilities::SessionReaderSharedPtr m_session;
    SpatialDomains::MeshGraphSharedPtr m_graph;
    std::vector<MultiRegions::ExpListSharedPtr> m_expansionLists;
    LibUtilities::FieldIOSharedPtr m_fieldIO;
    LibUtilities::FieldMetaDataMap m_fieldMetaDataMap;
    std::string m_sessionName;

    /// Session variables grouped by expansion list: group i holds the
    /// variables of m_expansionLists[i], which share its expansion, set by
    /// v_SetExpansionVariables().
    std::vector<std::vector<std::string>> m_variables;
    /// Whether each variable's system is singular, grouped as m_variables.
    /// The expansion list of a group takes the check of its first variable.
    std::vector<std::vector<bool>> m_checkIfSystemSingular;
    unsigned int m_coordim                        = 0;
    unsigned int m_expdim                         = 0;
    double m_time                                 = 0.0;
    MultiRegions::ProjectionType m_projectionType = MultiRegions::eGalerkin;

    /// Session SOLVERINFO ProjectInitialConditions; default Average for a
    /// continuous projection, None otherwise.
    InitialConditionProjection m_initialConditionProjection =
        InitialConditionProjection::Average;

    Math::MathHelper m_math;
    /// Solution in physical and coefficient space, one Field per expansion
    /// list holding the variables of its m_variables group.
    std::vector<LibUtilities::Field<double, FieldState::Phys>> m_phys;
    std::vector<LibUtilities::Field<double, FieldState::Coeff>> m_coeff;
    /// Workspaces laid out as m_phys and m_coeff, free between calls.
    std::vector<LibUtilities::Field<double, FieldState::Phys>> m_wsp_phys;
    std::vector<LibUtilities::Field<double, FieldState::Coeff>> m_wsp_coeff;

    /// One per expansion list, set for a singular system only: balances the
    /// forcing against the Neumann values of the system, if the solver builds
    /// it, and removes the mean of the reported error in PrintNorms(), which
    /// builds it otherwise.
    std::vector<std::shared_ptr<MeanRemovalOp<double>>> m_meanRemovalOp;

    /// HOMOGENEOUS setup
    HomogeneousType m_HomogeneousType;

    double m_LhomX; ///< physical length in X direction (if homogeneous)
    double m_LhomY; ///< physical length in Y direction (if homogeneous)
    double m_LhomZ; ///< physical length in Z direction (if homogeneous)

    int m_npointsX; ///< number of points in X direction (if homogeneous)
    int m_npointsY; ///< number of points in Y direction (if homogeneous)
    int m_npointsZ; ///< number of points in Z direction (if homogeneous)

    int m_HomoDirec; ///< number of homogenous directions

    /// Flag to determine if single homogeneous mode is used.
    bool m_singleMode;
    /// Flag to determine if half homogeneous mode is used.
    bool m_halfMode;
    /// Flag to determine if use multiple homogenenous modes are used.
    bool m_multipleModes;
    /// Flag to determine if FFT is used for homogeneous transform.
    bool m_useFFT;
    /// Flag to determine if dealiasing is used for homogeneous space.
    bool m_homogen_dealiasing;

    SOLVER_CORE_EXPORT EquationSystem(
        const LibUtilities::SessionReaderSharedPtr &session,
        const SpatialDomains::MeshGraphSharedPtr &graph);
    SOLVER_CORE_EXPORT virtual ~EquationSystem() = default;

    SOLVER_CORE_EXPORT virtual void v_InitObject(
        bool declareExpansionLists = true);
    SOLVER_CORE_EXPORT virtual void v_DoInitialise(
        bool dumpInitialConditions = true);
    SOLVER_CORE_EXPORT virtual void v_DoSolve();
    SOLVER_CORE_EXPORT virtual void v_Output();
    SOLVER_CORE_EXPORT virtual void v_GenerateSummary(SummaryList &summary);

    SOLVER_CORE_EXPORT virtual void v_InitialiseFields();
    SOLVER_CORE_EXPORT virtual void v_InitialiseOperators();
    SOLVER_CORE_EXPORT std::shared_ptr<FwdTransOp<double>> CreateFwdTransOp(
        const MultiRegions::ExpListSharedPtr &expList,
        const std::vector<std::string> &variables);
    SOLVER_CORE_EXPORT virtual std::vector<bool> v_GetSystemSingularChecks();
    SOLVER_CORE_EXPORT virtual void v_SetExpansionVariables();
    void DeclareExpansionLists();

    void CheckHomogeneousDimensions();

    /// Publish the fields of the system to the filters.
    SOLVER_CORE_EXPORT virtual FieldRegistry v_GetFieldRegistry();

    /// Return views of the output coefficients.
    SOLVER_CORE_EXPORT virtual CoefficientFieldViews v_GetOutputFieldViews();

    SOLVER_CORE_EXPORT virtual void v_WriteFld(const std::string &outname);

    /// Write coefficient fields, scaled by @p scale, to a field file.
    SOLVER_CORE_EXPORT void WriteOutputFields(
        const std::string &outname, const CoefficientFieldViews &fields,
        const LibUtilities::FieldMetaDataMap &metadata,
        const double scale = 1.0);
};

} // namespace Nektar::SolverCore
