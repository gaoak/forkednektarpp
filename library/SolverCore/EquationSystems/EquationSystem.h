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
#include <SpatialDomains/Conditions.h>
#include <iosfwd>

#include "Operators/ElmtOps/BwdTrans/BwdTransOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp"

namespace Nektar::SolverCore
{

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
    SOLVER_CORE_EXPORT virtual ~EquationSystem() = default;

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

    /*
     * Create m_fields and m_fields_coeff to hold solution.
     */
    void InitialiseFields()
    {
        v_InitialiseFields();
    }

    void InitialiseOperators()
    {
        v_InitialiseOperators();
    }

    /*
     * Evaluate L2 and Linf norm of solution.
     * Note the specific format of the error output is essential for the
     * regression tests to work.
     * This function may be reimplemented in child classes for different output.
     */
    void PrintNorms(std::ostream &out)
    {
        v_PrintNorms(out);
    }

    void DoProjection(MultiRegions::Field<double, FieldState::Phys> &in,
                      MultiRegions::Field<double, FieldState::Phys> &out,
                      [[maybe_unused]] const double time = 0.0)
    {
        v_DoProjection(in, out, time);
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

    void DeclareExpansionLists();

    void CheckHomogeneousDimensions();

protected:
    SOLVER_CORE_EXPORT EquationSystem(
        const LibUtilities::SessionReaderSharedPtr &session,
        const SpatialDomains::MeshGraphSharedPtr &graph);

    SOLVER_CORE_EXPORT virtual void v_InitObject(
        bool declareExpansionLists = true);
    SOLVER_CORE_EXPORT virtual void v_DoInitialise(
        bool dumpInitialConditions = true);
    SOLVER_CORE_EXPORT virtual void v_DoSolve();
    SOLVER_CORE_EXPORT virtual void v_Output();
    SOLVER_CORE_EXPORT virtual void v_GenerateSummary(SummaryList &summary);

    SOLVER_CORE_EXPORT virtual void v_InitialiseFields();
    SOLVER_CORE_EXPORT virtual void v_InitialiseOperators();
    SOLVER_CORE_EXPORT virtual std::vector<bool> v_GetSystemSingularChecks();
    SOLVER_CORE_EXPORT virtual void v_PrintNorms(std::ostream &out);
    SOLVER_CORE_EXPORT virtual void v_DoProjection(
        MultiRegions::Field<double, FieldState::Phys> &in,
        MultiRegions::Field<double, FieldState::Phys> &out,
        [[maybe_unused]] const double time);

    SOLVER_CORE_EXPORT virtual void v_WriteFld(const std::string &outname);

    LibUtilities::CommSharedPtr m_comm;
    LibUtilities::SessionReaderSharedPtr m_session;
    SpatialDomains::MeshGraphSharedPtr m_graph;
    std::vector<MultiRegions::ExpListSharedPtr> m_expansionLists;
    LibUtilities::FieldIOSharedPtr m_fieldIo;
    LibUtilities::FieldMetaDataMap m_fieldMetaDataMap;
    std::string m_sessionName;

    unsigned int m_nVariables                     = 0;
    unsigned int m_coordim                        = 0;
    unsigned int m_expdim                         = 0;
    double m_time                                 = 0.0;
    MultiRegions::ProjectionType m_projectionType = MultiRegions::eGalerkin;
    std::vector<bool> m_checkIfSystemSingular;
    std::vector<std::string> m_variables;

    Math::MathHelper m_math;
    MultiRegions::Field<double, FieldState::Phys> m_fields;
    MultiRegions::Field<double, FieldState::Coeff> m_fields_coeff;

    std::shared_ptr<BwdTransOp<double>> m_bwdTransOp = nullptr;
    std::shared_ptr<FwdTransOp<double>> m_fwdTransOp = nullptr;

    /// HOMOGENEOUS setup
    /// Parameter for homogeneous expansions
    enum HomogeneousType
    {
        eHomogeneous1D,
        eHomogeneous2D,
        eHomogeneous3D,
        eNotHomogeneous
    };

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
};

} // namespace Nektar::SolverCore
