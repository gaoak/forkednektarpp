///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionCFETraceFluxOpImpl.hpp
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
// Description: CFE Diffusion volume flux operator implementation.
//
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <boost/algorithm/string/predicate.hpp>

#include "DiffusionCFETraceFlux/DiffusionCFETraceFluxKernels.hpp"
#include "DiffusionCFETraceFlux/DiffusionCFETraceFluxOp.hpp"

namespace Nektar::detail
{

template <typename ExecSpace, typename EqnOfSParams, typename TData>
class DiffusionCFETraceFluxOpImpl : public DiffusionCFETraceFluxOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiffusionCFETraceFluxOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : DiffusionCFETraceFluxOp<TData>(std::move(expansionList), components),
          m_traceAver(MultiRegions::Field<TData, FieldState::Phys>(
              "Trace average",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_traceJump(MultiRegions::Field<TData, FieldState::Phys>(
              "Trace jump",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1))
    {
        m_nDim  = expansionList->GetCoordim(0);
        m_nComp = components.size();
        LibUtilities::SessionReaderSharedPtr session =
            expansionList->GetSession();

        SetUpEquationOfState(session, m_EoS);

        TData rhoInf, pInf;

        // reference values could probably be in seperate session section
        session->LoadReferenceValue("viscosity", m_muRef, 1.78e-05);
        session->LoadReferenceValue("density", rhoInf, 1.225);
        session->LoadReferenceValue("pressure", pInf, 101325);

        m_oneOverTstar = (rhoInf * m_EoS.gasConst()) / pInf;

        std::string viscosityType;
        session->LoadSolverInfo("ViscosityType", viscosityType, "Constant");
        if (boost::iequals(viscosityType, "Variable"))
        {
            m_isMuVariable = true;
            TData Tref, Tsuth;
            session->LoadReferenceValue("Temperature", Tref, 288.15);
            session->LoadReferenceValue("Tsutherland", Tsuth, 110.0);
            m_TRatioSutherland = Tsuth / Tref;
        }
        else
        {
            m_isMuVariable = false;
        }

        TData Cp = m_EoS.gamma() / (m_EoS.gamma() - 1.0) * m_EoS.gasConst();

        if (session->DefinesReferenceValue("thermalConductivity"))
        {
            ASSERTL0(!session->DefinesReferenceValue("Prandtl"),
                     "Cannot define both Prandtl and thermalConductivity.");
            TData thermalConductivityRef;
            session->LoadReferenceValue("thermalConductivity",
                                        thermalConductivityRef, 1.0);
            m_Prandtl = Cp * m_muRef / thermalConductivityRef;
        }
        else
        {
            session->LoadReferenceValue("Prandtl", m_Prandtl, 0.72);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            DiffusionCFETraceFluxOpImpl<ExecSpace, EqnOfSParams, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_nDim;
    unsigned int m_nComp;
    EqnOfSParams m_EoS;

    // viscosity parameters
    bool m_isMuVariable;
    TData m_muRef;
    TData m_Prandtl;

    // Sutherland's law parameters
    TData m_TRatioSutherland;
    TData m_oneOverTstar;

    MultiRegions::Field<TData, FieldState::Phys> m_traceAver, m_traceJump;

    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &fwd,
                 MultiRegions::Field<TData, FieldState::Phys> &bwd,
                 MultiRegions::Field<TData, FieldState::Phys> &derivFwd,
                 MultiRegions::Field<TData, FieldState::Phys> &derivBwd,
                 MultiRegions::Field<TData, FieldState::Phys> &numflux) override
    {
        for (unsigned blk = 0; blk < numflux.GetBlocks().size(); ++blk)
        {
            auto &fwdblock     = fwd.GetBlocks()[blk];
            auto &bwdblock     = bwd.GetBlocks()[blk];
            auto &numFluxblock = numflux.GetBlocks()[blk];

            auto &traceAverblock = m_traceAver.GetBlocks()[blk];
            auto &traceJumpblock = m_traceJump.GetBlocks()[blk];

            // These trace geometry/weight arrays are read-only mesh data, so
            // they live in the data warehouse instead of per-op Field storage.
            auto normalbase = this->m_dataWarehouse->template GetData<MemSpace>(
                Operators::IPTraceNormalKey<TData>(blk));
            auto bwdWeightAverBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    Operators::IPTraceScalarKey<TData>(
                        blk, Operators::IPTraceScalarData::BwdWeightAver));
            auto bwdWeightJumpBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    Operators::IPTraceScalarKey<TData>(
                        blk, Operators::IPTraceScalarData::BwdWeightJump));
            auto lengthRecipBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    Operators::IPTraceScalarKey<TData>(
                        blk, Operators::IPTraceScalarData::LengthRecip));
            auto penaltyFactorBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    Operators::IPTraceScalarKey<TData>(
                        blk, Operators::IPTraceScalarData::PenaltyFactor));

            auto fwdbase = fwdblock.template GetPtr<MemSpace, ReadOnly>();
            auto bwdbase = bwdblock.template GetPtr<MemSpace, ReadOnly>();
            auto derivTraceFwdbase =
                derivFwd.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
            auto derivTraceBwdbase =
                derivBwd.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
            auto traceAverbase =
                traceAverblock.template GetPtr<MemSpace, WriteOnly>();
            auto traceJumpbase =
                traceJumpblock.template GetPtr<MemSpace, WriteOnly>();
            auto numFluxbase =
                numFluxblock.template GetPtr<MemSpace, WriteOnly>();

            const size_t npts        = numFluxblock.CompSize();
            const size_t traceStride = numFluxblock.CompSize();
            const size_t derivStride = derivFwd.GetBlocks()[blk].CompSize();

            // For each trace point on this block, assemble:
            // - conservative average state
            // - conservative jump
            // - penalty-corrected derivative trace
            // - final normal viscous numerical flux
            DiffuseTraceFluxKernel<ExecSpace>(
                m_EoS, npts, m_nDim, m_nComp, traceStride, derivStride,
                this->m_Prandtl, this->m_muRef, this->m_isMuVariable,
                this->m_oneOverTstar, this->m_TRatioSutherland, normalbase,
                bwdWeightAverBase, bwdWeightJumpBase, lengthRecipBase,
                penaltyFactorBase, fwdbase, bwdbase, derivTraceFwdbase,
                derivTraceBwdbase, traceAverbase, traceJumpbase, numFluxbase);
        }
        ApplyFluxBndConds();
    }

    void v_Apply(MultiRegions::Field<TData, FieldState::Coeff> &out) override
    {
        // Placeholder for symmetric IP trace term.
        (void)out;
    }

    void ApplyFluxBndConds()
    {
        // Placeholder for future boundary trace-flux corrections.
        // Boundary trace-flux corrections such as WallAdiabatic are not
        // implemented in the current serial DiffusionIP operator path.
    }
};

} // namespace Nektar::detail
