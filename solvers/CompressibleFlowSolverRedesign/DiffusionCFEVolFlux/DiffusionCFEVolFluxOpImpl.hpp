///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionCFEVolFluxOpImpl.hpp
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

#include "DiffusionCFEVolFlux/DiffusionCFEVolFluxKernels.hpp"
#include "DiffusionCFEVolFlux/DiffusionCFEVolFluxOp.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename EqnOfSParams, typename TData>
class DiffusionCFEVolFluxOpImpl : public DiffusionCFEVolFluxOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiffusionCFEVolFluxOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : DiffusionCFEVolFluxOp<TData>(std::move(expansionList), components)
    {
        m_dimension = expansionList->GetCoordim(0);

        LibUtilities::SessionReaderSharedPtr session =
            expansionList->GetSession();

        SetUpEquationOfState(session, m_EoS);

        double rhoInf, pInf;

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
            double Tref, Tsuth;
            session->LoadReferenceValue("Temperature", Tref, 288.15);
            session->LoadReferenceValue("Tsutherland", Tsuth, 110.0);
            m_TRatioSutherland = Tsuth / Tref;
        }
        else
        {
            m_isMuVariable = false;
        }

        double Cp = m_EoS.gamma() / (m_EoS.gamma() - 1.0) * m_EoS.gasConst();

        if (session->DefinesReferenceValue("thermalConductivity"))
        {
            ASSERTL0(!session->DefinesReferenceValue("Prandtl"),
                     "Cannot define both Prandtl and thermalConductivity.");
            double thermalConductivityRef;
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
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            DiffusionCFEVolFluxOpImpl<ExecSpace, EqnOfSParams, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_dimension;
    EqnOfSParams m_EoS;
    // viscosity parameters
    bool m_isMuVariable;
    double m_muRef;
    double m_Prandtl;
    // Sutherland's law parameters
    double m_TRatioSutherland;
    double m_oneOverTstar;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Phys> &deriv,
                 Field<TData, FieldState::Phys> &out) override
    {
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblock    = in.GetBlocks()[blk];
            auto &derivblock = deriv.GetBlocks()[blk];
            auto &outblock   = out.GetBlocks()[blk];

            auto inbase  = inblock.template GetPtr<MemSpace, ReadOnly>();
            auto qbase   = derivblock.template GetPtr<MemSpace, ReadOnly>();
            auto outbase = outblock.template GetPtr<MemSpace, WriteOnly>();

            const auto npts        = outblock.CompSize();
            const auto inStride    = inblock.CompSize();
            const auto derivStride = derivblock.CompSize();
            const auto outStride   = outblock.CompSize();
            const auto nvarComps   = inblock.GetNumComponents();

            DiffusionCFEVolFluxKernel<ExecSpace, EqnOfSParams>(
                m_EoS, npts, m_dimension, nvarComps, inStride, derivStride,
                outStride, this->m_Prandtl, this->m_muRef, this->m_isMuVariable,
                this->m_oneOverTstar, this->m_TRatioSutherland, inbase, qbase,
                outbase);
        }
    }
};

} // namespace Nektar::Operators::detail
