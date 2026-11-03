///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffCFEVolFluxOpImpl.hpp
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
// Description: CFE combined Euler and viscous volume flux operator
// implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "AdvDiffCFEVolFlux/AdvDiffCFEVolFluxOp.hpp"
#include "DiffusionCFEVolFlux/CFEViscosityParams.hpp"
#include "DiffusionCFEVolFlux/DiffusionCFEVolFluxKernels.hpp"

namespace Nektar::detail
{

/**
 * @brief The viscous flux and the viscous minus the Euler flux from one pass
 * of DiffusionCFEVolFluxKernel(), which forms the Euler flux at each point
 * from the state it has loaded and the 1/rho and internal energy the viscous
 * flux has already computed.
 */
template <typename ExecSpace, typename EqnOfSParams, typename TData>
class AdvDiffCFEVolFluxOpImpl : public AdvDiffCFEVolFluxOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvDiffCFEVolFluxOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                            const std::vector<std::string> &components)
        : AdvDiffCFEVolFluxOp<TData>(expansionList, components)
    {
        m_dimension = expansionList->GetCoordim(0);

        LibUtilities::SessionReaderSharedPtr session =
            expansionList->GetSession();

        SetUpEquationOfState(session, m_EoS);

        m_visc = CFEViscosityParams<TData>::Read(session, m_EoS);

        ASSERTL0(components.size() == m_dimension + 2,
                 "The volume flux needs the dimension plus two components.");
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            AdvDiffCFEVolFluxOpImpl<ExecSpace, EqnOfSParams, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_dimension;
    EqnOfSParams m_EoS;
    /// The viscosity law and the Prandtl number.
    CFEViscosityParams<TData> m_visc;

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &deriv,
        LibUtilities::Field<TData, FieldState::Phys> &viscOut,
        LibUtilities::Field<TData, FieldState::Phys> &totalOut) override
    {
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;
            auto &inblock               = in.GetBlocks()[blk];
            auto &derivblock            = deriv.GetBlocks()[blk];
            auto &viscblock             = viscOut.GetBlocks()[blk];
            auto &totblock              = totalOut.GetBlocks()[blk];
            ASSERTL1(totblock.CompSize() == viscblock.CompSize(),
                     "The viscous and total fluxes share one layout.");
            DiffusionCFEVolFluxKernel<false, true, ExecSpace>(
                m_EoS, viscblock.CompSize(), m_dimension,
                inblock.GetNumComponents(), inblock.CompSize(),
                derivblock.CompSize(), viscblock.CompSize(), m_visc.prandtl,
                m_visc.muRef, m_visc.isMuVariable, m_visc.oneOverTstar,
                m_visc.tRatioSutherland,
                inblock.template GetPtr<MemSpace, ReadOnly>(streamID),
                derivblock.template GetPtr<MemSpace, ReadOnly>(streamID),
                viscblock.template GetPtr<MemSpace, WriteOnly>(streamID),
                totblock.template GetPtr<MemSpace, WriteOnly>(streamID),
                TData(-1.0), streamID);
        }
        // The kernel is pointwise: both fluxes have the layout of the state.
        viscOut.SetInterleaveWidth(in);
        totalOut.SetInterleaveWidth(in);
    }
};

} // namespace Nektar::detail
