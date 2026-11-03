///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffVolumeFluxOp.hpp
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
// Description: Combined advection-diffusion volume flux operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SolverCore/Advection/AdvectionVolumeFluxOp.hpp"
#include "SolverCore/Diffusion/DiffusionVolumeFluxOp.hpp"
#include "SolverCore/Flux/FluxOp.hpp"
#include <LibUtilities/BasicUtils/Math/Math.hpp>

#include <memory>

namespace Nektar::SolverCore
{

/**
 * @brief The volume flux of an advection-diffusion right-hand side, with
 * the viscous flux on its own: Apply() writes `F_v(u, grad u)` into
 * @p viscOut and `F_v - F_a(u)` into @p totalOut.
 *
 * The weak DG / interior-penalty pipeline needs both under the
 * average-of-normal-fluxes viscous trace form: the total for the derivative
 * inner product, the viscous flux for its normal component on the traces.
 * An implementation that knows both fluxes of its equations may form them
 * in one pass; AdvDiffVolumeFluxComposed is the generic one, from an
 * advection and a diffusion volume flux.
 */
template <typename TData> class AdvDiffVolumeFluxOp : public FluxOp<TData>
{
public:
    /// The viscous flux into @p viscOut and the total volume flux into
    /// @p totalOut; both are overwritten.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &deriv,
               LibUtilities::Field<TData, FieldState::Phys> &viscOut,
               LibUtilities::Field<TData, FieldState::Phys> &totalOut)
    {
        this->v_Apply(in, deriv, viscOut, totalOut);
    }

protected:
    AdvDiffVolumeFluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : FluxOp<TData>(expansionList, components)
    {
    }

    ~AdvDiffVolumeFluxOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &deriv,
        LibUtilities::Field<TData, FieldState::Phys> &viscOut,
        LibUtilities::Field<TData, FieldState::Phys> &totalOut) = 0;
};

/**
 * @brief AdvDiffVolumeFluxOp from an advection and a diffusion volume flux:
 * the viscous flux, copied, with the advective flux subtracted from the
 * copy. The default where no implementation for the equations is attached.
 */
template <typename ExecSpace, typename TData>
class AdvDiffVolumeFluxComposed final : public AdvDiffVolumeFluxOp<TData>
{
public:
    AdvDiffVolumeFluxComposed(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : AdvDiffVolumeFluxOp<TData>(expansionList, components)
    {
    }

    void SetAdvVolFlux(const std::shared_ptr<AdvectionVolumeFluxOp<TData>> &ptr)
    {
        m_advection = ptr;
    }

    void SetDiffVolFlux(
        const std::shared_ptr<DiffusionVolumeFluxOp<TData>> &ptr)
    {
        m_diffusion = ptr;
    }

protected:
    std::shared_ptr<AdvectionVolumeFluxOp<TData>> m_advection;
    std::shared_ptr<DiffusionVolumeFluxOp<TData>> m_diffusion;

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &deriv,
        LibUtilities::Field<TData, FieldState::Phys> &viscOut,
        LibUtilities::Field<TData, FieldState::Phys> &totalOut) override
    {
        ASSERTL1(m_advection && m_diffusion,
                 "AdvDiffVolumeFluxComposed needs SetAdvVolFlux() and "
                 "SetDiffVolFlux().");
        // The operators are shared with the caller, so their settings are
        // made here, where they are used.
        m_diffusion->SetAppend(false);
        m_diffusion->Apply(in, deriv, viscOut);
        Math::copy<ExecSpace>(viscOut, totalOut);
        m_advection->SetScale(-1.0);
        m_advection->SetAppend(true);
        m_advection->Apply(in, totalOut);
        m_advection->SetAppend(false);
    }
};

} // namespace Nektar::SolverCore
