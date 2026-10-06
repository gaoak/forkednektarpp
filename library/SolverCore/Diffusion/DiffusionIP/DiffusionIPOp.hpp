///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionIPOp.hpp
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
// Description: Interior-penalty diffusion operator interface.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/SolverCore.hpp>

#include "SolverCore/Diffusion/DiffusionOp.hpp"
#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

namespace Nektar::SolverCore
{

// Interior-penalty diffusion: the factory name, Create() and the scheme
// coefficients. Apply() is the base class's.
template <typename TData> class DiffusionIPOp : public DiffusionOp<TData>
{
public:
    static std::shared_ptr<DiffusionIPOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        // Force a genuine cross-library symbol reference into
        // libSolverCore - see EnsureLinked() in SolverCore.hpp. Without
        // it a linker that keeps only what is referenced can drop the
        // library, and the operator registrations go with it.
        EnsureLinked();

        return MultiRegions::Operator<TData>::template Create<DiffusionIPOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "DiffusionIP";

    void SetVolumeFluxOp(
        const std::shared_ptr<DiffusionVolumeFluxOp<TData>> &ptr)
    {
        this->m_volumeFluxOp = ptr;
    }

    void SetTraceFluxOp(const std::shared_ptr<TraceFluxOp<TData>> &ptr)
    {
        this->m_traceFluxOp = ptr;
    }

    /// Refresh time-dependent boundary values before the next Apply(). A no-op
    /// unless a boundary condition actually depends on time.
    void UpdateBndPhys(const TData &time = 0.0)
    {
        if (this->m_traceFluxOp)
        {
            this->m_traceFluxOp->UpdateBndPhys(time);
        }
    }

protected:
    std::shared_ptr<DiffusionVolumeFluxOp<TData>> m_volumeFluxOp;
    std::shared_ptr<TraceFluxOp<TData>> m_traceFluxOp;

    DiffusionIPOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : DiffusionOp<TData>(expansionList, components)
    {
    }

    ~DiffusionIPOp() override = default;
};

} // namespace Nektar::SolverCore
