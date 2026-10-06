///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionScalarVolFluxOpImpl.hpp
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
// Description: Scalar IP diffusion volume flux implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ADRSolverRedesign/DiffusionScalarVolFlux/DiffusionScalarVolFluxKernels.hpp"
#include "ADRSolverRedesign/DiffusionScalarVolFlux/DiffusionScalarVolFluxOp.hpp"

namespace Nektar::detail
{

template <typename ExecSpace, typename TData>
class DiffusionScalarVolFluxOpImpl : public DiffusionScalarVolFluxOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiffusionScalarVolFluxOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : DiffusionScalarVolFluxOp<TData>(std::move(expansionList), components)
    {
        m_nDim = expansionList->GetCoordim(0);

        std::vector<TData> diffCoeff(m_nDim * (m_nDim + 1) / 2, TData(0.0));
        for (unsigned int d = 0; d < m_nDim; ++d)
        {
            diffCoeff[d * (d + 3) / 2] = TData(1.0);
        }
        this->SetDiffCoeff(diffCoeff);
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DiffusionScalarVolFluxOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_nDim;
    LibUtilities::MemoryRegion<TData> m_diffCoeff;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &deriv,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inblock    = in.GetBlocks()[blk];
            auto &derivblock = deriv.GetBlocks()[blk];
            auto &outblock   = out.GetBlocks()[blk];

            auto derivbase =
                derivblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto outbase =
                outblock.template GetPtr<MemSpace, WriteOnly>(streamID);
            auto diffCoeffBase =
                m_diffCoeff.template GetPtr<MemSpace, ReadOnly>(streamID);

            DiffusionScalarVolFluxKernel<ExecSpace>(
                outblock.CompSize(), m_nDim, inblock.GetNumComponents(),
                derivblock.CompSize(), outblock.CompSize(), diffCoeffBase,
                derivbase, outbase, streamID);
        }
    }

    void v_SetDiffCoeff(std::vector<TData> &diffCoeff) override
    {
        const auto diffCoeffSize = m_nDim * (m_nDim + 1) / 2;
        ASSERTL0(diffCoeff.size() == diffCoeffSize,
                 "The number of diffusion coefficients must match 1, 3 or 6 "
                 "for a 1D, 2D or 3D case, respectively.");

        m_diffCoeff = LibUtilities::MemoryRegion<TData>::template FromVector<
            MemSpace, TData>(diffCoeff);
    }
};

} // namespace Nektar::detail