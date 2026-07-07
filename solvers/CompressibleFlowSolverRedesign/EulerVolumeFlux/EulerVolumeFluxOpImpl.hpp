///////////////////////////////////////////////////////////////////////////////
//
// File: EulerVolumeFluxOpImpl.hpp
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
// Description: Euler volume flux operator implementation.
//
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <boost/algorithm/string/predicate.hpp>

#include "EulerVolumeFlux/EulerVolumeFluxKernels.hpp"
#include "EulerVolumeFlux/EulerVolumeFluxOp.hpp"

#include "EquationOfState/SupportedEoS.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename EqnOfSParams, typename TData>
class EulerVolumeFluxOpImpl : public EulerVolumeFluxOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    EulerVolumeFluxOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                          const std::vector<std::string> &components)
        : EulerVolumeFluxOp<TData>(std::move(expansionList), components)
    {
        m_dimension = expansionList->GetExp(0)->GetShapeDimension();

        SetUpEquationOfState(expansionList->GetSession(), m_EoS);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            EulerVolumeFluxOpImpl<ExecSpace, EqnOfSParams, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_dimension;
    EqnOfSParams m_EoS;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Phys> &out) override
    {
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            auto inbase = inblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto outbase =
                outblock.template GetPtr<MemSpace, WriteOnly>(streamID);

            const auto npts      = outblock.CompSize();
            const auto inStride  = inblock.CompSize();
            const auto outStride = outblock.CompSize();

            const auto nvarComps = inblock.GetNumComponents();

            EulerVolumeFluxKernel<ExecSpace, EqnOfSParams>(
                m_EoS, npts, m_dimension, nvarComps, inStride, outStride,
                inbase, outbase, streamID);
        }
    }
};

} // namespace Nektar::Operators::detail
