///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceOpImpl.hpp
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

#pragma once

#include <MultiRegions/ExpListHomogeneous1D.h>

#include "Operators/ElmtOps/Divergence/DivergenceOp.hpp"
#include "Operators/ElmtOps/Divergence/DivergenceZOpImpl.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DivergenceOpImpl : public DivergenceOp<TData>
{
public:
    DivergenceOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : DivergenceOp<TData>(expansionList, components)
    {
        // A z-op is built only for a multi-plane 3DH1 expansion; m_zOp stays
        // null for 2D/3D, 3DH2 and single-plane 3DH1, where v_Apply does the
        // xy derivatives alone.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        if (homoExpList &&
            homoExpList->GetHomogeneousBasis()->GetNumPoints() > 1)
        {
            m_zOp = std::make_shared<DivergenceZOpImpl<ExecSpace, TData>>(
                expansionList);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DivergenceOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    std::shared_ptr<DivergenceZOpImpl<ExecSpace, TData>> m_zOp;

    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        if (m_zOp && in.GetNumHomoModes() > 1)
        {
            m_zOp->Launch(in, out);
        }
    }
};

} // namespace Nektar::Operators::detail
