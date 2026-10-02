///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondEnforceEntropyPressureCFEOpImpl.hpp
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

#include "../EnforceEntropyCommon/EnforceEntropyOpImplBase.hpp"
#include "BndCondEnforceEntropyPressureCFEOp.hpp"

namespace Nektar::detail
{

/// Which star state this condition imposes; see EnforceEntropyCFEOpImplBase.
template <typename TData> struct EnforceEntropyPressurePolicy
{
    static constexpr const char *tag         = "EnforceEntropyPressure";
    static constexpr bool hasSubsonicOutflow = true;

    template <typename... Args>
    NEK_HOSTDEVICE_INLINE static EntropyStarState<TData> InflowStar(
        Args... args)
    {
        return EntropyPressureStar<TData>(args...);
    }

    template <typename... Args>
    NEK_HOSTDEVICE_INLINE static EntropyStarState<TData> OutflowStar(
        Args... args)
    {
        return EntropyPressureOutflowStar<TData>(args...);
    }
};

template <typename ExecSpace, typename TData>
class BndCondEnforceEntropyPressureCFEOpImpl
    : public EnforceEntropyCFEOpImplBase<
          ExecSpace, TData, EnforceEntropyPressurePolicy<TData>,
          BndCondEnforceEntropyPressureCFEOp<TData>>
{
    using Base =
        EnforceEntropyCFEOpImplBase<ExecSpace, TData,
                                    EnforceEntropyPressurePolicy<TData>,
                                    BndCondEnforceEntropyPressureCFEOp<TData>>;

public:
    using Base::Base;

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            BndCondEnforceEntropyPressureCFEOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::detail
