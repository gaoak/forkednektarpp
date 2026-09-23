///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseOpImpl.hpp
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
// Description: Factory shell for the whole-field IProductWRTBase
// operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseOpImpl.hpp
 * @brief Factory shell for the whole-field IProductWRTBase operator.
 *
 * @details
 * The whole-field behaviour of the operator is implemented entirely in
 * IProductWRTBaseOp (IProductWRTBaseOp.hpp); the class here exists only so
 * that an "IProductWRTBase" + execution-space key ("IProductWRTBaseSerial",
 * "IProductWRTBaseAVX", "IProductWRTBaseDevice") can be registered with the
 * operator factory. The numerical work is done by the block operators
 * created alongside it (see IProductWRTBaseSerialAVXStdMat.hpp and its
 * siblings). CMake-generated translation units instantiate and register
 * this class for each enabled execution space.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTBaseOp.hpp and the
 * operator factory.
 */

#pragma once

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Execution-space registration shell of IProductWRTBaseOp; adds no
 * behaviour of its own.
 *
 * @tparam ExecSpace Execution space the instance is registered for; used
 *                   only to distinguish factory entries.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData>
class IProductWRTBaseOpImpl : public IProductWRTBaseOp<TData>
{
public:
    /// @brief Construct the operator for @p expansionList and
    /// @p components; forwards to IProductWRTBaseOp.
    IProductWRTBaseOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                          const std::vector<std::string> &components)
        : IProductWRTBaseOp<TData>(expansionList, components)
    {
    }

    /// Registration name for the operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the operator factory;
    /// builds one operator for the given expansion list and components.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<IProductWRTBaseOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::Operators::detail
