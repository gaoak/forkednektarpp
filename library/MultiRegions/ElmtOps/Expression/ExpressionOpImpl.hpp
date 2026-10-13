///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionOpImpl.hpp
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

/**
 * @file ExpressionOpImpl.hpp
 * @brief Factory-registration shim for the whole-field Expression
 * operator.
 *
 * The CMake-generated factory declarations instantiate
 * detail::ExpressionOpImpl per execution space and data type and
 * register it under the key "Expression" + execution space; see
 * ExpressionOp.hpp for the family overview.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through ExpressionOp.hpp and the
 * operator factory.
 */

#pragma once

#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Concrete whole-field Expression operator for one execution
 * space.
 *
 * As for every element-operator family, the execution-space class adds
 * no behaviour of its own: all evaluation state and logic lives in the
 * per-block operators (detail::ExpressionBlockOpImpl of
 * ExpressionBlockOpSerialAVXGeneric.hpp / ExpressionBlockOpDeviceGeneric.hpp),
 * which the inherited ExpressionOp machinery creates and drives. This
 * class exists so the operator factory has a distinct concrete type to
 * register for each execution space.
 *
 * @tparam ExecSpace Execution back-end (NektarSpaces::Serial, AVX or
 *                   Device); only forms the factory key here, since
 *                   the numerical work happens in the block operators.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData>
class ExpressionOpImpl : public ExpressionOp<TData>
{
public:
    /// @brief Construct the operator for @p expansionList and
    /// @p components; forwards to ExpressionOp.
    ExpressionOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : ExpressionOp<TData>(expansionList, components)
    {
    }

    /// Operator class name; defined by the generated factory code.
    static std::string className;

    /// Creator function registered with OperatorFactory.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<ExpressionOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::MultiRegions::detail
