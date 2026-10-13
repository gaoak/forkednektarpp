///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassOpImpl.hpp
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
 * @file MultiplyByElmtInvMassOpImpl.hpp
 * @brief Factory shell for the whole-field MultiplyByElmtInvMass operator.
 *
 * @details
 * The whole-field behaviour of the operator is implemented entirely in
 * MultiplyByElmtInvMassOp (MultiplyByElmtInvMassOp.hpp); the class here
 * exists only so that a "MultiplyByElmtInvMass" + execution-space key
 * ("MultiplyByElmtInvMassSerial", "MultiplyByElmtInvMassAVX",
 * "MultiplyByElmtInvMassDevice") can be registered with the operator
 * factory. The numerical work is done by the block operators created
 * alongside it (see MultiplyByElmtInvMassBlockOpSerialAVXGeneric.hpp and
 * MultiplyByElmtInvMassBlockOpDeviceGeneric.hpp). CMake-generated translation
 * units instantiate and register this class for each enabled execution
 * space.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through MultiplyByElmtInvMassOp.hpp and the
 * operator factory.
 */

#pragma once

#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Execution-space registration shell of MultiplyByElmtInvMassOp;
 * adds no behaviour of its own.
 *
 * @details
 * The whole-field stage of the operator is execution-space independent
 * (the block loop in MultiplyByElmtInvMassOp::v_Apply), and all numerical
 * work happens in the per-block detail::MultiplyByElmtInvMassBlockOpImpl
 * objects created alongside it by ElmtOp::Create. The ExecSpace template
 * parameter exists so that the generated factory declaration code can
 * register a separate entry for each execution space enabled in the build.
 *
 * @tparam ExecSpace Execution space the instance is registered for; used
 *                   only to distinguish factory entries.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData>
class MultiplyByElmtInvMassOpImpl : public MultiplyByElmtInvMassOp<TData>
{
public:
    /// @brief Construct the operator for @p expansionList and
    /// @p components; forwards to MultiplyByElmtInvMassOp.
    MultiplyByElmtInvMassOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : MultiplyByElmtInvMassOp<TData>(expansionList, components)
    {
    }

    /// Operator class name; defined by the generated factory code.
    static std::string className;

    /// Creator function registered with OperatorFactory.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MultiplyByElmtInvMassOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::MultiRegions::detail
