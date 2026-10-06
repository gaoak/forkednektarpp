///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledOpImpl.hpp
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
 * @file PhysInterp1DScaledOpImpl.hpp
 * @brief Factory-registered whole-field implementation of the scaled
 * physical-space interpolation operator.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through PhysInterp1DScaledOp.hpp and the
 * operator factory.
 */

#pragma once

#include <MultiRegions/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Whole-field implementation of PhysInterp1DScaledOp, registered
 * with the operator factory once per execution space.
 *
 * The class adds no behaviour of its own: the whole-field stage of the
 * interpolation is execution-space independent (the block loops of
 * PhysInterp1DScaledOp::v_Apply and PhysInterp1DScaledOp::SetScaleFactor),
 * and all numerical work happens in the per-block
 * detail::PhysInterp1DScaledBlockOpImpl objects created alongside it by
 * ElmtOp::Create. The ExecSpace template parameter exists so that the
 * CMake-generated factory declaration code can register a separate entry
 * ("PhysInterp1DScaledSerial", "PhysInterp1DScaledAVX",
 * "PhysInterp1DScaledDevice") for each execution space enabled in the
 * build.
 *
 * @tparam ExecSpace Execution space the factory entry is registered for.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData>
class PhysInterp1DScaledOpImpl : public PhysInterp1DScaledOp<TData>
{
public:
    /// @brief Construct the operator for @p expansionList and
    /// @p components; forwards to PhysInterp1DScaledOp.
    PhysInterp1DScaledOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : PhysInterp1DScaledOp<TData>(expansionList, components)
    {
    }

    /// Operator class name; defined by the generated factory code.
    static std::string className;

    /// Creator function registered with OperatorFactory.
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<PhysInterp1DScaledOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::MultiRegions::detail
