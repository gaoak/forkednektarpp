///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransOpImpl.hpp
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
// Description: Factory-registered whole-field implementation of the
// backward transform
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransOpImpl.hpp
 * @brief Factory-registered whole-field implementation of the
 * backward-transform operator.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, one per execution space and
 * data type; it should not normally be included by any other
 * translation unit. Other code goes through BwdTransOp.hpp and the
 * operator factory.
 */

#pragma once

#include "Operators/ElmtOps/BwdTrans/BwdTransOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Whole-field implementation of BwdTransOp, registered with the
 * operator factory once per execution space.
 *
 * The class adds no behaviour of its own: the whole-field stage of the
 * backward transform is execution-space independent -- it is the block
 * loop in BwdTransOp::v_Apply -- and all numerical work happens in the
 * per-block detail::BwdTransBlockOpImpl objects that ElmtOp::Create
 * builds alongside it. The ExecSpace template parameter exists so that
 * the CMake-generated factory declaration code can register a separate
 * entry ("BwdTransSerial", "BwdTransAVX", "BwdTransDevice") for each
 * execution space enabled in the build.
 *
 * @tparam ExecSpace Execution space the factory entry is registered
 *                   for.
 * @tparam TData     Floating-point type of the field data.
 *
 * @see BwdTransOp for the whole-field interface this class completes.
 */
template <typename ExecSpace, typename TData>
class BwdTransOpImpl : public BwdTransOp<TData>
{
public:
    /**
     * @brief Construct the whole-field operator; called by the
     * factory-registered creator function Instantiate().
     *
     * @param   expansionList   Expansion list the operator runs over.
     * @param   components      Names of the field components the
     *                          operator is applied to.
     */
    BwdTransOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : BwdTransOp<TData>(expansionList, components)
    {
    }

    /// Operator-factory registration key, `"BwdTrans" + execution space`.
    /// Defined by the CMake-generated declaration file, whose
    /// initialiser performs the registration.
    static std::string className;

    /**
     * @brief Creator function registered with the operator factory
     * under #className.
     *
     * @param   expansionList   Expansion list the operator runs over.
     * @param   components      Names of the field components the
     *                          operator is applied to.
     *
     * @return The newly created whole-field operator.
     */
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BwdTransOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }
};

} // namespace Nektar::Operators::detail
