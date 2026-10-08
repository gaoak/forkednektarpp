///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractOpImpl.hpp
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
 * @file PhysTraceExtractOpImpl.hpp
 * @brief Factory-registered whole-field implementation of the
 * PhysTraceExtract operator: one registrable type per execution space.
 *
 * @details
 * The whole-field level of this operator carries no execution-space
 * specific behaviour. PhysTraceExtractOp already implements the loop over
 * the element blocks, and every arithmetic decision belongs to the block
 * operators, which are selected independently through the block-operator
 * factory key. This header therefore only supplies the distinct type per
 * execution space that the operator factory needs in order to have
 * something to register.
 *
 * CMake generates one translation unit per execution space and data type
 * from Common/OpFactoryDec.cpp.in. That unit includes this header and
 * defines PhysTraceExtractOpImpl::className, whose initialiser registers
 * Instantiate() with the operator factory under the key
 * `PhysTraceExtractOp::name + <ExecSpace>`, which is exactly the key
 * PhysTraceExtractOp::Create() asks for.
 *
 * @see PhysTraceExtractOp.hpp for what the operator computes and for the
 * layout of the family.
 */

#pragma once

#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Execution-space instantiation of the whole-field trace
 * extraction; adds no behaviour of its own to PhysTraceExtractOp.
 *
 * @tparam ExecSpace   NektarSpaces::Serial, NektarSpaces::AVX or
 *                     NektarSpaces::Device. It selects the type the
 *                     factory registers and the key it is registered
 *                     under, and is otherwise unused in this class: the
 *                     execution space reaches the numerical work through
 *                     the block-operator factory key instead.
 * @tparam TData       Floating-point type of the field data.
 *
 * @see PhysTraceExtractOp for the interface this class inherits whole.
 */
template <typename ExecSpace, typename TData>
class PhysTraceExtractOpImpl : public PhysTraceExtractOp<TData>
{
public:
    /**
     * @brief Forward the expansion list and component names to the
     * interface base; there is nothing else to set up.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    PhysTraceExtractOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : PhysTraceExtractOp<TData>(expansionList, components)
    {
    }

    /// Factory registration token. Only declared here: the generated
    /// per-execution-space translation unit defines it, and the
    /// definition's initialiser is what performs the registration.
    // className - for OperatorFactory
    static std::string className;

    /**
     * @brief Creator function handed to the operator factory.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     *
     * @return The new operator, owned by the caller.
     */
    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<PhysTraceExtractOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::MultiRegions::detail
