///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceOpImpl.hpp
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
 * @file IProductWRTPhysNormalDerivTraceOpImpl.hpp
 * @brief Factory-registered whole-field implementation of the
 * IProductWRTPhysNormalDerivTrace operator: one registrable type per
 * execution space.
 *
 * @details
 * IProductWRTPhysNormalDerivTraceOp already implements the loop over the
 * element blocks, and every arithmetic decision belongs to the block
 * operators, which are selected independently through the block-operator
 * factory key. This header therefore supplies the distinct type per
 * execution space that the operator factory needs in order to have
 * something to register, plus the one piece of behaviour the block
 * operators cannot carry: v_CopyDirection(), the staging copy
 * ApplyVector() makes between whole-field blocks, which has to be issued
 * in the memory space the execution space names and so cannot be written
 * once for every space.
 *
 * CMake generates one translation unit per execution space and data type
 * from Common/OpFactoryDec.cpp.in. That unit includes this header and
 * defines IProductWRTPhysNormalDerivTraceOpImpl::className, whose
 * initialiser registers Instantiate() with the operator factory under
 * the key `IProductWRTPhysNormalDerivTraceOp::name + <ExecSpace>`, which
 * is exactly the key IProductWRTPhysNormalDerivTraceOp::Create() asks
 * for.
 *
 * @see IProductWRTPhysNormalDerivTraceOp.hpp for what the operator
 * computes and for the layout of the family.
 */

#pragma once

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"

#include <MultiRegions/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Execution-space instantiation of the whole-field
 * normal-derivative trace lift; adds the per-direction staging copy to
 * IProductWRTPhysNormalDerivTraceOp and nothing else.
 *
 * @tparam ExecSpace   NektarSpaces::Serial, NektarSpaces::AVX or
 *                     NektarSpaces::Device. It selects the type the
 *                     factory registers, the key it is registered under,
 *                     and the memory space and copy kernel
 *                     v_CopyDirection() issues in. The numerical work
 *                     reaches its execution space through the
 *                     block-operator factory key instead.
 * @tparam TData       Floating-point type of the field data.
 *
 * @see IProductWRTPhysNormalDerivTraceOp for the interface this class
 * inherits.
 */
template <typename ExecSpace, typename TData>
class IProductWRTPhysNormalDerivTraceOpImpl
    : public IProductWRTPhysNormalDerivTraceOp<TData>
{
    /// Memory space the staging copy reads and writes in.
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Forward the expansion list and component names to the
     * interface base; there is nothing else to set up.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    IProductWRTPhysNormalDerivTraceOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : IProductWRTPhysNormalDerivTraceOp<TData>(expansionList, components)
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
        return std::make_unique<
            IProductWRTPhysNormalDerivTraceOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    /**
     * @brief Copy direction @p dir of @p inblock into @p outblock, in the
     * memory space #MemSpace names.
     *
     * Component n of @p outblock takes component n * @p dim + @p dir of
     * @p inblock. A component slice is CompSize values whatever the
     * interleave, so the copy is layout-blind; @p outblock is then given
     * @p inblock's interleave.
     *
     * The copy is issued on the stream block @p blk applies on, so it is
     * ordered ahead of that apply without a synchronization.
     */
    void v_CopyDirection(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int dir, const unsigned int dim,
        const unsigned int blk) override
    {
        const unsigned int streamID = blk + 1;

        const unsigned int ncomp = outblock.GetNumComponents();

        const auto compSize = inblock.CompSize();
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(streamID);

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            Math::copyKernel<ExecSpace>(compSize,
                                        inptr + (n * dim + dir) * compSize,
                                        outptr + n * compSize, streamID);
        }

        outblock.template SetInterleaveWidth<TData>(
            inblock.GetInterleaveWidth());
    }
};

} // namespace Nektar::MultiRegions::detail
