///////////////////////////////////////////////////////////////////////////////
//
// File: RiemannSolverOpImpl.hpp
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
// Description: Riemann Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/LoopExecution/LoopExecution.hpp"

#include "Operators/SolverUtilsOps/RiemannSolvers/RiemannSolverOp.hpp"

namespace Nektar::Operators::detail
{

template <template <typename> typename RiemannKernel, typename ExecSpace,
          typename TData>
NEK_FORCE_INLINE static void RiemannKernelLauncher(
    const size_t blksize, const unsigned int velComps,
    const unsigned int fluxComps, const TData *velbase, const TData *normbase,
    const TData *fwdbase, const TData *bwdbase, TData *fluxbase,
    const unsigned int streamID)
{
    // Explicit vectorisation for AVX backend,
    // vec_t = tinysimd::simd<TData> for AVX,
    // vec_t = TData otherwise.
    // using vec_t = typename data_type_if<
    //    std::is_same_v<ExecSpace, NektarSpaces::AVX>, TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize = blksize / vec_width;

    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            RiemannKernel<ExecSpace>()(
                blksize, velComps, fluxComps, velbase + i * vec_width,
                normbase + i * vec_width, fwdbase + i * vec_width,
                bwdbase + i * vec_width, fluxbase + i * vec_width);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <template <typename> typename RiemannKernel, typename ExecSpace,
          typename TData>
class RiemannSolverOpImpl : public RiemannSolverOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    RiemannSolverOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : RiemannSolverOp<TData>(std::move(expansionList), components)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            RiemannSolverOpImpl<RiemannKernel, ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    void v_Apply(Field<TData, FieldState::Phys> &Fwd,
                 Field<TData, FieldState::Phys> &Bwd,
                 Field<TData, FieldState::Phys> &flux) override
    {
        // Loop over blocks
        for (unsigned int blk = 0; blk < flux.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            // Get block references
            auto &velblock    = this->m_traceAdvVel.GetBlocks()[blk];
            auto &normalblock = this->m_traceNormals.GetBlocks()[blk];
            auto &fwdblock    = Fwd.GetBlocks()[blk];
            auto &bwdblock    = Bwd.GetBlocks()[blk];
            auto &fluxblock   = flux.GetBlocks()[blk];

            // Base pointers in MemSpace corresponding to ExecSpace
            auto velbase =
                velblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto normbase =
                normalblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto fwdbase =
                fwdblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto bwdbase =
                bwdblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto fluxbase =
                fluxblock.template GetPtr<MemSpace, WriteOnly>(streamID);

            // Sizes / strides
            const auto blksize   = fluxblock.CompSize();
            const auto velComps  = velblock.GetNumComponents();
            const auto fluxComps = fluxblock.GetNumComponents();

            // Launch kernel
            RiemannKernelLauncher<RiemannKernel, ExecSpace>(
                blksize, velComps, fluxComps, velbase, normbase, fwdbase,
                bwdbase, fluxbase, streamID);
        }
    }
};

} // namespace Nektar::Operators::detail
