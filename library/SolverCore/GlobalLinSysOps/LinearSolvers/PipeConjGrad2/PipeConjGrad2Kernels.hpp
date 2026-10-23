///////////////////////////////////////////////////////////////////////////////
//
// File: PipeConjGrad2Kernels.hpp
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
// Description: Fused updates of the Gropp pipelined conjugate gradient method.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Update the solution, out += alpha p, r -= alpha s, u -= alpha q,
 * and, in the same pass, dots[0] = (u, u) and dots[1] = (u, r), in the memory
 * space of ExecSpace.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpdatePipeConjGrad2Solution(
    const TData alpha, LibUtilities::Field<TData, FieldState::Coeff> &p,
    LibUtilities::Field<TData, FieldState::Coeff> &s,
    LibUtilities::Field<TData, FieldState::Coeff> &q,
    LibUtilities::Field<TData, FieldState::Coeff> &out,
    LibUtilities::Field<TData, FieldState::Coeff> &r,
    LibUtilities::Field<TData, FieldState::Coeff> &u, TData *dots)
{
    using MemSpace = typename ExecSpace::memory_space;

    Math::zeroKernel<ExecSpace>(2, dots);

    for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = out.GetBlocks()[blk].CompSize() * out.GetNumComponents() *
                    out.GetNumHomoModes();
        auto pptr =
            p.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto sptr =
            s.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto qptr =
            q.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto outptr =
            out.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto rptr =
            r.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto uptr =
            u.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        // The padding is zero, so the whole block is reduced.
        Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 2>(
            0, size,
            NEKTAR_LAMBDA(const size_t idx, TData *vals) {
                const auto r1 = rptr[idx] - alpha * sptr[idx];
                const auto u1 = uptr[idx] - alpha * qptr[idx];

                outptr[idx] += alpha * pptr[idx];
                rptr[idx] = r1;
                uptr[idx] = u1;

                vals[0] = u1 * u1;
                vals[1] = u1 * r1;
            },
            dots);
    }

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Update the search directions, p = beta p + u, s = beta s + w, and,
 * in the same pass, dots[0] = (p, s), in the memory space of ExecSpace.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpdatePipeConjGrad2SearchDirection(
    const TData beta, LibUtilities::Field<TData, FieldState::Coeff> &u,
    LibUtilities::Field<TData, FieldState::Coeff> &w,
    LibUtilities::Field<TData, FieldState::Coeff> &p,
    LibUtilities::Field<TData, FieldState::Coeff> &s, TData *dots)
{
    using MemSpace = typename ExecSpace::memory_space;

    Math::zeroKernel<ExecSpace>(1, dots);

    for (unsigned int blk = 0; blk < p.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = p.GetBlocks()[blk].CompSize() * p.GetNumComponents() *
                    p.GetNumHomoModes();
        auto uptr =
            u.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto wptr =
            w.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto pptr =
            p.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto sptr =
            s.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        // The padding is zero, so the whole block is reduced.
        Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 1>(
            0, size,
            NEKTAR_LAMBDA(const size_t idx, TData *vals) {
                const auto p1 = beta * pptr[idx] + uptr[idx];
                const auto s1 = beta * sptr[idx] + wptr[idx];

                pptr[idx] = p1;
                sptr[idx] = s1;

                vals[0] = p1 * s1;
            },
            dots);
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
