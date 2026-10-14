///////////////////////////////////////////////////////////////////////////////
//
// File: PipeConjGradKernels.hpp
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
// Description: Fused update of the pipelined conjugate gradient method.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Update the search directions and the solution:
 *
 *   z = beta z + n, q = beta q + m, s = beta s + w, p = beta p + u,
 *   out += alpha p, r -= alpha s, u -= alpha q, w -= alpha z,
 *
 * and, in the same pass, the products of the next iteration:
 * dots[0] = (u, u), dots[1] = (u, r) and dots[2] = (w, u), in the memory
 * space of ExecSpace.
 *
 * @p STAGE is 0 on the first update, where z, q, s and p are not read:
 * z = n, q = m, s = w and p = u. It is 1 on every later update.
 */
template <typename ExecSpace, unsigned int STAGE, typename TData>
NEK_FORCE_INLINE static void UpdatePipeConjGradSearchDirection(
    const TData alpha, [[maybe_unused]] const TData beta,
    LibUtilities::Field<TData, FieldState::Coeff> &n,
    LibUtilities::Field<TData, FieldState::Coeff> &m,
    LibUtilities::Field<TData, FieldState::Coeff> &z,
    LibUtilities::Field<TData, FieldState::Coeff> &q,
    LibUtilities::Field<TData, FieldState::Coeff> &s,
    LibUtilities::Field<TData, FieldState::Coeff> &p,
    LibUtilities::Field<TData, FieldState::Coeff> &u,
    LibUtilities::Field<TData, FieldState::Coeff> &w,
    LibUtilities::Field<TData, FieldState::Coeff> &r,
    LibUtilities::Field<TData, FieldState::Coeff> &out, TData *dots)
{
    using MemSpace = typename ExecSpace::memory_space;
    using DirAccess =
        typename std::conditional<STAGE == 0, WriteOnly, ReadWrite>::type;

    Math::zeroKernel<ExecSpace>(3, dots);

    for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = out.GetBlocks()[blk].CompSize() * out.GetNumComponents() *
                    out.GetNumHomoModes();
        auto nptr =
            n.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto mptr =
            m.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto qptr =
            q.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto sptr =
            s.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto pptr =
            p.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto uptr =
            u.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto wptr =
            w.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto rptr =
            r.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto outptr =
            out.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        if constexpr (STAGE == 0)
        {
            Nektar::parallel_reduce_sum<ExecSpace, false, 3>(
                0, size,
                NEKTAR_LAMBDA(const size_t idx, TData *sums) {
                    auto z0 = nptr[idx];
                    auto q0 = mptr[idx];
                    auto s0 = wptr[idx];
                    auto p0 = uptr[idx];

                    const auto r1 = rptr[idx] - alpha * s0;
                    const auto u1 = p0 - alpha * q0;
                    const auto w1 = s0 - alpha * z0;

                    outptr[idx] += alpha * p0;
                    rptr[idx] = r1;
                    uptr[idx] = u1;
                    wptr[idx] = w1;

                    zptr[idx] = z0;
                    qptr[idx] = q0;
                    sptr[idx] = s0;
                    pptr[idx] = p0;

                    sums[0] += u1 * u1;
                    sums[1] += u1 * r1;
                    sums[2] += w1 * u1;
                },
                dots);
        }
        else
        {
            Nektar::parallel_reduce_sum<ExecSpace, false, 3>(
                0, size,
                NEKTAR_LAMBDA(const size_t idx, TData *sums) {
                    auto u0 = uptr[idx];
                    auto w0 = wptr[idx];

                    auto z0 = beta * zptr[idx] + nptr[idx];
                    auto q0 = beta * qptr[idx] + mptr[idx];
                    auto s0 = beta * sptr[idx] + w0;
                    auto p0 = beta * pptr[idx] + u0;

                    const auto r1 = rptr[idx] - alpha * s0;
                    const auto u1 = u0 - alpha * q0;
                    const auto w1 = w0 - alpha * z0;

                    outptr[idx] += alpha * p0;
                    rptr[idx] = r1;
                    uptr[idx] = u1;
                    wptr[idx] = w1;

                    zptr[idx] = z0;
                    qptr[idx] = q0;
                    sptr[idx] = s0;
                    pptr[idx] = p0;

                    sums[0] += u1 * u1;
                    sums[1] += u1 * r1;
                    sums[2] += w1 * u1;
                },
                dots);
        }

        if constexpr (STAGE == 0)
        {
            const auto width = u.GetBlocks()[blk].GetInterleaveWidth();
            z.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
            q.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
            s.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
            p.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
        }
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
