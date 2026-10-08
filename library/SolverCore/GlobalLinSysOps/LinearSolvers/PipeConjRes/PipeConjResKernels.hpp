///////////////////////////////////////////////////////////////////////////////
//
// File: PipeConjResKernels.hpp
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
// Description: Fused update of the pipelined conjugate residual method.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Update the search directions and the solution:
 *
 *   z = beta z + s, q = beta q + wk, p = beta p + r,
 *   out += alpha p, r -= alpha q, w -= alpha z,
 *
 * @p STAGE is 0 on the first update, where z, q and p are not read:
 * z = s, q = wk and p = r. It is 1 on every later update.
 */
template <typename ExecSpace, unsigned int STAGE, typename TData>
NEK_FORCE_INLINE static void UpdatePipeConjResSearchDirection(
    const TData alpha, [[maybe_unused]] const TData beta,
    LibUtilities::Field<TData, FieldState::Coeff> &s,
    LibUtilities::Field<TData, FieldState::Coeff> &wk,
    LibUtilities::Field<TData, FieldState::Coeff> &z,
    LibUtilities::Field<TData, FieldState::Coeff> &q,
    LibUtilities::Field<TData, FieldState::Coeff> &p,
    LibUtilities::Field<TData, FieldState::Coeff> &r,
    LibUtilities::Field<TData, FieldState::Coeff> &w,
    LibUtilities::Field<TData, FieldState::Coeff> &out)
{
    using MemSpace = typename ExecSpace::memory_space;
    using DirAccess =
        typename std::conditional<STAGE == 0, WriteOnly, ReadWrite>::type;

    for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = out.GetBlocks()[blk].CompSize() * out.GetNumComponents() *
                    out.GetNumHomoModes();
        auto sptr =
            s.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto wkptr =
            wk.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto qptr =
            q.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto pptr =
            p.GetBlocks()[blk].template GetPtr<MemSpace, DirAccess>(streamID);
        auto rptr =
            r.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto wptr =
            w.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto outptr =
            out.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        if constexpr (STAGE == 0)
        {
            Nektar::parallel_for<ExecSpace>(
                0, size, NEKTAR_LAMBDA(const size_t idx) {
                    auto z0 = sptr[idx];
                    auto q0 = wkptr[idx];
                    auto p0 = rptr[idx];

                    outptr[idx] += alpha * p0;
                    rptr[idx] = p0 - alpha * q0;
                    wptr[idx] -= alpha * z0;

                    zptr[idx] = z0;
                    qptr[idx] = q0;
                    pptr[idx] = p0;
                });
        }
        else
        {
            Nektar::parallel_for<ExecSpace>(
                0, size, NEKTAR_LAMBDA(const size_t idx) {
                    auto r0 = rptr[idx];

                    auto z0 = beta * zptr[idx] + sptr[idx];
                    auto q0 = beta * qptr[idx] + wkptr[idx];
                    auto p0 = beta * pptr[idx] + r0;

                    outptr[idx] += alpha * p0;
                    rptr[idx] = r0 - alpha * q0;
                    wptr[idx] -= alpha * z0;

                    zptr[idx] = z0;
                    qptr[idx] = q0;
                    pptr[idx] = p0;
                });
        }

        if constexpr (STAGE == 0)
        {
            const auto width = r.GetBlocks()[blk].GetInterleaveWidth();
            z.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
            q.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
            p.GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
        }
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
