///////////////////////////////////////////////////////////////////////////////
//
// File: CGSKernels.hpp
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
// Description: Fused vector updates of the CGS solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief u = beta q + r and p = beta q + beta^2 p + u, in one pass.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpdateCGSSearchDirection(
    const TData beta, LibUtilities::Field<TData, FieldState::Coeff> &q,
    LibUtilities::Field<TData, FieldState::Coeff> &r,
    LibUtilities::Field<TData, FieldState::Coeff> &u,
    LibUtilities::Field<TData, FieldState::Coeff> &p)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < p.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = p.GetBlocks()[blk].CompSize() * p.GetNumComponents() *
                    p.GetNumHomoModes();
        auto qptr =
            q.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto rptr =
            r.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto uptr =
            u.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto pptr =
            p.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        Nektar::parallel_for<ExecSpace>(
            0, size, NEKTAR_LAMBDA(const size_t idx) {
                auto q0 = qptr[idx];
                auto u0 = beta * q0 + rptr[idx];

                pptr[idx] = beta * q0 + beta * beta * pptr[idx] + u0;
                uptr[idx] = u0;
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief q = u - alpha s and w = u + q, in one pass.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpdateCGSHalfStep(
    const TData alpha, LibUtilities::Field<TData, FieldState::Coeff> &s,
    LibUtilities::Field<TData, FieldState::Coeff> &u,
    LibUtilities::Field<TData, FieldState::Coeff> &q,
    LibUtilities::Field<TData, FieldState::Coeff> &w)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < w.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = w.GetBlocks()[blk].CompSize() * w.GetNumComponents() *
                    w.GetNumHomoModes();
        auto sptr =
            s.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto uptr =
            u.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto qptr =
            q.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto wptr =
            w.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        Nektar::parallel_for<ExecSpace>(
            0, size, NEKTAR_LAMBDA(const size_t idx) {
                auto u0 = uptr[idx];
                auto q0 = -alpha * sptr[idx] + u0;

                qptr[idx] = q0;
                wptr[idx] = u0 + q0;
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
