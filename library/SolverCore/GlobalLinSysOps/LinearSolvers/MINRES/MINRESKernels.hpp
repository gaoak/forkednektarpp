///////////////////////////////////////////////////////////////////////////////
//
// File: MINRESKernels.hpp
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
// Description: Fused vector updates of the MINRES solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The new search direction p0 and the solution update out += d p0, in
 * one pass.
 *
 * p0 = b w at the first iteration (ITER 0), b w + a p1 at the second (ITER 1)
 * and a p1 + b w + c p0 after (ITER 2); p1 and p0 are read only when used.
 */
template <typename ExecSpace, unsigned int ITER, typename TData>
NEK_FORCE_INLINE static void UpdateMINRESSolution(
    [[maybe_unused]] const TData a, const TData b,
    [[maybe_unused]] const TData c, const TData d,
    LibUtilities::Field<TData, FieldState::Coeff> &w,
    LibUtilities::Field<TData, FieldState::Coeff> &p1,
    LibUtilities::Field<TData, FieldState::Coeff> &p0,
    LibUtilities::Field<TData, FieldState::Coeff> &out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = out.GetBlocks()[blk].CompSize() * out.GetNumComponents() *
                    out.GetNumHomoModes();
        auto wptr =
            w.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto outptr =
            out.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        if constexpr (ITER == 0)
        {
            auto p0ptr =
                p0.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(
                    streamID);
            Nektar::parallel_for<ExecSpace>(
                0, size, NEKTAR_LAMBDA(const size_t idx) {
                    auto q0 = b * wptr[idx];

                    p0ptr[idx]  = q0;
                    outptr[idx] = d * q0 + outptr[idx];
                });
        }
        else if constexpr (ITER == 1)
        {
            auto p1ptr =
                p1.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            auto p0ptr =
                p0.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(
                    streamID);
            Nektar::parallel_for<ExecSpace>(
                0, size, NEKTAR_LAMBDA(const size_t idx) {
                    auto q0 = b * wptr[idx] + a * p1ptr[idx];

                    p0ptr[idx]  = q0;
                    outptr[idx] = d * q0 + outptr[idx];
                });
        }
        else
        {
            auto p1ptr =
                p1.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            auto p0ptr =
                p0.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(
                    streamID);
            Nektar::parallel_for<ExecSpace>(
                0, size, NEKTAR_LAMBDA(const size_t idx) {
                    auto q0 = a * p1ptr[idx] + b * wptr[idx] + c * p0ptr[idx];

                    p0ptr[idx]  = q0;
                    outptr[idx] = d * q0 + outptr[idx];
                });
        }
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
