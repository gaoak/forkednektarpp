///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradKernels.hpp
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

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpdateConjGradSearchDirection(
    const TData alpha, const TData beta,
    LibUtilities::Field<TData, FieldState::Coeff> &w,
    LibUtilities::Field<TData, FieldState::Coeff> &s,
    LibUtilities::Field<TData, FieldState::Coeff> &p,
    LibUtilities::Field<TData, FieldState::Coeff> &q,
    LibUtilities::Field<TData, FieldState::Coeff> &r,
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
        auto sptr =
            s.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto pptr =
            p.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto qptr =
            q.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto rptr =
            r.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto outptr =
            out.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);

        Nektar::LoopExecutionSetStreamID(streamID);

        Nektar::parallel_for<ExecSpace>(
            0, size, NEKTAR_LAMBDA(const size_t idx) {
                auto w0 = wptr[idx];
                auto p0 = pptr[idx];

                p0 = beta * p0 + w0;
                outptr[idx] += alpha * p0;
                pptr[idx] = p0;

                auto s0 = sptr[idx];
                auto q0 = qptr[idx];

                q0 = beta * q0 + s0;
                rptr[idx] += -alpha * q0;
                qptr[idx] = q0;
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}
