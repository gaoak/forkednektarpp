///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondSlipWallCFEKernels.hpp
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
// Description: Block kernels of the inviscid slip wall boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Mirror the momentum about the boundary plane.
 *
 * The caller has put the interior trace in the storage, so this works
 * pointwise on values already in the right place and orientation. Density
 * and energy pass through: only the normal component of the momentum is
 * reversed, which is what leaves the averaged normal velocity zero and the
 * tangential velocity free.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void MirrorNormalMomentumBlock(
    const size_t stride, const unsigned coordDim, const TData *norms,
    TData *bndPtr, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            TData momDotN = TData(0.0);
            for (unsigned d = 1; d <= coordDim; ++d)
            {
                momDotN += bndPtr[d * stride + i] * norms[(d - 1) * stride + i];
            }

            for (unsigned d = 1; d <= coordDim; ++d)
            {
                bndPtr[d * stride + i] -=
                    TData(2.0) * momDotN * norms[(d - 1) * stride + i];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
