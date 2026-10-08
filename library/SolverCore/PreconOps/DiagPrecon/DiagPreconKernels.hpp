///////////////////////////////////////////////////////////////////////////////
//
// File: DiagPreconKernels.hpp
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

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
void SetModeBlkKernel(const unsigned mode, const size_t nelmt,
                      const unsigned numdata, const TData val, TData *blkptr,
                      const bool isInterleaved, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    if (isInterleaved)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt, NEKTAR_LAMBDA(size_t idx) {
                constexpr auto vector_width =
                    NektarSpaces::vector_width<ExecSpace, TData>::value;
                size_t e = idx / vector_width;
                size_t i = idx % vector_width;

                blkptr[(e * numdata + mode) * vector_width + i] = val;
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt, NEKTAR_LAMBDA(size_t idx) {
                blkptr[(idx * numdata + mode)] = val;
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void CopyModeBlkKernel(const unsigned mode, const size_t nelmt,
                       const unsigned numdata, const TData *fromblkptr,
                       TData *toblkptr, const bool isInterleaved,
                       const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    if (isInterleaved)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt, NEKTAR_LAMBDA(size_t idx) {
                constexpr auto vector_width =
                    NektarSpaces::vector_width<ExecSpace, TData>::value;
                size_t e = idx / vector_width;
                size_t i = idx % vector_width;

                toblkptr[(e * numdata + mode) * vector_width + i] =
                    fromblkptr[(e * numdata + mode) * vector_width + i];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt, NEKTAR_LAMBDA(size_t idx) {
                toblkptr[idx * numdata + mode] =
                    fromblkptr[idx * numdata + mode];
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void InvDiagBlkKernel(const size_t nsize, TData *diagblkptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(size_t idx) {
            // Set any zero terms to 1.0 - arises in variable p case.
            diagblkptr[idx] = (diagblkptr[idx] == 0.0) ? diagblkptr[idx]
                                                       : 1.0 / diagblkptr[idx];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
