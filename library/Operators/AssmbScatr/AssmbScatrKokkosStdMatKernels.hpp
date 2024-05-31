///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrKokkosStdMatKernels.hpp
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

#include "Operators/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
AssembleKernel([[maybe_unused]] const size_t gridSize,
               [[maybe_unused]] const size_t blockSize,
               const unsigned int ncoeff, const unsigned int nelmt,
               const unsigned int offset, const int *assmbPtr,
               const TData *signPtr, const TData *inPtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt, KOKKOS_LAMBDA(int e) {
            unsigned int index = offset + e * ncoeff;

            for (unsigned int i = 0; i < ncoeff; i++)
            {
                Kokkos::atomic_add(outPtr + assmbPtr[index + i],
                                   signPtr[index + i] * inPtr[index + i]);
            }
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
AssembleKernel([[maybe_unused]] const size_t gridSize,
               [[maybe_unused]] const size_t blockSize,
               const unsigned int ncoeff, const unsigned int nelmt,
               const unsigned int offset, const int *assmbPtr, const TData sign,
               const TData *inPtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt, KOKKOS_LAMBDA(int e) {
            unsigned int index = offset + e * ncoeff;

            for (unsigned int i = 0; i < ncoeff; i++)
            {
                Kokkos::atomic_add(outPtr + assmbPtr[index + i],
                                   sign * inPtr[index + i]);
            }
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
AssembleKernel([[maybe_unused]] const size_t gridSize,
               [[maybe_unused]] const size_t blockSize,
               const unsigned int ncoeff, const unsigned int nelmt,
               const unsigned int offset, const int *assmbPtr,
               const TData *inPtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt, KOKKOS_LAMBDA(int e) {
            unsigned int index = offset + e * ncoeff;

            for (unsigned int i = 0; i < ncoeff; i++)
            {
                Kokkos::atomic_add(outPtr + assmbPtr[index + i],
                                   inPtr[index + i]);
            }
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
GlobalToLocalKernel([[maybe_unused]] const size_t gridSize,
                    [[maybe_unused]] const size_t blockSize,
                    const unsigned int ncoeff, const unsigned int nelmt,
                    const unsigned int offset, const int *assmbPtr,
                    const TData *signPtr, const TData *inPtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt, KOKKOS_LAMBDA(int e) {
            unsigned int index = offset + e * ncoeff;

            for (unsigned int i = 0; i < ncoeff; i++)
            {
                outPtr[index + i] =
                    signPtr[index + i] * inPtr[assmbPtr[index + i]];
            }
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
GlobalToLocalKernel([[maybe_unused]] const size_t gridSize,
                    [[maybe_unused]] const size_t blockSize,
                    const unsigned int ncoeff, const unsigned int nelmt,
                    const unsigned int offset, const int *assmbPtr,
                    const TData sign, const TData *inPtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt, KOKKOS_LAMBDA(int e) {
            unsigned int index = offset + e * ncoeff;

            for (unsigned int i = 0; i < ncoeff; i++)
            {
                outPtr[index + i] = sign * inPtr[assmbPtr[index + i]];
            }
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
GlobalToLocalKernel([[maybe_unused]] const size_t gridSize,
                    [[maybe_unused]] const size_t blockSize,
                    const unsigned int ncoeff, const unsigned int nelmt,
                    const unsigned int offset, const int *assmbPtr,
                    const TData *inPtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt, KOKKOS_LAMBDA(int e) {
            unsigned int index = offset + e * ncoeff;

            for (unsigned int i = 0; i < ncoeff; i++)
            {
                outPtr[index + i] = inPtr[assmbPtr[index + i]];
            }
        });
}

} // namespace Nektar::Operators::detail
