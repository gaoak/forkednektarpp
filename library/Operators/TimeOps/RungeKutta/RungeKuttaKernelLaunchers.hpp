///////////////////////////////////////////////////////////////////////////////
//
// File: RungeKuttaKernelLaunchers.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static void StageSolutionRungeKuttaKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... explicits)
{
    constexpr unsigned int stage    = sizeof...(explicits);
    constexpr unsigned int indStart = (stage * (stage - 1)) / 2;

    // 2nd order
    if constexpr (IntOrder == 2)
    {
        // clang-format off
        constexpr TData coeff[] =
                {  1./2. };
                // {  1.0. }; // SSP
        // clang-format on

        inout[idx] =
            solution[idx] + ((explicits[idx] * coeff[indStart + ind]) + ...);
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        // clang-format off
        constexpr TData coeff[] =
                {  1./2.,
                      0.,   3./4. };
                //{   1.,
                //    1./4.,   1./4. }; // SSP
        // clang-format on

        inout[idx] =
            solution[idx] + ((explicits[idx] * coeff[indStart + ind]) + ...);
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        // clang-format off
        constexpr TData coeff[] =
                {  1./2.,
                      0.,   1./2.,
                      0.,      0.,      1. };
        // clang-format on

        inout[idx] =
            solution[idx] + ((explicits[idx] * coeff[indStart + ind]) + ...);
    }
    // 5th order
    else if constexpr (IntOrder == 5)
    {
        // clang-format off
        constexpr TData coeff[] = 
                {  1./4.,
                   1./8.,   1./8.,
                      0.,  -1./2.,      1.,
                  3./16.,      0.,      0.,  9./16.,
                  -3./7.,   2./7.,  12./7., -12./7.,   8./7. };
        // clang-format on

        inout[idx] =
            solution[idx] + ((explicits[idx] * coeff[indStart + ind]) + ...);
    }
}

template <unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static void UpdateRungeKuttaKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... explicits)
{
    // 1st order
    if constexpr (IntOrder == 1)
    {
        constexpr TData coeff[] = {1.0};

        inout[idx] = solution[idx] + ((explicits[idx] * coeff[ind]) + ...);
    }
    // 2nd order
    if constexpr (IntOrder == 2)
    {
        constexpr TData coeff[] = {0.0, 1.0};
        // constexpr TData coeff[] = {1.0 / 2.0, 1.0 / 2.0}; // SSP

        inout[idx] = solution[idx] + ((explicits[idx] * coeff[ind]) + ...);
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        constexpr TData coeff[] = {2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0};
        // constexpr TData coeff[] = {1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0}; // SSP

        inout[idx] = solution[idx] + ((explicits[idx] * coeff[ind]) + ...);
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        constexpr TData coeff[] = {1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0};

        inout[idx] = solution[idx] + ((explicits[idx] * coeff[ind]) + ...);
    }
    // 5th order
    else if constexpr (IntOrder == 5)
    {
        constexpr TData coeff[] = {7.0 / 90.0,  0.0,         32.0 / 90.0,
                                   12.0 / 90.0, 32.0 / 90.0, 7.0 / 90.0};

        inout[idx] = solution[idx] + ((explicits[idx] * coeff[ind]) + ...);
    }
}

// Kernel Launchers.
#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
// Currently, argument pack can't be captured in a device lambda. Explicit
// kernel must be used (instead of nektar::parallel_for)
template <unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void StageSolutionRungeKuttaKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TData *__restrict solution, const TDatas *__restrict... explicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        StageSolutionRungeKuttaKernelImpl<IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(explicits)>(),
            explicits...);
    }
}

template <unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void UpdateRungeKuttaKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TData *__restrict solution, const TDatas *__restrict... explicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateRungeKuttaKernelImpl<IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(explicits)>(),
            explicits...);
    }
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void StageSolutionRungeKuttaKernel(
    const size_t nsize, TData *inout, const TData *solution,
    const TDatas *...explicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    StageSolutionRungeKuttaKernelLauncher<IntOrder>
        <<<gridSize, blockSize>>>(nsize, inout, solution, explicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateRungeKuttaKernel(const size_t nsize,
                                                    TData *inout,
                                                    const TData *solution,
                                                    const TDatas *...explicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    UpdateRungeKuttaKernelLauncher<IntOrder>
        <<<gridSize, blockSize>>>(nsize, inout, solution, explicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}
#else
template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void StageSolutionRungeKuttaKernel(
    const size_t nsize, TData *inout, const TData *solution,
    const TDatas *...explicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            StageSolutionRungeKuttaKernelImpl<IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(explicits)>(),
                explicits...);
        });
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateRungeKuttaKernel(const size_t nsize,
                                                    TData *inout,
                                                    const TData *solution,
                                                    const TDatas *...explicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateRungeKuttaKernelImpl<IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(explicits)>(),
                explicits...);
        });
}
#endif

} // namespace Nektar::Operators::detail
