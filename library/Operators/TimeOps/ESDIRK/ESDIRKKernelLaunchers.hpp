///////////////////////////////////////////////////////////////////////////////
//
// File: ESDIRKKernelLaunchers.hpp
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
NEK_DEVICE_INLINE static void StageSolutionESDIRKKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    constexpr TData ConstSqrt2 = 1.414213562373095;

    constexpr unsigned int stage    = sizeof...(implicits);
    constexpr unsigned int indStart = (stage * (stage - 1)) / 2;

    // 2nd order
    if constexpr (IntOrder == 2)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.

        constexpr TData lambda = (2.0 - ConstSqrt2) / 2.0;

        // clang-format off
        constexpr TData coeff[] =
                {   // 1st stage
                    lambda,
                    // 2nd stage
                    (-1.0 + 6.0 * lambda - 4 * lambda * lambda) / (4 * lambda),
                    ( 1.0 - 2.0 * lambda) / (4 * lambda)
                };
        // clang-format on

        inout[idx] =
            solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.

        // clang-format off
        constexpr TData coeff[] =
                {   // 1st stage
                    9.0 / 40.0,
                    // 2n stage
                    9.0 * (1.0 + ConstSqrt2) / 80.0,
                    9.0 * (1.0 + ConstSqrt2) / 80.0,
                    // 3rd stage
                    (22.0 + 15.0 * ConstSqrt2) / (80.0 * (1.0 + ConstSqrt2)),
                    (22.0 + 15.0 * ConstSqrt2) / (80.0 * (1.0 + ConstSqrt2)),
                    -7.0 / (40.0 * (1.0 + ConstSqrt2)),
                    // 4th stage
                    (2398.0 + 1205.0 * ConstSqrt2) / (2835.0 * (4.0 + 3.0 *
    ConstSqrt2)), (2398.0 + 1205.0 * ConstSqrt2) / (2835.0 * (4.0 + 3.0 *
    ConstSqrt2)), -2374.0 * (1.0 + 2.0 * ConstSqrt2) / (2835.0 * (5.0 + 3.0 *
    ConstSqrt2)), 5827.0 / 7560.0
                };
        // clang-format on

        inout[idx] =
            solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        constexpr TData coeff[] =
                {  // 1st stage
                   0.25,
                   // 2nd stage
                   (1.0 - ConstSqrt2) / 8.0, (1.0 - ConstSqrt2) / 8.0,
                   // 3rd stage
                   (5.0 - 7.0 * ConstSqrt2) / 64.0,
                   (5.0 - 7.0 * ConstSqrt2) / 64.0,
                   7.0 * (1.0 + ConstSqrt2) / 32.0,
                   // 4th stage
                   (-13796.0 - 54539.0 * ConstSqrt2) / 125000.0,
                   (-13796.0 - 54539.0 * ConstSqrt2) / 125000.0,
                   (506605.0 + 132109.0 * ConstSqrt2) / 437500.0,
                   166.0 * (-97.0 + 376.0 * ConstSqrt2) / 109375.0,
                   // 5th stage
                   (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
                   (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
                   47.0 * (-267.0 + 1783.0 * ConstSqrt2) / 273343.0,
                  -16.0 * (-22922.0 + 3525.0 * ConstSqrt2) / 571953.0,
                  -15625.0 * (97.0 + 376.0 * ConstSqrt2) / 90749876.0};
        // clang-format on

        inout[idx] =
            solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
    }
}

template <unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static void UpdateESDIRKKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    constexpr TData ConstSqrt2 = 1.414213562373095;

    // 2nd order
    if constexpr (IntOrder == 2)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        constexpr TData lambda = (2.0 - ConstSqrt2) / 2.0;

        constexpr TData coeff[] = {
                    (-1.0 + 6.0 * lambda - 4 * lambda * lambda) / (4 * lambda),
                    ( 1.0 - 2.0 * lambda) / (4 * lambda), lambda};

        inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        constexpr TData coeff[] = {(2398.0 + 1205.0 * ConstSqrt2) /
                                       (2835.0 * (4.0 + 3.0 * ConstSqrt2)),
                                   (2398.0 + 1205.0 * ConstSqrt2) /
                                       (2835.0 * (4.0 + 3.0 * ConstSqrt2)),
                                   -2374.0 * (1.0 + 2.0 * ConstSqrt2) /
                                       (2835.0 * (5.0 + 3.0 * ConstSqrt2)),
                                   5827.0 / 7560.0, 9.0 / 40.0};

        inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        constexpr TData coeff[] = {
            (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
            (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
            47.0 * (-267.0 + 1783.0 * ConstSqrt2) / 273343.0,
            -16.0 * (-22922.0 + 3525.0 * ConstSqrt2) / 571953.0,
            -15625.0 * (97.0 + 376.0 * ConstSqrt2) / 90749876.0,
            0.25};

        inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
    }
}

// Kernel Launchers.
#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
// Currently, argument pack can't be captured in a device lambda. Explicit
// kernel must be used (instead of nektar::parallel_for)
template <unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void StageSolutionESDIRKKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TData *__restrict solution, const TDatas *__restrict... implicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        StageSolutionESDIRKKernelImpl<IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(implicits)>(),
            implicits...);
    }
}

template <unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void UpdateESDIRKKernelLauncher(const size_t nsize,
                                         TData *__restrict inout,
                                         const TData *__restrict solution,
                                         const TDatas *__restrict... implicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateESDIRKKernelImpl<IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(implicits)>(),
            implicits...);
    }
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void StageSolutionESDIRKKernel(const size_t nsize,
                                                     TData *inout,
                                                     const TData *solution,
                                                     const TDatas *...implicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    StageSolutionESDIRKKernelLauncher<IntOrder>
        <<<gridSize, blockSize>>>(nsize, inout, solution, implicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateESDIRKKernel(const size_t nsize, TData *inout,
                                              const TData *solution,
                                              const TDatas *...implicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    UpdateESDIRKKernelLauncher<IntOrder>
        <<<gridSize, blockSize>>>(nsize, inout, solution, implicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}
#else
template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void StageSolutionESDIRKKernel(const size_t nsize,
                                                     TData *inout,
                                                     const TData *solution,
                                                     const TDatas *...implicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            StageSolutionESDIRKKernelImpl<IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(implicits)>(),
                implicits...);
        });
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateESDIRKKernel(const size_t nsize, TData *inout,
                                              const TData *solution,
                                              const TDatas *...implicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateESDIRKKernelImpl<IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(implicits)>(),
                implicits...);
        });
}
#endif

} // namespace Nektar::Operators::detail
