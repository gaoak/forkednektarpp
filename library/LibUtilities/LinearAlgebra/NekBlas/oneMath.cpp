///////////////////////////////////////////////////////////////////////////////
//
// File: oneMath.cpp
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

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

#if defined(NEKTAR_ENABLE_ONEMATH)
#include "oneapi/math.hpp"
using namespace oneapi::math;
#elif defined(NEKTAR_ENABLE_ONEMKL)
#include "oneapi/mkl.hpp"
using namespace oneapi::mkl;
#endif

namespace Nektar::NekBlas
{
[[maybe_unused]] static std::vector<
    sycl::event> inline setOneMathExecutionDependency(const unsigned int
                                                          streamID)
{
    // Set SYCL depedencies to reproduce CUDA/HIP default stream behavior.
    std::vector<sycl::event> dependencies;
    if (streamID == 0)
    {
        // Tasks in the "default" queue depend on all "non-default" queue.
        for (auto &item : SYCLQueue::GetAllEvents())
        {
            if (item.first != 0)
            {
                dependencies.push_back(item.second);
            }
        }
    }
    else
    {
        // Tasks in "non-default" queue depend on the "default" queue.
        dependencies.push_back(SYCLQueue::GetEvent(0));
    }

    return dependencies;
}

template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>>
void Gemm([[maybe_unused]] THandle handle, std::string transposeA,
          std::string transposeB, const std::int64_t M, const std::int64_t N,
          const std::int64_t K, const TData alpha, const TData *a,
          const std::int64_t lda, const TData *b, const std::int64_t ldb,
          const TData beta, TData *c, const std::int64_t ldc)
{
    sycl::queue &Q = handle.GetQueue();
#if defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL)
    auto transA   = (transposeA == "N") ? transpose::N : transpose::T;
    auto transB   = (transposeB == "N") ? transpose::N : transpose::T;
    sycl::event e = blas::column_major::gemm(
        Q, transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc,
        setOneMathExecutionDependency(handle.GetStreamID()));
#else // SYCL-CPU
    // clang-format off
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(handle.GetStreamID(), cgh);
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(
#else
        cgh.host_task(
#endif
            [=]([[maybe_unused]] sycl::interop_handle ih) {
                Gemm(blasHandle_t(), transposeA, transposeB, M, N, K, alpha, 
                        a, lda, b, ldb, beta, c, ldc);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>>
void GemmStridedBatched([[maybe_unused]] THandle handle, std::string transposeA,
                        std::string transposeB, const std::int64_t M,
                        const std::int64_t N, const std::int64_t K,
                        const TData alpha, const TData *a,
                        const std::int64_t lda, const std::int64_t strideA,
                        const TData *b, const std::int64_t ldb,
                        const std::int64_t strideB, const TData beta, TData *c,
                        const std::int64_t ldc, const std::int64_t strideC,
                        const std::int64_t batchSize)
{
    sycl::queue &Q = handle.GetQueue();
#if defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL)
    auto transA   = (transposeA == "N") ? transpose::N : transpose::T;
    auto transB   = (transposeB == "N") ? transpose::N : transpose::T;
    sycl::event e = blas::column_major::gemm_batch(
        Q, transA, transB, M, N, K, alpha, a, lda, strideA, b, ldb, strideB,
        beta, c, ldc, strideC, batchSize,
        setOneMathExecutionDependency(handle.GetStreamID()));
#else // SYCL-CPU
    // clang-format off
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(handle.GetStreamID(), cgh);
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(
#else
        cgh.host_task(
#endif
            [=]([[maybe_unused]] sycl::interop_handle ih) {
                GemmStridedBatched(blasHandle_t(), transposeA, transposeB, 
                    M, N, K, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
                    ldc, strideC, batchSize);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>>
void Gemv([[maybe_unused]] THandle handle, std::string transpose,
          const std::int64_t M, const std::int64_t N, const TData alpha,
          const TData *a, const std::int64_t lda, const TData *x,
          const std::int64_t incx, const TData beta, TData *y,
          const std::int64_t incy)
{
    sycl::queue &Q = handle.GetQueue();
#if defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL)
    auto trans    = (transpose == "N") ? transpose::N : transpose::T;
    sycl::event e = blas::column_major::gemv(
        Q, trans, M, N, alpha, a, lda, x, incx, beta, y, incy,
        setOneMathExecutionDependency(handle.GetStreamID()));
#else // SYCL-CPU
    // clang-format off
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(handle.GetStreamID(), cgh);
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(
#else
        cgh.host_task(
#endif
            [=]([[maybe_unused]] sycl::interop_handle ih) {
                Gemv(blasHandle_t(), transpose, M, N, alpha, a, lda, x, incx,
                        beta, y, incy);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>>
void GemvStridedBatched([[maybe_unused]] THandle handle, std::string transpose,
                        const std::int64_t M, const std::int64_t N,
                        const TData alpha, const TData *a,
                        const std::int64_t lda, const std::int64_t strideA,
                        const TData *x, const std::int64_t incx,
                        const std::int64_t strideX, const TData beta, TData *y,
                        const std::int64_t incy, const std::int64_t strideY,
                        const std::int64_t batchSize)
{
    sycl::queue &Q = handle.GetQueue();
#if defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL)
#if defined(NEKTAR_ENABLE_ONEMATH)
    ASSERTL0(false, "gemv_batch not yet implemented in oneMath")
#endif
    auto trans    = (transpose == "N") ? transpose::N : transpose::T;
    sycl::event e = blas::column_major::gemv_batch(
        Q, trans, M, N, alpha, a, lda, strideA, x, incx, strideX, beta, y, incy,
        strideY, batchSize,
        setOneMathExecutionDependency(handle.GetStreamID()));
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
#else // SYCL-CPU
    // clang-format off
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(handle.GetStreamID(), cgh);
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(
#else
        cgh.host_task(
#endif
            [=]([[maybe_unused]] sycl::interop_handle ih) {
                GemvStridedBatched(blasHandle_t(), transpose, M, N, alpha,
                    a, lda, strideA, x, incx, strideX, beta, y, incy, strideY,
                    batchSize);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template void Gemm<oneMathHandle_t, float>(
    oneMathHandle_t handle, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const float alpha, const float *a, const std::int64_t lda, const float *b,
    const std::int64_t ldb, const float beta, float *c, const std::int64_t ldc);

template void Gemm<oneMathHandle_t, double>(
    oneMathHandle_t handle, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const double alpha, const double *a, const std::int64_t lda,
    const double *b, const std::int64_t ldb, const double beta, double *c,
    const std::int64_t ldc);

template void GemmStridedBatched<oneMathHandle_t, float>(
    oneMathHandle_t handle, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const float alpha, const float *a, const std::int64_t lda,
    const std::int64_t strideA, const float *b, const std::int64_t ldb,
    const std::int64_t strideB, const float beta, float *c,
    const std::int64_t ldc, const std::int64_t strideC,
    const std::int64_t batchSize);

template void GemmStridedBatched<oneMathHandle_t, double>(
    oneMathHandle_t handle, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const double alpha, const double *a, const std::int64_t lda,
    const std::int64_t strideA, const double *b, const std::int64_t ldb,
    const std::int64_t strideB, const double beta, double *c,
    const std::int64_t ldc, const std::int64_t strideC,
    const std::int64_t batchSize);

template void Gemv<oneMathHandle_t, float>(
    oneMathHandle_t handle, std::string transpose, const std::int64_t M,
    const std::int64_t N, const float alpha, const float *a,
    const std::int64_t lda, const float *x, const std::int64_t incx,
    const float beta, float *y, const std::int64_t incy);

template void Gemv<oneMathHandle_t, double>(
    oneMathHandle_t handle, std::string transpose, const std::int64_t M,
    const std::int64_t N, const double alpha, const double *a,
    const std::int64_t lda, const double *x, const std::int64_t incx,
    const double beta, double *y, const std::int64_t incy);

template void GemvStridedBatched<oneMathHandle_t, float>(
    oneMathHandle_t handle, std::string transpose, const std::int64_t M,
    const std::int64_t N, const float alpha, const float *a,
    const std::int64_t lda, const std::int64_t strideA, const float *x,
    const std::int64_t incx, const std::int64_t strideX, const float beta,
    float *y, const std::int64_t incy, const std::int64_t strideY,
    const std::int64_t batchSize);

template void GemvStridedBatched<oneMathHandle_t, double>(
    oneMathHandle_t handle, std::string transpose, const std::int64_t M,
    const std::int64_t N, const double alpha, const double *a,
    const std::int64_t lda, const std::int64_t strideA, const double *x,
    const std::int64_t incx, const std::int64_t strideX, const double beta,
    double *y, const std::int64_t incy, const std::int64_t strideY,
    const std::int64_t batchSize);
} // namespace Nektar::NekBlas
