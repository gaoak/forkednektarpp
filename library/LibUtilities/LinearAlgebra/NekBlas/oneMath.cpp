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
// Description: NekBlas backend for the Device execution space (oneMath/oneMKL).
// Vendor headers are included here, never in NekBlas.hpp.
//
///////////////////////////////////////////////////////////////////////////////

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

#include <LibUtilities/Backends/SYCLQueue.hpp>

#include <stdexcept>
#include <vector>

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
    // Set SYCL dependencies to reproduce CUDA/HIP default stream behavior.
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

template <typename TData>
void Gemm(Handle<NektarSpaces::Device> handle, std::string transposeA,
          std::string transposeB, const int M, const int N, const int K,
          const TData alpha, const TData *a, const int lda, const TData *b,
          const int ldb, const TData beta, TData *c, const int ldc)
{
    sycl::queue &Q = SYCLQueue::GetInstance(handle.GetStreamID());
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
                Gemm(Handle<NektarSpaces::Serial>::GetInstance(0),
                    transposeA, transposeB, M, N, K, alpha, a, lda, b, ldb,
                    beta, c, ldc);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename TData>
void GemmStridedBatched(Handle<NektarSpaces::Device> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const int K,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *b, const int ldb,
                        const int strideB, const TData beta, TData *c,
                        const int ldc, const int strideC, const int batchSize)
{
    sycl::queue &Q = SYCLQueue::GetInstance(handle.GetStreamID());
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
                GemmStridedBatched(
                    Handle<NektarSpaces::Serial>::GetInstance(0), transposeA,
                    transposeB, M, N, K, alpha, a, lda, strideA, b, ldb,
                    strideB, beta, c, ldc, strideC, batchSize);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename TData>
void Gemv(Handle<NektarSpaces::Device> handle, std::string transpose,
          const int M, const int N, const TData alpha, const TData *a,
          const int lda, const TData *x, const int incx, const TData beta,
          TData *y, const int incy)
{
    sycl::queue &Q = SYCLQueue::GetInstance(handle.GetStreamID());
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
                Gemv(Handle<NektarSpaces::Serial>::GetInstance(0),
                    transpose, M, N, alpha, a, lda, x, incx, beta, y, incy);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename TData>
void GemvStridedBatched(Handle<NektarSpaces::Device> handle,
                        std::string transpose, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *x, const int incx,
                        const int strideX, const TData beta, TData *y,
                        const int incy, const int strideY, const int batchSize)
{
    sycl::queue &Q = SYCLQueue::GetInstance(handle.GetStreamID());
#if defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL)
#if defined(NEKTAR_ENABLE_ONEMATH)
    throw std::runtime_error("gemv_batch not yet implemented in oneMath");
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
                GemvStridedBatched(
                    Handle<NektarSpaces::Serial>::GetInstance(0), transpose,
                    M, N, alpha, a, lda, strideA, x, incx, strideX, beta, y,
                    incy, strideY, batchSize);
            });
    });
    // clang-format on
#endif
    SYCLQueue::SetEvent(handle.GetStreamID(), e);
}

template <typename TData>
void GeamStridedBatched(Handle<NektarSpaces::Device>, std::string, std::string,
                        const int, const int, const TData, const TData *,
                        const int, const int, const TData, const TData *,
                        const int, const int, TData *, const int, const int,
                        const int)
{
    throw std::runtime_error("GeamStridedBatched is not available for oneMath");
}

template void Gemm<float>(Handle<NektarSpaces::Device> handle,
                          std::string transposeA, std::string transposeB,
                          const int M, const int N, const int K,
                          const float alpha, const float *a, const int lda,
                          const float *b, const int ldb, const float beta,
                          float *c, const int ldc);

template void Gemm<double>(Handle<NektarSpaces::Device> handle,
                           std::string transposeA, std::string transposeB,
                           const int M, const int N, const int K,
                           const double alpha, const double *a, const int lda,
                           const double *b, const int ldb, const double beta,
                           double *c, const int ldc);

template void GemmStridedBatched<float>(
    Handle<NektarSpaces::Device> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const int K,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *b, const int ldb, const int strideB, const float beta,
    float *c, const int ldc, const int strideC, const int batchSize);

template void GemmStridedBatched<double>(
    Handle<NektarSpaces::Device> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const int K,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *b, const int ldb, const int strideB, const double beta,
    double *c, const int ldc, const int strideC, const int batchSize);

template void Gemv<float>(Handle<NektarSpaces::Device> handle,
                          std::string transpose, const int M, const int N,
                          const float alpha, const float *a, const int lda,
                          const float *x, const int incx, const float beta,
                          float *y, const int incy);

template void Gemv<double>(Handle<NektarSpaces::Device> handle,
                           std::string transpose, const int M, const int N,
                           const double alpha, const double *a, const int lda,
                           const double *x, const int incx, const double beta,
                           double *y, const int incy);

template void GemvStridedBatched<float>(
    Handle<NektarSpaces::Device> handle, std::string transpose, const int M,
    const int N, const float alpha, const float *a, const int lda,
    const int strideA, const float *x, const int incx, const int strideX,
    const float beta, float *y, const int incy, const int strideY,
    const int batchSize);

template void GemvStridedBatched<double>(
    Handle<NektarSpaces::Device> handle, std::string transpose, const int M,
    const int N, const double alpha, const double *a, const int lda,
    const int strideA, const double *x, const int incx, const int strideX,
    const double beta, double *y, const int incy, const int strideY,
    const int batchSize);

template void GeamStridedBatched<float>(
    Handle<NektarSpaces::Device> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const float alpha,
    const float *a, const int lda, const int strideA, const float beta,
    const float *b, const int ldb, const int strideB, float *c, const int ldc,
    const int strideC, const int batchSize);

template void GeamStridedBatched<double>(
    Handle<NektarSpaces::Device> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const double alpha,
    const double *a, const int lda, const int strideA, const double beta,
    const double *b, const int ldb, const int strideB, double *c, const int ldc,
    const int strideC, const int batchSize);
} // namespace Nektar::NekBlas
