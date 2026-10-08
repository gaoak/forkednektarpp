///////////////////////////////////////////////////////////////////////////////
//
// File: magma.cpp
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
// Description: NekBlas backend for the Device execution space (MAGMA).
// Vendor headers are included here, never in NekBlas.hpp.
//
///////////////////////////////////////////////////////////////////////////////

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include <magma_v2.h>

#if defined(NEKTAR_ENABLE_CUDA)
#include <LibUtilities/Backends/CUDAStream.hpp>

#define CUBLAS_CHECK(condition)                                                \
    {                                                                          \
        const cublasStatus_t status = condition;                               \
        if (status != CUBLAS_STATUS_SUCCESS)                                   \
        {                                                                      \
            std::cerr << "cuBLAS error encountered: \""                        \
                      << cublasGetStatusString(status) << "\" at " << __FILE__ \
                      << ':' << __LINE__ << std::endl;                         \
            exit(0);                                                           \
        }                                                                      \
    }
#elif defined(NEKTAR_ENABLE_HIP)
#include <LibUtilities/Backends/HIPStream.hpp>

#define HIPBLAS_CHECK(condition)                                               \
    {                                                                          \
        const hipblasStatus_t status = condition;                              \
        if (status != HIPBLAS_STATUS_SUCCESS)                                  \
        {                                                                      \
            std::cerr << "hipBLAS error encountered: \""                       \
                      << hipblasStatusToString(status) << "\" at " << __FILE__ \
                      << ':' << __LINE__ << std::endl;                         \
            exit(0);                                                           \
        }                                                                      \
    }
#endif

namespace Nektar::NekBlas
{
namespace
{
magma_queue_t GetMagmaQueue(const unsigned int streamID)
{
    static std::unordered_map<unsigned int, magma_queue_t> handle;

    if (handle.find(streamID) == handle.end())
    {
        magma_queue_t magma_queue;
        int device_rank = 0;
#if defined(NEKTAR_ENABLE_CUDA)
        (void)cudaGetDevice(&device_rank);
        cudaStream_t stream = CUDAStream::GetInstance(streamID);
        cublasHandle_t cublas_handle;
        cusparseHandle_t cusparse_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        (void)cusparseCreate(&cusparse_handle);
        CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
        (void)cusparseSetStream(cusparse_handle, stream);
        magma_queue_create_from_cuda(device_rank, stream, cublas_handle,
                                     cusparse_handle, &magma_queue);
#elif defined(NEKTAR_ENABLE_HIP)
        (void)hipGetDevice(&device_rank);
        hipStream_t stream = HIPStream::GetInstance(streamID);
        hipblasHandle_t hipblas_handle;
        hipsparseHandle_t hipsparse_handle;
        HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
        (void)hipsparseCreate(&hipsparse_handle);
        HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
        (void)hipsparseSetStream(hipsparse_handle, stream);
        magma_queue_create_from_hip(device_rank, stream, hipblas_handle,
                                    hipsparse_handle, &magma_queue);
#endif
        handle[streamID] = magma_queue;
    }

    return handle[streamID];
}
} // namespace

template <typename TData>
void Gemm(Handle<NektarSpaces::Device> handle, std::string transposeA,
          std::string transposeB, const int M, const int N, const int K,
          const TData alpha, const TData *a, const int lda, const TData *b,
          const int ldb, const TData beta, TData *c, const int ldc)
{
    auto queue = GetMagmaQueue(handle.GetStreamID());

    auto transA = (transposeA == "N") ? MagmaNoTrans : MagmaTrans;
    auto transB = (transposeB == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magma_sgemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c,
                    ldc, queue);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magma_dgemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c,
                    ldc, queue);
    }
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
    auto queue = GetMagmaQueue(handle.GetStreamID());

    auto transA = (transposeA == "N") ? MagmaNoTrans : MagmaTrans;
    auto transB = (transposeB == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magmablas_sgemm_batched_strided(transA, transB, M, N, K, alpha, a, lda,
                                        strideA, b, ldb, strideB, beta, c, ldc,
                                        strideC, batchSize, queue);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magmablas_dgemm_batched_strided(transA, transB, M, N, K, alpha, a, lda,
                                        strideA, b, ldb, strideB, beta, c, ldc,
                                        strideC, batchSize, queue);
    }
}

template <typename TData>
void Gemv(Handle<NektarSpaces::Device> handle, std::string transpose,
          const int M, const int N, const TData alpha, const TData *a,
          const int lda, const TData *x, const int incx, const TData beta,
          TData *y, const int incy)
{
    auto queue = GetMagmaQueue(handle.GetStreamID());

    auto trans = (transpose == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magma_sgemv(trans, M, N, alpha, a, lda, x, incx, beta, y, incy, queue);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magma_dgemv(trans, M, N, alpha, a, lda, x, incx, beta, y, incy, queue);
    }
}

template <typename TData>
void GemvStridedBatched(Handle<NektarSpaces::Device> handle,
                        std::string transpose, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *x, const int incx,
                        const int strideX, const TData beta, TData *y,
                        const int incy, const int strideY, const int batchSize)
{
    auto queue = GetMagmaQueue(handle.GetStreamID());

    auto trans = (transpose == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magmablas_sgemv_batched_strided(trans, M, N, alpha, a, lda, strideA, x,
                                        incx, strideX, beta, y, incy, strideY,
                                        batchSize, queue);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magmablas_dgemv_batched_strided(trans, M, N, alpha, a, lda, strideA, x,
                                        incx, strideX, beta, y, incy, strideY,
                                        batchSize, queue);
    }
}

template <typename TData>
void GeamStridedBatched(
    [[maybe_unused]] Handle<NektarSpaces::Device> handle,
    [[maybe_unused]] std::string transposeA,
    [[maybe_unused]] std::string transposeB, [[maybe_unused]] const int M,
    [[maybe_unused]] const int N, [[maybe_unused]] const TData alpha,
    [[maybe_unused]] const TData *a, [[maybe_unused]] const int lda,
    [[maybe_unused]] const int strideA, [[maybe_unused]] const TData beta,
    [[maybe_unused]] const TData *b, [[maybe_unused]] const int ldb,
    [[maybe_unused]] const int strideB, [[maybe_unused]] TData *c,
    [[maybe_unused]] const int ldc, [[maybe_unused]] const int strideC,
    [[maybe_unused]] const int batchSize)
{
#if defined(NEKTAR_ENABLE_CUDA)
    throw std::runtime_error(
        "GeamStridedBatched is not available for MAGMA with CUDA");
#elif defined(NEKTAR_ENABLE_HIP)
    auto queue         = GetMagmaQueue(handle.GetStreamID());
    auto hipblasHandle = magma_queue_get_hipblas_handle(queue);
    auto transA        = (transposeA == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    auto transB        = (transposeB == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        HIPBLAS_CHECK(hipblasSgeamStridedBatched(
            hipblasHandle, transA, transB, M, N, &alpha, a, lda, strideA, &beta,
            b, ldb, strideB, c, ldc, strideC, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        HIPBLAS_CHECK(hipblasDgeamStridedBatched(
            hipblasHandle, transA, transB, M, N, &alpha, a, lda, strideA, &beta,
            b, ldb, strideB, c, ldc, strideC, batchSize));
    }
#endif
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
