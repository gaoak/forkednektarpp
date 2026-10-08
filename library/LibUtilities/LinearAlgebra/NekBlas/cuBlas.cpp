///////////////////////////////////////////////////////////////////////////////
//
// File: cuBlas.cpp
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
// Description: NekBlas backend for the Device execution space (cuBLAS).
// Vendor headers are included here, never in NekBlas.hpp.
//
///////////////////////////////////////////////////////////////////////////////

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

#include <LibUtilities/Backends/CUDAStream.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <cublas_v2.h>

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

namespace Nektar::NekBlas
{
namespace
{
cublasHandle_t GetCuBlasHandle(const unsigned int streamID)
{
    static cublasHandle_t handle = nullptr;

    if (!handle)
    {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        {
            printf("cuBLAS initialization failed\n");
        }
    }

    CUBLAS_CHECK(cublasSetStream(handle, CUDAStream::GetInstance(streamID)));

    return handle;
}
} // namespace

template <typename TData>
void Gemm(Handle<NektarSpaces::Device> handle, std::string transposeA,
          std::string transposeB, const int M, const int N, const int K,
          const TData alpha, const TData *a, const int lda, const TData *b,
          const int ldb, const TData beta, TData *c, const int ldc)
{
    auto cublasHandle = GetCuBlasHandle(handle.GetStreamID());

    auto transA = (transposeA == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = (transposeB == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemm(cublasHandle, transA, transB, M, N, K, &alpha,
                                 a, lda, b, ldb, &beta, c, ldc));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemm(cublasHandle, transA, transB, M, N, K, &alpha,
                                 a, lda, b, ldb, &beta, c, ldc));
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
    auto cublasHandle = GetCuBlasHandle(handle.GetStreamID());

    auto transA = (transposeA == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = (transposeB == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublasHandle, transA, transB, M, N, K, &alpha, a, lda, strideA, b,
            ldb, strideB, &beta, c, ldc, strideC, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            cublasHandle, transA, transB, M, N, K, &alpha, a, lda, strideA, b,
            ldb, strideB, &beta, c, ldc, strideC, batchSize));
    }
}

template <typename TData>
void Gemv(Handle<NektarSpaces::Device> handle, std::string transpose,
          const int M, const int N, const TData alpha, const TData *a,
          const int lda, const TData *x, const int incx, const TData beta,
          TData *y, const int incy)
{
    auto cublasHandle = GetCuBlasHandle(handle.GetStreamID());

    auto trans = (transpose == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemv(cublasHandle, trans, M, N, &alpha, a, lda, x,
                                 incx, &beta, y, incy));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemv(cublasHandle, trans, M, N, &alpha, a, lda, x,
                                 incx, &beta, y, incy));
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
    auto cublasHandle = GetCuBlasHandle(handle.GetStreamID());

    auto trans = (transpose == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemvStridedBatched(
            cublasHandle, trans, M, N, &alpha, a, lda, strideA, x, incx,
            strideX, &beta, y, incy, strideY, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemvStridedBatched(
            cublasHandle, trans, M, N, &alpha, a, lda, strideA, x, incx,
            strideX, &beta, y, incy, strideY, batchSize));
    }
}

template <typename TData>
void GeamStridedBatched(Handle<NektarSpaces::Device>, std::string, std::string,
                        const int, const int, const TData, const TData *,
                        const int, const int, const TData, const TData *,
                        const int, const int, TData *, const int, const int,
                        const int)
{
    throw std::runtime_error("GeamStridedBatched is not available for cuBLAS");
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
