///////////////////////////////////////////////////////////////////////////////
//
// File: hipBlas.cpp
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

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"
#include "Operators/NekBlas/NekBlas.hpp"

void NekBlasSetStream(hipblasHandle_t handle, const unsigned int streamID)
{
    HIPBLAS_CHECK(hipblasSetStream(handle, HIPStream::GetInstance(streamID)));
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
NekGemm(THandle handle, std::string transposeA, std::string transposeB,
        const int M, const int N, const int K, const TData alpha,
        const TData *a, const int lda, const TData *b, const int ldb,
        const TData beta, TData *c, const int ldc)
{
    auto transA = (transposeA == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    auto transB = (transposeB == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        HIPBLAS_CHECK(hipblasSgemm(handle, transA, transB, M, N, K, &alpha, a,
                                   lda, b, ldb, &beta, c, ldc));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        HIPBLAS_CHECK(hipblasDgemm(handle, transA, transB, M, N, K, &alpha, a,
                                   lda, b, ldb, &beta, c, ldc));
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize)
{
    auto transA = (transposeA == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    auto transB = (transposeB == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        HIPBLAS_CHECK(hipblasSgemmStridedBatched(
            handle, transA, transB, M, N, K, &alpha, a, lda, strideA, b, ldb,
            strideB, &beta, c, ldc, strideC, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        HIPBLAS_CHECK(hipblasDgemmStridedBatched(
            handle, transA, transB, M, N, K, &alpha, a, lda, strideA, b, ldb,
            strideB, &beta, c, ldc, strideC, batchSize));
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
NekGemv(THandle handle, std::string transpose, const int M, const int N,
        const TData alpha, const TData *a, const int lda, const TData *x,
        const int incx, const TData beta, TData *y, const int incy)
{
    auto trans = (transpose == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        HIPBLAS_CHECK(hipblasSgemv(handle, trans, M, N, &alpha, a, lda, x, incx,
                                   &beta, y, incy));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        HIPBLAS_CHECK(hipblasDgemv(handle, trans, M, N, &alpha, a, lda, x, incx,
                                   &beta, y, incy));
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
NekGemvStridedBatched(THandle handle, std::string transpose, const int M,
                      const int N, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *x,
                      const int incx, const int strideX, const TData beta,
                      TData *y, const int incy, const int strideY,
                      const int batchSize)
{
    auto trans = (transpose == "N") ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        HIPBLAS_CHECK(hipblasSgemvStridedBatched(
            handle, trans, M, N, &alpha, a, lda, strideA, x, incx, strideX,
            &beta, y, incy, strideY, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        HIPBLAS_CHECK(hipblasDgemvStridedBatched(
            handle, trans, M, N, &alpha, a, lda, strideA, x, incx, strideX,
            &beta, y, incy, strideY, batchSize));
    }
}

template void NekGemm<hipblasHandle_t, float>(
    hipblasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const float *b, const int ldb, const float beta, float *c,
    const int ldc);

template void NekGemm<hipblasHandle_t, double>(
    hipblasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const double *b, const int ldb, const double beta, double *c,
    const int ldc);

template void NekGemmStridedBatched<hipblasHandle_t, float>(
    hipblasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const int strideA, const float *b, const int ldb,
    const int strideB, const float beta, float *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmStridedBatched<hipblasHandle_t, double>(
    hipblasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const int strideA, const double *b, const int ldb,
    const int strideB, const double beta, double *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemv<hipblasHandle_t, float>(
    hipblasHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const float *x,
    const int incx, const float beta, float *y, const int incy);

template void NekGemv<hipblasHandle_t, double>(
    hipblasHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const double *x,
    const int incx, const double beta, double *y, const int incy);

template void NekGemvStridedBatched<hipblasHandle_t, float>(
    hipblasHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *x, const int incx, const int strideX, const float beta,
    float *y, const int incy, const int strideY, const int batchSize);

template void NekGemvStridedBatched<hipblasHandle_t, double>(
    hipblasHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *x, const int incx, const int strideX, const double beta,
    double *y, const int incy, const int strideY, const int batchSize);
