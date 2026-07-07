///////////////////////////////////////////////////////////////////////////////
//
// File: blas.cpp
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

#include "Operators/NekBlas/NekBlas.hpp"

#include <LibUtilities/LinearAlgebra/Blas.hpp>

void NekBlasSetStream([[maybe_unused]] blasHandle_t handle,
                      [[maybe_unused]] const unsigned int streamID)
{
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type NekGemm(
    [[maybe_unused]] THandle handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const int K,
    const TData alpha, const TData *a, const int lda, const TData *b,
    const int ldb, const TData beta, TData *c, const int ldc)
{
    auto transA = *transposeA.c_str();
    auto transB = *transposeB.c_str();

    Blas::Gemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type
NekGemmStridedBatched([[maybe_unused]] THandle handle, std::string transposeA,
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize)
{
    auto transA = *transposeA.c_str();
    auto transB = *transposeB.c_str();

    for (int i = 0; i < batchSize; i++)
    {
        Blas::Gemm(transA, transB, M, N, K, alpha, a + strideA * i, lda,
                   b + strideB * i, ldb, beta, c + strideC * i, ldc);
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type NekGemv(
    [[maybe_unused]] THandle handle, std::string transpose, const int M,
    const int N, const TData alpha, const TData *a, const int lda,
    const TData *x, const int incx, const TData beta, TData *y, const int incy)
{
    auto trans = *transpose.c_str();

    Blas::Gemv(trans, M, N, alpha, a, lda, x, incx, beta, y, incy);
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type
NekGemvStridedBatched([[maybe_unused]] THandle handle, std::string transpose,
                      const int M, const int N, const TData alpha,
                      const TData *a, const int lda, const int strideA,
                      const TData *x, const int incx, const int strideX,
                      const TData beta, TData *y, const int incy,
                      const int strideY, const int batchSize)
{
    auto trans = *transpose.c_str();

    for (int i = 0; i < batchSize; i++)
    {
        Blas::Gemv(trans, M, N, alpha, a + strideA * i, lda, x + strideX * i,
                   incx, beta, y + strideY * i, incy);
    }
}

template void NekGemm<blasHandle_t, float>(
    blasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const float *b, const int ldb, const float beta, float *c,
    const int ldc);

template void NekGemm<blasHandle_t, double>(
    blasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const double *b, const int ldb, const double beta, double *c,
    const int ldc);

template void NekGemmStridedBatched<blasHandle_t, float>(
    blasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const int strideA, const float *b, const int ldb,
    const int strideB, const float beta, float *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmStridedBatched<blasHandle_t, double>(
    blasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const int strideA, const double *b, const int ldb,
    const int strideB, const double beta, double *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemv<blasHandle_t, float>(
    blasHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const float *x,
    const int incx, const float beta, float *y, const int incy);

template void NekGemv<blasHandle_t, double>(
    blasHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const double *x,
    const int incx, const double beta, double *y, const int incy);

template void NekGemvStridedBatched<blasHandle_t, float>(
    blasHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *x, const int incx, const int strideX, const float beta,
    float *y, const int incy, const int strideY, const int batchSize);

template void NekGemvStridedBatched<blasHandle_t, double>(
    blasHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *x, const int incx, const int strideX, const double beta,
    double *y, const int incy, const int strideY, const int batchSize);
