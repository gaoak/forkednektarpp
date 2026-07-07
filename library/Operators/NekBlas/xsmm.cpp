///////////////////////////////////////////////////////////////////////////////
//
// File: xsmm.cpp
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

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"
#include "libxsmm.h"

void NekBlasSetStream([[maybe_unused]] xsmmHandle_t handle,
                      [[maybe_unused]] const unsigned int streamID)
{
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type NekGemm(
    [[maybe_unused]] THandle handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const int K,
    const TData alpha, const TData *a, const int lda, const TData *b,
    const int ldb, const TData beta, TData *c, const int ldc)
{
    ASSERTL0(transposeA == "N" && transposeB == "N",
             "libxsmm: matrix tranpose is not supported");
    ASSERTL0(alpha == 1.0, "libxsmm: alpha must be equal to 1.0");
    ASSERTL0(beta == 0.0 || beta == 1.0,
             "libxsmm: beta must be equal to 0.0 or 1.0");
    if (alpha != 1.0)
    {
        ASSERTL0(beta == 0.0,
                 "libxsmm: beta must be equal to 0.0 when alpha != 1.0");
    }

    // Dispatch kernel.
    int lda0 = static_cast<int>(lda);
    int ldb0 = static_cast<int>(ldb);
    int ldc0 = static_cast<int>(ldc);
    libxsmm_gemm(nullptr, nullptr, M, N, K, &alpha, a, &lda0, b, &ldb0, &beta,
                 c, &ldc0);
    if (alpha != 1.0 && beta == 0.0)
    {
        for (int i = 0; i < M * N; i++)
        {
            c[i] *= alpha;
        }
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
NekGemmStridedBatched([[maybe_unused]] THandle handle, std::string transposeA,
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize)
{
    ASSERTL0(transposeA == "N" && transposeB == "N",
             "libxsmm: matrix tranpose is not supported");
    ASSERTL0(alpha == 1.0, "libxsmm: alpha must be equal to 1.0");
    ASSERTL0(beta == 0.0 || beta == 1.0,
             "libxsmm: beta must be equal to 0.0 or 1.0");
    if (alpha != 1.0)
    {
        ASSERTL0(beta == 0.0,
                 "libxsmm: beta must be equal to 0.0 when alpha != 1.0");
    }

    // Dispatch kernel.
    int lda0 = static_cast<int>(lda);
    int ldb0 = static_cast<int>(ldb);
    int ldc0 = static_cast<int>(ldc);
    for (int i = 0; i < batchSize; i++)
    {
        libxsmm_gemm(nullptr, nullptr, M, N, K, &alpha, a + strideA * i, &lda0,
                     b + strideB * i, &ldb0, &beta, c + strideC * i, &ldc0);
        if (alpha != 1.0 && beta == 0.0)
        {
            for (int j = 0; j < M * N; j++)
            {
                c[strideC * i + j] *= alpha;
            }
        }
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type NekGemv(
    [[maybe_unused]] THandle, std::string transpose, const int M, const int N,
    const TData alpha, const TData *A, const int lda, const TData *x,
    const int incx, const TData beta, TData *y, const int incy)
{
    ASSERTL0(transpose == "N", "libxsmm: transpose not supported in Gemv");

    // Gemv is just Gemm with a single column vector
    int lda0  = lda;
    int incx0 = incx;
    int incy0 = incy;
    int n     = 1;
    libxsmm_gemm(nullptr, nullptr, M, n, N, &alpha, A, &lda0, x, &incx0, &beta,
                 y, &incy0);
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
NekGemvStridedBatched([[maybe_unused]] THandle handle, std::string transpose,
                      const int M, const int N, const TData alpha,
                      const TData *a, const int lda, const int strideA,
                      const TData *x, const int incx, const int strideX,
                      const TData beta, TData *y, const int incy,
                      const int strideY, const int batchSize)
{
    ASSERTL0(transpose == "N", "libxsmm: transpose not supported in Gemv");

    for (int i = 0; i < batchSize; i++)
    {
        // Gemv is just Gemm with a single column vector
        int lda0  = lda;
        int incx0 = incx;
        int incy0 = incy;
        int n     = 1;
        libxsmm_gemm(nullptr, nullptr, M, n, N, &alpha, a + strideA * i, &lda0,
                     x + strideX * i, &incx0, &beta, y + strideY * i, &incy0);
    }
}

template void NekGemm<xsmmHandle_t, float>(
    xsmmHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const float *b, const int ldb, const float beta, float *c,
    const int ldc);

template void NekGemm<xsmmHandle_t, double>(
    xsmmHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const double *b, const int ldb, const double beta, double *c,
    const int ldc);

template void NekGemmStridedBatched<xsmmHandle_t, float>(
    xsmmHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const int strideA, const float *b, const int ldb,
    const int strideB, const float beta, float *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmStridedBatched<xsmmHandle_t, double>(
    xsmmHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const int strideA, const double *b, const int ldb,
    const int strideB, const double beta, double *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemv<xsmmHandle_t, float>(xsmmHandle_t, std::string, int, int,
                                           float, const float *, int,
                                           const float *, int, float, float *,
                                           int);

template void NekGemv<xsmmHandle_t, double>(xsmmHandle_t, std::string, int, int,
                                            double, const double *, int,
                                            const double *, int, double,
                                            double *, int);

template void NekGemvStridedBatched<xsmmHandle_t, float>(
    xsmmHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *x, const int incx, const int strideX, const float beta,
    float *y, const int incy, const int strideY, const int batchSize);

template void NekGemvStridedBatched<xsmmHandle_t, double>(
    xsmmHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *x, const int incx, const int strideX, const double beta,
    double *y, const int incy, const int strideY, const int batchSize);
