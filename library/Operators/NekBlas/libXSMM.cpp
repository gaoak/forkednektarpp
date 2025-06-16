///////////////////////////////////////////////////////////////////////////////
//
// File: libXSMM.cpp
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

#include "Operators/Common/Spaces.hpp"
#include "Operators/NekBlas/NekBlas.hpp"

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"
#include "libxsmm.h"

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, xsmmHandle>, void>::type NekGemm(
    [[maybe_unused]] THandle handle, std::string transposeA,
    std::string transposeB, const size_t M, const size_t N, const size_t K,
    const TData alpha, const TData *a, const size_t lda, const TData *b,
    const size_t ldb, const TData beta, TData *c, const size_t ldc)
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
        for (size_t i = 0; i < M * N; i++)
        {
            c[i] *= alpha;
        }
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, xsmmHandle>, void>::type
NekGemmStridedBatched([[maybe_unused]] THandle handle, std::string transposeA,
                      std::string transposeB, const size_t M, const size_t N,
                      const size_t K, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *b,
                      const size_t ldb, const size_t strideB, const TData beta,
                      TData *c, const size_t ldc, const size_t strideC,
                      const size_t batchSize)
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
    for (size_t i = 0; i < batchSize; i++)
    {
        libxsmm_gemm(nullptr, nullptr, M, N, K, &alpha, a + strideA * i, &lda0,
                     b + strideB * i, &ldb0, &beta, c + strideC * i, &ldc0);
        if (alpha != 1.0 && beta == 0.0)
        {
            for (size_t j = 0; j < M * N; j++)
            {
                c[strideC * i + j] *= alpha;
            }
        }
    }
}

template void NekGemm<xsmmHandle, float>(
    xsmmHandle handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const float alpha,
    const float *a, const size_t lda, const float *b, const size_t ldb,
    const float beta, float *c, const size_t ldc);

template void NekGemm<xsmmHandle, double>(
    xsmmHandle handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const double alpha,
    const double *a, const size_t lda, const double *b, const size_t ldb,
    const double beta, double *c, const size_t ldc);

template void NekGemmStridedBatched<xsmmHandle, float>(
    xsmmHandle handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const float alpha,
    const float *a, const size_t lda, const size_t strideA, const float *b,
    const size_t ldb, const size_t strideB, const float beta, float *c,
    const size_t ldc, const size_t strideC, const size_t batchSize);

template void NekGemmStridedBatched<xsmmHandle, double>(
    xsmmHandle handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const double alpha,
    const double *a, const size_t lda, const size_t strideA, const double *b,
    const size_t ldb, const size_t strideB, const double beta, double *c,
    const size_t ldc, const size_t strideC, const size_t batchSize);
