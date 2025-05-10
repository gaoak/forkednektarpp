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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include "Operators/Common/Spaces.hpp"
#include "Operators/NekBlas/NekBlas.hpp"

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const TData alpha,
    const TData *a, const size_t lda, const TData *b, const size_t ldb,
    const TData beta, TData *c, const size_t ldc)
{
    auto transA = (transposeA == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = (transposeB == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemm(handle, transA, transB, M, N, K, &alpha, a,
                                 lda, b, ldb, &beta, c, ldc));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemm(handle, transA, transB, M, N, K, &alpha, a,
                                 lda, b, ldb, &beta, c, ldc));
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const size_t M, const size_t N,
                      const size_t K, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *b,
                      const size_t ldb, const size_t strideB, const TData beta,
                      TData *c, const size_t ldc, const size_t strideC,
                      const size_t batchSize)
{
    auto transA = (transposeA == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = (transposeB == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle, transA, transB, M, N, K, &alpha, a, lda, strideA, b, ldb,
            strideB, &beta, c, ldc, strideC, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            handle, transA, transB, M, N, K, &alpha, a, lda, strideA, b, ldb,
            strideB, &beta, c, ldc, strideC, batchSize));
    }
}

template void NekGemm<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const float alpha,
    const float *a, const size_t lda, const float *b, const size_t ldb,
    const float beta, float *c, const size_t ldc);

template void NekGemm<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const double alpha,
    const double *a, const size_t lda, const double *b, const size_t ldb,
    const double beta, double *c, const size_t ldc);

template void NekGemmStridedBatched<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const float alpha,
    const float *a, const size_t lda, const size_t strideA, const float *b,
    const size_t ldb, const size_t strideB, const float beta, float *c,
    const size_t ldc, const size_t strideC, const size_t batchSize);

template void NekGemmStridedBatched<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const double alpha,
    const double *a, const size_t lda, const size_t strideA, const double *b,
    const size_t ldb, const size_t strideB, const double beta, double *c,
    const size_t ldc, const size_t strideC, const size_t batchSize);
