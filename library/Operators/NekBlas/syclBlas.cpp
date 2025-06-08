///////////////////////////////////////////////////////////////////////////////
//
// File: syclBlas.cpp
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

#if __has_include("oneapi/math.hpp")
#include "oneapi/math.hpp"
using namespace oneapi::math;
#elif __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
using namespace oneapi::mkl;
#endif

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type NekGemm(
    THandle queue, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const TData alpha,
    const TData *a, const size_t lda, const TData *b, const size_t ldb,
    const TData beta, TData *c, const size_t ldc)
{
    auto transA = (transposeA == "N") ? transpose::N : transpose::T;
    auto transB = (transposeB == "N") ? transpose::N : transpose::T;
    blas::column_major::gemm(queue, transA, transB, M, N, K, alpha, a, lda, b,
                             ldb, beta, c, ldc);
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmStridedBatched(THandle queue, std::string transposeA,
                      std::string transposeB, const size_t M, const size_t N,
                      const size_t K, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *b,
                      const size_t ldb, const size_t strideB, const TData beta,
                      TData *c, const size_t ldc, const size_t strideC,
                      const size_t batchSize)
{
    auto transA = (transposeA == "N") ? transpose::N : transpose::T;
    auto transB = (transposeB == "N") ? transpose::N : transpose::T;
    blas::column_major::gemm_batch(queue, transA, transB, M, N, K, alpha, a,
                                   lda, strideA, b, ldb, strideB, beta, c, ldc,
                                   strideC, batchSize);
}

template void NekGemm<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const float alpha,
    const float *a, const size_t lda, const float *b, const size_t ldb,
    const float beta, float *c, const size_t ldc);

template void NekGemm<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const double alpha,
    const double *a, const size_t lda, const double *b, const size_t ldb,
    const double beta, double *c, const size_t ldc);

template void NekGemmStridedBatched<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const float alpha,
    const float *a, const size_t lda, const size_t strideA, const float *b,
    const size_t ldb, const size_t strideB, const float beta, float *c,
    const size_t ldc, const size_t strideC, const size_t batchSize);

template void NekGemmStridedBatched<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const double alpha,
    const double *a, const size_t lda, const size_t strideA, const double *b,
    const size_t ldb, const size_t strideB, const double beta, double *c,
    const size_t ldc, const size_t strideC, const size_t batchSize);
