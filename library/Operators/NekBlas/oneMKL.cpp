///////////////////////////////////////////////////////////////////////////////
//
// File: oneMKL.cpp
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
#include "oneapi/mkl.hpp"

using namespace oneapi::mkl;

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type NekGemm(
    THandle queue, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const TData alpha, const TData *a, const unsigned int lda, const TData *b,
    const unsigned int ldb, const TData beta, TData *c, const unsigned int ldc)
{
    auto transA = (transposeA == "N") ? oneapi::mkl::transpose::N
                                      : oneapi::mkl::transpose::T;
    auto transB = (transposeB == "N") ? oneapi::mkl::transpose::N
                                      : oneapi::mkl::transpose::T;
    blas::column_major::gemm(queue, transA, transB, M, N, K, alpha, a, lda, b,
                             ldb, beta, c, ldc);

#if defined(SYCL_ENABLE_SERIAL)
    queue.wait();
#endif
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmStridedBatched(THandle queue, std::string transposeA,
                      std::string transposeB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const TData alpha, const TData *a, const unsigned int lda,
                      const unsigned int strideA, const TData *b,
                      const unsigned int ldb, const unsigned int strideB,
                      const TData beta, TData *c, const unsigned int ldc,
                      const unsigned int strideC, const unsigned int batchSize)
{
    auto transA = (transposeA == "N") ? oneapi::mkl::transpose::N
                                      : oneapi::mkl::transpose::T;
    auto transB = (transposeB == "N") ? oneapi::mkl::transpose::N
                                      : oneapi::mkl::transpose::T;
    blas::column_major::gemm_batch(queue, transA, transB, M, N, K, alpha, a,
                                   lda, strideA, b, ldb, strideB, beta, c, ldc,
                                   strideC, batchSize);

#if defined(SYCL_ENABLE_SERIAL)
    queue.wait();
#endif
}

template void NekGemm<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float alpha, const float *a, const unsigned int lda, const float *b,
    const unsigned int ldb, const float beta, float *c, const unsigned int ldc);

template void NekGemm<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const double alpha, const double *a, const unsigned int lda,
    const double *b, const unsigned int ldb, const double beta, double *c,
    const unsigned int ldc);

template void NekGemmStridedBatched<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float alpha, const float *a, const unsigned int lda,
    const unsigned int strideA, const float *b, const unsigned int ldb,
    const unsigned int strideB, const float beta, float *c,
    const unsigned int ldc, const unsigned int strideC,
    const unsigned int batchSize);

template void NekGemmStridedBatched<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const double alpha, const double *a, const unsigned int lda,
    const unsigned int strideA, const double *b, const unsigned int ldb,
    const unsigned int strideB, const double beta, double *c,
    const unsigned int ldc, const unsigned int strideC,
    const unsigned int batchSize);
