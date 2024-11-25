///////////////////////////////////////////////////////////////////////////////
//
// File: OneMKL.cpp
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

#include "OneMKL.hpp"
#include "oneapi/mkl.hpp"

namespace OneMKL
{

using namespace oneapi::mkl;

void gemm(sycl::queue queue, std::string transposeA, std::string transposeB,
          const unsigned int M, const unsigned int N, const unsigned int K,
          const float alpha, const float *a, const unsigned int lda,
          const float *b, const unsigned int ldb, const float beta, float *c,
          const unsigned int ldc)
{
    if (transposeA == "N" && transposeB == "N")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::N,
                                 oneapi::mkl::transpose::N, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "N")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::T,
                                 oneapi::mkl::transpose::N, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
    else if (transposeA == "N" && transposeB == "T")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::N,
                                 oneapi::mkl::transpose::T, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "T")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::T,
                                 oneapi::mkl::transpose::T, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
}

void gemm(sycl::queue queue, std::string transposeA, std::string transposeB,
          const unsigned int M, const unsigned int N, const unsigned int K,
          const double alpha, const double *a, const unsigned int lda,
          const double *b, const unsigned int ldb, const double beta, double *c,
          const unsigned int ldc)
{
    if (transposeA == "N" && transposeB == "N")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::N,
                                 oneapi::mkl::transpose::N, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "N")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::T,
                                 oneapi::mkl::transpose::N, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
    else if (transposeA == "N" && transposeB == "T")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::N,
                                 oneapi::mkl::transpose::T, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "T")
    {
        blas::column_major::gemm(queue, oneapi::mkl::transpose::T,
                                 oneapi::mkl::transpose::T, M, N, K, alpha, a,
                                 lda, b, ldb, beta, c, ldc)
            .wait();
    }
}

void gemm_batch(sycl::queue queue, std::string transposeA,
                std::string transposeB, const unsigned int M,
                const unsigned int N, const unsigned int K, const float alpha,
                const float *a, const unsigned int lda,
                const unsigned int strideA, const float *b,
                const unsigned int ldb, const unsigned int strideB,
                const float beta, float *c, const unsigned int ldc,
                const unsigned int strideC, const unsigned int batchSize)
{
    if (transposeA == "N" && transposeB == "N")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::N,
                                       oneapi::mkl::transpose::N, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "N")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::T,
                                       oneapi::mkl::transpose::N, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
    else if (transposeA == "N" && transposeB == "T")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::N,
                                       oneapi::mkl::transpose::T, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "T")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::T,
                                       oneapi::mkl::transpose::T, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
}

void gemm_batch(sycl::queue queue, std::string transposeA,
                std::string transposeB, const unsigned int M,
                const unsigned int N, const unsigned int K, const double alpha,
                const double *a, const unsigned int lda,
                const unsigned int strideA, const double *b,
                const unsigned int ldb, const unsigned int strideB,
                const double beta, double *c, const unsigned int ldc,
                const unsigned int strideC, const unsigned int batchSize)
{
    if (transposeA == "N" && transposeB == "N")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::N,
                                       oneapi::mkl::transpose::N, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "N")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::T,
                                       oneapi::mkl::transpose::N, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
    else if (transposeA == "N" && transposeB == "T")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::N,
                                       oneapi::mkl::transpose::T, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
    else if (transposeA == "T" && transposeB == "T")
    {
        blas::column_major::gemm_batch(queue, oneapi::mkl::transpose::T,
                                       oneapi::mkl::transpose::T, M, N, K,
                                       alpha, a, lda, strideA, b, ldb, strideB,
                                       beta, c, ldc, strideC, batchSize)
            .wait();
    }
}

} // namespace OneMKL
