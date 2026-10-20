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
// Description: NekBlas backend for the Serial execution space (host BLAS,
// with NVPL batched extensions when available). Vendor headers are included
// here, never in NekBlas.hpp.
//
///////////////////////////////////////////////////////////////////////////////

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

#include <LibUtilities/LinearAlgebra/Blas.hpp>

#include <stdexcept>

#ifdef NEKTAR_USE_NVPL
#include <nvpl_blas_cblas.h>

#include <type_traits>
#endif

namespace Nektar::NekBlas
{
#ifdef NEKTAR_USE_NVPL
namespace
{
template <typename TData>
void NvplGemmStridedBatched(const char transA, const char transB, const int M,
                            const int N, const int K, const TData alpha,
                            const TData *a, const int lda, const int strideA,
                            const TData *b, const int ldb, const int strideB,
                            const TData beta, TData *c, const int ldc,
                            const int strideC, const int batchSize)
{
    auto opA = transA == 'N' ? CblasNoTrans : CblasTrans;
    auto opB = transB == 'N' ? CblasNoTrans : CblasTrans;

    if constexpr (std::is_same_v<TData, double>)
    {
        cblas_dgemm_batch_strided(CblasColMajor, opA, opB, M, N, K, alpha, a,
                                  lda, strideA, b, ldb, strideB, beta, c, ldc,
                                  strideC, batchSize);
    }
    else
    {
        cblas_sgemm_batch_strided(CblasColMajor, opA, opB, M, N, K, alpha, a,
                                  lda, strideA, b, ldb, strideB, beta, c, ldc,
                                  strideC, batchSize);
    }
}
} // namespace
#endif

template <typename TData>
void Gemm([[maybe_unused]] Handle<NektarSpaces::Serial> handle,
          std::string transposeA, std::string transposeB, const int M,
          const int N, const int K, const TData alpha, const TData *a,
          const int lda, const TData *b, const int ldb, const TData beta,
          TData *c, const int ldc)
{
    auto transA = *transposeA.c_str();
    auto transB = *transposeB.c_str();

    Blas::Gemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename TData>
void GemmStridedBatched([[maybe_unused]] Handle<NektarSpaces::Serial> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const int K,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *b, const int ldb,
                        const int strideB, const TData beta, TData *c,
                        const int ldc, const int strideC, const int batchSize)
{
    auto transA = *transposeA.c_str();
    auto transB = *transposeB.c_str();

#ifdef NEKTAR_USE_NVPL
    NvplGemmStridedBatched(transA, transB, M, N, K, alpha, a, lda, strideA, b,
                           ldb, strideB, beta, c, ldc, strideC, batchSize);
#else
    for (int i = 0; i < batchSize; i++)
    {
        Blas::Gemm(transA, transB, M, N, K, alpha, a + strideA * i, lda,
                   b + strideB * i, ldb, beta, c + strideC * i, ldc);
    }
#endif
}

template <typename TData>
void Gemv([[maybe_unused]] Handle<NektarSpaces::Serial> handle,
          std::string transpose, const int M, const int N, const TData alpha,
          const TData *a, const int lda, const TData *x, const int incx,
          const TData beta, TData *y, const int incy)
{
    auto trans = *transpose.c_str();

    Blas::Gemv(trans, M, N, alpha, a, lda, x, incx, beta, y, incy);
}

template <typename TData>
void GemvStridedBatched([[maybe_unused]] Handle<NektarSpaces::Serial> handle,
                        std::string transpose, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *x, const int incx,
                        const int strideX, const TData beta, TData *y,
                        const int incy, const int strideY, const int batchSize)
{
    auto trans = *transpose.c_str();

#ifdef NEKTAR_USE_NVPL
    // Unit-stride Gemv is a Gemm with a single column vector.
    if (incx == 1 && incy == 1)
    {
        const int m = trans == 'N' ? M : N;
        const int k = trans == 'N' ? N : M;
        NvplGemmStridedBatched(trans, 'N', m, 1, k, alpha, a, lda, strideA, x,
                               k, strideX, beta, y, m, strideY, batchSize);
        return;
    }
#endif

    for (int i = 0; i < batchSize; i++)
    {
        Blas::Gemv(trans, M, N, alpha, a + strideA * i, lda, x + strideX * i,
                   incx, beta, y + strideY * i, incy);
    }
}

template <typename TData>
void GeamStridedBatched(Handle<NektarSpaces::Serial>, std::string, std::string,
                        const int, const int, const TData, const TData *,
                        const int, const int, const TData, const TData *,
                        const int, const int, TData *, const int, const int,
                        const int)
{
    throw std::runtime_error("GeamStridedBatched is not available for BLAS");
}

template void Gemm<float>(Handle<NektarSpaces::Serial> handle,
                          std::string transposeA, std::string transposeB,
                          const int M, const int N, const int K,
                          const float alpha, const float *a, const int lda,
                          const float *b, const int ldb, const float beta,
                          float *c, const int ldc);

template void Gemm<double>(Handle<NektarSpaces::Serial> handle,
                           std::string transposeA, std::string transposeB,
                           const int M, const int N, const int K,
                           const double alpha, const double *a, const int lda,
                           const double *b, const int ldb, const double beta,
                           double *c, const int ldc);

template void GemmStridedBatched<float>(
    Handle<NektarSpaces::Serial> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const int K,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *b, const int ldb, const int strideB, const float beta,
    float *c, const int ldc, const int strideC, const int batchSize);

template void GemmStridedBatched<double>(
    Handle<NektarSpaces::Serial> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const int K,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *b, const int ldb, const int strideB, const double beta,
    double *c, const int ldc, const int strideC, const int batchSize);

template void Gemv<float>(Handle<NektarSpaces::Serial> handle,
                          std::string transpose, const int M, const int N,
                          const float alpha, const float *a, const int lda,
                          const float *x, const int incx, const float beta,
                          float *y, const int incy);

template void Gemv<double>(Handle<NektarSpaces::Serial> handle,
                           std::string transpose, const int M, const int N,
                           const double alpha, const double *a, const int lda,
                           const double *x, const int incx, const double beta,
                           double *y, const int incy);

template void GemvStridedBatched<float>(
    Handle<NektarSpaces::Serial> handle, std::string transpose, const int M,
    const int N, const float alpha, const float *a, const int lda,
    const int strideA, const float *x, const int incx, const int strideX,
    const float beta, float *y, const int incy, const int strideY,
    const int batchSize);

template void GemvStridedBatched<double>(
    Handle<NektarSpaces::Serial> handle, std::string transpose, const int M,
    const int N, const double alpha, const double *a, const int lda,
    const int strideA, const double *x, const int incx, const int strideX,
    const double beta, double *y, const int incy, const int strideY,
    const int batchSize);

template void GeamStridedBatched<float>(
    Handle<NektarSpaces::Serial> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const float alpha,
    const float *a, const int lda, const int strideA, const float beta,
    const float *b, const int ldb, const int strideB, float *c, const int ldc,
    const int strideC, const int batchSize);

template void GeamStridedBatched<double>(
    Handle<NektarSpaces::Serial> handle, std::string transposeA,
    std::string transposeB, const int M, const int N, const double alpha,
    const double *a, const int lda, const int strideA, const double beta,
    const double *b, const int ldb, const int strideB, double *c, const int ldc,
    const int strideC, const int batchSize);
} // namespace Nektar::NekBlas
