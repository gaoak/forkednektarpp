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

#include "Operators/Common/Memory/MemoryAlloc.hpp"
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
    [[maybe_unused]] THandle queue, std::string transposeA,
    std::string transposeB, const std::int64_t M, const std::int64_t N,
    const std::int64_t K, const TData alpha, const TData *a,
    const std::int64_t lda, const TData *b, const std::int64_t ldb,
    const TData beta, TData *c, const std::int64_t ldc)
{
#if __has_include("oneapi/mkl.hpp") || __has_include("oneapi/math.hpp")
    auto transA = (transposeA == "N") ? transpose::N : transpose::T;
    auto transB = (transposeB == "N") ? transpose::N : transpose::T;
    blas::column_major::gemm(queue, transA, transB, M, N, K, alpha, a, lda, b,
                             ldb, beta, c, ldc);
#else
    queue.wait();
    NekGemm(blasHandle_t(), transposeA, transposeB, M, N, K, alpha, a, lda, b,
            ldb, beta, c, ldc);
#endif
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmStridedBatched([[maybe_unused]] THandle queue, std::string transposeA,
                      std::string transposeB, const std::int64_t M,
                      const std::int64_t N, const std::int64_t K,
                      const TData alpha, const TData *a, const std::int64_t lda,
                      const std::int64_t strideA, const TData *b,
                      const std::int64_t ldb, const std::int64_t strideB,
                      const TData beta, TData *c, const std::int64_t ldc,
                      const std::int64_t strideC, const std::int64_t batchSize)
{
#if __has_include("oneapi/mkl.hpp") || __has_include("oneapi/math.hpp")
    std::vector<std::int64_t> Mvec(batchSize), Nvec(batchSize), Kvec(batchSize);
    std::vector<std::int64_t> lda_vec(batchSize), ldb_vec(batchSize),
        ldc_vec(batchSize);
    std::fill(Mvec.begin(), Mvec.end(), M);
    std::fill(Nvec.begin(), Nvec.end(), N);
    std::fill(Kvec.begin(), Kvec.end(), K);
    std::fill(lda_vec.begin(), lda_vec.end(), lda);
    std::fill(ldb_vec.begin(), ldb_vec.end(), ldb);
    std::fill(ldc_vec.begin(), ldc_vec.end(), ldc);

    std::vector<const TData *> Avec(batchSize), Bvec(batchSize);
    std::vector<TData *> Cvec(batchSize);
    for (std::int64_t i = 0; i < batchSize; i++)
    {
        Avec[i] = a + i * strideA;
        Bvec[i] = b + i * strideB;
        Cvec[i] = c + i * strideC;
    }
    NekGemmGroupedBatched(queue, transposeA, transposeB, Mvec.data(),
                          Nvec.data(), Kvec.data(), alpha, Avec.data(),
                          lda_vec.data(), Bvec.data(), ldb_vec.data(), beta,
                          Cvec.data(), ldc_vec.data(), batchSize);

    // Not working
    /*auto transA = (transposeA == "N") ? transpose::N : transpose::T;
    auto transB = (transposeB == "N") ? transpose::N : transpose::T;
    blas::column_major::gemm_batch(queue, transA, transB, M, N, K, alpha, a,
                                   lda, strideA, b, ldb, strideB, beta, c, ldc,
                                   strideC, batchSize);*/
#else
    queue.wait();
    NekGemmStridedBatched(blasHandle_t(), transposeA, transposeB, M, N, K,
                          alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc,
                          strideC, batchSize);
#endif
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmGroupedBatched([[maybe_unused]] THandle queue, std::string transposeA,
                      std::string transposeB, const std::int64_t *M,
                      const std::int64_t *N, const std::int64_t *K,
                      const TData alpha, TData const *const *Aarray,
                      const std::int64_t *lda, TData const *const *Barray,
                      const std::int64_t *ldb, const TData beta, TData **Carray,
                      const std::int64_t *ldc, const std::int64_t batchSize)
{
#if __has_include("oneapi/mkl.hpp") || __has_include("oneapi/math.hpp")
    std::vector<transpose> transA(batchSize);
    std::vector<transpose> transB(batchSize);
    std::vector<std::int64_t> groupSize(batchSize);
    std::vector<TData> alpha_array(batchSize);
    std::vector<TData> beta_array(batchSize);
    std::fill(transA.begin(), transA.end(),
              (transposeA == "N") ? transpose::N : transpose::T);
    std::fill(transB.begin(), transB.end(),
              (transposeB == "N") ? transpose::N : transpose::T);
    std::fill(alpha_array.begin(), alpha_array.end(), alpha);
    std::fill(beta_array.begin(), beta_array.end(), beta);
    std::fill(groupSize.begin(), groupSize.end(), 1);

    TData const **Adev;
    TData const **Bdev;
    TData **Cdev;
    GetDeviceProperties::CheckGlobalMemoryUsage(3 * sizeof(TData *) *
                                                batchSize);
    Adev =
        (const TData **)sycl::malloc_device(sizeof(TData *) * batchSize, queue);
    Bdev =
        (const TData **)sycl::malloc_device(sizeof(TData *) * batchSize, queue);
    Cdev = (TData **)sycl::malloc_device(sizeof(TData *) * batchSize, queue);
    GetDeviceProperties::TotalGlobalMemory() -= 3 * sizeof(TData *) * batchSize;
    queue.memcpy(Adev, Aarray, sizeof(TData *) * batchSize).wait();
    queue.memcpy(Bdev, Barray, sizeof(TData *) * batchSize).wait();
    queue.memcpy(Cdev, Carray, sizeof(TData *) * batchSize).wait();

#if __has_include("oneapi/math.hpp")
    blas::column_major::gemm_batch(
        queue, transA.data(), transB.data(), (std::int64_t *)M,
        (std::int64_t *)N, (std::int64_t *)K, alpha_array.data(), Adev,
        (std::int64_t *)lda, Bdev, (std::int64_t *)ldb, beta_array.data(), Cdev,
        (std::int64_t *)ldc, batchSize, groupSize.data());
#elif __has_include("oneapi/mkl.hpp")
    blas::column_major::gemm_batch(queue, transA.data(), transB.data(), M, N, K,
                                   alpha_array.data(), Adev, lda, Bdev, ldb,
                                   beta_array.data(), Cdev, ldc, batchSize,
                                   groupSize.data());
#endif

    queue.wait();
    sycl::free(Adev, queue);
    sycl::free(Bdev, queue);
    sycl::free(Cdev, queue);
    GetDeviceProperties::TotalGlobalMemory() += 3 * sizeof(TData *) * batchSize;
#else
    ASSERTL0(false, "NekGemmGroupedBatched not yet implemented")
    queue.wait();
    NekGemmGroupedBatched(blasHandle_t(), transposeA, transposeB,
                          (const int *)M, (const int *)N, (const int *)K, alpha,
                          Aarray, (const int *)lda, Barray, (const int *)ldb,
                          beta, Carray, (const int *)ldc, batchSize);
#endif
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type NekGemv(
    [[maybe_unused]] THandle queue, std::string transpose, const std::int64_t M,
    const std::int64_t N, const TData alpha, const TData *a,
    const std::int64_t lda, const TData *x, const std::int64_t incx,
    const TData beta, TData *y, const std::int64_t incy)
{
#if __has_include("oneapi/mkl.hpp") || __has_include("oneapi/math.hpp")
    auto trans = (transpose == "N") ? transpose::N : transpose::T;

    blas::column_major::gemv(queue, trans, M, N, alpha, a, lda, x, incx, beta,
                             y, incy);
#else
    queue.wait();
    NekGemv(blasHandle_t(), transpose, M, N, alpha, a, lda, x, incx, beta, y,
            incy);
#endif
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemvStridedBatched([[maybe_unused]] THandle queue, std::string transpose,
                      const std::int64_t M, const std::int64_t N,
                      const TData alpha, const TData *a, const std::int64_t lda,
                      const std::int64_t strideA, const TData *x,
                      const std::int64_t incx, const std::int64_t strideX,
                      const TData beta, TData *y, const std::int64_t incy,
                      const std::int64_t strideY, const std::int64_t batchSize)
{
#if __has_include("oneapi/mkl.hpp") || __has_include("oneapi/math.hpp")
#if __has_include("oneapi/math.hpp")
    ASSERTL0(false, "gemv_batch not yet implemented in oneMath")
#endif
    auto trans = (transpose == "N") ? transpose::N : transpose::T;

    blas::column_major::gemv_batch(queue, trans, M, N, alpha, a, lda, strideA,
                                   x, incx, strideX, beta, y, incy, strideY,
                                   batchSize);
#else
    queue.wait();
    NekGemvStridedBatched(blasHandle_t(), transpose, M, N, alpha, a, lda,
                          strideA, x, incx, strideX, beta, y, incy, strideY,
                          batchSize);
#endif
}

template void NekGemm<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const float alpha, const float *a, const std::int64_t lda, const float *b,
    const std::int64_t ldb, const float beta, float *c, const std::int64_t ldc);

template void NekGemm<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const double alpha, const double *a, const std::int64_t lda,
    const double *b, const std::int64_t ldb, const double beta, double *c,
    const std::int64_t ldc);

template void NekGemmStridedBatched<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const float alpha, const float *a, const std::int64_t lda,
    const std::int64_t strideA, const float *b, const std::int64_t ldb,
    const std::int64_t strideB, const float beta, float *c,
    const std::int64_t ldc, const std::int64_t strideC,
    const std::int64_t batchSize);

template void NekGemmStridedBatched<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const double alpha, const double *a, const std::int64_t lda,
    const std::int64_t strideA, const double *b, const std::int64_t ldb,
    const std::int64_t strideB, const double beta, double *c,
    const std::int64_t ldc, const std::int64_t strideC,
    const std::int64_t batchSize);

template void NekGemmGroupedBatched<sycl::queue, float>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const std::int64_t *m, const std::int64_t *n, const std::int64_t *k,
    const float alpha, float const *const *Aarray, const std::int64_t *lda,
    float const *const *Barray, const std::int64_t *ldb, const float beta,
    float **Carray, const std::int64_t *ldc, const std::int64_t batchSize);

template void NekGemmGroupedBatched<sycl::queue, double>(
    sycl::queue queue, std::string transposeA, std::string transposeB,
    const std::int64_t *M, const std::int64_t *N, const std::int64_t *K,
    const double alpha, double const *const *Aarray, const std::int64_t *lda,
    double const *const *Barray, const std::int64_t *ldb, const double beta,
    double **Carray, const std::int64_t *ldc, const std::int64_t batchSize);

template void NekGemv<sycl::queue, float>(
    sycl::queue queue, std::string transpose, const std::int64_t M,
    const std::int64_t N, const float alpha, const float *a,
    const std::int64_t lda, const float *x, const std::int64_t incx,
    const float beta, float *y, const std::int64_t incy);

template void NekGemv<sycl::queue, double>(
    sycl::queue queue, std::string transpose, const std::int64_t M,
    const std::int64_t N, const double alpha, const double *a,
    const std::int64_t lda, const double *x, const std::int64_t incx,
    const double beta, double *y, const std::int64_t incy);

template void NekGemvStridedBatched<sycl::queue, float>(
    sycl::queue queue, std::string transpose, const std::int64_t M,
    const std::int64_t N, const float alpha, const float *a,
    const std::int64_t lda, const std::int64_t strideA, const float *x,
    const std::int64_t incx, const std::int64_t strideX, const float beta,
    float *y, const std::int64_t incy, const std::int64_t strideY,
    const std::int64_t batchSize);

template void NekGemvStridedBatched<sycl::queue, double>(
    sycl::queue queue, std::string transpose, const std::int64_t M,
    const std::int64_t N, const double alpha, const double *a,
    const std::int64_t lda, const std::int64_t strideA, const double *x,
    const std::int64_t incx, const std::int64_t strideX, const double beta,
    double *y, const std::int64_t incy, const std::int64_t strideY,
    const std::int64_t batchSize);
