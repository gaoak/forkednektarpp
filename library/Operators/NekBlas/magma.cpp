///////////////////////////////////////////////////////////////////////////////
//
// File: magma.cpp
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

#include "Operators/Field/MemoryAlloc.hpp"
#include "Operators/NekBlas/NekBlas.hpp"

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB, const int M,
    const int N, const int K, const TData alpha, const TData *a, const int lda,
    const TData *b, const int ldb, const TData beta, TData *c, const int ldc)
{
    auto transA = (transposeA == "N") ? MagmaNoTrans : MagmaTrans;
    auto transB = (transposeB == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magma_sgemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c,
                    ldc, handle);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magma_dgemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c,
                    ldc, handle);
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize)
{
    auto transA = (transposeA == "N") ? MagmaNoTrans : MagmaTrans;
    auto transB = (transposeB == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magmablas_sgemm_batched_strided(transA, transB, M, N, K, alpha, a, lda,
                                        strideA, b, ldb, strideB, beta, c, ldc,
                                        strideC, batchSize, handle);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magmablas_dgemm_batched_strided(transA, transB, M, N, K, alpha, a, lda,
                                        strideA, b, ldb, strideB, beta, c, ldc,
                                        strideC, batchSize, handle);
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
NekGemmGroupedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const int *M, const int *N,
                      const int *K, const TData alpha,
                      TData const *const *Aarray, const int *lda,
                      TData const *const *Barray, const int *ldb,
                      const TData beta, TData **Carray, const int *ldc,
                      const int batchSize)
{
    auto transA = (transposeA == "N") ? MagmaNoTrans : MagmaTrans;
    auto transB = (transposeB == "N") ? MagmaNoTrans : MagmaTrans;

    TData const **Adev;
    TData const **Bdev;
    TData **Cdev;

    Nektar::deviceMalloc(&Adev, sizeof(TData *) * batchSize);
    Nektar::deviceMalloc(&Bdev, sizeof(TData *) * batchSize);
    Nektar::deviceMalloc(&Cdev, sizeof(TData *) * batchSize);

    Nektar::deviceMemcpy<Nektar::HostToDevice>(Adev, Aarray,
                                               sizeof(TData *) * batchSize);
    Nektar::deviceMemcpy<Nektar::HostToDevice>(Bdev, Barray,
                                               sizeof(TData *) * batchSize);
    Nektar::deviceMemcpy<Nektar::HostToDevice>(Cdev, Carray,
                                               sizeof(TData *) * batchSize);

    if constexpr (std::is_same_v<TData, float>)
    {
        magmablas_sgemm_vbatched(transA, transB, (int *)M, (int *)N, (int *)K,
                                 alpha, Adev, (int *)lda, Bdev, (int *)ldb,
                                 beta, Cdev, (int *)ldc, batchSize, handle);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magmablas_dgemm_vbatched(transA, transB, (int *)M, (int *)N, (int *)K,
                                 alpha, Adev, (int *)lda, Bdev, (int *)ldb,
                                 beta, Cdev, (int *)ldc, batchSize, handle);
    }

    Nektar::deviceFree(Adev, sizeof(TData *) * batchSize);
    Nektar::deviceFree(Bdev, sizeof(TData *) * batchSize);
    Nektar::deviceFree(Cdev, sizeof(TData *) * batchSize);
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type NekGemv(
    THandle handle, std::string transpose, const int M, const int N,
    const TData alpha, const TData *a, const int lda, const TData *x,
    const int incx, const TData beta, TData *y, const int incy)
{
    auto trans = (transpose == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magma_sgemv(trans, M, N, alpha, a, lda, x, incx, beta, y, incy, handle);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magma_dgemv(trans, M, N, alpha, a, lda, x, incx, beta, y, incy, handle);
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
NekGemvStridedBatched(THandle handle, std::string transpose, const int M,
                      const int N, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *x,
                      const int incx, const int strideX, const TData beta,
                      TData *y, const int incy, const int strideY,
                      const int batchSize)
{
    auto trans = (transpose == "N") ? MagmaNoTrans : MagmaTrans;

    if constexpr (std::is_same_v<TData, float>)
    {
        magmablas_sgemv_batched_strided(trans, M, N, alpha, a, lda, strideA, x,
                                        incx, strideX, beta, y, incy, strideY,
                                        batchSize, handle);
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        magmablas_dgemv_batched_strided(trans, M, N, alpha, a, lda, strideA, x,
                                        incx, strideX, beta, y, incy, strideY,
                                        batchSize, handle);
    }
}

template void NekGemm<magma_queue_t, float>(
    magma_queue_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const float *b, const int ldb, const float beta, float *c,
    const int ldc);

template void NekGemm<magma_queue_t, double>(
    magma_queue_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const double *b, const int ldb, const double beta, double *c,
    const int ldc);

template void NekGemmStridedBatched<magma_queue_t, float>(
    magma_queue_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const int strideA, const float *b, const int ldb,
    const int strideB, const float beta, float *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmStridedBatched<magma_queue_t, double>(
    magma_queue_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const int strideA, const double *b, const int ldb,
    const int strideB, const double beta, double *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmGroupedBatched<magma_queue_t, float>(
    magma_queue_t handle, std::string transposeA, std::string transposeB,
    const int *M, const int *N, const int *K, const float alpha,
    float const *const *Aarray, const int *lda, float const *const *Barray,
    const int *ldb, const float beta, float **Carray, const int *ldc,
    const int batchSize);

template void NekGemmGroupedBatched<magma_queue_t, double>(
    magma_queue_t handle, std::string transposeA, std::string transposeB,
    const int *M, const int *N, const int *K, const double alpha,
    double const *const *Aarray, const int *lda, double const *const *Barray,
    const int *ldb, const double beta, double **Carray, const int *ldc,
    const int batchSize);

template void NekGemv<magma_queue_t, float>(
    magma_queue_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const float *x,
    const int incx, const float beta, float *y, const int incy);

template void NekGemv<magma_queue_t, double>(
    magma_queue_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const double *x,
    const int incx, const double beta, double *y, const int incy);

template void NekGemvStridedBatched<magma_queue_t, float>(
    magma_queue_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *x, const int incx, const int strideX, const float beta,
    float *y, const int incy, const int strideY, const int batchSize);

template void NekGemvStridedBatched<magma_queue_t, double>(
    magma_queue_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *x, const int incx, const int strideX, const double beta,
    double *y, const int incy, const int strideY, const int batchSize);
