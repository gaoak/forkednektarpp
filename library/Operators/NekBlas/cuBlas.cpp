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

#include "Operators/Field/MemoryAlloc.hpp"
#include "Operators/NekBlas/NekBlas.hpp"

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB, const int M,
    const int N, const int K, const TData alpha, const TData *a, const int lda,
    const TData *b, const int ldb, const TData beta, TData *c, const int ldc)
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
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize)
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

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
NekGemmGroupedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const int *M, const int *N,
                      const int *K, const TData alpha,
                      TData const *const *Aarray, const int *lda,
                      TData const *const *Barray, const int *ldb,
                      const TData beta, TData **Carray, const int *ldc,
                      const int batchSize)
{
    std::vector<cublasOperation_t> transA(batchSize);
    std::vector<cublasOperation_t> transB(batchSize);
    std::vector<int> groupSize(batchSize);
    std::vector<TData> alpha_array(batchSize);
    std::vector<TData> beta_array(batchSize);
    std::fill(transA.begin(), transA.end(),
              (transposeA == "N") ? CUBLAS_OP_N : CUBLAS_OP_T);
    std::fill(transB.begin(), transB.end(),
              (transposeB == "N") ? CUBLAS_OP_N : CUBLAS_OP_T);
    std::fill(alpha_array.begin(), alpha_array.end(), alpha);
    std::fill(beta_array.begin(), beta_array.end(), beta);
    std::fill(groupSize.begin(), groupSize.end(), 1);

    TData const **Adev;
    TData const **Bdev;
    TData **Cdev;
    Nektar::deviceMalloc(&Adev, sizeof(TData *) * batchSize,
                         NektarSpaces::Device::alignment);
    Nektar::deviceMalloc(&Bdev, sizeof(TData *) * batchSize,
                         NektarSpaces::Device::alignment);
    Nektar::deviceMalloc(&Cdev, sizeof(TData *) * batchSize,
                         NektarSpaces::Device::alignment);

    Nektar::deviceMemcpy<Nektar::HostToDevice>(Adev, Aarray,
                                               sizeof(TData *) * batchSize);
    Nektar::deviceMemcpy<Nektar::HostToDevice>(Bdev, Barray,
                                               sizeof(TData *) * batchSize);
    Nektar::deviceMemcpy<Nektar::HostToDevice>(Cdev, Carray,
                                               sizeof(TData *) * batchSize);

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemmGroupedBatched(
            handle, transA.data(), transB.data(), M, N, K, alpha_array.data(),
            Adev, lda, Bdev, ldb, beta_array.data(), Cdev, ldc, batchSize,
            groupSize.data()));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemmGroupedBatched(
            handle, transA.data(), transB.data(), M, N, K, alpha_array.data(),
            Adev, lda, Bdev, ldb, beta_array.data(), Cdev, ldc, batchSize,
            groupSize.data()));
    }

    Nektar::deviceFree(Adev, sizeof(TData *) * batchSize,
                       NektarSpaces::Device::alignment);
    Nektar::deviceFree(Bdev, sizeof(TData *) * batchSize,
                       NektarSpaces::Device::alignment);
    Nektar::deviceFree(Cdev, sizeof(TData *) * batchSize,
                       NektarSpaces::Device::alignment);
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type NekGemv(
    THandle handle, std::string transpose, const int M, const int N,
    const TData alpha, const TData *a, const int lda, const TData *x,
    const int incx, const TData beta, TData *y, const int incy)
{
    auto trans = (transpose == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemv(handle, trans, M, N, &alpha, a, lda, x, incx,
                                 &beta, y, incy));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemv(handle, trans, M, N, &alpha, a, lda, x, incx,
                                 &beta, y, incy));
    }
}

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
NekGemvStridedBatched(THandle handle, std::string transpose, const int M,
                      const int N, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *x,
                      const int incx, const int strideX, const TData beta,
                      TData *y, const int incy, const int strideY,
                      const int batchSize)
{
    auto trans = (transpose == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    if constexpr (std::is_same_v<TData, float>)
    {
        CUBLAS_CHECK(cublasSgemvStridedBatched(
            handle, trans, M, N, &alpha, a, lda, strideA, x, incx, strideX,
            &beta, y, incy, strideY, batchSize));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        CUBLAS_CHECK(cublasDgemvStridedBatched(
            handle, trans, M, N, &alpha, a, lda, strideA, x, incx, strideX,
            &beta, y, incy, strideY, batchSize));
    }
}

template void NekGemm<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const float *b, const int ldb, const float beta, float *c,
    const int ldc);

template void NekGemm<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const double *b, const int ldb, const double beta, double *c,
    const int ldc);

template void NekGemmStridedBatched<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const float alpha, const float *a,
    const int lda, const int strideA, const float *b, const int ldb,
    const int strideB, const float beta, float *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmStridedBatched<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const int M, const int N, const int K, const double alpha, const double *a,
    const int lda, const int strideA, const double *b, const int ldb,
    const int strideB, const double beta, double *c, const int ldc,
    const int strideC, const int batchSize);

template void NekGemmGroupedBatched<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const int *M, const int *N, const int *K, const float alpha,
    float const *const *Aarray, const int *lda, float const *const *Barray,
    const int *ldb, const float beta, float **Carray, const int *ldc,
    const int batchSize);

template void NekGemmGroupedBatched<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transposeA, std::string transposeB,
    const int *M, const int *N, const int *K, const double alpha,
    double const *const *Aarray, const int *lda, double const *const *Barray,
    const int *ldb, const double beta, double **Carray, const int *ldc,
    const int batchSize);

template void NekGemv<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const float *x,
    const int incx, const float beta, float *y, const int incy);

template void NekGemv<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const double *x,
    const int incx, const double beta, double *y, const int incy);

template void NekGemvStridedBatched<cublasHandle_t, float>(
    cublasHandle_t handle, std::string transpose, const int M, const int N,
    const float alpha, const float *a, const int lda, const int strideA,
    const float *x, const int incx, const int strideX, const float beta,
    float *y, const int incy, const int strideY, const int batchSize);

template void NekGemvStridedBatched<cublasHandle_t, double>(
    cublasHandle_t handle, std::string transpose, const int M, const int N,
    const double alpha, const double *a, const int lda, const int strideA,
    const double *x, const int incx, const int strideX, const double beta,
    double *y, const int incy, const int strideY, const int batchSize);
