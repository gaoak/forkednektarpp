///////////////////////////////////////////////////////////////////////////////
//
// File: NekBlas.hpp
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

#pragma once

#include <string>

#include "Operators/NekBlas/libXSMMDispatchWrapper.hpp"

#include "Operators/NekBlas/blasHandle.hpp"
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
#include "Operators/NekBlas/xsmmHandle.hpp"
#elif defined(NEKTAR_ENABLE_MAGMA)
#include "Operators/NekBlas/magmaHandle.hpp"
#elif defined(NEKTAR_ENABLE_CUDA)
#include "Operators/NekBlas/cuBlasHandle.hpp"
#elif defined(NEKTAR_ENABLE_HIP)
#include "Operators/NekBlas/hipBlasHandle.hpp"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Common/SYCLQueue.hpp"
#endif

#include "Operators/Common/Spaces.hpp"

template <typename ExecSpace> class NekHandle
{
public:
    typedef int index_type;

    static blasHandle_t GetInstance(void)
    {
        return blasHandle::GetInstance();
    }
};

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
template <> class NekHandle<NektarSpaces::AVX>
{
public:
    typedef int index_type;

    static xsmmHandle_t GetInstance(void)
    {
        return xsmmHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_MAGMA)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    typedef int index_type;

    static magma_queue_t GetInstance(void)
    {
        return magmaHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_CUDA)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    typedef int index_type;

    static cublasHandle_t GetInstance(void)
    {
        return cuBlasHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_HIP)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    typedef int index_type;

    static hipblasHandle_t GetInstance(void)
    {
        return hipBlasHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_SYCL)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    typedef std::int64_t index_type;

    static sycl::queue GetInstance(void)
    {
        return SYCLQueue::GetInstance();
    }
};
#endif

// NekGemm
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB, const int M,
    const int N, const int K, const TData alpha, const TData *a, const int lda,
    const TData *b, const int ldb, const TData beta, TData *c, const int ldc);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_MAGMA)
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
#elif defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemm(THandle handle, std::string transposeA, std::string transposeB,
        const int M, const int N, const int K, const TData alpha,
        const TData *a, const int lda, const TData *b, const int ldb,
        const TData beta, TData *c, const int ldc);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB,
    const std::int64_t M, const std::int64_t N, const std::int64_t K,
    const TData alpha, const TData *a, const std::int64_t lda, const TData *b,
    const std::int64_t ldb, const TData beta, TData *c, const std::int64_t ldc);
#endif

// NekGemmStridedBatched
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_MAGMA)
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
#elif defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const int M, const int N,
                      const int K, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *b,
                      const int ldb, const int strideB, const TData beta,
                      TData *c, const int ldc, const int strideC,
                      const int batchSize);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const std::int64_t M,
                      const std::int64_t N, const std::int64_t K,
                      const TData alpha, const TData *a, const std::int64_t lda,
                      const std::int64_t strideA, const TData *b,
                      const std::int64_t ldb, const std::int64_t strideB,
                      const TData beta, TData *c, const std::int64_t ldc,
                      const std::int64_t strideC, const std::int64_t batchSize);
#endif

// NekGemmGroupedBatched
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type
NekGemmGroupedBatched(THandle, std::string transposeA, std::string transposeB,
                      const int *m, const int *n, const int *k,
                      const TData alpha, TData const *const *Aarray,
                      const int *lda, TData const *const *Barray,
                      const int *ldb, const TData beta, TData **Carray,
                      const int *ldc, const int batchSize);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_MAGMA)
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
#elif defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemmGroupedBatched(THandle, std::string transposeA, std::string transposeB,
                      const int *M, const int *N, const int *K,
                      const TData alpha, TData const *const *Aarray,
                      const int *lda, TData const *const *Barray,
                      const int *ldb, const TData beta, TData **Carray,
                      const int *ldc, const int batchSize);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmGroupedBatched(THandle, std::string transposeA, std::string transposeB,
                      const std::int64_t *M, const std::int64_t *N,
                      const std::int64_t *K, const TData alpha,
                      TData const *const *Aarray, const std::int64_t *lda,
                      TData const *const *Barray, const std::int64_t *ldb,
                      const TData beta, TData **Carray, const std::int64_t *ldc,
                      const std::int64_t batchSize);
#endif

// NekGemv
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type NekGemv(
    THandle handle, std::string transpose, const int M, const int N,
    const TData alpha, const TData *a, const int lda, const TData *x,
    const int incx, const TData beta, TData *y, const int incy);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_MAGMA)
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
#elif defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemv(THandle handle, std::string transpose, const int M, const int N,
        const TData alpha, const TData *a, const int lda, const TData *x,
        const int incx, const TData beta, TData *y, const int incy);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type NekGemv(
    THandle handle, std::string transpose, const std::int64_t M,
    const std::int64_t N, const TData alpha, const TData *a,
    const std::int64_t lda, const TData *x, const std::int64_t incx,
    const TData beta, TData *y, const std::int64_t incy);
#endif

// NekGemvStridedBatched
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle_t>, void>::type
NekGemvStridedBatched(THandle handle, std::string transpose, const int M,
                      const int N, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *x,
                      const int incx, const int strideX, const TData beta,
                      TData *y, const int incy, const int strideY,
                      const int batchSize);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
typename std::enable_if<std::is_same_v<THandle, xsmmHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_MAGMA)
typename std::enable_if<std::is_same_v<THandle, magma_queue_t>, void>::type
#elif defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemvStridedBatched(THandle handle, std::string transpose, const int M,
                      const int N, const TData alpha, const TData *a,
                      const int lda, const int strideA, const TData *x,
                      const int incx, const int strideX, const TData beta,
                      TData *y, const int incy, const int strideY,
                      const int batchSize);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemvStridedBatched(THandle handle, std::string transpose,
                      const std::int64_t M, const std::int64_t N,
                      const TData alpha, const TData *a, const std::int64_t lda,
                      const std::int64_t strideA, const TData *x,
                      const std::int64_t incx, const std::int64_t strideX,
                      const TData beta, TData *y, const std::int64_t incy,
                      const std::int64_t strideY, const std::int64_t batchSize);
#endif
