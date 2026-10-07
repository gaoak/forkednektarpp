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

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp"

#include "LibUtilities/LinearAlgebra/NekBlas/blas.hpp"
#if defined(NEKTAR_ENABLE_SIMD)
#include "LibUtilities/LinearAlgebra/NekBlas/xsmm.hpp"
#endif
#if defined(NEKTAR_USE_MAGMA)
#include "LibUtilities/LinearAlgebra/NekBlas/magma.hpp"
#elif defined(NEKTAR_ENABLE_CUDA)
#include "LibUtilities/LinearAlgebra/NekBlas/cuBlas.hpp"
#elif defined(NEKTAR_ENABLE_HIP)
#include "LibUtilities/LinearAlgebra/NekBlas/hipBlas.hpp"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "LibUtilities/LinearAlgebra/NekBlas/oneMath.hpp"
#endif

#include <LibUtilities/Backends/Backends.hpp>

namespace Nektar::NekBlas
{
template <typename ExecSpace> class Handle
{
public:
    typedef int index_type;

    static blasHandle_t GetInstance(
        [[maybe_unused]] const unsigned int streamID)
    {
        return blasHandle::GetInstance();
    }
};

#if defined(NEKTAR_ENABLE_SIMD)
template <> class Handle<NektarSpaces::AVX>
{
public:
    typedef int index_type;

    static xsmmHandle_t GetInstance(
        [[maybe_unused]] const unsigned int streamID)
    {
        return xsmmHandle::GetInstance();
    }
};
#endif

#if defined(NEKTAR_USE_MAGMA)
template <> class Handle<NektarSpaces::Device>
{
public:
    typedef int index_type;

    static magma_queue_t GetInstance(const unsigned int streamID)
    {
        return magmaHandle::GetInstance(streamID);
    }
};
#elif defined(NEKTAR_ENABLE_CUDA)
template <> class Handle<NektarSpaces::Device>
{
public:
    typedef int index_type;

    static cublasHandle_t GetInstance(const unsigned int streamID)
    {
        return cuBlasHandle::GetInstance(streamID);
    }
};
#elif defined(NEKTAR_ENABLE_HIP)
template <> class Handle<NektarSpaces::Device>
{
public:
    typedef int index_type;

    static hipblasHandle_t GetInstance(const unsigned int streamID)
    {
        return hipBlasHandle::GetInstance(streamID);
    }
};
#elif defined(NEKTAR_ENABLE_SYCL)
template <> class Handle<NektarSpaces::Device>
{
public:
    typedef std::int64_t index_type;

    static oneMathHandle_t GetInstance(const unsigned int streamID)
    {
        return oneMathHandle::GetInstance(streamID);
    }
};
#endif

// Gemm
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, blasHandle_t>, bool> Enable = true>
void Gemm(THandle handle, std::string transposeA, std::string transposeB,
          const int M, const int N, const int K, const TData alpha,
          const TData *a, const int lda, const TData *b, const int ldb,
          const TData beta, TData *c, const int ldc);
#if defined(NEKTAR_ENABLE_SIMD)
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, xsmmHandle_t>, bool> Enable = true>
void Gemm(THandle handle, std::string transposeA, std::string transposeB,
          const int M, const int N, const int K, const TData alpha,
          const TData *a, const int lda, const TData *b, const int ldb,
          const TData beta, TData *c, const int ldc);
#endif
template <
    typename THandle, typename TData,
#if defined(NEKTAR_USE_MAGMA)
    std::enable_if_t<std::is_same_v<THandle, magma_queue_t>, bool> Enable = true
#elif defined(NEKTAR_ENABLE_CUDA)
    std::enable_if_t<std::is_same_v<THandle, cublasHandle_t>, bool> Enable =
        true
#elif defined(NEKTAR_ENABLE_HIP)
    std::enable_if_t<std::is_same_v<THandle, hipblasHandle_t>, bool> Enable =
        true
#else
    std::enable_if_t<std::is_same_v<THandle, std::nullptr_t>, bool> Enable =
        true
#endif
    >
void Gemm(THandle handle, std::string transposeA, std::string transposeB,
          const int M, const int N, const int K, const TData alpha,
          const TData *a, const int lda, const TData *b, const int ldb,
          const TData beta, TData *c, const int ldc);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>
              Enable = true>
void Gemm(THandle handle, std::string transposeA, std::string transposeB,
          const std::int64_t M, const std::int64_t N, const std::int64_t K,
          const TData alpha, const TData *a, const std::int64_t lda,
          const TData *b, const std::int64_t ldb, const TData beta, TData *c,
          const std::int64_t ldc);
#endif

// GemmStridedBatched
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, blasHandle_t>, bool> Enable = true>
void GemmStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const int M, const int N,
                        const int K, const TData alpha, const TData *a,
                        const int lda, const int strideA, const TData *b,
                        const int ldb, const int strideB, const TData beta,
                        TData *c, const int ldc, const int strideC,
                        const int batchSize);
#if defined(NEKTAR_ENABLE_SIMD)
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, xsmmHandle_t>, bool> Enable = true>
void GemmStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const int M, const int N,
                        const int K, const TData alpha, const TData *a,
                        const int lda, const int strideA, const TData *b,
                        const int ldb, const int strideB, const TData beta,
                        TData *c, const int ldc, const int strideC,
                        const int batchSize);
#endif
template <
    typename THandle, typename TData,
#if defined(NEKTAR_USE_MAGMA)
    std::enable_if_t<std::is_same_v<THandle, magma_queue_t>, bool> Enable = true
#elif defined(NEKTAR_ENABLE_CUDA)
    std::enable_if_t<std::is_same_v<THandle, cublasHandle_t>, bool> Enable =
        true
#elif defined(NEKTAR_ENABLE_HIP)
    std::enable_if_t<std::is_same_v<THandle, hipblasHandle_t>, bool> Enable =
        true
#else
    std::enable_if_t<std::is_same_v<THandle, std::nullptr_t>, bool> Enable =
        true
#endif
    >
void GemmStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const int M, const int N,
                        const int K, const TData alpha, const TData *a,
                        const int lda, const int strideA, const TData *b,
                        const int ldb, const int strideB, const TData beta,
                        TData *c, const int ldc, const int strideC,
                        const int batchSize);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>
              Enable = true>
void GemmStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const std::int64_t M,
                        const std::int64_t N, const std::int64_t K,
                        const TData alpha, const TData *a,
                        const std::int64_t lda, const std::int64_t strideA,
                        const TData *b, const std::int64_t ldb,
                        const std::int64_t strideB, const TData beta, TData *c,
                        const std::int64_t ldc, const std::int64_t strideC,
                        const std::int64_t batchSize);
#endif

// Gemv
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, blasHandle_t>, bool> Enable = true>
void Gemv(THandle handle, std::string transpose, const int M, const int N,
          const TData alpha, const TData *a, const int lda, const TData *x,
          const int incx, const TData beta, TData *y, const int incy);
#if defined(NEKTAR_ENABLE_SIMD)
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, xsmmHandle_t>, bool> Enable = true>
void Gemv(THandle handle, std::string transpose, const int M, const int N,
          const TData alpha, const TData *a, const int lda, const TData *x,
          const int incx, const TData beta, TData *y, const int incy);
#endif
template <
    typename THandle, typename TData,
#if defined(NEKTAR_USE_MAGMA)
    std::enable_if_t<std::is_same_v<THandle, magma_queue_t>, bool> Enable = true
#elif defined(NEKTAR_ENABLE_CUDA)
    std::enable_if_t<std::is_same_v<THandle, cublasHandle_t>, bool> Enable =
        true
#elif defined(NEKTAR_ENABLE_HIP)
    std::enable_if_t<std::is_same_v<THandle, hipblasHandle_t>, bool> Enable =
        true
#else
    std::enable_if_t<std::is_same_v<THandle, std::nullptr_t>, bool> Enable =
        true
#endif
    >
void Gemv(THandle handle, std::string transpose, const int M, const int N,
          const TData alpha, const TData *a, const int lda, const TData *x,
          const int incx, const TData beta, TData *y, const int incy);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>
              Enable = true>
void Gemv(THandle handle, std::string transpose, const std::int64_t M,
          const std::int64_t N, const TData alpha, const TData *a,
          const std::int64_t lda, const TData *x, const std::int64_t incx,
          const TData beta, TData *y, const std::int64_t incy);
#endif

// GemvStridedBatched
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, blasHandle_t>, bool> Enable = true>
void GemvStridedBatched(THandle handle, std::string transpose, const int M,
                        const int N, const TData alpha, const TData *a,
                        const int lda, const int strideA, const TData *x,
                        const int incx, const int strideX, const TData beta,
                        TData *y, const int incy, const int strideY,
                        const int batchSize);
#if defined(NEKTAR_ENABLE_SIMD)
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, xsmmHandle_t>, bool> Enable = true>
void GemvStridedBatched(THandle handle, std::string transpose, const int M,
                        const int N, const TData alpha, const TData *a,
                        const int lda, const int strideA, const TData *x,
                        const int incx, const int strideX, const TData beta,
                        TData *y, const int incy, const int strideY,
                        const int batchSize);
#endif
template <
    typename THandle, typename TData,
#if defined(NEKTAR_USE_MAGMA)
    std::enable_if_t<std::is_same_v<THandle, magma_queue_t>, bool> Enable = true
#elif defined(NEKTAR_ENABLE_CUDA)
    std::enable_if_t<std::is_same_v<THandle, cublasHandle_t>, bool> Enable =
        true
#elif defined(NEKTAR_ENABLE_HIP)
    std::enable_if_t<std::is_same_v<THandle, hipblasHandle_t>, bool> Enable =
        true
#else
    std::enable_if_t<std::is_same_v<THandle, std::nullptr_t>, bool> Enable =
        true
#endif
    >
void GemvStridedBatched(THandle handle, std::string transpose, const int M,
                        const int N, const TData alpha, const TData *a,
                        const int lda, const int strideA, const TData *x,
                        const int incx, const int strideX, const TData beta,
                        TData *y, const int incy, const int strideY,
                        const int batchSize);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>
              Enable = true>
void GemvStridedBatched(THandle handle, std::string transpose,
                        const std::int64_t M, const std::int64_t N,
                        const TData alpha, const TData *a,
                        const std::int64_t lda, const std::int64_t strideA,
                        const TData *x, const std::int64_t incx,
                        const std::int64_t strideX, const TData beta, TData *y,
                        const std::int64_t incy, const std::int64_t strideY,
                        const std::int64_t batchSize);
#endif

// GeamStridedBatched
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, blasHandle_t>, bool> Enable = true>
void GeamStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData beta, const TData *b,
                        const int ldb, const int strideB, TData *c,
                        const int ldc, const int strideC, const int batchSize);
#if defined(NEKTAR_ENABLE_SIMD)
template <
    typename THandle, typename TData,
    std::enable_if_t<std::is_same_v<THandle, xsmmHandle_t>, bool> Enable = true>
void GeamStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData beta, const TData *b,
                        const int ldb, const int strideB, TData *c,
                        const int ldc, const int strideC, const int batchSize);
#endif
template <
    typename THandle, typename TData,
#if defined(NEKTAR_USE_MAGMA)
    std::enable_if_t<std::is_same_v<THandle, magma_queue_t>, bool> Enable = true
#elif defined(NEKTAR_ENABLE_CUDA)
    std::enable_if_t<std::is_same_v<THandle, cublasHandle_t>, bool> Enable =
        true
#elif defined(NEKTAR_ENABLE_HIP)
    std::enable_if_t<std::is_same_v<THandle, hipblasHandle_t>, bool> Enable =
        true
#else
    std::enable_if_t<std::is_same_v<THandle, std::nullptr_t>, bool> Enable =
        true
#endif
    >
void GeamStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData beta, const TData *b,
                        const int ldb, const int strideB, TData *c,
                        const int ldc, const int strideC, const int batchSize);
#if defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData,
          std::enable_if_t<std::is_same_v<THandle, oneMathHandle_t>, bool>
              Enable = true>
void GeamStridedBatched(THandle handle, std::string transposeA,
                        std::string transposeB, const std::int64_t M,
                        const std::int64_t N, const TData alpha, const TData *a,
                        const std::int64_t lda, const std::int64_t strideA,
                        const TData beta, const TData *b,
                        const std::int64_t ldb, const std::int64_t strideB,
                        TData *c, const std::int64_t ldc,
                        const std::int64_t strideC,
                        const std::int64_t batchSize);
#endif
} // namespace Nektar::NekBlas
