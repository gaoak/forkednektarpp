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
#if defined(NEKTAR_ENABLE_CUDA)
#include "Operators/NekBlas/cuBlasHandle.hpp"
#elif defined(NEKTAR_ENABLE_HIP)
#include "Operators/NekBlas/hipBlasHandle.hpp"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Common/SYCLQueue.hpp"
#endif

#include "Operators/Common/Spaces.hpp"

struct blasHandle
{
};

template <typename ExecSpace> class NekHandle
{
public:
    static blasHandle GetInstance(void)
    {
        return blasHandle();
    }
};

#if defined(NEKTAR_ENABLE_CUDA)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    static cublasHandle_t GetInstance(void)
    {
        return cuBlasHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_HIP)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    static hipblasHandle_t GetInstance(void)
    {
        return hipBlasHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_SYCL)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    static sycl::queue GetInstance(void)
    {
        return SYCLQueue::GetInstance();
    }
};
#endif

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB,
    const size_t M, const size_t N, const size_t K, const TData alpha,
    const TData *a, const size_t lda, const TData *b, const size_t ldb,
    const TData beta, TData *c, const size_t ldc);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_SYCL)
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemm(THandle handle, std::string transposeA, std::string transposeB,
        const size_t M, const size_t N, const size_t K, const TData alpha,
        const TData *a, const size_t lda, const TData *b, const size_t ldb,
        const TData beta, TData *c, const size_t ldc);

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const size_t M, const size_t N,
                      const size_t K, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *b,
                      const size_t ldb, const size_t strideB, const TData beta,
                      TData *c, const size_t ldc, const size_t strideC,
                      const size_t batchSize);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_SYCL)
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const size_t M, const size_t N,
                      const size_t K, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *b,
                      const size_t ldb, const size_t strideB, const TData beta,
                      TData *c, const size_t ldc, const size_t strideC,
                      const size_t batchSize);

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle>, void>::type NekGemv(
    THandle handle, std::string transpose, const size_t M, const size_t N,
    const TData alpha, const TData *a, const size_t lda, const TData *x,
    const size_t incx, const TData beta, TData *y, const size_t incy);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_SYCL)
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemv(THandle handle, std::string transpose, const size_t M, const size_t N,
        const TData alpha, const TData *a, const size_t lda, const TData *x,
        const size_t incx, const TData beta, TData *y, const size_t incy);

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle>, void>::type
NekGemvStridedBatched(THandle handle, std::string transpose, const size_t M,
                      const size_t N, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *x,
                      const size_t incx, const size_t strideX, const TData beta,
                      TData *y, const size_t incy, const size_t strideY,
                      const size_t batchSize);
template <typename THandle, typename TData>
#if defined(NEKTAR_ENABLE_CUDA)
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_HIP)
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
#elif defined(NEKTAR_ENABLE_SYCL)
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
#else
typename std::enable_if<std::is_same_v<THandle, std::nullptr_t>, void>::type
#endif
NekGemvStridedBatched(THandle handle, std::string transpose, const size_t M,
                      const size_t N, const TData alpha, const TData *a,
                      const size_t lda, const size_t strideA, const TData *x,
                      const size_t incx, const size_t strideX, const TData beta,
                      TData *y, const size_t incy, const size_t strideY,
                      const size_t batchSize);
