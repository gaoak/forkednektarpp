///////////////////////////////////////////////////////////////////////////////
//
// File: MathKernels.hpp
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

#include "Operators/Field.hpp"
#include "Operators/MathKernels/MathAVXKernels.hpp"
#include "Operators/MathKernels/MathCUDAKernels.cuh"
#include "Operators/MathKernels/MathKokkosKernels.hpp"
#include "Operators/MathKernels/MathSerialKernels.hpp"

namespace Nektar //::Operators
{

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
neg(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::neg - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, WriteOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        negKernel<ExecSpace>(nElmts * nPts, xptr, yptr);
        xptr += block.block_size;
        yptr += block.block_size;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    neg(MemoryRegion<TData> &x, MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::neg - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, WriteOnly>();
    negKernel<ExecSpace>(nsize, xptr, yptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
add(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
    Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::add - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        addKernel<ExecSpace>(nElmts * nPts, xptr, yptr, zptr);
        xptr += block.block_size;
        yptr += block.block_size;
        zptr += block.block_size;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    add(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::add - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    addKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
sub(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
    Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::sub - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        subKernel<ExecSpace>(nElmts * nPts, xptr, yptr, zptr);
        xptr += block.block_size;
        yptr += block.block_size;
        zptr += block.block_size;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    sub(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::sub - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    subKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
daxpy(const TData alpha, Field<TData, TFieldState> &x,
      Field<TData, TFieldState> &y, Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::daxpy - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        daxpyKernel<ExecSpace>(nElmts * nPts, alpha, xptr, yptr, zptr);
        xptr += block.block_size;
        yptr += block.block_size;
        zptr += block.block_size;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    daxpy(const TData alpha, MemoryRegion<TData> &x, MemoryRegion<TData> &y,
          MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::daxpy - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    daxpyKernel<ExecSpace>(nsize, alpha, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
div(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
    Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        divKernel<ExecSpace>(nElmts * nPts, xptr, yptr, zptr);
        xptr += block.block_size;
        yptr += block.block_size;
        zptr += block.block_size;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    div(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    auto *zptr = z.template GetPtr<MemSpace, WriteOnly>();
    divKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
reduceSum(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = 0.0;

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        reduceSumKernel<ExecSpace>(nElmts * nPts, xptr, &reduce);
        xptr += block.block_size;
        *out += reduce;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    reduceSum(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    reduceSumKernel<ExecSpace>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
reduceMax(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = std::numeric_limits<TData>::min();

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        reduceMaxKernel<ExecSpace>(nElmts * nPts, xptr, &reduce);
        xptr += block.block_size;
        *out = std::max(*out, reduce);
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    reduceMax(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    reduceMaxKernel<ExecSpace>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
reduceMin(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = std::numeric_limits<TData>::max();

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        reduceMinKernel<ExecSpace>(nElmts * nPts, xptr, &reduce);
        xptr += block.block_size;
        *out = std::min(*out, reduce);
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    reduceMin(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    reduceMinKernel<ExecSpace>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
ddot(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = 0.0;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        ddotKernel<ExecSpace>(nElmts * nPts, xptr, yptr, &reduce);
        xptr += block.block_size;
        yptr += block.block_size;
        *out += reduce;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    ddot(MemoryRegion<TData> &x, MemoryRegion<TData> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    auto *yptr = y.template GetPtr<MemSpace, ReadOnly>();
    ddotKernel<ExecSpace>(nsize, xptr, yptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
l1norm(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = 0.0;

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        l1norm<ExecSpace>(nElmts * nPts, xptr, &reduce);
        xptr += block.block_size;
        *out += reduce;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    l1norm(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    l1norm<ExecSpace>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
l2norm(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = 0.0;

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        l2norm<ExecSpace>(nElmts * nPts, xptr, &reduce);
        xptr += block.block_size;
        *out += reduce;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    l2norm(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    l2norm<ExecSpace>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
lpnorm(const unsigned int p, Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = 0.0;

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        lpnorm<ExecSpace>(nElmts * nPts, p, xptr, &reduce);
        xptr += block.block_size;
        *out += reduce;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    lpnorm(const unsigned int p, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    lpnorm<ExecSpace>(nsize, p, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
linfnorm(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    TData reduce;
    *out = std::numeric_limits<TData>::min();

    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    for (auto const &block : x.GetBlocks())
    {
        auto nElmts = block.num_elements;
        auto nPts   = block.num_pts;
        linfnorm<ExecSpace>(nElmts * nPts, xptr, &reduce);
        xptr += block.block_size;
        *out = std::max(*out, reduce);
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    linfnorm(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto nsize = x.size();
    auto *xptr = x.template GetPtr<MemSpace, ReadOnly>();
    linfnorm<ExecSpace>(nsize, xptr, out);
}

} // namespace Nektar
