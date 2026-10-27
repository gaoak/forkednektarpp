///////////////////////////////////////////////////////////////////////////////
//
// File: allocator.hpp
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
// Description: Vector type custom allocator. It is needed to allocate vectors
// to the required alignment.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_LIBUTILITES_SIMDLIB_ALLOCATOR_H
#define NEKTAR_LIB_LIBUTILITES_SIMDLIB_ALLOCATOR_H

#include "traits.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace tinysimd
{

template <typename T, std::size_t Alignment> class aligned_allocator
{
public:
    using value_type      = T;
    using pointer         = T *;
    using const_pointer   = const T *;
    using reference       = T &;
    using const_reference = const T &;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal                        = std::true_type;

    static constexpr std::size_t alignment = std::max(Alignment, alignof(T));
    static_assert((alignment & (alignment - 1)) == 0,
                  "Alignment must be a power of two.");

    // Needed by std::allocator_traits as Alignment is a non-type parameter.
    template <typename U> struct rebind
    {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;

    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment> &) noexcept
    {
    }

    T *allocate(size_type n)
    {
        if (n > max_size())
        {
            throw std::bad_array_new_length();
        }
        return static_cast<T *>(
            ::operator new(n * sizeof(T), std::align_val_t{alignment}));
    }

    void deallocate(T *p, size_type) noexcept
    {
        if (p != nullptr)
        {
            ::operator delete(p, std::align_val_t{alignment});
        }
    }

    constexpr size_type max_size() const noexcept
    {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

// Disable array-bounds and stringop-overflow warnings as GCC incorrectly
// triggers them on the zero-length path of std::vector value-initialisation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
    template <typename U, typename... Args> void construct(U *p, Args &&...args)
    {
        ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

    template <typename U> void destroy(U *p)
    {
        p->~U();
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment> &,
                const aligned_allocator<U, Alignment> &) noexcept
{
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment> &,
                const aligned_allocator<U, Alignment> &) noexcept
{
    return false;
}

// should be enabled only for vector types
template <typename T, std::enable_if_t<is_vector_v<T>, bool> Enable = true>
using allocator = aligned_allocator<T, T::alignment>;

} // namespace tinysimd

#endif
