///////////////////////////////////////////////////////////////////////////////
//
// File: neon.hpp
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
// Description: Vector type using Arm NEON (AArch64) extension.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_LIBUTILITES_SIMDLIB_NEON_H
#define NEKTAR_LIB_LIBUTILITES_SIMDLIB_NEON_H

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "allocator.hpp"
#include "traits.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

namespace tinysimd::abi
{

template <typename scalarType, int width = 0> struct neon
{
    using type = void;
};

} // namespace tinysimd::abi

// NEON on AArch64 is the only variant carrying full IEEE double precision
// (the 32 bit A-profile NEON has no float64x2_t), so restrict to AArch64.
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(NEKTAR_ENABLE_SIMD_NEON)

namespace tinysimd
{

// forward declaration of concrete types
template <typename T> struct neonInt2;
template <typename T> struct neonInt4;
template <typename T> struct neonLong2;
struct neonDouble2;
struct neonFloat4;
struct neonMask2;
struct neonMask4;

namespace abi
{

// mapping between abstract types and concrete floating point types
template <> struct neon<double>
{
    using type = neonDouble2;
};
template <> struct neon<float>
{
    using type = neonFloat4;
};
// generic index mapping
// assumes index type width same as floating point type
template <> struct neon<std::int64_t>
{
    using type = neonLong2<std::int64_t>;
};
template <> struct neon<std::uint64_t>
{
    using type = neonLong2<std::uint64_t>;
};
#if defined(__APPLE__)
template <> struct neon<std::size_t>
{
    using type = neonLong2<std::size_t>;
};
#endif
template <> struct neon<std::int32_t>
{
    using type = neonInt4<std::int32_t>;
};
template <> struct neon<std::uint32_t>
{
    using type = neonInt4<std::uint32_t>;
};
// specialized index mapping
template <> struct neon<std::int64_t, 2>
{
    using type = neonLong2<std::int64_t>;
};
template <> struct neon<std::uint64_t, 2>
{
    using type = neonLong2<std::uint64_t>;
};
#if defined(__APPLE__)
template <> struct neon<std::size_t, 2>
{
    using type = neonLong2<std::size_t>;
};
#endif
template <> struct neon<std::int32_t, 2>
{
    using type = neonInt2<std::int32_t>;
};
template <> struct neon<std::uint32_t, 2>
{
    using type = neonInt2<std::uint32_t>;
};
template <> struct neon<std::int32_t, 4>
{
    using type = neonInt4<std::int32_t>;
};
template <> struct neon<std::uint32_t, 4>
{
    using type = neonInt4<std::uint32_t>;
};
// bool mapping
template <> struct neon<bool, 2>
{
    using type = neonMask2;
};
template <> struct neon<bool, 4>
{
    using type = neonMask4;
};

} // namespace abi

// concrete types

// 2 lanes of 32 bits packed in a 64 bit D register; used as the index type
// when gathering/scattering doubles with 32 bit indices.
template <typename T> struct neonInt2
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 4,
                  "4 bytes Integral required.");

    static constexpr unsigned int width     = 2;
    static constexpr unsigned int alignment = 8;

    using scalarType  = T;
    using vectorType  = int32x2_t;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline neonInt2()                    = default;
    inline neonInt2(const neonInt2 &rhs) = default;
    inline neonInt2(const vectorType &rhs) : _data(rhs)
    {
    }
    inline neonInt2(const scalarType rhs)
    {
        _data = vdup_n_s32(static_cast<std::int32_t>(rhs));
    }
    explicit inline neonInt2(scalarArray &rhs)
    {
        _data = vld1_s32(reinterpret_cast<const std::int32_t *>(rhs));
    }

    // copy assignment
    inline neonInt2 &operator=(const neonInt2 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        vst1_s32(reinterpret_cast<std::int32_t *>(p), _data);
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1_s32(reinterpret_cast<std::int32_t *>(p), _data);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1_s32(reinterpret_cast<std::int32_t *>(p), _data);
    }

    inline void load(const scalarType *p)
    {
        _data = vld1_s32(reinterpret_cast<const std::int32_t *>(p));
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1_s32(reinterpret_cast<const std::int32_t *>(p));
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1_s32(reinterpret_cast<const std::int32_t *>(p));
    }

    inline void broadcast(const scalarType rhs)
    {
        _data = vdup_n_s32(static_cast<std::int32_t>(rhs));
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }
};

template <typename T>
inline neonInt2<T> operator+(neonInt2<T> lhs, neonInt2<T> rhs)
{
    return vadd_s32(lhs._data, rhs._data);
}

template <typename T, typename U,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> Enable = true>
inline neonInt2<T> operator+(neonInt2<T> lhs, U rhs)
{
    return vadd_s32(lhs._data, vdup_n_s32(static_cast<std::int32_t>(rhs)));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct neonInt4
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 4,
                  "4 bytes Integral required.");

    static constexpr unsigned int width     = 4;
    static constexpr unsigned int alignment = 16;

    using scalarType  = T;
    using vectorType  = int32x4_t;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline neonInt4()                    = default;
    inline neonInt4(const neonInt4 &rhs) = default;
    inline neonInt4(const vectorType &rhs) : _data(rhs)
    {
    }
    inline neonInt4(const scalarType rhs)
    {
        _data = vdupq_n_s32(static_cast<std::int32_t>(rhs));
    }
    explicit inline neonInt4(scalarArray &rhs)
    {
        _data = vld1q_s32(reinterpret_cast<const std::int32_t *>(rhs));
    }

    // copy assignment
    inline neonInt4 &operator=(const neonInt4 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        vst1q_s32(reinterpret_cast<std::int32_t *>(p), _data);
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_s32(reinterpret_cast<std::int32_t *>(p), _data);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_s32(reinterpret_cast<std::int32_t *>(p), _data);
    }

    inline void load(const scalarType *p)
    {
        _data = vld1q_s32(reinterpret_cast<const std::int32_t *>(p));
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_s32(reinterpret_cast<const std::int32_t *>(p));
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_s32(reinterpret_cast<const std::int32_t *>(p));
    }

    inline void broadcast(const scalarType rhs)
    {
        _data = vdupq_n_s32(static_cast<std::int32_t>(rhs));
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }
};

template <typename T>
inline neonInt4<T> operator+(neonInt4<T> lhs, neonInt4<T> rhs)
{
    return vaddq_s32(lhs._data, rhs._data);
}

template <typename T, typename U,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> Enable = true>
inline neonInt4<T> operator+(neonInt4<T> lhs, U rhs)
{
    return vaddq_s32(lhs._data, vdupq_n_s32(static_cast<std::int32_t>(rhs)));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct neonLong2
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 8,
                  "8 bytes Integral required.");

    static constexpr unsigned int width     = 2;
    static constexpr unsigned int alignment = 16;

    using scalarType  = T;
    using vectorType  = int64x2_t;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline neonLong2()                     = default;
    inline neonLong2(const neonLong2 &rhs) = default;
    inline neonLong2(const vectorType &rhs) : _data(rhs)
    {
    }
    inline neonLong2(const scalarType rhs)
    {
        _data = vdupq_n_s64(static_cast<std::int64_t>(rhs));
    }
    explicit inline neonLong2(scalarArray &rhs)
    {
        _data = vld1q_s64(reinterpret_cast<const std::int64_t *>(rhs));
    }

    // copy assignment
    inline neonLong2 &operator=(const neonLong2 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        vst1q_s64(reinterpret_cast<std::int64_t *>(p), _data);
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_s64(reinterpret_cast<std::int64_t *>(p), _data);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_s64(reinterpret_cast<std::int64_t *>(p), _data);
    }

    inline void load(const scalarType *p)
    {
        _data = vld1q_s64(reinterpret_cast<const std::int64_t *>(p));
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_s64(reinterpret_cast<const std::int64_t *>(p));
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_s64(reinterpret_cast<const std::int64_t *>(p));
    }

    inline void broadcast(const scalarType rhs)
    {
        _data = vdupq_n_s64(static_cast<std::int64_t>(rhs));
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }
};

template <typename T>
inline neonLong2<T> operator+(neonLong2<T> lhs, neonLong2<T> rhs)
{
    return vaddq_s64(lhs._data, rhs._data);
}

template <typename T, typename U,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> Enable = true>
inline neonLong2<T> operator+(neonLong2<T> lhs, U rhs)
{
    return vaddq_s64(lhs._data, vdupq_n_s64(static_cast<std::int64_t>(rhs)));
}

////////////////////////////////////////////////////////////////////////////////

struct neonDouble2
{
    static constexpr unsigned width     = 2;
    static constexpr unsigned alignment = 16;

    using scalarType      = double;
    using scalarIndexType = std::uint64_t;
    using vectorType      = float64x2_t;
    using scalarArray     = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline neonDouble2()                       = default;
    inline neonDouble2(const neonDouble2 &rhs) = default;
    inline neonDouble2(const vectorType &rhs) : _data(rhs)
    {
    }
    inline neonDouble2(const scalarType rhs)
    {
        _data = vdupq_n_f64(rhs);
    }

    // copy assignment
    inline neonDouble2 &operator=(const neonDouble2 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        vst1q_f64(p, _data);
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_f64(p, _data);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_f64(p, _data);
    }

    // NEON has no non-temporal store intrinsic, fall back to a regular store
    template <class flag,
              std::enable_if_t<is_streaming_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_f64(p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = vld1q_f64(p);
    }

    template <class flag, std::enable_if_t<is_requiring_alignment_v<flag>, bool>
                              Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_f64(p);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_f64(p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = vdupq_n_f64(rhs);
    }

    // gather/scatter with 32 bit indices
    template <typename T>
    inline void gather(scalarType const *p, const neonInt2<T> &indices)
    {
        // no gather intrinsics for NEON
        _data = vsetq_lane_f64(
            p[static_cast<T>(vget_lane_s32(indices._data, 0))], _data, 0);
        _data = vsetq_lane_f64(
            p[static_cast<T>(vget_lane_s32(indices._data, 1))], _data, 1);
    }

    template <typename T>
    inline void scatter(scalarType *out, const neonInt2<T> &indices) const
    {
        // no scatter intrinsics for NEON
        out[static_cast<T>(vget_lane_s32(indices._data, 0))] =
            vgetq_lane_f64(_data, 0);
        out[static_cast<T>(vget_lane_s32(indices._data, 1))] =
            vgetq_lane_f64(_data, 1);
    }

    // gather/scatter with 64 bit indices
    template <typename T>
    inline void gather(scalarType const *p, const neonLong2<T> &indices)
    {
        // no gather intrinsics for NEON
        _data = vsetq_lane_f64(
            p[static_cast<T>(vgetq_lane_s64(indices._data, 0))], _data, 0);
        _data = vsetq_lane_f64(
            p[static_cast<T>(vgetq_lane_s64(indices._data, 1))], _data, 1);
    }

    template <typename T>
    inline void scatter(scalarType *out, const neonLong2<T> &indices) const
    {
        // no scatter intrinsics for NEON
        out[static_cast<T>(vgetq_lane_s64(indices._data, 0))] =
            vgetq_lane_f64(_data, 0);
        out[static_cast<T>(vgetq_lane_s64(indices._data, 1))] =
            vgetq_lane_f64(_data, 1);
    }

    // fma
    // this = this + a * b
    inline void fma(const neonDouble2 &a, const neonDouble2 &b)
    {
        _data = vfmaq_f64(_data, a._data, b._data);
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }

    // unary ops
    inline void operator+=(neonDouble2 rhs)
    {
        _data = vaddq_f64(_data, rhs._data);
    }

    inline void operator-=(neonDouble2 rhs)
    {
        _data = vsubq_f64(_data, rhs._data);
    }

    inline void operator*=(neonDouble2 rhs)
    {
        _data = vmulq_f64(_data, rhs._data);
    }

    inline void operator/=(neonDouble2 rhs)
    {
        _data = vdivq_f64(_data, rhs._data);
    }
};

inline neonDouble2 operator+(neonDouble2 lhs, neonDouble2 rhs)
{
    return vaddq_f64(lhs._data, rhs._data);
}

inline neonDouble2 operator-(neonDouble2 lhs, neonDouble2 rhs)
{
    return vsubq_f64(lhs._data, rhs._data);
}

inline neonDouble2 operator-(neonDouble2 in)
{
    return vnegq_f64(in._data);
}

inline neonDouble2 operator*(neonDouble2 lhs, neonDouble2 rhs)
{
    return vmulq_f64(lhs._data, rhs._data);
}

inline neonDouble2 operator/(neonDouble2 lhs, neonDouble2 rhs)
{
    return vdivq_f64(lhs._data, rhs._data);
}

inline neonDouble2 sqrt(neonDouble2 in)
{
    return vsqrtq_f64(in._data);
}

inline neonDouble2 abs(neonDouble2 in)
{
    return vabsq_f64(in._data);
}

inline neonDouble2 min(neonDouble2 lhs, neonDouble2 rhs)
{
    return vminq_f64(lhs._data, rhs._data);
}

inline neonDouble2 max(neonDouble2 lhs, neonDouble2 rhs)
{
    return vmaxq_f64(lhs._data, rhs._data);
}

inline neonDouble2 log(neonDouble2 in)
{
    // there is no NEON log intrinsic
    // this is a dreadful implementation and is simply a stop gap measure
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp;
    in.store(tmp);
    tmp[0] = std::log(tmp[0]);
    tmp[1] = std::log(tmp[1]);
    neonDouble2 ret;
    ret.load(tmp);
    return ret;
}

inline void load_unalign_interleave(
    const double *in, const std::uint32_t dataLen,
    std::vector<neonDouble2, allocator<neonDouble2>> &out)
{
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        tmp[0] = in[i];
        tmp[1] = in[i + dataLen];
        out[i].load(tmp);
    }
}

inline void load_unalign_interleave_skipPads(
    const double *in, const std::uint32_t dataLen, const std::uint32_t skipPads,
    std::vector<neonDouble2, allocator<neonDouble2>> &out)
{
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp;
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp1;
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp2;
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp3;

    size_t nBlocks = dataLen / 4;
    const neonDouble2 zero{0.0};

    for (size_t i = 0; i < nBlocks; ++i)
    {
        zero.store(tmp);
        zero.store(tmp1);
        zero.store(tmp2);
        zero.store(tmp3);
        for (size_t j = 0; j < neonDouble2::width - skipPads; ++j)
        {
            tmp[j]  = in[j * dataLen + 4 * i];
            tmp1[j] = in[j * dataLen + 4 * i + 1];
            tmp2[j] = in[j * dataLen + 4 * i + 2];
            tmp3[j] = in[j * dataLen + 4 * i + 3];
        }
        out[4 * i].load(tmp);
        out[4 * i + 1].load(tmp1);
        out[4 * i + 2].load(tmp2);
        out[4 * i + 3].load(tmp3);
    }

    for (size_t i = nBlocks * 4; i < dataLen; ++i)
    {
        zero.store(tmp);
        for (size_t j = 0; j < neonDouble2::width - skipPads; ++j)
        {
            tmp[j] = in[i + j * dataLen];
        }
        out[i].load(tmp);
    }
}

inline void load_interleave(
    const double *in, std::uint32_t dataLen,
    std::vector<neonDouble2, allocator<neonDouble2>> &out)
{
    alignas(neonDouble2::alignment)
        size_t tmp[neonDouble2::width] = {0, dataLen};
    using index_t                      = neonLong2<size_t>;
    index_t index0(tmp);
    index_t index1 = index0 + 1;

    // 2x unrolled loop
    constexpr uint16_t unrl = 2;
    size_t nBlocks          = dataLen / unrl;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        out[unrl * i + 0].gather(in, index0);
        out[unrl * i + 1].gather(in, index1);
        index0 = index0 + unrl;
        index1 = index1 + unrl;
    }

    // spillover loop
    for (size_t i = unrl * nBlocks; i < dataLen; ++i)
    {
        out[i].gather(in, index0);
        index0 = index0 + 1;
    }
}

inline void deinterleave_unalign_store(
    const std::vector<neonDouble2, allocator<neonDouble2>> &in,
    const std::uint32_t dataLen, double *out)
{
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].store(tmp);
        out[i]           = tmp[0];
        out[i + dataLen] = tmp[1];
    }
}

inline void deinterleave_unalign_store_skipPads(
    const std::vector<neonDouble2, allocator<neonDouble2>> &in,
    const std::uint32_t dataLen, const std::uint32_t skipPads, double *out)
{
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp;
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp1;
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp2;
    alignas(neonDouble2::alignment) neonDouble2::scalarArray tmp3;

    size_t nBlocks = dataLen / 4;

    for (size_t i = 0; i < nBlocks; ++i)
    {
        in[4 * i].store(tmp);
        in[4 * i + 1].store(tmp1);
        in[4 * i + 2].store(tmp2);
        in[4 * i + 3].store(tmp3);
        for (size_t j = 0; j < neonDouble2::width - skipPads; ++j)
        {
            out[j * dataLen + 4 * i]     = tmp[j];
            out[j * dataLen + 4 * i + 1] = tmp1[j];
            out[j * dataLen + 4 * i + 2] = tmp2[j];
            out[j * dataLen + 4 * i + 3] = tmp3[j];
        }
    }

    // spill over loop
    for (size_t i = nBlocks * 4; i < dataLen; ++i)
    {
        in[i].store(tmp);
        for (size_t j = 0; j < neonDouble2::width - skipPads; ++j)
        {
            out[j * dataLen + i] = tmp[j];
        }
    }
}

inline void deinterleave_store(
    const std::vector<neonDouble2, allocator<neonDouble2>> &in,
    std::uint32_t dataLen, double *out)
{
    alignas(neonDouble2::alignment)
        size_t tmp[neonDouble2::width] = {0, dataLen};
    using index_t                      = neonLong2<size_t>;
    index_t index0(tmp);

    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].scatter(out, index0);
        index0 = index0 + 1;
    }
}

//////////////////////////////////////////////////////////////////////////////

struct neonFloat4
{
    static constexpr unsigned width     = 4;
    static constexpr unsigned alignment = 16;

    using scalarType      = float;
    using scalarIndexType = std::uint32_t;
    using vectorType      = float32x4_t;
    using scalarArray     = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline neonFloat4()                      = default;
    inline neonFloat4(const neonFloat4 &rhs) = default;
    inline neonFloat4(const vectorType &rhs) : _data(rhs)
    {
    }
    inline neonFloat4(const scalarType rhs)
    {
        _data = vdupq_n_f32(rhs);
    }

    // copy assignment
    inline neonFloat4 &operator=(const neonFloat4 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        vst1q_f32(p, _data);
    }

    template <class flag,
              std::enable_if_t<
                  is_requiring_alignment_v<flag> && !is_streaming_v<flag>, bool>
                  Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_f32(p, _data);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_f32(p, _data);
    }

    // NEON has no non-temporal store intrinsic, fall back to a regular store
    template <class flag,
              std::enable_if_t<is_streaming_v<flag>, bool> Enable = true>
    inline void store(scalarType *p, flag) const
    {
        vst1q_f32(p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = vld1q_f32(p);
    }

    template <class flag, std::enable_if_t<is_requiring_alignment_v<flag>, bool>
                              Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_f32(p);
    }

    template <
        class flag,
        std::enable_if_t<!is_requiring_alignment_v<flag>, bool> Enable = true>
    inline void load(const scalarType *p, flag)
    {
        _data = vld1q_f32(p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = vdupq_n_f32(rhs);
    }

    // gather/scatter with 32 bit indices
    template <typename T>
    inline void gather(scalarType const *p, const neonInt4<T> &indices)
    {
        // no gather intrinsics for NEON
        _data = vsetq_lane_f32(
            p[static_cast<T>(vgetq_lane_s32(indices._data, 0))], _data, 0);
        _data = vsetq_lane_f32(
            p[static_cast<T>(vgetq_lane_s32(indices._data, 1))], _data, 1);
        _data = vsetq_lane_f32(
            p[static_cast<T>(vgetq_lane_s32(indices._data, 2))], _data, 2);
        _data = vsetq_lane_f32(
            p[static_cast<T>(vgetq_lane_s32(indices._data, 3))], _data, 3);
    }

    template <typename T>
    inline void scatter(scalarType *out, const neonInt4<T> &indices) const
    {
        // no scatter intrinsics for NEON
        out[static_cast<T>(vgetq_lane_s32(indices._data, 0))] =
            vgetq_lane_f32(_data, 0);
        out[static_cast<T>(vgetq_lane_s32(indices._data, 1))] =
            vgetq_lane_f32(_data, 1);
        out[static_cast<T>(vgetq_lane_s32(indices._data, 2))] =
            vgetq_lane_f32(_data, 2);
        out[static_cast<T>(vgetq_lane_s32(indices._data, 3))] =
            vgetq_lane_f32(_data, 3);
    }

    // fma
    // this = this + a * b
    inline void fma(const neonFloat4 &a, const neonFloat4 &b)
    {
        _data = vfmaq_f32(_data, a._data, b._data);
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }

    inline void operator+=(neonFloat4 rhs)
    {
        _data = vaddq_f32(_data, rhs._data);
    }

    inline void operator-=(neonFloat4 rhs)
    {
        _data = vsubq_f32(_data, rhs._data);
    }

    inline void operator*=(neonFloat4 rhs)
    {
        _data = vmulq_f32(_data, rhs._data);
    }

    inline void operator/=(neonFloat4 rhs)
    {
        _data = vdivq_f32(_data, rhs._data);
    }
};

inline neonFloat4 operator+(neonFloat4 lhs, neonFloat4 rhs)
{
    return vaddq_f32(lhs._data, rhs._data);
}

inline neonFloat4 operator-(neonFloat4 lhs, neonFloat4 rhs)
{
    return vsubq_f32(lhs._data, rhs._data);
}

inline neonFloat4 operator-(neonFloat4 in)
{
    return vnegq_f32(in._data);
}

inline neonFloat4 operator*(neonFloat4 lhs, neonFloat4 rhs)
{
    return vmulq_f32(lhs._data, rhs._data);
}

inline neonFloat4 operator/(neonFloat4 lhs, neonFloat4 rhs)
{
    return vdivq_f32(lhs._data, rhs._data);
}

inline neonFloat4 sqrt(neonFloat4 in)
{
    return vsqrtq_f32(in._data);
}

inline neonFloat4 abs(neonFloat4 in)
{
    return vabsq_f32(in._data);
}

inline neonFloat4 min(neonFloat4 lhs, neonFloat4 rhs)
{
    return vminq_f32(lhs._data, rhs._data);
}

inline neonFloat4 max(neonFloat4 lhs, neonFloat4 rhs)
{
    return vmaxq_f32(lhs._data, rhs._data);
}

inline neonFloat4 log(neonFloat4 in)
{
    // there is no NEON log intrinsic
    // this is a dreadful implementation and is simply a stop gap measure
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp;
    in.store(tmp);
    tmp[0] = std::log(tmp[0]);
    tmp[1] = std::log(tmp[1]);
    tmp[2] = std::log(tmp[2]);
    tmp[3] = std::log(tmp[3]);
    neonFloat4 ret;
    ret.load(tmp);
    return ret;
}

inline void load_unalign_interleave(
    const double *in, const std::uint32_t dataLen,
    std::vector<neonFloat4, allocator<neonFloat4>> &out)
{
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        tmp[0] = in[i];
        tmp[1] = in[i + dataLen];
        tmp[2] = in[i + 2 * dataLen];
        tmp[3] = in[i + 3 * dataLen];
        out[i].load(tmp);
    }
}

inline void load_unalign_interleave_skipPads(
    const double *in, const std::uint32_t dataLen, const std::uint32_t skipPads,
    std::vector<neonFloat4, allocator<neonFloat4>> &out)
{
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp;
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp1;
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp2;
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp3;

    size_t nBlocks = dataLen / 4;
    const neonFloat4 zero{0.0f};

    for (size_t i = 0; i < nBlocks; ++i)
    {
        zero.store(tmp);
        zero.store(tmp1);
        zero.store(tmp2);
        zero.store(tmp3);
        for (size_t j = 0; j < neonFloat4::width - skipPads; ++j)
        {
            tmp[j]  = in[j * dataLen + 4 * i];
            tmp1[j] = in[j * dataLen + 4 * i + 1];
            tmp2[j] = in[j * dataLen + 4 * i + 2];
            tmp3[j] = in[j * dataLen + 4 * i + 3];
        }
        out[4 * i].load(tmp);
        out[4 * i + 1].load(tmp1);
        out[4 * i + 2].load(tmp2);
        out[4 * i + 3].load(tmp3);
    }

    for (size_t i = nBlocks * 4; i < dataLen; ++i)
    {
        zero.store(tmp);
        for (size_t j = 0; j < neonFloat4::width - skipPads; ++j)
        {
            tmp[j] = in[i + j * dataLen];
        }
        out[i].load(tmp);
    }
}

inline void load_interleave(const float *in, std::uint32_t dataLen,
                            std::vector<neonFloat4, allocator<neonFloat4>> &out)
{
    alignas(neonFloat4::alignment) neonFloat4::scalarIndexType tmp[4] = {
        0, dataLen, 2 * dataLen, 3 * dataLen};

    using index_t = neonInt4<neonFloat4::scalarIndexType>;
    index_t index0(tmp);
    index_t index1 = index0 + 1;

    // 2x unrolled loop
    size_t nBlocks = dataLen / 2;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        out[2 * i + 0].gather(in, index0);
        out[2 * i + 1].gather(in, index1);
        index0 = index0 + 2;
        index1 = index1 + 2;
    }

    // spillover loop
    for (size_t i = 2 * nBlocks; i < dataLen; ++i)
    {
        out[i].gather(in, index0);
        index0 = index0 + 1;
    }
}

inline void deinterleave_unalign_store(
    const std::vector<neonFloat4, allocator<neonFloat4>> &in,
    const std::uint32_t dataLen, double *out)
{
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].store(tmp);
        out[i]               = tmp[0];
        out[i + dataLen]     = tmp[1];
        out[i + 2 * dataLen] = tmp[2];
        out[i + 3 * dataLen] = tmp[3];
    }
}

inline void deinterleave_unalign_store_skipPads(
    const std::vector<neonFloat4, allocator<neonFloat4>> &in,
    const std::uint32_t dataLen, const std::uint32_t skipPads, double *out)
{
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp;
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp1;
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp2;
    alignas(neonFloat4::alignment) neonFloat4::scalarArray tmp3;

    size_t nBlocks = dataLen / 4;

    for (size_t i = 0; i < nBlocks; ++i)
    {
        in[4 * i].store(tmp);
        in[4 * i + 1].store(tmp1);
        in[4 * i + 2].store(tmp2);
        in[4 * i + 3].store(tmp3);
        for (size_t j = 0; j < neonFloat4::width - skipPads; ++j)
        {
            out[j * dataLen + 4 * i]     = tmp[j];
            out[j * dataLen + 4 * i + 1] = tmp1[j];
            out[j * dataLen + 4 * i + 2] = tmp2[j];
            out[j * dataLen + 4 * i + 3] = tmp3[j];
        }
    }

    // spill over loop
    for (size_t i = nBlocks * 4; i < dataLen; ++i)
    {
        in[i].store(tmp);
        for (size_t j = 0; j < neonFloat4::width - skipPads; ++j)
        {
            out[j * dataLen + i] = tmp[j];
        }
    }
}

inline void deinterleave_store(
    const std::vector<neonFloat4, allocator<neonFloat4>> &in,
    std::uint32_t dataLen, float *out)
{
    alignas(neonFloat4::alignment) neonFloat4::scalarIndexType tmp[4] = {
        0, dataLen, 2 * dataLen, 3 * dataLen};
    using index_t = neonInt4<neonFloat4::scalarIndexType>;
    index_t index0(tmp);

    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].scatter(out, index0);
        index0 = index0 + 1;
    }
}

////////////////////////////////////////////////////////////////////////////////

// mask type
// mask is a int type with special properties (broad boolean vector)
// broad boolean vectors defined and allowed values are:
// false=0x0 and true=0xFFFFFFFF
//
// VERY LIMITED SUPPORT...just enough to make cubic eos work...
//
struct neonMask2 : neonLong2<std::uint64_t>
{
    // bring in ctors
    using neonLong2::neonLong2;

    static constexpr scalarType true_v  = -1;
    static constexpr scalarType false_v = 0;
};

inline neonMask2 operator>(neonDouble2 lhs, neonDouble2 rhs)
{
    return vreinterpretq_s64_u64(vcgtq_f64(lhs._data, rhs._data));
}

inline bool operator&&(neonMask2 lhs, bool rhs)
{
    // all lanes are true iff every bit of the mask is set
    bool tmp = vminvq_u32(vreinterpretq_u32_s64(lhs._data)) == 0xFFFFFFFFu;

    return tmp && rhs;
}

struct neonMask4 : neonInt4<std::uint32_t>
{
    // bring in ctors
    using neonInt4::neonInt4;

    static constexpr scalarType true_v  = -1;
    static constexpr scalarType false_v = 0;
};

inline neonMask4 operator>(neonFloat4 lhs, neonFloat4 rhs)
{
    return vreinterpretq_s32_u32(vcgtq_f32(lhs._data, rhs._data));
}

inline bool operator&&(neonMask4 lhs, bool rhs)
{
    // all lanes are true iff every bit of the mask is set
    bool tmp = vminvq_u32(vreinterpretq_u32_s32(lhs._data)) == 0xFFFFFFFFu;

    return tmp && rhs;
}

} // namespace tinysimd

#endif // defined(__aarch64__) && defined(__ARM_NEON) &&
       // defined(NEKTAR_ENABLE_SIMD_NEON)
#endif
