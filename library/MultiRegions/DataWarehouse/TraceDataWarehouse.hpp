///////////////////////////////////////////////////////////////////////////////
//
// File: TraceDataWarehouse.hpp
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

#include <LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp>
#include <LibUtilities/Foundations/Interp.h>
#include <LibUtilities/Foundations/Points.h>
#include <MultiRegions/AssemblyMap/LocTraceToTraceMap.h>

namespace Nektar::MultiRegions
{

class ExpList;

/// Shared pointer to an ExpList object.
typedef std::shared_ptr<ExpList> ExpListSharedPtr;

class TraceEssentialCreator;

enum class IPTraceScalarData
{
    BwdWeightAver,
    BwdWeightJump,
    LengthRecip,
    PenaltyFactor
};

template <typename TData> class IPTraceNormalKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~IPTraceNormalKey() override = default;

    IPTraceNormalKey(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "IPTraceNormalKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class IPTraceScalarKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~IPTraceScalarKey() override = default;

    IPTraceScalarKey(const unsigned int block_idx, const IPTraceScalarData type)
        : m_block_idx(block_idx), m_type(type)
    {
        hash_combine(m_hash, m_block_idx, static_cast<unsigned int>(m_type),
                     typeid(value_type).name(), "IPTraceScalarKey");
    }

private:
    unsigned int m_block_idx;
    IPTraceScalarData m_type;
};

template <typename TData>
class IPTraceDerivBaseKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~IPTraceDerivBaseKey() override = default;

    // Stores the physical derivative of each elemental basis function on the
    // local trace, multiplied by the trace quadrature metric. The scalar IP
    // symmetric term uses this as:
    //   sum_p jump(p) * (D n)_d(p) * dphi_i/dx_d(p) * wJ_trace(p).
    // Layout is [direction][coefficient][local trace point] for one element
    // block shape.
    IPTraceDerivBaseKey(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "IPTraceDerivBaseKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class LocTracePhysToElmtMapsKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~LocTracePhysToElmtMapsKey() override = default;

    LocTracePhysToElmtMapsKey(const unsigned int block_idx,
                              const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "LocTracePhysToElmtMapsKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class OrientationMapsKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~OrientationMapsKey() override = default;

    OrientationMapsKey(const unsigned int block_idx,
                       const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "OrientationMapsKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class OrientationMapsOffsetKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef size_t value_type;

    ~OrientationMapsOffsetKey() override = default;

    OrientationMapsOffsetKey(const unsigned int block_idx,
                             const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "OrientationMapsOffsetKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class LocToTracePhysOffsetKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef size_t value_type;

    ~LocToTracePhysOffsetKey() override = default;

    LocToTracePhysOffsetKey(const unsigned int block_idx,
                            const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "LocToTracePhysOffsetKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class IsLocTraceLeftAdjacentKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef bool value_type;

    ~IsLocTraceLeftAdjacentKey() override = default;

    IsLocTraceLeftAdjacentKey(const unsigned int block_idx,
                              const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "IsLocTraceLeftAdjacentKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData>
class InterpTraceIndexKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpTraceIndexKey() override = default;

    InterpTraceIndexKey(const unsigned int block_idx,
                        const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "InterpTraceIndexKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class InterpPointsKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpPointsKey() override = default;

    InterpPointsKey(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpPointsKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class InterpTypesKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpTypesKey() override = default;

    InterpTypesKey(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpTypesKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class QuadRangeKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~QuadRangeKey() override = default;

    QuadRangeKey(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "QuadRangeKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class InterpTraceKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef InterpLocTraceToTrace value_type;

    ~InterpTraceKey() override = default;

    InterpTraceKey(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpTraceKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class InterpTraceI0Key : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~InterpTraceI0Key() override = default;

    InterpTraceI0Key(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpTraceI0Key");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class InterpTraceI0OffsetKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpTraceI0OffsetKey() override = default;

    InterpTraceI0OffsetKey(const unsigned int block_idx)
        : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpTraceI0OffsetKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class InterpTraceI1Key : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~InterpTraceI1Key() override = default;

    InterpTraceI1Key(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpTraceI1Key");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class InterpTraceI1OffsetKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpTraceI1OffsetKey() override = default;

    InterpTraceI1OffsetKey(const unsigned int block_idx)
        : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpTraceI1OffsetKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class InterpFromTraceI0Key : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~InterpFromTraceI0Key() override = default;

    InterpFromTraceI0Key(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpFromTraceI0Key");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class InterpFromTraceI1Key : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef DNekMatSharedPtr value_type;

    ~InterpFromTraceI1Key() override = default;

    InterpFromTraceI1Key(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpFromTraceI1Key");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class InterpEndPtI0Key : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~InterpEndPtI0Key() override = default;

    InterpEndPtI0Key(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpEndPtI0Key");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class InterpEndPtI0OffsetKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpEndPtI0OffsetKey() override = default;

    InterpEndPtI0OffsetKey(const unsigned int block_idx)
        : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpEndPtI0OffsetKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class InterpEndPtI1Key : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~InterpEndPtI1Key() override = default;

    InterpEndPtI1Key(const unsigned int block_idx) : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpEndPtI1Key");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData>
class InterpEndPtI1OffsetKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef unsigned int value_type;

    ~InterpEndPtI1OffsetKey() override = default;

    InterpEndPtI1OffsetKey(const unsigned int block_idx)
        : m_block_idx(block_idx)
    {
        hash_combine(m_hash, m_block_idx, typeid(value_type).name(),
                     "InterpEndPtI1OffsetKey");
    }

private:
    unsigned int m_block_idx;
};

template <typename TData> class Interp1DKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~Interp1DKey() override = default;

    Interp1DKey(const LibUtilities::PointsKey fromKey,
                const LibUtilities::PointsKey toKey)
        : m_fromKey(fromKey), m_toKey(toKey)
    {
        hash_combine(m_hash, m_fromKey.GetNumPoints(), m_toKey.GetNumPoints(),
                     typeid(value_type).name(), "Interp1DKey");
    }

private:
    LibUtilities::PointsKey m_fromKey;
    LibUtilities::PointsKey m_toKey;
};

template <typename TData> class Interp2DKey : public LibUtilities::BaseKey
{
    friend class TraceEssentialCreator;

public:
    using creator = TraceEssentialCreator;
    typedef TData value_type;

    ~Interp2DKey() override = default;

    Interp2DKey(const LibUtilities::PointsKey fromKey0,
                const LibUtilities::PointsKey fromKey1,
                const LibUtilities::PointsKey toKey0,
                const LibUtilities::PointsKey toKey1)
        : m_fromKey0(fromKey0), m_fromKey1(fromKey1), m_toKey0(toKey0),
          m_toKey1(toKey1)
    {
        hash_combine(m_hash, m_fromKey0.GetNumPoints(),
                     m_fromKey1.GetNumPoints(), m_toKey0.GetNumPoints(),
                     m_toKey1.GetNumPoints(), typeid(value_type).name(),
                     "Interp2DKey");
    }

private:
    LibUtilities::PointsKey m_fromKey0;
    LibUtilities::PointsKey m_fromKey1;
    LibUtilities::PointsKey m_toKey0;
    LibUtilities::PointsKey m_toKey1;
};

class TraceEssentialCreator : public LibUtilities::DataCreatorClass
{
public:
    ~TraceEssentialCreator() override = default;
    TraceEssentialCreator(const ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const LocTracePhysToElmtMapsKey<TData> &locTracePhysToElmtMapsKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const OrientationMapsKey<TData> &orientationMapsKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<size_t> Create(
        const OrientationMapsOffsetKey<TData> &orientationMapsOffsetKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<size_t> Create(
        const LocToTracePhysOffsetKey<TData> &locToTracePhysOffsetKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<bool> Create(
        const IsLocTraceLeftAdjacentKey<TData> &isLocTraceLeftAdjacentKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpTraceIndexKey<TData> &interpTraceIndexKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpPointsKey<TData> &interpPointsKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpTypesKey<TData> &interpTypesKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const QuadRangeKey<TData> &quadRangeKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<InterpLocTraceToTrace> Create(
        const InterpTraceKey<TData> &interpTraceKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const InterpTraceI0Key<TData> &interpTraceI0Key);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpTraceI0OffsetKey<TData> &interpTraceI0OffsetKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const InterpTraceI1Key<TData> &interpTraceI1Key);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpTraceI1OffsetKey<TData> &interpTraceI1OffsetKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<DNekMatSharedPtr> Create(
        const InterpFromTraceI0Key<TData> &interpFromTraceI0Key);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<DNekMatSharedPtr> Create(
        const InterpFromTraceI1Key<TData> &interpFromTraceI1Key);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const InterpEndPtI0Key<TData> &interpEndPtI0Key);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpEndPtI0OffsetKey<TData> &interpEndPtI0OffsetKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const InterpEndPtI1Key<TData> &interpEndPtI1Key);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<unsigned int> Create(
        const InterpEndPtI1OffsetKey<TData> &interpEndPtI1OffsetKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const Interp1DKey<TData> &interp1DKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const Interp2DKey<TData> &interp2DKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const IPTraceNormalKey<TData> &ipTraceNormalKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const IPTraceScalarKey<TData> &ipTraceScalarKey);

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(
        const IPTraceDerivBaseKey<TData> &ipTraceDerivBaseKey);

    inline static const std::string m_name = "TraceEssentialCreator";

private:
    ExpListSharedPtr m_expansionList;
};

class PermuteKernel
{
public:
    NEK_DEVICE_INLINE PermuteKernel() = default;

    NEK_DEVICE_INLINE PermuteKernel(const unsigned int orient,
                                    const unsigned int nq0,
                                    const unsigned int nq1 = 1)
    {
        InitCoeffs(orient, nq0, nq1);
    }

    NEK_DEVICE_INLINE void InitCoeffs(const unsigned int orient,
                                      const unsigned int nq0,
                                      const unsigned int nq1 = 1)
    {
        m_nq0 = nq0;
        m_nq1 = nq1;
        if (orient >= 5) // 2D face
        {
            if (orient >= 9) // swap if transposed
            {
                m_nq0 = nq1;
                m_nq1 = nq0;
            }
            unsigned int isTransposed =
                (orient - 5) / 4; // 1 if transposed, 0 otherwise
            unsigned int isDir1Bwd =
                ((orient - 5) / 2) % 2;                // 1 for bwd, 0 otherwise
            unsigned int isDir2Bwd = (orient - 5) % 2; // 1 for bwd, 0 otherwise

            m_coeffA = (1 - isTransposed) * (1 - 2 * isDir1Bwd) +
                       isTransposed * nq0 * (1 - 2 * isDir1Bwd);
            m_coeffB = isTransposed * (1 - 2 * isDir2Bwd) +
                       (1 - isTransposed) * nq0 * (1 - 2 * isDir2Bwd);
            m_coeffC = (isDir1Bwd * (nq0 - 1) * nq0 + isDir2Bwd * (nq1 - 1)) *
                           isTransposed +
                       (isDir1Bwd * (nq0 - 1) + isDir2Bwd * (nq1 - 1) * nq0) *
                           (1 - isTransposed);
        }
        else // 1D edge
        {
            if (orient == 3)
            {
                m_coeffA = 1; // result is i
                m_coeffB = 0;
                m_coeffC = 0;
            }
            else if (orient == 4)
            {
                m_coeffA = -1; // result is nq0 - i - 1
                m_coeffB = 0;
                m_coeffC = nq0 - 1;
            }
        }
    }

    NEK_DEVICE_INLINE unsigned int GetLocTraceToTraceIndex(
        const unsigned int &i, const unsigned int &j) const
    {
        return i * m_coeffA + j * m_coeffB + m_coeffC;
    }

    template <typename T>
    NEK_DEVICE_INLINE void PermuteTraceToLocTrace2DKernel(
        const T *in, T *out, unsigned int nvars = 1) const
    {
        for (unsigned int j = 0; j < m_nq1; ++j)
        {
            for (unsigned int i = 0; i < m_nq0; ++i)
            {
                for (unsigned int v = 0; v < nvars; ++v)
                {
                    out[(i + j * m_nq0) * nvars + v] =
                        in[(i * m_coeffA + j * m_coeffB + m_coeffC) * nvars +
                           v];
                }
            }
        }
    }

private:
    unsigned int m_nq0 = 1, m_nq1 = 1;
    int m_coeffA = 0, m_coeffB = 0, m_coeffC = 0;
};

} // namespace Nektar::MultiRegions
