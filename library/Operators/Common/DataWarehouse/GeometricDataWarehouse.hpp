///////////////////////////////////////////////////////////////////////////////
//
// File: GeometricDataWarehouse.hpp
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

#include "Operators/Common/DataWarehouse/NekDataWarehouse.hpp"

namespace Nektar::Operators
{

class GeometricDataCreator;

template <typename TData> class WeightsKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~WeightsKey() override = default;

    WeightsKey(const unsigned int block_idx,
               const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "WeightsKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class JacobianKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianKey() override = default;

    JacobianKey(const unsigned int block_idx,
                const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "JacobianKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class DerivFactorKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~DerivFactorKey() override = default;

    DerivFactorKey(const unsigned int block_idx,
                   const unsigned int interleave_width, const bool transpose)
        : m_block_idx(block_idx), m_interleave_width(interleave_width),
          m_transpose(transpose)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width, m_transpose,
                     typeid(value_type).name(), "DerivFactorKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
    bool m_transpose;
};

template <typename TData> class CoordKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~CoordKey() override = default;

    CoordKey(const unsigned block_idx, const unsigned int interleave_width,
             const bool transpose)
        : m_block_idx(block_idx), m_interleave_width(interleave_width),
          m_transpose(transpose)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width, m_transpose,
                     typeid(value_type).name(), "CoordKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
    bool m_transpose;
};

template <typename TData> class OrientKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef unsigned int value_type;

    ~OrientKey() override = default;

    OrientKey(const unsigned int block_idx, const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "OrientKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class TraceToElmtMapKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef unsigned int value_type;

    ~TraceToElmtMapKey() override = default;

    TraceToElmtMapKey(const unsigned int block_idx,
                      const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "TraceToElmtMapKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class TraceToElmtSignKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef int value_type;

    ~TraceToElmtSignKey() override = default;

    TraceToElmtSignKey(const unsigned int block_idx,
                       const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "TraceToElmtSignKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class InteriorMapKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef unsigned int value_type;

    ~InteriorMapKey() override = default;

    InteriorMapKey(const unsigned int block_idx,
                   const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "InteriorMapKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class JacobianTraceKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianTraceKey() override = default;

    JacobianTraceKey(const unsigned int block_idx,
                     const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "JacobianTraceKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

template <typename TData> class JacobianLocTraceKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianLocTraceKey() override = default;

    JacobianLocTraceKey(const unsigned int block_idx,
                        const unsigned int interleave_width)
        : m_block_idx(block_idx), m_interleave_width(interleave_width)
    {
        hash_combine(m_hash, m_block_idx, m_interleave_width,
                     typeid(value_type).name(), "JacobianLocTraceKey");
    }

private:
    unsigned int m_block_idx;
    unsigned int m_interleave_width;
};

class GeometricDataCreator : public DataCreatorClass
{
public:
    ~GeometricDataCreator() override = default;
    GeometricDataCreator(const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const WeightsKey<TData> &weightsKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const JacobianKey<TData> &jacobianKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const DerivFactorKey<TData> &derivFactorKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const CoordKey<TData> &coordKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<unsigned int> Create(const OrientKey<TData> &orientKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<unsigned int> Create(
        const TraceToElmtMapKey<TData> &traceToElmtMapKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<int> Create(
        const TraceToElmtSignKey<TData> &traceToElmtSignKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<unsigned int> Create(
        const InteriorMapKey<TData> &interiorMapKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const JacobianTraceKey<TData> &jacobianTraceKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(
        const JacobianLocTraceKey<TData> &jacobianLocTraceKey);

    inline static const std::string m_name = "GeometricDataCreator";

private:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
